"""GEPAAdapter for real, public ALFWorld -- chosen specifically because this
project already has a proven, working recipe here (zero-shot success 0.07,
with-recipe-prompt success 0.50, from the earlier Step 1b probe), meaning
there's real, already-demonstrated headroom for instruction optimization to
matter, and no Docker/sandbox needed (pure text env, fast per-step).

The GEPA-optimizable component is the agent's SYSTEM instructions (candidate
text) -- the same slot the SYS recipe prompt in alf_common.py fills. Reuses
the same 10 diagnostic questions from the Terminal-Bench run, UNCHANGED, so
this run doubles as a real out-of-domain generalization test of those
questions (coding tasks -> embodied household tasks), not a re-fit.
"""
from __future__ import annotations

import re
from typing import TypedDict

import litellm
from gepa.core.adapter import EvaluationBatch, GEPAAdapter

MAX_STEPS = 15  # capped low -- each GEPA "rollout" here is a FULL episode
                 # (up to this many LLM calls), so cost scales with this


class ALFDataInst(TypedDict):
    game_idx: int


class ALFStep(TypedDict):
    turn: int
    action: str
    observation: str
    agent_message: str


class ALFTrajectory(TypedDict):
    data: ALFDataInst
    task: str
    steps: list[ALFStep]
    won: bool


class ALFRolloutOutput(TypedDict):
    won: bool
    n_turns: int


def _parse_action(text: str, adm: list[str]) -> str:
    m = re.search(r"ACTION:\s*(.+)", text)
    cand = (m.group(1) if m else (text.strip().splitlines() or [""])[-1]).strip().strip('"').strip()
    if cand in adm:
        return cand
    low = cand.lower()
    for a in adm:
        if a.lower() == low:
            return a
    for a in adm:
        if a.lower() in low or low in a.lower():
            return a
    return max(adm, key=lambda a: len(set(a.lower().split()) & set(low.split())))


def _build_user_msg(task: str, hist: list[str], obs: str, adm: list[str]) -> str:
    hb = "\n".join(hist[-6:])
    return (f"Task: {task}\n\n{hb}\n\nObservation: {obs[:600]}\n\n"
            f"Admissible commands: {adm}\n\n"
            "Think briefly then end with 'ACTION: <one admissible command>'.")


def _run_episode(make_env, model: str, api_base: str, api_key: str,
                  system_prompt: str, game_idx: int) -> ALFTrajectory:
    env = make_env(game_idx)
    obs, info = env.reset()
    task = obs[0].split("Your task is to:")[-1].strip().split("\n")[0]
    hist: list[str] = []
    steps: list[ALFStep] = []
    won = False

    for turn in range(1, MAX_STEPS + 1):
        adm = info["admissible_commands"][0]
        user_msg = _build_user_msg(task, hist, obs[0], adm)
        try:
            resp = litellm.completion(
                model=model, api_base=api_base, api_key=api_key,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_msg}],
                timeout=60,
            )
            reply = resp.choices[0].message.content or ""
        except Exception as e:
            reply = f"[LLM error: {e}]"

        action = _parse_action(reply, adm)
        hist.append(f"Observation: {obs[0][:200]}\nACTION: {action}")
        steps.append({"turn": turn, "action": action, "observation": obs[0][:300], "agent_message": reply})
        obs, sc, dn, info = env.step([action])
        if info["won"][0] or dn[0]:
            won = bool(info["won"][0])
            break

    return {"data": {"game_idx": game_idx}, "task": task, "steps": steps, "won": won}


class ALFWorldAdapter(GEPAAdapter[ALFDataInst, ALFTrajectory, ALFRolloutOutput]):
    def __init__(self, make_env, model: str, api_base: str, api_key: str):
        self.make_env, self.model, self.api_base, self.api_key = make_env, model, api_base, api_key

    def evaluate(self, batch, candidate, capture_traces=False):
        system_prompt = next(iter(candidate.values()))
        outputs, scores, trajectories = [], [], [] if capture_traces else None
        for data in batch:
            traj = _run_episode(self.make_env, self.model, self.api_base, self.api_key,
                                 system_prompt, data["game_idx"])
            score = 1.0 if traj["won"] else 0.0
            outputs.append({"won": traj["won"], "n_turns": len(traj["steps"])})
            scores.append(score)
            if capture_traces:
                trajectories.append(traj)
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        component = components_to_update[0]
        records = []
        for traj in eval_batch.trajectories or []:
            step_text = "\n\n".join(
                f"[turn {s['turn']}] {s['agent_message'][:400]}\n-> action taken: {s['action']}"
                for s in traj["steps"]
            )
            records.append({
                "Inputs": traj["task"],
                "Generated Outputs": f"won={traj['won']}",
                "Feedback": step_text,
            })
        return {component: records}


class JevALFWorldAdapter(ALFWorldAdapter):
    def __init__(self, make_env, model: str, api_base: str, api_key: str, jev, questions: dict[str, str]):
        super().__init__(make_env, model, api_base, api_key)
        self.jev, self.questions = jev, questions

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        component = components_to_update[0]
        records = []
        for traj in eval_batch.trajectories or []:
            lines = []
            for s in traj["steps"]:
                premise = s["agent_message"][:600]
                pairs = [(premise, stmt) for stmt in self.questions.values()]
                preds = self.jev.predict(pairs)
                tags = ", ".join(f"{name}={float(p[1]):.2f}" for name, p in zip(self.questions, preds))
                lines.append(
                    f"[turn {s['turn']}] {s['agent_message'][:400]}\n"
                    f"-> action taken: {s['action']}\n-> diagnostic tags: {tags}"
                )
            records.append({
                "Inputs": traj["task"],
                "Generated Outputs": f"won={traj['won']}",
                "Feedback": "\n\n".join(lines),
            })
        return {component: records}
