"""GEPAAdapter for the local debug-task environment (debug_tasks.py).

Purpose-built to make a REAL multi-iteration GEPA comparison affordable
today: each rollout is a few LLM calls + microsecond-scale local code
execution -- no Docker, no cloud sandbox, so dozens of GEPA iterations cost
cents and minutes instead of the dollars and hours Terminal-Bench trials need.

The GEPA-optimizable component is the agent's SYSTEM INSTRUCTIONS (candidate
text). Two adapter variants share the same evaluate() (same rollouts, same
scores) but differ in make_reflective_dataset():
  - DebugAdapter: baseline, feeds GEPA raw trace text (what vanilla GEPA gets).
  - JevDebugAdapter: same trace, PLUS Jev-scored diagnostic tags per step,
    to test whether that extra signal changes what GEPA proposes / how fast.
"""
from __future__ import annotations

import re
from typing import Any, TypedDict

import litellm
from gepa.core.adapter import EvaluationBatch, GEPAAdapter

from debug_tasks import _run_tests

MAX_TURNS = 4  # small on purpose -- forces real difficulty/variance, keeps cost down


class DebugDataInst(TypedDict):
    name: str
    buggy_code: str
    test_code: str
    instruction: str


class Step(TypedDict):
    turn: int
    agent_message: str
    proposed_code: str | None
    test_result: str
    score: float


class DebugTrajectory(TypedDict):
    data: DebugDataInst
    steps: list[Step]
    final_score: float


class DebugRolloutOutput(TypedDict):
    final_score: float
    n_turns: int


def _extract_code(text: str) -> str | None:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else None


def _run_agent_loop(model: str, api_base: str, api_key: str, system_prompt: str,
                     task: DebugDataInst) -> list[Step]:
    steps: list[Step] = []
    current_code = task["buggy_code"]
    score, msg = _run_tests(current_code, task["test_code"])

    for turn in range(1, MAX_TURNS + 1):
        user_msg = (
            f"Task: {task['instruction']}\n\n"
            f"Current code:\n```python\n{current_code}\n```\n\n"
            f"Test result: {msg}\n\n"
            "Reply with your analysis, then the corrected full function in a "
            "```python code block. If tests already pass, say so and repeat the "
            "current code."
        )
        try:
            resp = litellm.completion(
                model=model, api_base=api_base, api_key=api_key,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_msg}],
                timeout=60,
            )
            reply = resp.choices[0].message.content or ""
        except Exception as e:
            steps.append({"turn": turn, "agent_message": f"[LLM error: {e}]",
                          "proposed_code": None, "test_result": msg, "score": score})
            break

        proposed = _extract_code(reply)
        if proposed:
            current_code = proposed
            score, msg = _run_tests(current_code, task["test_code"])
        steps.append({"turn": turn, "agent_message": reply,
                      "proposed_code": proposed, "test_result": msg, "score": score})
        if score == 1.0:
            break

    return steps


class DebugAdapter(GEPAAdapter[DebugDataInst, DebugTrajectory, DebugRolloutOutput]):
    """Baseline: GEPA's reflection sees only the raw agent trace text."""

    def __init__(self, model: str, api_base: str, api_key: str):
        self.model, self.api_base, self.api_key = model, api_base, api_key

    def evaluate(self, batch, candidate, capture_traces=False):
        system_prompt = next(iter(candidate.values()))
        outputs, scores, trajectories = [], [], [] if capture_traces else None
        for data in batch:
            steps = _run_agent_loop(self.model, self.api_base, self.api_key, system_prompt, data)
            final_score = steps[-1]["score"] if steps else 0.0
            outputs.append({"final_score": final_score, "n_turns": len(steps)})
            scores.append(final_score)
            if capture_traces:
                trajectories.append({"data": data, "steps": steps, "final_score": final_score})
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        component = components_to_update[0]
        records = []
        for traj in eval_batch.trajectories or []:
            step_text = "\n\n".join(
                f"[turn {s['turn']}] {s['agent_message'][:500]}\n-> test result: {s['test_result']}"
                for s in traj["steps"]
            )
            records.append({
                "Inputs": traj["data"]["instruction"],
                "Generated Outputs": f"final_score={traj['final_score']}",
                "Feedback": step_text,
            })
        return {component: records}


class JevDebugAdapter(DebugAdapter):
    """Same rollouts as DebugAdapter, but make_reflective_dataset() enriches
    the feedback with Jev-scored diagnostic tags per step -- the actual test
    of whether cheap NLI-scored signal changes what GEPA proposes.
    """

    def __init__(self, model: str, api_base: str, api_key: str, jev, questions: dict[str, str]):
        super().__init__(model, api_base, api_key)
        self.jev = jev
        self.questions = questions  # name -> statement

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        component = components_to_update[0]
        records = []
        for traj in eval_batch.trajectories or []:
            lines = []
            for s in traj["steps"]:
                premise = s["agent_message"][:600]
                pairs = [(premise, stmt) for stmt in self.questions.values()]
                preds = self.jev.predict(pairs)
                tags = ", ".join(
                    f"{name}={float(p[1]):.2f}" for name, p in zip(self.questions, preds)
                )
                lines.append(
                    f"[turn {s['turn']}] {s['agent_message'][:500]}\n"
                    f"-> test result: {s['test_result']}\n"
                    f"-> diagnostic tags: {tags}"
                )
            records.append({
                "Inputs": traj["data"]["instruction"],
                "Generated Outputs": f"final_score={traj['final_score']}",
                "Feedback": "\n\n".join(lines),
            })
        return {component: records}
