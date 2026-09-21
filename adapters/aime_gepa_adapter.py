"""GEPAAdapter for GEPA's own bundled AIME benchmark (real problems from
AI-MO/aimo-validation-aime + MathArena/aime_2025 -- one of GEPA's own paper
tasks, chosen specifically to be fast: one LLM call per rollout, no
environment, no Docker). Reasoning is chunked into sentences so Jev has
per-chunk text to score, same mechanism as the trace-based adapters, applied
to a single chain-of-thought instead of a multi-turn agent trace.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import TypedDict

import litellm
from gepa.core.adapter import EvaluationBatch, GEPAAdapter

# Real bug found live, twice, burning real budget both times: litellm's own
# `timeout=` kwarg did NOT reliably fire -- two separate calls hung for
# 15-73+ minutes with zero CPU activity (confirmed via `ps -o etimes,time`:
# wall time climbing, process time frozen), well past the 270s timeout that
# was supposedly set. Every real successful call in this whole project has
# taken 57-90s; a hard, EXTERNALLY-enforced timeout via a worker thread
# (which the caller can walk away from even if the thread itself never
# returns) is the actual fix -- it can't be silently bypassed by whatever
# layer (httpx connection pool, DNS, the remote server itself) was eating
# litellm's own timeout.
#
# SECOND real bug found live, burning more budget on top of the above: the
# first version of this fix used one SHARED, bounded ThreadPoolExecutor
# (max_workers=8). The comment below used to claim an abandoned thread
# "just wastes one worker slot" was harmless -- it is not. Every timed-out
# call permanently occupies a slot forever (the abandoned thread is never
# awaited or killed), so after ~8 real hangs the entire pool is starved:
# every NEW call then has to wait the full timeout just to get a worker
# that's never free, then gives up anyway -- which looks identical to a
# hang from the outside (frozen CPU time, no progress) even though the
# timeout mechanism is technically "working". Fixed by giving every call
# its OWN single-use, single-worker executor instead of sharing one pool --
# an abandoned thread then only ever wastes its own disposable executor,
# never blocking any other call.


def _with_hard_timeout(fn, timeout_s: float = 150.0, retries: int = 1, rate_limit_retries: int = 15):
    # THIRD real bug found live: switching to Vertex AI (gemini-2.5-flash-lite),
    # both concurrent arms hit a real, uncaught litellm.RateLimitError (HTTP
    # 429 RESOURCE_EXHAUSTED) almost immediately and the whole script crashed
    # -- this function only ever handled TIMEOUTS, never retried on a real
    # exception raised promptly by the call itself. Rate limits are ordinary
    # and expected, especially running two arms concurrently against a fresh
    # project's quota. Kept as a SEPARATE retry budget from the timeout one:
    # a rate-limit retry only costs a sleep (cheap, worth retrying many
    # times with backoff), whereas a timeout retry re-attempts a call that
    # may genuinely be stuck (expensive, kept bounded for budget safety).
    backoff = 5.0
    rl_attempt = 0
    while True:
        for attempt in range(retries + 1):
            executor = ThreadPoolExecutor(max_workers=1)  # single-use, never shared
            fut = executor.submit(fn)
            try:
                return fut.result(timeout=timeout_s)
            except FutureTimeoutError:
                if attempt == retries:
                    return None  # give up; caller treats this as an empty/failed response
                # this call's executor (and its one abandoned thread, if the call
                # ever does return) is simply dropped here -- it cannot starve any
                # future call, since every call gets a fresh executor.
            except litellm.RateLimitError:
                if rl_attempt >= rate_limit_retries:
                    raise  # out of rate-limit retries -- let it surface for real
                rl_attempt += 1
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                break  # restart the outer while loop with a fresh timeout-attempt count
            finally:
                executor.shutdown(wait=False)


class AIMEDataInst(TypedDict):
    input: str
    additional_context: dict
    answer: str


class AIMETrajectory(TypedDict):
    data: AIMEDataInst
    response: str
    chunks: list[str]
    correct: bool


class AIMERolloutOutput(TypedDict):
    response: str
    correct: bool


def _extract_int(text: str) -> int | None:
    # AIME answers are 0-999, dataset stores them zero-padded ("### 073").
    # Real bug found live: a correct, unpadded "### 73" from the model was
    # being scored wrong by a literal "### 073" in response substring check
    # -- compare as integers instead, using the LAST "### <n>" match (the
    # final-answer line per the seed prompt's own instructions).
    matches = re.findall(r"###\s*(\d+)", text)
    if matches:
        return int(matches[-1])
    # Real second bug found live: Gemini 2.5 Flash-Lite sometimes ignores the
    # requested "### " format and writes \boxed{95} instead (real example: a
    # correct answer that the seed prompt's own instructions never mentioned
    # \boxed{} for). Fall back to it rather than silently score a correct
    # answer as a miss purely because of format drift.
    boxed = re.findall(r"\\boxed\{(\d+)\}", text)
    return int(boxed[-1]) if boxed else None


def _chunk(text: str, cap: int = 24) -> list[dict]:
    # split into rough reasoning "steps" (sentences/short paragraphs) for
    # per-chunk Jev scoring -- same idea as per-turn scoring on agent traces,
    # applied to a single chain-of-thought. Returns {"text": <chunk for
    # display>, "premise": <chunk prefixed with ~200 chars of the PRECEDING
    # chunk, for scoring>}.
    #
    # Real bug found live, offline, for free (no GPU spend): scoring each
    # chunk bare/context-free left most of the diagnostic bank near-dead --
    # e.g. made_concrete_progress never exceeded 0.29 confidence across 720
    # real chunks, commits_to_final_answer exceeded 0.5 in only 0.1% of
    # cases -- because those are PRAGMATIC/functional claims ("commits to",
    # "verifies", "abandons") that need to know what came before to judge,
    # unlike the one strong tag (made_arithmetic_step, a literal, self-
    # contained, content-checkable claim, which never needed context).
    # Validated fix (720-item A/B on real saved chunks, same GPU, ~5 min):
    # prefixing ~200 chars of the preceding chunk as premise context nearly
    # doubled the overall >0.5-confidence rate (5.5%->8.3%) and helped every
    # pragmatic tag substantially (e.g. commits_to_final_answer's max jumped
    # 0.62->0.93); the two tags that stayed weak regardless
    # (made_concrete_progress, restates_problem) are weak in the statement
    # itself, not from missing context.
    #
    # Earlier bug found live: a reasoning model's full response (content +
    # reasoning_content) runs up to ~15-17K chars -- a naive [:12] truncation
    # only ever sampled the opening ~10-15% of the trace, meaning Jev's tags
    # for "verified_own_work" and "commits_to_final_answer" -- both
    # necessarily LATE-trace behaviors -- could almost never fire. Fixed by
    # evenly sampling across the whole trace instead of always taking the
    # prefix, so the diagnostic tags actually cover beginning/middle/end.
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text) if len(p.strip()) > 15]
    if len(parts) <= cap:
        idxs = list(range(len(parts)))
    else:
        idxs = list(dict.fromkeys(round(i * (len(parts) - 1) / (cap - 1)) for i in range(cap)))
    out = []
    for i in idxs:
        prev = parts[i - 1] if i > 0 else ""
        premise = (prev[-200:] + " " + parts[i]).strip() if prev else parts[i]
        out.append({"text": parts[i], "premise": premise})
    return out


def _solve(model: str, litellm_kwargs: dict, system_prompt: str, problem: str, max_tokens: int = 20000) -> str:
    # Real finding: standard Gemini pay-as-you-go access on Vertex AI uses
    # Dynamic Shared Quota (DSQ), not a fixed per-project number -- confirmed
    # by searching the actual quota catalog for "gemini-2.5-flash-lite" and
    # finding no explicit, raisable entry (only the unrelated -tts variant
    # has one). 429s under DSQ reflect real-time shared-capacity contention,
    # not something a quota-increase request fixes. Reducing max_tokens from
    # the earlier 32000 to 20000 (real observed usage tops out ~14-16K even
    # on hard problems) eases output-token pressure without real truncation
    # risk, on top of the retry/backoff already in _with_hard_timeout.
    def call():
        return litellm.completion(
            model=model,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": problem}],
            timeout=270,
            # root cause of 3 straight saturated-at-zero AIME runs, confirmed by
            # direct inspection: Qwen3.5-4B is a REASONING model -- it was solving
            # problems correctly (verified: n_4=73 for the real answer 073) inside
            # reasoning_content, but 1500/6000 max_tokens cut generation off via
            # finish_reason="length" before it ever reached the formatted "### "
            # answer in `content`, which came back None every time.
            #
            # Same class of bug re-confirmed live on Gemini 2.5 Flash via Vertex
            # AI: its hidden "thinking" tokens ate 7681 of an 8000 budget, same
            # empty-content-despite-real-progress symptom. Gemini 2.5 Flash-Lite
            # has no hidden thinking budget (reasoning_tokens always None) but is
            # simply verbose in VISIBLE text on hard problems -- confirmed a real
            # hard AIME problem converges correctly by ~14K tokens, so 32000 is
            # generous headroom, not a guess.
            max_tokens=max_tokens,
            **litellm_kwargs,
        )

    resp = _with_hard_timeout(call)
    if resp is None:
        return ""  # hard-timed-out twice; treated as a real (empty, incorrect) response
    msg = resp.choices[0].message
    content = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", None) or ""
    # fall back to reasoning text so a real answer that never made it into the
    # formatted `content` field (still cut short) isn't scored as a miss when
    # it was actually reached -- append rather than replace so a well-formed
    # `content` answer is still found first/normally.
    return (content + "\n" + reasoning) if reasoning else content


class AIMEAdapter(GEPAAdapter[AIMEDataInst, AIMETrajectory, AIMERolloutOutput]):
    def __init__(self, model: str, litellm_kwargs: dict):
        self.model, self.litellm_kwargs = model, litellm_kwargs

    def evaluate(self, batch, candidate, capture_traces=False):
        system_prompt = next(iter(candidate.values()))
        outputs, scores, trajectories = [], [], [] if capture_traces else None
        for data in batch:
            response = _solve(self.model, self.litellm_kwargs, system_prompt, data["input"])
            expected = _extract_int(data["answer"])
            got = _extract_int(response)
            correct = expected is not None and got is not None and expected == got
            score = 1.0 if correct else 0.0
            outputs.append({"response": response, "correct": correct})
            scores.append(score)
            if capture_traces:
                trajectories.append({"data": data, "response": response,
                                      "chunks": _chunk(response), "correct": correct})
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        component = components_to_update[0]
        records = []
        for traj in eval_batch.trajectories or []:
            feedback = traj["response"][:1500]
            correct_str = "CORRECT" if traj["correct"] else "INCORRECT"
            additional = "\n".join(f"{k}: {v}" for k, v in traj["data"]["additional_context"].items())
            records.append({
                "Inputs": traj["data"]["input"][:500],
                "Generated Outputs": feedback,
                "Feedback": f"The answer was {correct_str}. Expected answer: {traj['data']['answer']}. "
                            f"{('Reference solution: ' + additional) if additional else ''}",
            })
        return {component: records}


class JevAIMEAdapter(AIMEAdapter):
    def __init__(self, model: str, litellm_kwargs: dict, jev, questions: dict[str, str]):
        super().__init__(model, litellm_kwargs)
        self.jev, self.questions = jev, questions

    # Real dilution hypothesis found live, for free (offline analysis of
    # already-collected tags from the n=30 run): 76.5% of all tag values
    # across 720 real chunks are near-zero (<0.15), and the unfiltered
    # diagnostic-tags block added ~6,183 chars of mostly-noise text per
    # example to what the reflection model has to read. Thresholding to
    # only report tags >=0.2, and skipping chunks where nothing clears that
    # bar, cut the block to ~1,797 chars (-71%) while keeping 80.8% of
    # chunks (those with real signal) -- testing whether the RAW ADDITION
    # of near-zero tags was itself hurting reflection by drowning out the
    # few genuinely informative ones, not helping it.
    TAG_THRESHOLD = 0.2

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        component = components_to_update[0]
        records = []
        for traj in eval_batch.trajectories or []:
            correct_str = "CORRECT" if traj["correct"] else "INCORRECT"
            tag_lines = []
            for chunk in traj["chunks"]:
                # score with context-augmented premise (validated: nearly
                # doubles confident-tag rate vs. the bare chunk alone), but
                # display just the chunk's own text for readability.
                pairs = [(chunk["premise"], stmt) for stmt in self.questions.values()]
                preds = self.jev.predict(pairs)
                strong = [(name, float(p[1])) for name, p in zip(self.questions, preds)
                          if float(p[1]) >= self.TAG_THRESHOLD]
                if not strong:
                    continue  # no informative tag on this chunk -- omit it entirely
                tags = ", ".join(f"{name}={v:.2f}" for name, v in strong)
                tag_lines.append(f"[{chunk['text'][:150]}] -> {tags}")
            additional = "\n".join(f"{k}: {v}" for k, v in traj["data"]["additional_context"].items())
            tag_block = ("\n".join(tag_lines) if tag_lines
                         else "(no chunk cleared the diagnostic-tag confidence threshold)")
            records.append({
                "Inputs": traj["data"]["input"][:500],
                "Generated Outputs": traj["response"][:1000],
                "Feedback": f"The answer was {correct_str}. Expected answer: {traj['data']['answer']}. "
                            f"{('Reference solution: ' + additional) if additional else ''}\n\n"
                            f"Per-step diagnostic tags (only shown where confidence >= {self.TAG_THRESHOLD}):\n" + tag_block,
            })
        return {component: records}
