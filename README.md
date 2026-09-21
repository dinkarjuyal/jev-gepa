# Jev + GEPA: fast NLI judges as an in-the-loop training signal

Wiring [Jev](https://huggingface.co/AlexWortega/openjev) (a small, fast, local NLI cross-encoder) into [GEPA](https://arxiv.org/abs/2507.19457) (a real, published reflective prompt optimizer, ICLR 2026 Oral), to test whether cheap per-step diagnostic tags improve GEPA's reflection step over raw trace text alone.

**The idea.** LLM judges are usually too slow/expensive to run inside a training or optimization loop — they show up at evaluation time, after the fact. Jev is small and local enough (~9GB, one GPU, no per-call API cost) to plausibly tag every step of every rollout. This repo tests whether that lets a reflective prompt optimizer's feedback channel carry a real, structured, cheap semantic signal instead of just raw text.

## Layout

```
adapters/   GEPAAdapter implementations (baseline vs. Jev-enriched, per task)
drivers/    scripts that actually run gepa.optimize() for each experiment
results/    real result JSONs and captured example traces from completed runs
docs/       supporting data (e.g. the offline chunk-context validation set)
```

- `adapters/aime_gepa_adapter.py` — AIME (GEPA's own paper benchmark). Contains the two most important pieces of engineering in this project: `_chunk()` (splits a single chain-of-thought response into pseudo-steps for per-chunk Jev scoring, with a validated fix for representative sampling across a full trace and context-augmented premises — see below) and `_with_hard_timeout()` (an externally-enforced call timeout, added after litellm's own `timeout=` parameter failed to fire on two real, separate hangs).
- `adapters/alfworld_gepa_adapter.py` — ALFWorld (embodied agent benchmark; the Jev-enriched arm here was lost to a real GPU hardware fault, documented as an honest negative result).
- `adapters/debug_gepa_adapter.py` + `debug_tasks.py` — the original small pilot task (6 hand-verified buggy-Python-function tasks).
- `drivers/run_baseline_only.py` / `run_jev_only.py` — the current, correct pattern: run each arm of the A/B comparison as an **independent, concurrent process** (not sequential steps in one script), since the two arms are fully independent and running them concurrently roughly halves wall-clock time for a paired comparison.

## What was found, in order

1. **Debug-task pilot**: a real positive result at tiny scale (n=2 val) — a pilot, not a result on its own.
2. **ALFWorld**: baseline arm valid, Jev-enriched arm lost to a real CUDA ECC hardware fault. No comparison obtained.
3. **AIME, small scale**: three consecutive runs saturated at 0.0 everywhere, traced to two real bugs — a reasoning model's `content` field silently truncated by a `max_tokens` cap before it ever reached its formatted answer, and a dataset-answer-format mismatch (`"### 073"` vs a correct-but-unpadded `"### 73"`) defeating naive substring grading. Both fixed.
4. **AIME, full scale (v1)**: baseline appeared to win. Investigating why found a *third* bug — `_chunk()` truncated every response to its first 12 sentences, so tags describing late-trace behavior (verifying work, committing to a final answer) were structurally almost never sampled once real responses ran 15-38K+ characters.
5. **AIME, full scale, chunking fixed**: Jev-enriched won 3x (0.6 vs 0.2) at n=15 validation.
6. **AIME, replication check (n=30)**: the 3x margin did *not* replicate — near-parity, baseline nominally ahead (0.4 vs 0.367) — a real reversal at double the sample size, run specifically to stress-test the n=15 result rather than trust it. Per-problem analysis showed the two arms are still genuinely different (not interchangeable): of 5 disagreements on 30 problems, Jev-enriched was right and baseline wasn't on 4, the reverse on 1.
7. **Offline diagnosis + context-augmentation fix**: analyzing the real captured Jev tags from run 6 (for free, no GPU) found 6 of 8 diagnostic tags were near-dead weight (e.g. `made_concrete_progress` never exceeded 0.29 confidence across 720 real chunks). The pattern: the one strong tag (`made_arithmetic_step`) is a literal, self-contained claim; the weak ones are pragmatic/functional claims that need to know what came *before* a chunk to judge — but each chunk was being scored with zero context. Validated on the same 720 real chunks (bare vs. chunk-plus-preceding-context as premise): nearly doubled the overall confident-tag rate (5.5%→8.3%) and helped every pragmatic tag substantially.
8. **Infra hardening**: two real, separate hangs (73+ minutes and 15+ minutes, both with zero CPU activity) revealed that litellm's own `timeout=` parameter was not reliably firing. Replaced with `_with_hard_timeout()`, an externally-enforced timeout via a worker thread the caller can walk away from regardless of what the underlying call does — verified in isolation before redeploying.

## Reusable engineering lessons

- litellm strips everything before the first `/` in a model ID as a routing prefix — a self-referential ID like `openai/gpt-oss-120b` needs doubling (`openai/openai/gpt-oss-120b`) to survive.
- GEPA's `reflection_lm` needs a plain callable, not a bare model string, to route through a custom `api_base`.
- A reasoning model's `content` field can come back empty under a `max_tokens` cap that looks generous, because the model spends its budget on a separate `reasoning_content` field first.
- Dataset answer strings should never be trusted to match a model's literal output format without checking.
- Chunking a long generated trace for per-step scoring must sample representatively across the whole trace, not truncate to a prefix.
- Pragmatic/functional NLI claims ("commits to", "verifies", "abandons") need surrounding context to score meaningfully; purely literal, content-checkable claims don't.
- litellm's own `timeout=` is not something to trust blindly for hang protection — wrap remote calls in an externally-enforced timeout if a hang would be costly.
- Run independent comparison arms as concurrent processes, not sequential steps in one script.

## Status

Full experimental writeup (papers with all figures, tables, and example traces) lives in Google Docs; ask the repo owner for the current link. This repo has the actual code and result data behind that writeup.
