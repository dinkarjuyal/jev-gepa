"""Fast GEPA baseline-vs-Jev-enriched comparison on GEPA's own bundled AIME
benchmark -- real math problems from AI-MO/aimo-validation-aime, one of
GEPA's own paper tasks. Chosen for speed: single LLM call per rollout, no
environment, no Docker -- should run in minutes, not hours.
"""
import glob
import json
import sys
from pathlib import Path

import litellm

sys.path.insert(0, str(Path.home()))
import gepa
from gepa.examples.aime import init_dataset
from aime_gepa_adapter import AIMEAdapter, JevAIMEAdapter

PIT_KEY = json.load(open(Path.home() / "pit_config.json"))["api_key"]
PIT_BASE = "https://api.pinference.ai/api/v1"
TASK_MODEL = "openai/Qwen/Qwen3.5-4B"  # picked for real speed after a live
                                           # timing test: Qwen3.5-4B took ~240s
                                           # per problem (heavy reasoning mode),
                                           # this one ~60s
REFLECTION_MODEL = "openai/openai/gpt-oss-120b"
MAX_METRIC_CALLS = 80

print("Loading AIME dataset...", flush=True)
trainset, valset, testset = init_dataset()
trainset = trainset[:20]
valset = valset[:15]
print(f"trainset={len(trainset)} valset={len(valset)}", flush=True)

seed_candidate = {
    "instructions": "You are a mathematics expert. Solve the given AIME problem step by step "
                    "and give your final numeric answer prefixed with '### '."
}

snap_dirs = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--AlexWortega--openjev/snapshots/*"))
assert snap_dirs, "OpenJev not downloaded"
sys.path.insert(0, snap_dirs[0])
from modeling_openjev import OpenJevCrossEncoder

print("Loading OpenJev...", flush=True)
jev = OpenJevCrossEncoder(snap_dirs[0], subfolder="qwen3.5-4b-nli")
print("Loaded.", flush=True)

# math-reasoning-appropriate diagnostic questions -- some carried over
# (general enough to apply), some newly written for this domain
QUESTIONS = {
    "made_concrete_progress": "The step makes concrete progress toward solving the problem.",
    "still_exploring": "The step is exploring a possible approach rather than committing to a solution.",
    "expressed_uncertainty": "The step expresses doubt or uncertainty about the current approach.",
    "verified_own_work": "The step checks or verifies a previous calculation or claim.",
    "made_arithmetic_step": "The step performs a concrete arithmetic or algebraic calculation.",
    "restates_problem": "The step merely restates or rephrases the problem without adding new reasoning.",
    "commits_to_final_answer": "The step states or commits to a final numeric answer.",
    "abandons_approach": "The step abandons a previous approach and switches to a different one.",
}


def reflection_lm_fn(prompt: str) -> str:
    resp = litellm.completion(
        model=REFLECTION_MODEL, api_base=PIT_BASE, api_key=PIT_KEY,
        messages=[{"role": "user", "content": prompt}], timeout=240,
    )
    return resp.choices[0].message.content or ""


results = {}
for label, adapter in [
    ("baseline", AIMEAdapter(TASK_MODEL, PIT_BASE, PIT_KEY)),
    ("jev_enriched", JevAIMEAdapter(TASK_MODEL, PIT_BASE, PIT_KEY, jev, QUESTIONS)),
]:
    print(f"\n{'='*70}\nRunning GEPA on AIME: {label}\n{'='*70}", flush=True)
    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm_fn,
        max_metric_calls=MAX_METRIC_CALLS,
        display_progress_bar=True,
        seed=0,
    )
    results[label] = {
        "best_candidate": result.best_candidate,
        "val_aggregate_scores": result.val_aggregate_scores,
        "best_idx": result.best_idx,
    }
    print(f"\n{label} val_aggregate_scores:", result.val_aggregate_scores, flush=True)
    print(f"{label} best_idx:", result.best_idx, flush=True)

with open(Path.home() / "gepa_jev_aime_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n\nDONE. Saved to ~/gepa_jev_aime_results.json", flush=True)
