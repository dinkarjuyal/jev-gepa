"""Baseline-only arm, split out so it can run concurrently with the
jev_enriched arm instead of sequentially after it -- same trainset/valset/seed
as run_jev_only.py so the two remain a valid paired comparison."""
import json
import sys
from pathlib import Path

import litellm

sys.path.insert(0, str(Path.home()))
import gepa
from gepa.examples.aime import init_dataset
from aime_gepa_adapter import AIMEAdapter, _with_hard_timeout

import os
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", os.path.expanduser("~/adc.json"))
GCP_PROJECT = "intrepid-app-509303-p9"
GCP_LOCATION = "europe-west4"
VERTEX_KWARGS = {"vertex_project": GCP_PROJECT, "vertex_location": GCP_LOCATION}
TASK_MODEL = "vertex_ai/gemini-2.5-flash-lite"  # $0.10/$0.40 per 1M -- cheapest
                                                  # real option; no hidden
                                                  # thinking-token budget (unlike
                                                  # gemini-2.5-flash, which burned
                                                  # 7681/8000 tokens on hidden
                                                  # reasoning in a live test)
REFLECTION_MODEL = "vertex_ai/gemini-2.5-pro"
MAX_METRIC_CALLS = 60

print("Loading AIME dataset...", flush=True)
trainset, valset, testset = init_dataset()
trainset = trainset[:20]
valset = valset[:15]
print(f"trainset={len(trainset)} valset={len(valset)}", flush=True)

seed_candidate = {
    "instructions": "You are a mathematics expert. Solve the given AIME problem step by step "
                    "and give your final numeric answer prefixed with '### '."
}


def reflection_lm_fn(prompt: str) -> str:
    def call():
        return litellm.completion(
            model=REFLECTION_MODEL,
            messages=[{"role": "user", "content": prompt}], timeout=240,
            max_tokens=16000,  # gemini-2.5-pro has hidden thinking tokens too;
                               # reflection calls are infrequent so the extra
                               # headroom costs little, but a truncated proposed
                               # edit would be a real, silent failure mode.
            **VERTEX_KWARGS,
        )
    resp = _with_hard_timeout(call)
    return (resp.choices[0].message.content or "") if resp else ""


adapter = AIMEAdapter(TASK_MODEL, VERTEX_KWARGS)

print(f"\n{'='*70}\nRunning GEPA on AIME: baseline\n{'='*70}", flush=True)
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

out = {
    "best_candidate": result.best_candidate,
    "val_aggregate_scores": result.val_aggregate_scores,
    "best_idx": result.best_idx,
}
print("\nbaseline val_aggregate_scores:", result.val_aggregate_scores, flush=True)
print("baseline best_idx:", result.best_idx, flush=True)

with open(Path.home() / "gepa_jev_aime_results_baseline.json", "w") as f:
    json.dump(out, f, indent=2)

# Same example-capture pass as run_jev_only.py, for the paper's trace examples
# and for a direct per-problem baseline-vs-jev comparison on the same valset.
print("\nCapturing example traces for the final best candidate...", flush=True)
eval_batch = adapter.evaluate(valset, result.best_candidate, capture_traces=True)
examples = []
for data, output, score, traj in zip(valset, eval_batch.outputs, eval_batch.scores, eval_batch.trajectories):
    examples.append({
        "input": data["input"],
        "answer": data["answer"],
        "correct": bool(output["correct"]),
        "response": output["response"],
    })

with open(Path.home() / "gepa_jev_aime_examples_baseline.json", "w") as f:
    json.dump(examples, f, indent=2)

print("\n\nDONE. Saved to ~/gepa_jev_aime_results_baseline.json and ~/gepa_jev_aime_examples_baseline.json", flush=True)
