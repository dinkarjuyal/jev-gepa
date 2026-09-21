"""jev_enriched-only arm, split out so it can run concurrently with the
baseline arm instead of sequentially after it -- same trainset/valset/seed as
run_baseline_only.py so the two remain a valid paired comparison."""
import glob
import json
import sys
from pathlib import Path

import litellm

sys.path.insert(0, str(Path.home()))
import gepa
from gepa.examples.aime import init_dataset
from aime_gepa_adapter import JevAIMEAdapter

PIT_KEY = json.load(open(Path.home() / "pit_config.json"))["api_key"]
PIT_BASE = "https://api.pinference.ai/api/v1"
TASK_MODEL = "openai/Qwen/Qwen3.5-4B"
REFLECTION_MODEL = "openai/openai/gpt-oss-120b"
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

snap_dirs = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--AlexWortega--openjev/snapshots/*"))
assert snap_dirs, "OpenJev not downloaded"
sys.path.insert(0, snap_dirs[0])
from modeling_openjev import OpenJevCrossEncoder

print("Loading OpenJev...", flush=True)
jev = OpenJevCrossEncoder(snap_dirs[0], subfolder="qwen3.5-4b-nli")
print("Loaded.", flush=True)

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


from aime_gepa_adapter import _with_hard_timeout  # noqa: E402  (same hard-timeout fix as _solve())


def reflection_lm_fn(prompt: str) -> str:
    def call():
        return litellm.completion(
            model=REFLECTION_MODEL, api_base=PIT_BASE, api_key=PIT_KEY,
            messages=[{"role": "user", "content": prompt}], timeout=240,
        )
    resp = _with_hard_timeout(call)
    return (resp.choices[0].message.content or "") if resp else ""


adapter = JevAIMEAdapter(TASK_MODEL, PIT_BASE, PIT_KEY, jev, QUESTIONS)

print(f"\n{'='*70}\nRunning GEPA on AIME: jev_enriched\n{'='*70}", flush=True)
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
print("\njev_enriched val_aggregate_scores:", result.val_aggregate_scores, flush=True)
print("jev_enriched best_idx:", result.best_idx, flush=True)

with open(Path.home() / "gepa_jev_aime_results_jev.json", "w") as f:
    json.dump(out, f, indent=2)

# Re-run the best candidate on the full valset with trace capture ON, purely to
# get real, concrete per-example examples (response text, chunks, Jev tags) for
# the paper -- not part of the optimization loop itself, GEPA's own accept/
# reject logic already picked this candidate honestly before this call runs.
print("\nCapturing example traces for the final best candidate...", flush=True)
eval_batch = adapter.evaluate(valset, result.best_candidate, capture_traces=True)
examples = []
for data, output, score, traj in zip(valset, eval_batch.outputs, eval_batch.scores, eval_batch.trajectories):
    tag_lines = []
    for chunk in traj["chunks"]:
        pairs = [(chunk["premise"], stmt) for stmt in QUESTIONS.values()]
        preds = jev.predict(pairs)
        tags = ", ".join(f"{name}={float(p[1]):.2f}" for name, p in zip(QUESTIONS, preds))
        tag_lines.append({"chunk": chunk["text"], "tags": tags})
    examples.append({
        "input": data["input"],
        "answer": data["answer"],
        "correct": bool(output["correct"]),
        "response": output["response"],
        "tagged_chunks": tag_lines,
    })

with open(Path.home() / "gepa_jev_aime_examples_jev.json", "w") as f:
    json.dump(examples, f, indent=2)

print("\n\nDONE. Saved to ~/gepa_jev_aime_results_jev.json and ~/gepa_jev_aime_examples_jev.json", flush=True)
