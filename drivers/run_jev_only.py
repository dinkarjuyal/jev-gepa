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

import os
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", os.path.expanduser("~/adc.json"))
GCP_PROJECT = "intrepid-app-509303-p9"
GCP_LOCATION = "europe-west4"
VERTEX_KWARGS = {"vertex_project": GCP_PROJECT, "vertex_location": GCP_LOCATION}
TASK_MODEL = "vertex_ai/gemini-2.5-flash-lite"  # see run_baseline_only.py for why
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

snap_dirs = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--AlexWortega--openjev/snapshots/*"))
assert snap_dirs, "OpenJev not downloaded"
sys.path.insert(0, snap_dirs[0])
from modeling_openjev import OpenJevCrossEncoder

print("Loading OpenJev...", flush=True)
# v5 checkpoint (2026-09): trained on a larger/harder mixture, higher JevBench
# accuracy (0.814 vs v4's 0.779) and stronger faithfulness-benchmark AUROC
# (RAGTruth 0.932, HaluBench 0.937) than the base qwen3.5-4b-nli checkpoint
# used in rounds 1-4 -- same repo/interface, HF-card-recommended upgrade.
jev = OpenJevCrossEncoder(snap_dirs[0], subfolder="qwen3.5-4b-nli-v5")
print("Loaded (v5 checkpoint).", flush=True)

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

# Real finding from reading actual traces (both arms, both reflection models
# tested -- gpt-oss-120b on Qwen runs AND gemini-2.5-pro): the reflection
# model reliably turns ANY feedback into a 15-35x longer, checklist-style
# instruction (seed=128 chars; every single proposed mutation across every
# run so far was 1,450-4,689 chars) -- and the tiny seed beat every one of
# those elaborations on held-out validation, in both arms, every time this
# was checked. This isn't Jev-specific: it's what this reflection model does
# with any input. Constraining its OUTPUT directly -- not the diagnostic
# input -- is the fix this finding actually points to. Applied identically
# in run_baseline_only.py so this stays a fair, controlled comparison.
CONCISENESS_CONSTRAINT = (
    "\n\nIMPORTANT: Your proposed instruction text must be concise -- at most "
    "2-3 sentences, well under 400 characters total. Do NOT add procedural "
    "checklists, a 'toolbox' of theorems, worked-example templates, or "
    "mandated multi-step verification sections. A short, direct instruction "
    "close in length and style to this one is strongly preferred over an "
    "elaborate one: 'You are a mathematics expert. Solve the given AIME "
    "problem step by step and give your final numeric answer prefixed with "
    "\"### \".' Do not pad the instructions with extra structure."
)


def reflection_lm_fn(prompt: str) -> str:
    def call():
        return litellm.completion(
            model=REFLECTION_MODEL,
            messages=[{"role": "user", "content": prompt + CONCISENESS_CONSTRAINT}], timeout=240,
            max_tokens=16000, **VERTEX_KWARGS,
        )
    resp = _with_hard_timeout(call)
    return (resp.choices[0].message.content or "") if resp else ""


adapter = JevAIMEAdapter(TASK_MODEL, VERTEX_KWARGS, jev, QUESTIONS)

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

with open(Path.home() / "gepa_jev_aime_results_jev_v5.json", "w") as f:
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

with open(Path.home() / "gepa_jev_aime_examples_jev_v5.json", "w") as f:
    json.dump(examples, f, indent=2)

print("\n\nDONE. Saved to ~/gepa_jev_aime_results_jev_v5.json and ~/gepa_jev_aime_examples_jev_v5.json", flush=True)
