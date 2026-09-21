"""The actual experiment: does feeding GEPA's reflection step Jev-scored
diagnostic tags (instead of just raw trace text) produce a better-optimized
agent instruction, at equal rollout budget, on the local debug-task env?

Two gepa.optimize() runs, same seed candidate, same trainset/valset, same
task LM, same reflection LM, same max_metric_calls -- the ONLY difference is
whether make_reflective_dataset() includes Jev tags. Runs entirely on this
pod (local debug-task execution + local Jev scoring + remote LLM calls via
Prime Intellect).
"""
import json
import sys
import glob
from pathlib import Path

sys.path.insert(0, str(Path.home() / "debug_exp"))
import gepa
from debug_tasks import TASKS, verify_all_tasks
from debug_gepa_adapter import DebugAdapter, JevDebugAdapter

verify_all_tasks()

PIT_KEY = json.load(open(Path.home() / "pit_config.json"))["api_key"]
PIT_BASE = "https://api.pinference.ai/api/v1"
TASK_MODEL = "openai/Qwen/Qwen3.5-4B"        # deliberately weaker -- needs to leave real room to fail
REFLECTION_MODEL = "openai/openai/gpt-oss-120b"  # needs to reason well about traces

# train/val split -- small on purpose (n=6 real tasks total), honestly a pilot
trainset = TASKS[:4]
valset = TASKS[4:]

seed_candidate = {
    "instructions": "You are a Python debugging assistant. Fix the bug in the given function."
}

# --- Jev setup (shared by the Jev-enriched run) ---
snap_dirs = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--AlexWortega--openjev/snapshots/*"))
assert snap_dirs, "OpenJev not downloaded"
MODEL_PATH = snap_dirs[0]
sys.path.insert(0, MODEL_PATH)
from modeling_openjev import OpenJevCrossEncoder

print("Loading OpenJev...")
jev = OpenJevCrossEncoder(MODEL_PATH, subfolder="qwen3.5-4b-nli")
print("Loaded.")

# the 4 questions that showed real pass/fail signal on the real TB2.1 batch,
# plus the general (non-task-specific) subset of the model-generated ones
QUESTIONS = {
    "still_exploring": "The agent is still exploring and gathering information rather than writing the final solution.",
    "made_concrete_progress": "The agent made concrete progress toward completing the task in this step.",
    "encountered_missing_dependency": "The agent discovered that an expected library or tool was not available in the environment.",
    "wrote_final_solution_file": "The agent wrote or edited the actual solution file the task asked for.",
    "missing_dependency_report": "The step reports that a required dependency or tool is missing.",
    "proposes_install_missing_dependency": "The step proposes installing the missing dependency or tool.",
    "waits_for_long_running_process": "The step indicates that a long-running process is currently executing and the agent decides to wait.",
    "searches_filesystem_for_clues": "The step describes searching the filesystem for clues or hint files.",
    "confirms_successful_command_execution": "The step confirms that a previously issued command completed successfully.",
    "provides_concrete_command_without_speculation": "The step provides a specific shell command to run without additional speculation.",
}

MAX_METRIC_CALLS = 24  # equal, small budget for both arms -- a real pilot, not a full study

results = {}

def reflection_lm_fn(prompt: str) -> str:
    # gepa's reflection_lm needs api_base/key threaded via a callable, not a bare
    # string -- a plain "provider/model" string routes through litellm's defaults,
    # which don't know about a custom api_base.
    import litellm
    resp = litellm.completion(
        model=REFLECTION_MODEL, api_base=PIT_BASE, api_key=PIT_KEY,
        messages=[{"role": "user", "content": prompt}], timeout=120,
    )
    return resp.choices[0].message.content or ""


for label, adapter in [
    ("baseline", DebugAdapter(TASK_MODEL, PIT_BASE, PIT_KEY)),
    ("jev_enriched", JevDebugAdapter(TASK_MODEL, PIT_BASE, PIT_KEY, jev, QUESTIONS)),
]:
    print(f"\n{'='*70}\nRunning GEPA: {label}\n{'='*70}", flush=True)
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
        "val_scores_over_time": getattr(result, "val_aggregate_scores", None),
    }
    print(f"\n{label} best candidate:", result.best_candidate, flush=True)
    print(f"{label} val scores over time:", getattr(result, "val_aggregate_scores", None), flush=True)

with open(Path.home() / "gepa_jev_experiment_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n\nDONE. Saved to ~/gepa_jev_experiment_results.json", flush=True)
