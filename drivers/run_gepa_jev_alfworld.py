"""Real GEPA baseline-vs-Jev-enriched comparison on real, public ALFWorld --
the direct fix for using a self-built toy task instead of a public benchmark.
Reuses the SAME 10 diagnostic questions from the Terminal-Bench run unchanged,
so this also tests whether they generalize out-of-domain.
"""
import glob
import json
import os
import random
import sys
from pathlib import Path

import yaml

os.environ.setdefault("ALFWORLD_DATA", os.path.expanduser("~/alfworld_data"))
from alfworld.agents.environment import get_environment

sys.path.insert(0, str(Path.home() / "debug_exp"))
import gepa
from alfworld_gepa_adapter import ALFWorldAdapter, JevALFWorldAdapter
from alf_common import SYS as SEED_SYS_PROMPT

PIT_KEY = json.load(open(Path.home() / "pit_config.json"))["api_key"]
PIT_BASE = "https://api.pinference.ai/api/v1"
TASK_MODEL = "openai/Qwen/Qwen3.5-4B"
REFLECTION_MODEL = "openai/openai/gpt-oss-120b"
MAX_METRIC_CALLS = 12  # each call = one full ALFWorld episode (up to 15 turns) -- kept small

random.seed(0)
cfg = yaml.safe_load(os.path.expandvars(open(os.path.expanduser("~/base_config.yaml")).read()))
obj = get_environment(cfg["env"]["type"])(cfg, train_eval="eval_out_of_distribution")
games = list(obj.game_files)
random.shuffle(games)
print(f"{len(games)} games available")

def make_env(game_idx):
    obj.game_files = [games[game_idx]]
    return obj.init_env(batch_size=1)

trainset = [{"game_idx": i} for i in range(4)]
valset = [{"game_idx": i} for i in range(4, 7)]

seed_candidate = {"instructions": SEED_SYS_PROMPT}

snap_dirs = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--AlexWortega--openjev/snapshots/*"))
assert snap_dirs, "OpenJev not downloaded"
sys.path.insert(0, snap_dirs[0])
from modeling_openjev import OpenJevCrossEncoder

print("Loading OpenJev...")
jev = OpenJevCrossEncoder(snap_dirs[0], subfolder="qwen3.5-4b-nli")
print("Loaded.")

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


def reflection_lm_fn(prompt: str) -> str:
    resp = litellm.completion(
        model=REFLECTION_MODEL, api_base=PIT_BASE, api_key=PIT_KEY,
        messages=[{"role": "user", "content": prompt}], timeout=120,
    )
    return resp.choices[0].message.content or ""


import litellm

results = {}
# baseline already completed for real on the previous pod (val_aggregate_scores=[0.0],
# best_idx=0 -- seed scored 0/3, two proposed mutations both legitimately rejected on
# subsample) -- that pod's GPU then hit a real hardware fault (CUDA uncorrectable ECC
# error) on the jev_enriched arm's very first reflective-dataset build, before it ever
# produced one usable data point. Only re-running jev_enriched here, on fresh hardware,
# same seed/train/val split, so it stays directly comparable to the already-valid baseline.
for label, adapter in [
    ("jev_enriched", JevALFWorldAdapter(make_env, TASK_MODEL, PIT_BASE, PIT_KEY, jev, QUESTIONS)),
]:
    print(f"\n{'='*70}\nRunning GEPA on ALFWorld: {label}\n{'='*70}", flush=True)
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

with open(Path.home() / "gepa_jev_alfworld_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n\nDONE. Saved to ~/gepa_jev_alfworld_results.json", flush=True)
