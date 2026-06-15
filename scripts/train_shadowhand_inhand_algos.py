"""Train in-hand manipulation tasks with the SB3 parallel training stack.

This entry point intentionally keeps ShadowHand/Adroit in-hand experiments
separate from MyoChallenge muscle-hand experiments.  It reuses the generic
parallel algorithm implementation from train_parallel_algos.py and adds a small
task-family guard so configs do not silently switch robot morphology.
"""

import argparse
from pathlib import Path

from train_parallel_algos import load_config, train


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "pen" / "train_pen_ppo_parallel.json"

SHADOWHAND_INHAND_ENVS = {
    "pen-v1": "Adroit ShadowHand pen reorientation",
    "baoding-v1": "Adroit ShadowHand baoding balls",
    "baoding4th-v1": "Adroit ShadowHand baoding balls, 4 shifts per period",
    "baoding8th-v1": "Adroit ShadowHand baoding balls, 8 shifts per period",
}

MYOCHALLENGE_DIE_REORIENT_ENVS = {
    "myoChallengeDieReorientDemo-v0": "MyoChallenge muscle-hand die reorientation demo",
    "myoChallengeDieReorientP1-v0": "MyoChallenge muscle-hand die reorientation phase 1",
    "myoChallengeDieReorientP2-v0": "MyoChallenge muscle-hand die reorientation phase 2",
}


def print_known_tasks():
    print("ShadowHand in-hand tasks:")
    for env_id, desc in SHADOWHAND_INHAND_ENVS.items():
        print(f"  {env_id}: {desc}")

    print("\nMyoChallenge die reorient tasks, not ShadowHand:")
    for env_id, desc in MYOCHALLENGE_DIE_REORIENT_ENVS.items():
        print(f"  {env_id}: {desc}")


def validate_task_family(config):
    env_id = config.get("env_id")
    if env_id in SHADOWHAND_INHAND_ENVS:
        print(f"task_family=shadowhand_inhand")
        print(f"task_description={SHADOWHAND_INHAND_ENVS[env_id]}")
        return

    if env_id in MYOCHALLENGE_DIE_REORIENT_ENVS:
        if bool(config.get("allow_myochallenge", False)):
            print("task_family=myochallenge_reference")
            print(f"task_description={MYOCHALLENGE_DIE_REORIENT_ENVS[env_id]}")
            print("warning=This is not a ShadowHand task; use it only as a reference run.")
            return

        raise ValueError(
            f"{env_id} is a MyoChallenge muscle-hand task, not ShadowHand. "
            "Use pen-v1 for the current ShadowHand in-hand reorientation line, "
            "or set allow_myochallenge=true in the config for an explicit reference run."
        )

    supported = sorted(SHADOWHAND_INHAND_ENVS)
    myo_reference = sorted(MYOCHALLENGE_DIE_REORIENT_ENVS)
    raise ValueError(
        f"Unsupported env_id for this entry point: {env_id}. "
        f"ShadowHand supported: {supported}. "
        f"MyoChallenge reference ids: {myo_reference}."
    )


def resolve_project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to a ShadowHand in-hand training JSON config.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print known ShadowHand and MyoChallenge reference task ids.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_tasks:
        print_known_tasks()
        return

    config_path = resolve_project_path(args.config)
    config = load_config(config_path)
    print(f"config={config_path}")
    validate_task_family(config)
    train(config)


if __name__ == "__main__":
    main()
