"""Build deterministic RPent MT50 and ML45 task pools from native resets."""

import argparse
import json
from importlib.resources import files
from pathlib import Path

from prepare_tasks import parse_seeds


def write_pools(output_dir: Path, seeds: list[int]) -> None:
    import metaworld
    from robots.metaworld.initial_state import initial_scene

    if not seeds or any(not 0 <= seed < 50 for seed in seeds):
        raise ValueError("MetaWorld task indices must be in [0, 49]")
    config = json.loads(
        (files("rlinf") / "envs/sim/metaworld/metaworld_config.json").read_text()
    )
    descriptions = config["TASK_DESCRIPTIONS"]
    pools = {
        "mt50": sorted(descriptions),
        "ml45-train": sorted(config["ML45"]["train"]),
        "ml45-ood": sorted(config["ML45"]["test"]),
    }
    rows = {}
    for name in pools["mt50"]:
        benchmark = metaworld.MT1(name, seed=0)
        rows[name] = []
        for seed in seeds:
            env, identity = initial_scene(name, seed, benchmark)
            env.close()
            task = {"environment": "metaworld", "task_name": name, "seed": seed, "init_state_id": seed}
            rows[name].append({
                "query_id": f"metaworld-{name}-s{seed}",
                "prompt": descriptions[name],
                "answer": json.dumps(task, sort_keys=True),
                "initial_state": identity,
            })
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, names in pools.items():
        data = "".join(json.dumps(row, sort_keys=True) + "\n" for name in names for row in rows[name])
        (output_dir / f"{split}.jsonl").write_text(data, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="49", help="Native task indices or inclusive ranges")
    args = parser.parse_args()
    write_pools(args.output_dir, parse_seeds(args.seeds))
