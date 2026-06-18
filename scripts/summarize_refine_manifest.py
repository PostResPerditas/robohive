import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: str) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_csv(path: Optional[Path], rows):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["object_name", "mode", "count", "indices"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"csv={path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mode", default=None, choices=["tabletop", "unconstrained", "all"])
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    manifest_path = resolve_project_path(args.manifest)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    entries = manifest.get("entries", [])
    if args.mode and args.mode != "all":
        entries = [entry for entry in entries if entry.get("mode") == args.mode]

    by_key = defaultdict(list)
    for entry in entries:
        key = (entry.get("object_name", ""), entry.get("mode", ""))
        by_key[key].append(int(entry.get("index", len(by_key[key]))))

    rows = [
        {
            "object_name": object_name,
            "mode": mode,
            "count": len(indices),
            "indices": " ".join(str(index) for index in sorted(indices)),
        }
        for (object_name, mode), indices in by_key.items()
    ]
    rows.sort(key=lambda row: (-int(row["count"]), row["mode"], row["object_name"]))

    histogram = Counter(int(row["count"]) for row in rows)
    print(f"manifest={manifest_path}")
    print(f"entries={len(entries)}")
    print(f"object_mode_groups={len(rows)}")
    print(f"objects_ge2={sum(int(row['count']) >= 2 for row in rows)}")
    print(f"objects_ge3={sum(int(row['count']) >= 3 for row in rows)}")
    print(f"objects_ge5={sum(int(row['count']) >= 5 for row in rows)}")
    print(f"histogram={dict(sorted(histogram.items()))}")
    print("top_counts")
    for row in rows[: max(args.top, 0)]:
        print(f"{row['count']} {row['mode']} {row['object_name']} indices={row['indices']}")

    write_csv(resolve_project_path(args.csv) if args.csv else None, rows)


if __name__ == "__main__":
    main()
