from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.rir_experiment import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATA_ROOT,
    DEFAULT_DEPTH_ROOT,
    DEFAULT_METADATA_ROOT,
    cap_split_records,
    discover_samples,
    prepare_runtime_environment,
    resolve_model_names,
    select_device,
    set_seed,
    split_records_by_room,
    summarize_split_sizes,
    train_model,
    validate_depth_maps_available,
    validate_uniform_rir_shape,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RIR models and save checkpoints.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--depth-root", type=Path, default=DEFAULT_DEPTH_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--target-length", type=int, default=22050)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overall_start = time.perf_counter()
    prepare_runtime_environment()
    set_seed(args.seed)

    records = discover_samples(
        data_root=args.data_root,
        metadata_root=args.metadata_root,
        depth_root=args.depth_root,
        sample_limit=args.max_samples,
    )
    sample_rate, output_dim = validate_uniform_rir_shape(
        records,
        target_length=args.target_length,
    )
    splits = split_records_by_room(records, seed=args.seed)
    train_records = cap_split_records(splits["train"], args.max_train_samples)
    val_records = cap_split_records(splits["val"], args.max_val_samples)
    test_records = cap_split_records(splits["test"], args.max_test_samples)
    device = select_device(args.device)

    summaries = {
        "sample_rate": sample_rate,
        "output_dim": output_dim,
        "split_sizes": summarize_split_sizes(
            {"train": train_records, "val": val_records, "test": test_records}
        ),
        "models": [],
    }

    for model_name in resolve_model_names(args.models):
        if model_name == "rir_depth":
            validate_depth_maps_available([*train_records, *val_records, *test_records])
        checkpoint_path = args.checkpoint_dir / f"{model_name}.pt"
        result = train_model(
            model_name=model_name,
            train_records=train_records,
            val_records=val_records,
            output_dim=output_dim,
            checkpoint_path=checkpoint_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
        )
        summaries["models"].append(result)

    summaries["total_minutes"] = (time.perf_counter() - overall_start) / 60.0
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
