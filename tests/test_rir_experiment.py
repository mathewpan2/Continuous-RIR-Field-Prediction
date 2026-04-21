from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from utils.rir_experiment import (
    create_model,
    discover_samples,
    export_predictions,
    split_records_by_room,
    validate_depth_maps_available,
    validate_uniform_rir_shape,
)


def _write_sample(
    base_dir: Path,
    room_id: str,
    sample_id: str,
    waveform: np.ndarray,
    with_depth_map: bool = True,
) -> None:
    metadata_path = base_dir / "metadata" / "Apartments_Metadata" / "Apartments" / room_id
    data_path = base_dir / "data" / "Apartments_RIR" / "Apartments" / room_id
    depth_path = base_dir / "depth_map" / "Apartments_depthmap" / "Apartments" / room_id
    metadata_path.mkdir(parents=True, exist_ok=True)
    data_path.mkdir(parents=True, exist_ok=True)
    depth_path.mkdir(parents=True, exist_ok=True)

    with (metadata_path / f"{sample_id}.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "src_loc": [0.0, 1.0, 2.0],
                "rec_loc": [3.0, 4.0, 5.0],
            },
            handle,
        )

    sf.write(data_path / f"{sample_id}_hybrid_IR.wav", waveform, 8000)
    if with_depth_map:
        receiver_index = int(sample_id.split("_R", maxsplit=1)[1])
        np.save(depth_path / f"{receiver_index}.npy", np.ones((4, 6), dtype=np.float32))


def test_discover_samples_and_split_by_room(tmp_path: Path) -> None:
    waveform = np.linspace(-1.0, 1.0, num=8, dtype=np.float32)
    for room_id in ("room_a", "room_b", "room_c"):
        _write_sample(tmp_path, room_id, "S000_R000", waveform)

    records = discover_samples(
        data_root=tmp_path / "data" / "Apartments_RIR" / "Apartments",
        metadata_root=tmp_path / "metadata" / "Apartments_Metadata" / "Apartments",
        depth_root=tmp_path / "depth_map" / "Apartments_depthmap" / "Apartments",
    )

    assert len(records) == 3
    sample_rate, rir_length = validate_uniform_rir_shape(records)
    assert sample_rate == 8000
    assert rir_length == 8

    splits = split_records_by_room(records, seed=42)
    assert len(splits["train"]) == 1
    assert len(splits["val"]) == 1
    assert len(splits["test"]) == 1
    validate_depth_maps_available(records)
    assert records[0].depth_map_path is not None


def test_create_model_supports_rir_baseline_shape() -> None:
    model = create_model("base_model", output_dim=8)
    coordinates = torch.randn(2, 6)

    output = model(coordinates)

    assert output.shape == (2, 8)


def test_create_model_supports_depth_conditioned_shape() -> None:
    model = create_model("rir_depth", output_dim=8)
    coordinates = torch.randn(2, 6)
    depth_map = torch.randn(2, 32, 64)

    output = model(coordinates, depth_map)

    assert output.shape == (2, 8)


def test_export_predictions_writes_npz_contract(tmp_path: Path) -> None:
    waveform = np.linspace(-1.0, 1.0, num=8, dtype=np.float32)
    for room_id in ("room_a", "room_b", "room_c"):
        _write_sample(tmp_path, room_id, "S000_R000", waveform)

    records = discover_samples(
        data_root=tmp_path / "data" / "Apartments_RIR" / "Apartments",
        metadata_root=tmp_path / "metadata" / "Apartments_Metadata" / "Apartments",
        depth_root=tmp_path / "depth_map" / "Apartments_depthmap" / "Apartments",
    )
    model = create_model("rir_small", output_dim=8)
    checkpoint_path = tmp_path / "rir_small.pt"
    torch.save(
        {
            "model_name": "rir_small",
            "model_state_dict": model.state_dict(),
            "output_dim": 8,
            "epoch": 1,
            "train_loss": 0.0,
            "val_loss": 0.0,
        },
        checkpoint_path,
    )

    output_path = tmp_path / "result" / "rir_small_predictions.npz"
    export_predictions(
        model_name="rir_small",
        checkpoint_path=checkpoint_path,
        records=records,
        output_path=output_path,
        batch_size=2,
        device="cpu",
    )

    with np.load(output_path, allow_pickle=True) as data:
        assert set(data.files) == {"method_name", "sample_id", "sample_rate", "y_pred", "y_true"}
        assert data["y_true"].shape == (3, 8)
        assert data["y_pred"].shape == (3, 8)
        assert str(data["method_name"].item()) == "rir_small"
