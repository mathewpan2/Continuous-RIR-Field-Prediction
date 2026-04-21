from __future__ import annotations

import json
import os
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import soundfile as sf
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from models import BaseModel, DepthConditionedRIRNetwork, RIRNetwork

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - exercised when tqdm is unavailable
    tqdm = None


DEFAULT_DATA_ROOT = Path("data/Apartments_RIR/Apartments")
DEFAULT_METADATA_ROOT = Path("metadata/Apartments_Metadata/Apartments")
DEFAULT_DEPTH_ROOT = Path("depth_map/Apartments_depthmap/Apartments")
DEFAULT_RESULT_DIR = Path("result")
DEFAULT_CHECKPOINT_DIR = Path("artifacts/checkpoints")
DEFAULT_RANDOM_SEED = 42
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_DEPTH_MAP_SHAPE = (256, 512)


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    room_id: str
    metadata_path: Path
    rir_path: Path
    depth_map_path: Path | None
    coordinates: tuple[float, float, float, float, float, float]
    sample_rate: int
    rir_length: int


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    kwargs: dict[str, int | float]


MODEL_SPECS: dict[str, ModelSpec] = {
    "base_model": ModelSpec(
        name="base_model",
        family="base_model",
        kwargs={
            "input_dim": 6,
            "hidden_dim": 128,
            "dense_hidden_dim": 1024,
            "dropout": 0.2,
            "num_lstm_layers": 1,
        },
    ),
    "rir_small": ModelSpec(
        name="rir_small",
        family="rir_network",
        kwargs={
            "input_dim": 6,
            "hidden_dim": 256,
            "num_hidden_layers": 3,
            "num_frequencies": 6,
        },
    ),
    "rir_large": ModelSpec(
        name="rir_large",
        family="rir_network",
        kwargs={
            "input_dim": 6,
            "hidden_dim": 512,
            "num_hidden_layers": 4,
            "num_frequencies": 8,
        },
    ),
    "rir_depth": ModelSpec(
        name="rir_depth",
        family="rir_depth_network",
        kwargs={
            "input_dim": 6,
            "hidden_dim": 384,
            "num_hidden_layers": 4,
            "num_frequencies": 8,
            "depth_embedding_dim": 128,
        },
    ),
}


def prepare_runtime_environment() -> None:
    # Some Windows Conda setups ship multiple OpenMP runtimes.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def set_seed(seed: int = DEFAULT_RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _normalize_waveform(rir: np.ndarray) -> np.ndarray:
    rir = np.asarray(rir, dtype=np.float32)
    if rir.ndim == 2:
        rir = rir.mean(axis=1)
    if rir.ndim != 1:
        raise ValueError(f"Expected 1D waveform after normalization, got shape {rir.shape}")
    peak = float(np.max(np.abs(rir))) if rir.size else 0.0
    if peak > 0.0:
        rir = rir / peak
    return rir


def _fit_waveform_length(rir: np.ndarray, target_length: int) -> np.ndarray:
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    if rir.shape[0] == target_length:
        return rir
    if rir.shape[0] > target_length:
        return rir[:target_length]

    padded = np.zeros(target_length, dtype=np.float32)
    padded[: rir.shape[0]] = rir
    return padded


def _load_sample_metadata(metadata_path: Path) -> dict:
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_receiver_index(sample_id: str) -> int:
    try:
        return int(sample_id.split("_R", maxsplit=1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Sample id does not contain a valid receiver index: {sample_id}") from exc


def _resolve_depth_map_path(
    depth_root: Path | str,
    room_id: str,
    sample_id: str,
) -> Path | None:
    depth_root = Path(depth_root)
    candidate = depth_root / room_id / f"{_parse_receiver_index(sample_id)}.npy"
    return candidate if candidate.exists() else None


def discover_samples(
    data_root: Path | str = DEFAULT_DATA_ROOT,
    metadata_root: Path | str = DEFAULT_METADATA_ROOT,
    depth_root: Path | str = DEFAULT_DEPTH_ROOT,
    expected_sample_rate: int | None = None,
    sample_limit: int | None = None,
) -> list[SampleRecord]:
    data_root = Path(data_root)
    metadata_root = Path(metadata_root)
    depth_root = Path(depth_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if not metadata_root.exists():
        raise FileNotFoundError(f"Metadata root not found: {metadata_root}")

    records: list[SampleRecord] = []
    metadata_paths = sorted(metadata_root.rglob("*.json"))
    for metadata_path in metadata_paths:
        room_id = metadata_path.parent.name
        sample_id = metadata_path.stem
        rir_path = data_root / room_id / f"{sample_id}_hybrid_IR.wav"
        if not rir_path.exists():
            continue

        metadata = _load_sample_metadata(metadata_path)
        src_loc = metadata.get("src_loc")
        rec_loc = metadata.get("rec_loc")
        if src_loc is None or rec_loc is None:
            continue

        info = sf.info(str(rir_path))
        sample_rate = int(info.samplerate)
        if expected_sample_rate is not None and sample_rate != expected_sample_rate:
            continue

        records.append(
            SampleRecord(
                sample_id=sample_id,
                room_id=room_id,
                metadata_path=metadata_path,
                rir_path=rir_path,
                depth_map_path=_resolve_depth_map_path(depth_root, room_id, sample_id),
                coordinates=tuple(float(v) for v in [*src_loc, *rec_loc]),
                sample_rate=sample_rate,
                rir_length=int(info.frames),
            )
        )
        if sample_limit is not None and len(records) >= sample_limit:
            break

    if not records:
        raise RuntimeError("No paired metadata/RIR samples were discovered.")
    return records


def infer_target_length(
    records: Sequence[SampleRecord],
    preferred_length: int | None = None,
) -> int:
    if preferred_length is not None:
        return int(preferred_length)
    lengths = [record.rir_length for record in records]
    if not lengths:
        raise ValueError("At least one record is required to infer target length.")
    return Counter(lengths).most_common(1)[0][0]


def validate_uniform_rir_shape(
    records: Sequence[SampleRecord],
    target_length: int | None = None,
) -> tuple[int, int]:
    sample_rates = {record.sample_rate for record in records}
    if len(sample_rates) != 1:
        raise ValueError(f"Mixed sample rates are not supported: {sorted(sample_rates)}")
    return next(iter(sample_rates)), infer_target_length(records, preferred_length=target_length)


def split_records_by_room(
    records: Sequence[SampleRecord],
    seed: int = DEFAULT_RANDOM_SEED,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, list[SampleRecord]]:
    rooms = sorted({record.room_id for record in records})
    if len(rooms) < 3:
        raise ValueError("At least three rooms are required for train/val/test splits.")

    rng = random.Random(seed)
    rng.shuffle(rooms)

    total_rooms = len(rooms)
    train_count = max(1, int(total_rooms * train_ratio))
    val_count = max(1, int(total_rooms * val_ratio))
    test_count = total_rooms - train_count - val_count

    while test_count < 1:
        if train_count > val_count and train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1
        else:
            break
        test_count = total_rooms - train_count - val_count

    if train_count + val_count + test_count != total_rooms:
        raise ValueError("Room split counts do not add up to the total number of rooms.")

    train_rooms = set(rooms[:train_count])
    val_rooms = set(rooms[train_count : train_count + val_count])
    test_rooms = set(rooms[train_count + val_count :])

    splits = {"train": [], "val": [], "test": []}
    for record in records:
        if record.room_id in train_rooms:
            splits["train"].append(record)
        elif record.room_id in val_rooms:
            splits["val"].append(record)
        elif record.room_id in test_rooms:
            splits["test"].append(record)

    for split_name, split_records in splits.items():
        if not split_records:
            raise ValueError(f"Split '{split_name}' is empty.")
    return splits


def cap_split_records(
    records: Sequence[SampleRecord],
    limit: int | None,
) -> list[SampleRecord]:
    if limit is None:
        return list(records)
    return list(records[:limit])


class RIRDataset(Dataset[tuple[Tensor, Tensor, str]]):
    def __init__(self, records: Sequence[SampleRecord], target_length: int) -> None:
        self.records = list(records)
        if not self.records:
            raise ValueError("RIRDataset requires at least one record.")
        self.target_length = int(target_length)
        self.depth_map_shape = self._infer_depth_map_shape()

    def _infer_depth_map_shape(self) -> tuple[int, int]:
        for record in self.records:
            if record.depth_map_path is not None:
                depth_map = np.load(record.depth_map_path)
                if depth_map.ndim != 2:
                    raise ValueError(
                        f"Expected 2D depth map, got shape {depth_map.shape} from {record.depth_map_path}"
                    )
                return (int(depth_map.shape[0]), int(depth_map.shape[1]))
        return DEFAULT_DEPTH_MAP_SHAPE

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, str]:
        record = self.records[index]
        rir, _ = sf.read(record.rir_path)
        waveform = _fit_waveform_length(_normalize_waveform(rir), self.target_length)
        if record.depth_map_path is not None:
            depth_map = np.load(record.depth_map_path).astype(np.float32, copy=False)
        else:
            depth_map = np.zeros(self.depth_map_shape, dtype=np.float32)
        coordinates = np.asarray(record.coordinates, dtype=np.float32)
        return (
            torch.from_numpy(coordinates),
            torch.from_numpy(depth_map),
            torch.from_numpy(waveform),
            record.sample_id,
        )


def build_dataloader(
    records: Sequence[SampleRecord],
    batch_size: int,
    shuffle: bool,
    target_length: int,
) -> DataLoader:
    return DataLoader(
        RIRDataset(records, target_length=target_length),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def create_model(model_name: str, output_dim: int) -> nn.Module:
    if model_name not in MODEL_SPECS:
        raise KeyError(f"Unknown model name: {model_name}")
    spec = MODEL_SPECS[model_name]
    kwargs = dict(spec.kwargs)
    kwargs["output_dim"] = output_dim
    if spec.family == "base_model":
        return BaseModel(**kwargs)
    if spec.family == "rir_network":
        return RIRNetwork(**kwargs)
    if spec.family == "rir_depth_network":
        return DepthConditionedRIRNetwork(**kwargs)
    raise ValueError(f"Unsupported model family: {spec.family}")


def forward_model(
    model_name: str,
    model: nn.Module,
    coordinates: Tensor,
    depth_maps: Tensor,
) -> Tensor:
    if model_name == "rir_depth":
        return model(coordinates, depth_maps)
    return model(coordinates)


def train_one_epoch(
    model_name: str,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    criterion = nn.MSELoss()
    iterator = dataloader
    if tqdm is not None:
        iterator = tqdm(
            dataloader,
            desc=f"[{model_name}] Train {epoch}/{epochs}",
            leave=False,
        )
    for batch_index, (coordinates, depth_maps, targets, _) in enumerate(iterator, start=1):
        coordinates = coordinates.to(device=device, dtype=torch.float32)
        depth_maps = depth_maps.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        predictions = forward_model(model_name, model, coordinates, depth_maps)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        batch_size = coordinates.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size
        if tqdm is not None:
            iterator.set_postfix(loss=float(loss.detach().cpu()), batch=batch_index)
    return total_loss / max(total_items, 1)


@torch.no_grad()
def evaluate(
    model_name: str,
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0
    criterion = nn.MSELoss()
    iterator = dataloader
    if tqdm is not None:
        iterator = tqdm(
            dataloader,
            desc=f"[{model_name}] Val {epoch}/{epochs}",
            leave=False,
        )
    for batch_index, (coordinates, depth_maps, targets, _) in enumerate(iterator, start=1):
        coordinates = coordinates.to(device=device, dtype=torch.float32)
        depth_maps = depth_maps.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)
        predictions = forward_model(model_name, model, coordinates, depth_maps)
        loss = criterion(predictions, targets)
        batch_size = coordinates.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size
        if tqdm is not None:
            iterator.set_postfix(loss=float(loss.detach().cpu()), batch=batch_index)
    return total_loss / max(total_items, 1)


def train_model(
    model_name: str,
    train_records: Sequence[SampleRecord],
    val_records: Sequence[SampleRecord],
    output_dim: int,
    checkpoint_path: Path | str,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> dict[str, float | int | str]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model_start = time.perf_counter()

    torch_device = torch.device(device)
    model = create_model(model_name, output_dim=output_dim).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = build_dataloader(
        train_records,
        batch_size=batch_size,
        shuffle=True,
        target_length=output_dim,
    )
    val_loader = build_dataloader(
        val_records,
        batch_size=batch_size,
        shuffle=False,
        target_length=output_dim,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    epoch_iterator = range(1, epochs + 1)
    if tqdm is not None:
        epoch_iterator = tqdm(epoch_iterator, desc=f"[{model_name}] Epochs", leave=True)

    for epoch in epoch_iterator:
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(
            model_name,
            model,
            train_loader,
            optimizer,
            torch_device,
            epoch,
            epochs,
        )
        val_loss = evaluate(
            model_name,
            model,
            val_loader,
            torch_device,
            epoch,
            epochs,
        )
        epoch_seconds = time.perf_counter() - epoch_start
        elapsed_total_seconds = time.perf_counter() - model_start
        average_epoch_seconds = elapsed_total_seconds / epoch
        remaining_epochs = epochs - epoch
        eta_seconds = average_epoch_seconds * remaining_epochs
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if tqdm is not None:
            epoch_iterator.set_postfix(
                train_loss=f"{train_loss:.6f}",
                val_loss=f"{val_loss:.6f}",
                eta_min=f"{eta_seconds / 60.0:.1f}",
            )
        print(
            f"[{model_name}] epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"elapsed={epoch_seconds:.1f}s "
            f"eta={eta_seconds / 60.0:.2f}min"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "output_dim": output_dim,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            print(
                f"[{model_name}] new best checkpoint saved at epoch {epoch} "
                f"with val_loss={val_loss:.6f}"
            )

    total_seconds = time.perf_counter() - model_start
    print(
        f"[{model_name}] training completed in {total_seconds / 60.0:.2f} min "
        f"(best_epoch={best_epoch}, best_val_loss={best_val_loss:.6f})"
    )

    return {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "checkpoint_path": str(checkpoint_path),
        "epochs": epochs,
        "train_minutes": total_seconds / 60.0,
    }


def load_model_checkpoint(
    checkpoint_path: Path | str,
    device: str = "cpu",
) -> tuple[nn.Module, dict]:
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=device)
    model_name = payload["model_name"]
    output_dim = int(payload["output_dim"])
    model = create_model(model_name, output_dim=output_dim)
    model.load_state_dict(payload["model_state_dict"])
    model.to(torch.device(device))
    model.eval()
    return model, payload


@torch.no_grad()
def export_predictions(
    model_name: str,
    checkpoint_path: Path | str,
    records: Sequence[SampleRecord],
    output_path: Path | str,
    batch_size: int = 16,
    device: str = "cpu",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model, payload = load_model_checkpoint(checkpoint_path, device=device)
    target_length = int(payload["output_dim"])
    dataloader = build_dataloader(
        records,
        batch_size=batch_size,
        shuffle=False,
        target_length=target_length,
    )
    torch_device = torch.device(device)

    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_ids: list[str] = []

    for coordinates, depth_maps, targets, sample_ids in dataloader:
        coordinates = coordinates.to(device=torch_device, dtype=torch.float32)
        depth_maps = depth_maps.to(device=torch_device, dtype=torch.float32)
        predictions = forward_model(model_name, model, coordinates, depth_maps).cpu().numpy()
        all_pred.append(predictions.astype(np.float32))
        all_true.append(targets.numpy().astype(np.float32))
        all_ids.extend(list(sample_ids))

    sample_rate, _ = validate_uniform_rir_shape(records, target_length=target_length)
    np.savez(
        output_path,
        y_true=np.concatenate(all_true, axis=0),
        y_pred=np.concatenate(all_pred, axis=0),
        sample_id=np.asarray(all_ids, dtype=object),
        method_name=np.asarray(model_name, dtype=object),
        sample_rate=np.asarray(sample_rate),
    )
    return output_path


def summarize_split_sizes(splits: dict[str, Sequence[SampleRecord]]) -> dict[str, int]:
    return {name: len(records) for name, records in splits.items()}


def validate_depth_maps_available(records: Sequence[SampleRecord]) -> None:
    missing = [record.sample_id for record in records if record.depth_map_path is None]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Missing depth maps for {len(missing)} sample(s): {preview}")


def select_device(requested_device: str | None = None) -> str:
    if requested_device:
        return requested_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_model_names(model_names: Iterable[str] | None) -> list[str]:
    if not model_names:
        return list(MODEL_SPECS.keys())
    resolved = list(model_names)
    unknown = [name for name in resolved if name not in MODEL_SPECS]
    if unknown:
        raise KeyError(f"Unknown model names: {unknown}")
    return resolved
