"""
Training script for predicting continuous Room Impulse Response (RIR).
Uses the RIRNetwork model to predict RIR waveforms from source-receiver coordinates.
"""

import argparse
import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import soundfile as sf

from models.rir_network import RIRNetwork


def schroeder_edc(rir: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Computes normalized log-EDC (Schroeder integration) in dB-like scale."""
    energy = rir ** 2
    # Reverse cumulative sum of energy
    edc = torch.flip(torch.cumsum(torch.flip(energy, dims=[-1]), dim=-1), dims=[-1])
    edc = edc / torch.clamp(edc[..., :1], min=eps)
    return 10.0 * torch.log10(torch.clamp(edc, min=eps))


def compute_effective_edc_weight(
    base_weight: float,
    wave_loss: torch.Tensor,
    edc_loss: torch.Tensor,
    auto_balance: bool,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Returns effective EDC weight; when auto_balance is enabled, scales by wave/edc ratio."""
    if base_weight <= 0.0:
        return torch.zeros(1, device=wave_loss.device, dtype=wave_loss.dtype).squeeze(0)

    if not auto_balance:
        return torch.tensor(base_weight, device=wave_loss.device, dtype=wave_loss.dtype)

    ratio = wave_loss.detach() / torch.clamp(edc_loss.detach(), min=eps)
    return torch.tensor(base_weight, device=wave_loss.device, dtype=wave_loss.dtype) * ratio


def pairwise_distance_delay_features(
    src_xyz: np.ndarray,
    rec_xyz: np.ndarray,
    speed_of_sound: float = 343.0,
) -> Tuple[float, float]:
    distance = float(np.linalg.norm(src_xyz - rec_xyz))
    delay = float(distance / max(speed_of_sound, 1e-6))
    return distance, delay


def compute_early_window_loss(
    predicted_rirs: torch.Tensor,
    target_rirs: torch.Tensor,
    criterion: nn.Module,
    early_window_samples: int,
) -> torch.Tensor:
    if early_window_samples <= 0:
        return torch.zeros(1, device=target_rirs.device, dtype=target_rirs.dtype).squeeze(0)

    early_n = min(
        early_window_samples,
        predicted_rirs.shape[-1],
        target_rirs.shape[-1],
    )
    if early_n <= 0:
        return torch.zeros(1, device=target_rirs.device, dtype=target_rirs.dtype).squeeze(0)

    return criterion(predicted_rirs[..., :early_n], target_rirs[..., :early_n])


class RIRDataset(Dataset):
    """PyTorch Dataset for loading RIR data from dataframe."""

    def __init__(
        self,
        df: pd.DataFrame,
        normalize: bool = True,
        room_context_columns: Optional[List[str]] = None,
    ):
        """
        Args:
            df: DataFrame with columns: rir_path, src_x, src_y, src_z, rec_x, rec_y, rec_z
            normalize: Whether to normalize RIR waveforms to [-1, 1]
        """
        self.df = df.reset_index(drop=True)
        self.normalize = normalize
        self.room_context_columns = room_context_columns or []
        self._load_rirs()

    def _load_rirs(self):
        """Pre-load all RIR waveforms into memory."""
        self.rirs = []
        self.coordinates = []
        self.room_contexts = []

        for idx, row in self.df.iterrows():
            # Load RIR waveform
            rir, sr = sf.read(row["rir_path"])
            
            # Ensure consistent length and format
            if len(rir.shape) > 1:
                rir = rir[:, 0]  # Take first channel if stereo
            
            # Normalize if requested
            if self.normalize:
                max_val = np.abs(rir).max()
                if max_val > 0:
                    rir = rir / max_val
            
            self.rirs.append(torch.from_numpy(rir).float())

            # Create coordinate tensor [src_x, src_y, src_z, rec_x, rec_y, rec_z]
            coords = torch.tensor([
                row["src_x"], row["src_y"], row["src_z"],
                row["rec_x"], row["rec_y"], row["rec_z"]
            ], dtype=torch.float32)
            self.coordinates.append(coords)

            if self.room_context_columns:
                room_ctx = torch.tensor(
                    row[self.room_context_columns].to_numpy(dtype=np.float32),
                    dtype=torch.float32,
                )
                self.room_contexts.append(room_ctx)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        """Returns (coordinates, room_context, rir_waveform) or (coordinates, rir_waveform)."""
        if self.room_context_columns:
            return self.coordinates[idx], self.room_contexts[idx], self.rirs[idx]
        return self.coordinates[idx], self.rirs[idx]


def _safe_bbox_features(points: List[List[float]]) -> Tuple[float, float, float, float, float]:
    if not points:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    arr = np.asarray(points, dtype=np.float32)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    extents = np.maximum(maxs - mins, 1e-6)
    dim_x, dim_y, dim_z = extents.tolist()
    volume = float(dim_x * dim_y * dim_z)
    diag = float(np.linalg.norm(extents))
    return float(dim_x), float(dim_y), float(dim_z), volume, diag


def _build_material_category_lookup(material_library_dir: str) -> Tuple[Dict[str, str], List[str]]:
    material_name_to_category: Dict[str, str] = {}
    categories = set()

    for file_path in sorted(glob.glob(os.path.join(material_library_dir, "*.json"))):
        with open(file_path, "r") as f:
            records = json.load(f)
        for rec in records:
            name = rec.get("name")
            cat = rec.get("category")
            if name is None or cat is None:
                continue
            material_name_to_category[name] = cat
            categories.add(cat)

    return material_name_to_category, sorted(categories)


def add_room_context_features(
    df: pd.DataFrame,
    simulation_info_base: str,
    material_library_dir: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """Adds per-room context features derived from geometry proxies and material assignments."""
    material_name_to_category, categories = _build_material_category_lookup(material_library_dir)
    category_to_idx = {cat: i for i, cat in enumerate(categories)}

    ctx_cols = [f"ctx_mat_{cat}" for cat in categories]
    ctx_cols += [
        "ctx_dim_x",
        "ctx_dim_y",
        "ctx_dim_z",
        "ctx_bbox_volume",
        "ctx_bbox_diag",
        "ctx_impulse_length_sec",
        "ctx_unknown_material_ratio",
        "ctx_unique_material_ratio",
        "ctx_has_simulation_info",
    ]

    if len(df) == 0:
        empty_df = df.copy()
        for col in ctx_cols:
            empty_df[col] = np.array([], dtype=np.float32)
        return empty_df, ctx_cols

    grouped = df.groupby(["room_type", "room_id"], sort=False)
    room_ctx_map: Dict[Tuple[str, str], np.ndarray] = {}

    for (room_type, room_id), group in grouped:
        mat_hist = np.zeros(len(categories), dtype=np.float32)

        # Fallback geometry proxy from source/receiver positions present in metadata.
        points = []
        points.extend(group[["src_x", "src_y", "src_z"]].to_numpy(dtype=np.float32).tolist())
        points.extend(group[["rec_x", "rec_y", "rec_z"]].to_numpy(dtype=np.float32).tolist())
        dim_x, dim_y, dim_z, bbox_volume, bbox_diag = _safe_bbox_features(points)

        impulse_length_sec = 0.0
        unknown_material_ratio = 0.0
        unique_material_ratio = 0.0
        has_sim_info = 0.0

        sim_path = os.path.join(simulation_info_base, room_type, room_id, "simulation.json")
        if os.path.exists(sim_path):
            has_sim_info = 1.0
            with open(sim_path, "r") as f:
                sim = json.load(f)

            impulse_length_sec = float(sim.get("impulseLengthSec", 0.0) or 0.0)

            sim_points = []
            for src in sim.get("sources", []):
                if all(k in src for k in ("x", "y", "z")):
                    sim_points.append([src["x"], src["y"], src["z"]])
            for rec in sim.get("receivers", []):
                if all(k in rec for k in ("x", "y", "z")):
                    sim_points.append([rec["x"], rec["y"], rec["z"]])
            if sim_points:
                dim_x, dim_y, dim_z, bbox_volume, bbox_diag = _safe_bbox_features(sim_points)

            assignments = sim.get("layerMaterialAssignments", [])
            total = len(assignments)
            unknown = 0
            used_cats = set()
            if total > 0:
                for row in assignments:
                    material_name = row.get("materialName")
                    cat = material_name_to_category.get(material_name)
                    if cat is None:
                        unknown += 1
                        continue
                    idx = category_to_idx[cat]
                    mat_hist[idx] += 1.0
                    used_cats.add(cat)

                mat_hist = mat_hist / max(mat_hist.sum(), 1.0)
                unknown_material_ratio = float(unknown / total)
                unique_material_ratio = float(len(used_cats) / max(len(categories), 1))

        features = np.concatenate(
            [
                mat_hist,
                np.array(
                    [
                        dim_x,
                        dim_y,
                        dim_z,
                        bbox_volume,
                        bbox_diag,
                        impulse_length_sec,
                        unknown_material_ratio,
                        unique_material_ratio,
                        has_sim_info,
                    ],
                    dtype=np.float32,
                ),
            ]
        )

        room_ctx_map[(room_type, room_id)] = features

    ctx_values = np.vstack(
        [room_ctx_map[(rt, rid)] for rt, rid in zip(df["room_type"].tolist(), df["room_id"].tolist())]
    )
    ctx_df = pd.DataFrame(ctx_values, columns=ctx_cols)
    out_df = pd.concat([df.reset_index(drop=True), ctx_df.reset_index(drop=True)], axis=1)
    return out_df, ctx_cols


def add_pair_geometry_features(
    df: pd.DataFrame,
    speed_of_sound: float = 343.0,
) -> Tuple[pd.DataFrame, List[str]]:
    out_df = df.copy()
    src = out_df[["src_x", "src_y", "src_z"]].to_numpy(dtype=np.float32)
    rec = out_df[["rec_x", "rec_y", "rec_z"]].to_numpy(dtype=np.float32)

    distances = np.linalg.norm(src - rec, axis=1)
    delays = distances / max(speed_of_sound, 1e-6)

    out_df["ctx_pair_distance_m"] = distances.astype(np.float32)
    out_df["ctx_pair_delay_s"] = delays.astype(np.float32)
    out_df["ctx_pair_inv_distance"] = (1.0 / np.maximum(distances, 1e-6)).astype(np.float32)
    return out_df, ["ctx_pair_distance_m", "ctx_pair_delay_s", "ctx_pair_inv_distance"]


def normalize_room_context(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ctx_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not ctx_cols:
        return train_df, val_df, test_df

    mean = train_df[ctx_cols].mean(axis=0)
    std = train_df[ctx_cols].std(axis=0).replace(0, 1.0)

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df[ctx_cols] = (train_df[ctx_cols] - mean) / std
    if len(val_df) > 0:
        val_df[ctx_cols] = (val_df[ctx_cols] - mean) / std
    if len(test_df) > 0:
        test_df[ctx_cols] = (test_df[ctx_cols] - mean) / std
    return train_df, val_df, test_df


def load_dataset(
    ir_base: str,
    meta_base: str,
    train_split: float = 0.8,
    val_split: float = 0.1,
    seed: int = 42,
    max_rooms: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load dataset from directory structure and split into train/val/test."""
    
    rows = []

    # Walk through room types (Apartments, Auditorium, etc.)
    for room_type in os.listdir(ir_base):
        room_type_path = os.path.join(ir_base, room_type)
        if not os.path.isdir(room_type_path):
            continue

        # Walk through each room index (Apartments_idx_0, etc.)
        for room_id in os.listdir(room_type_path):
            room_path = os.path.join(room_type_path, room_id)
            if not os.path.isdir(room_path):
                continue

            # Walk through each wav file
            for file in os.listdir(room_path):
                if not file.endswith(".wav"):
                    continue

                # S000_R000_hybrid_IR.wav -> S000_R000
                stem = file.replace("_hybrid_IR.wav", "")

                rir_path = os.path.join(room_path, file)
                json_path = os.path.join(meta_base, room_type, room_id, f"{stem}.json")

                if not os.path.exists(json_path):
                    continue

                with open(json_path, "r") as f:
                    meta = json.load(f)

                rows.append({
                    "room_type": room_type,
                    "room_id": room_id,
                    "stem": stem,
                    "rir_path": rir_path,
                    "src_x": meta["src_loc"][0],
                    "src_y": meta["src_loc"][1],
                    "src_z": meta["src_loc"][2],
                    "rec_x": meta["rec_loc"][0],
                    "rec_y": meta["rec_loc"][1],
                    "rec_z": meta["rec_loc"][2],
                })

    df = pd.DataFrame(rows)
    
    # Limit dataset size if requested
    if max_rooms is not None:
        rooms_list = df["room_id"].unique()
        if len(rooms_list) > max_rooms:
            np.random.seed(seed)
            selected_rooms = np.random.choice(rooms_list, size=max_rooms, replace=False)
            df = df[df["room_id"].isin(selected_rooms)].reset_index(drop=True)
            print(f"Limited to {max_rooms} unique rooms")
    
    if max_samples is not None:
        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
            print(f"Limited to {max_samples} total samples")

    # Split by unique rooms to avoid data leakage
    rooms = np.array(df["room_id"].unique())
    np.random.seed(seed)
    np.random.shuffle(rooms)

    n = len(rooms)
    
    # Ensure minimum 1 room per split for small datasets
    if n < 3:
        # For very small datasets, use 60/20/20 or adjust as needed
        train_idx = max(1, int(n * 0.6))
        val_idx = train_idx + max(1, int(n * 0.2))
    else:
        # Standard 80/10/10 split
        train_idx = int(n * train_split)
        val_idx = train_idx + max(1, int(n * val_split))
    
    train_rooms = rooms[:train_idx]
    val_rooms = rooms[train_idx:val_idx]
    test_rooms = rooms[val_idx:]

    train_df = df[df["room_id"].isin(train_rooms)].reset_index(drop=True)
    val_df = df[df["room_id"].isin(val_rooms)].reset_index(drop=True)
    test_df = df[df["room_id"].isin(test_rooms)].reset_index(drop=True)

    print(f"Rooms  — train: {len(train_rooms)}, val: {len(val_rooms)}, test: {len(test_rooms)}")
    print(f"Samples — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    return train_df, val_df, test_df


def collate_fn(batch):
    """Collate function to handle variable-length RIR waveforms."""
    has_room_context = len(batch[0]) == 3

    if has_room_context:
        coordinates, room_contexts, rirs = zip(*batch)
    else:
        coordinates, rirs = zip(*batch)
    
    # Find max length
    max_len = max(len(rir) for rir in rirs)
    
    # Pad all RIRs to the same length
    padded_rirs = []
    for rir in rirs:
        if len(rir) < max_len:
            rir = torch.nn.functional.pad(rir, (0, max_len - len(rir)))
        padded_rirs.append(rir)
    
    coords_tensor = torch.stack(coordinates)
    rirs_tensor = torch.stack(padded_rirs)

    if has_room_context:
        room_ctx_tensor = torch.stack(room_contexts)
        return coords_tensor, room_ctx_tensor, rirs_tensor
    return coords_tensor, rirs_tensor


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    edc_loss_weight: float = 0.0,
    edc_criterion: Optional[nn.Module] = None,
    edc_auto_balance: bool = False,
    early_loss_weight: float = 0.0,
    early_criterion: Optional[nn.Module] = None,
    early_window_samples: int = 0,
) -> Tuple[float, float, float, float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_wave_loss = 0.0
    total_edc_loss = 0.0
    total_early_loss = 0.0
    total_effective_edc_weight = 0.0

    for batch in loader:
        if len(batch) == 3:
            coords, room_ctx, rirs = batch
            room_ctx = room_ctx.to(device)
        else:
            coords, rirs = batch
            room_ctx = None

        coords = coords.to(device)
        rirs = rirs.to(device)

        optimizer.zero_grad()

        # Forward pass
        predicted_rirs = model(coords, room_context=room_ctx)
        
        # Trim or pad predicted RIRs to match target length
        target_len = rirs.shape[-1]
        if predicted_rirs.shape[-1] > target_len:
            predicted_rirs = predicted_rirs[..., :target_len]
        elif predicted_rirs.shape[-1] < target_len:
            predicted_rirs = torch.nn.functional.pad(
                predicted_rirs, (0, target_len - predicted_rirs.shape[-1])
            )

        # Waveform loss
        wave_loss = criterion(predicted_rirs, rirs)

        # Optional EDC loss to match decay behavior
        if edc_loss_weight > 0.0 and edc_criterion is not None:
            pred_edc = schroeder_edc(predicted_rirs)
            target_edc = schroeder_edc(rirs)
            edc_loss = edc_criterion(pred_edc, target_edc)
        else:
            edc_loss = torch.zeros(1, device=rirs.device, dtype=rirs.dtype).squeeze(0)

        effective_edc_weight = compute_effective_edc_weight(
            base_weight=edc_loss_weight,
            wave_loss=wave_loss,
            edc_loss=edc_loss,
            auto_balance=edc_auto_balance,
        )

        if early_loss_weight > 0.0 and early_criterion is not None:
            early_loss = compute_early_window_loss(
                predicted_rirs,
                rirs,
                criterion=early_criterion,
                early_window_samples=early_window_samples,
            )
        else:
            early_loss = torch.zeros(1, device=rirs.device, dtype=rirs.dtype).squeeze(0)

        loss = wave_loss + effective_edc_weight * edc_loss + early_loss_weight * early_loss

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_wave_loss += wave_loss.item()
        total_edc_loss += edc_loss.item()
        total_early_loss += early_loss.item()
        total_effective_edc_weight += effective_edc_weight.item()

    return (
        total_loss / len(loader),
        total_wave_loss / len(loader),
        total_edc_loss / len(loader),
        total_early_loss / len(loader),
        total_effective_edc_weight / len(loader),
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    edc_loss_weight: float = 0.0,
    edc_criterion: Optional[nn.Module] = None,
    edc_auto_balance: bool = False,
    early_loss_weight: float = 0.0,
    early_criterion: Optional[nn.Module] = None,
    early_window_samples: int = 0,
) -> Tuple[float, float, float, float, float]:
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0.0
    total_wave_loss = 0.0
    total_edc_loss = 0.0
    total_early_loss = 0.0
    total_effective_edc_weight = 0.0

    for batch in loader:
        if len(batch) == 3:
            coords, room_ctx, rirs = batch
            room_ctx = room_ctx.to(device)
        else:
            coords, rirs = batch
            room_ctx = None

        coords = coords.to(device)
        rirs = rirs.to(device)

        # Forward pass
        predicted_rirs = model(coords, room_context=room_ctx)
        
        # Trim or pad predicted RIRs to match target length
        target_len = rirs.shape[-1]
        if predicted_rirs.shape[-1] > target_len:
            predicted_rirs = predicted_rirs[..., :target_len]
        elif predicted_rirs.shape[-1] < target_len:
            predicted_rirs = torch.nn.functional.pad(
                predicted_rirs, (0, target_len - predicted_rirs.shape[-1])
            )

        # Waveform loss
        wave_loss = criterion(predicted_rirs, rirs)

        # Optional EDC loss for evaluation consistency
        if edc_loss_weight > 0.0 and edc_criterion is not None:
            pred_edc = schroeder_edc(predicted_rirs)
            target_edc = schroeder_edc(rirs)
            edc_loss = edc_criterion(pred_edc, target_edc)
        else:
            edc_loss = torch.zeros(1, device=rirs.device, dtype=rirs.dtype).squeeze(0)

        effective_edc_weight = compute_effective_edc_weight(
            base_weight=edc_loss_weight,
            wave_loss=wave_loss,
            edc_loss=edc_loss,
            auto_balance=edc_auto_balance,
        )

        if early_loss_weight > 0.0 and early_criterion is not None:
            early_loss = compute_early_window_loss(
                predicted_rirs,
                rirs,
                criterion=early_criterion,
                early_window_samples=early_window_samples,
            )
        else:
            early_loss = torch.zeros(1, device=rirs.device, dtype=rirs.dtype).squeeze(0)

        loss = wave_loss + effective_edc_weight * edc_loss + early_loss_weight * early_loss
        total_loss += loss.item()
        total_wave_loss += wave_loss.item()
        total_edc_loss += edc_loss.item()
        total_early_loss += early_loss.item()
        total_effective_edc_weight += effective_edc_weight.item()

    return (
        total_loss / len(loader),
        total_wave_loss / len(loader),
        total_edc_loss / len(loader),
        total_early_loss / len(loader),
        total_effective_edc_weight / len(loader),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train RIR prediction network"
    )
    parser.add_argument(
        "--ir-base",
        type=str,
        default="single_channel_ir_1",
        help="Path to single_channel_ir_1 directory",
    )
    parser.add_argument(
        "--meta-base",
        type=str,
        default="metadata",
        help="Path to metadata directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Hidden dimension for MLP",
    )
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=4,
        help="Number of hidden layers",
    )
    parser.add_argument(
        "--num-frequencies",
        type=int,
        default=8,
        help="Number of Fourier frequencies",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=16000,
        help="Output RIR length in samples",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--loss-fn",
        type=str,
        default="mse",
        choices=["mse", "l1"],
        help="Loss function to use",
    )
    parser.add_argument(
        "--max-rooms",
        type=int,
        default=None,
        help="Maximum number of unique rooms to use (for quick testing)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (for quick testing)",
    )
    parser.add_argument(
        "--use-room-context",
        action="store_true",
        help="Use room context features (materials + geometry proxies) as additional model input",
    )
    parser.add_argument(
        "--simulation-info-base",
        type=str,
        default="simulation_info",
        help="Path to simulation_info directory",
    )
    parser.add_argument(
        "--material-library-dir",
        type=str,
        default="AcousticRooms/material_library",
        help="Path to material library JSON files",
    )
    parser.add_argument(
        "--edc-loss-weight",
        type=float,
        default=0.0,
        help="Weight for EDC loss term; 0.0 disables EDC objective",
    )
    parser.add_argument(
        "--edc-loss-fn",
        type=str,
        default="l1",
        choices=["l1", "mse"],
        help="Loss function used for log-EDC matching",
    )
    parser.add_argument(
        "--edc-auto-balance",
        action="store_true",
        help="Automatically balance EDC loss scale against waveform loss each batch",
    )
    parser.add_argument(
        "--use-pair-geometry-features",
        action="store_true",
        help="Append source-receiver pair geometry features (distance, delay, inverse distance) to room context",
    )
    parser.add_argument(
        "--speed-of-sound",
        type=float,
        default=343.0,
        help="Speed of sound used for delay feature computation (m/s)",
    )
    parser.add_argument(
        "--early-loss-weight",
        type=float,
        default=0.0,
        help="Weight for early-window transient loss",
    )
    parser.add_argument(
        "--early-window-ms",
        type=float,
        default=20.0,
        help="Early window duration in milliseconds for transient-focused loss",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Sample rate used to convert early window ms to samples",
    )
    parser.add_argument(
        "--early-loss-fn",
        type=str,
        default="l1",
        choices=["l1", "mse"],
        help="Loss function for early-window transient matching",
    )

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RIR Prediction Training Script")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print()

    # Load dataset
    print("Loading dataset...")
    train_df, val_df, test_df = load_dataset(
        args.ir_base,
        args.meta_base,
        seed=args.seed,
        max_rooms=args.max_rooms,
        max_samples=args.max_samples,
    )

    room_context_columns: List[str] = []
    if args.use_room_context:
        print("Building room context features (materials + geometry)...")
        train_df, room_context_columns = add_room_context_features(
            train_df,
            simulation_info_base=args.simulation_info_base,
            material_library_dir=args.material_library_dir,
        )
        val_df, _ = add_room_context_features(
            val_df,
            simulation_info_base=args.simulation_info_base,
            material_library_dir=args.material_library_dir,
        )
        test_df, _ = add_room_context_features(
            test_df,
            simulation_info_base=args.simulation_info_base,
            material_library_dir=args.material_library_dir,
        )

        train_df, val_df, test_df = normalize_room_context(
            train_df,
            val_df,
            test_df,
            room_context_columns,
        )

        room_ctx_available = float(train_df["ctx_has_simulation_info"].gt(0).mean())
        print(
            f"Room context dim: {len(room_context_columns)} | "
            f"train rooms with simulation_info: {room_ctx_available * 100:.1f}%"
        )

        if args.use_pair_geometry_features:
            train_df, pair_cols = add_pair_geometry_features(
                train_df,
                speed_of_sound=args.speed_of_sound,
            )
            val_df, _ = add_pair_geometry_features(
                val_df,
                speed_of_sound=args.speed_of_sound,
            )
            test_df, _ = add_pair_geometry_features(
                test_df,
                speed_of_sound=args.speed_of_sound,
            )

            train_df, val_df, test_df = normalize_room_context(
                train_df,
                val_df,
                test_df,
                pair_cols,
            )
            room_context_columns = room_context_columns + pair_cols
            print(f"Added pair geometry features: {pair_cols}")
            print(f"Updated room context dim: {len(room_context_columns)}")
    print()

    # Create datasets
    print("Creating datasets...")
    if args.use_room_context:
        train_dataset = RIRDataset(
            train_df,
            normalize=True,
            room_context_columns=room_context_columns,
        )
        val_dataset = RIRDataset(
            val_df,
            normalize=True,
            room_context_columns=room_context_columns,
        )
        test_dataset = RIRDataset(
            test_df,
            normalize=True,
            room_context_columns=room_context_columns,
        )
    else:
        train_dataset = RIRDataset(train_df, normalize=True)
        val_dataset = RIRDataset(val_df, normalize=True)
        test_dataset = RIRDataset(test_df, normalize=True)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    print(f"Created {len(train_loader)} train batches")
    print(f"Created {len(val_loader)} val batches")
    print(f"Created {len(test_loader)} test batches")
    
    # Check for empty loaders
    if len(train_loader) == 0:
        print("ERROR: Training set is empty!")
        return
    
    if len(val_loader) == 0:
        print("WARNING: Validation set is empty. Skipping validation.")
        skip_validation = True
    else:
        skip_validation = False
    
    if len(test_loader) == 0:
        print("WARNING: Test set is empty.")
    print()

    # Create model
    print("Creating model...")
    model = RIRNetwork(
        input_dim=6,  # [src_x, src_y, src_z, rec_x, rec_y, rec_z]
        output_dim=args.output_dim,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        num_frequencies=args.num_frequencies,
        room_context_dim=len(room_context_columns),
    )
    model = model.to(args.device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print()

    # Create optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Create loss function
    if args.loss_fn == "mse":
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()
    print(f"Loss function: {args.loss_fn.upper()}")
    print(f"EDC loss weight: {args.edc_loss_weight}")
    print(f"EDC auto balance: {args.edc_auto_balance}")
    print(f"Early loss weight: {args.early_loss_weight}")

    if args.edc_loss_fn == "mse":
        edc_criterion = nn.MSELoss()
    else:
        edc_criterion = nn.L1Loss()
    print(f"EDC loss fn: {args.edc_loss_fn.upper()}")

    if args.early_loss_fn == "mse":
        early_criterion = nn.MSELoss()
    else:
        early_criterion = nn.L1Loss()

    early_window_samples = int(round(args.early_window_ms * args.sample_rate / 1000.0))
    print(f"Early loss fn: {args.early_loss_fn.upper()}")
    print(f"Early window: {args.early_window_ms} ms ({early_window_samples} samples)")
    print()

    # Training loop
    print("Starting training...")
    print("=" * 60)

    best_val_loss = float("inf")
    best_epoch = 0
    best_val_wave_loss = float("inf")
    best_val_edc_loss = float("inf")
    best_val_early_loss = float("inf")
    best_val_effective_edc_weight = 0.0

    for epoch in range(args.epochs):
        train_loss, train_wave_loss, train_edc_loss, train_early_loss, train_effective_edc_weight = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            args.device,
            edc_loss_weight=args.edc_loss_weight,
            edc_criterion=edc_criterion,
            edc_auto_balance=args.edc_auto_balance,
            early_loss_weight=args.early_loss_weight,
            early_criterion=early_criterion,
            early_window_samples=early_window_samples,
        )
        
        if skip_validation:
            val_loss = train_loss  # Use train loss as proxy if no validation set
            val_wave_loss = train_wave_loss
            val_edc_loss = train_edc_loss
            val_early_loss = train_early_loss
            val_effective_edc_weight = train_effective_edc_weight
            print(
                f"Epoch {epoch + 1:3d}/{args.epochs} | "
                f"Train Total: {train_loss:.6f} | "
                f"Train Wave: {train_wave_loss:.6f} | "
                f"Train EDC: {train_edc_loss:.6f} | "
                f"Train Early: {train_early_loss:.6f} | "
                f"Train EDC w: {train_effective_edc_weight:.6f}"
            )
        else:
            val_loss, val_wave_loss, val_edc_loss, val_early_loss, val_effective_edc_weight = evaluate(
                model,
                val_loader,
                criterion,
                args.device,
                edc_loss_weight=args.edc_loss_weight,
                edc_criterion=edc_criterion,
                edc_auto_balance=args.edc_auto_balance,
                early_loss_weight=args.early_loss_weight,
                early_criterion=early_criterion,
                early_window_samples=early_window_samples,
            )
            print(
                f"Epoch {epoch + 1:3d}/{args.epochs} | "
                f"Train Total: {train_loss:.6f} | "
                f"Val Total: {val_loss:.6f} | "
                f"Val Wave: {val_wave_loss:.6f} | "
                f"Val EDC: {val_edc_loss:.6f} | "
                f"Val Early: {val_early_loss:.6f} | "
                f"Val EDC w: {val_effective_edc_weight:.6f}"
            )

        scheduler.step()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_wave_loss = val_wave_loss
            best_val_edc_loss = val_edc_loss
            best_val_early_loss = val_early_loss
            best_val_effective_edc_weight = val_effective_edc_weight
            best_epoch = epoch
            checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_wave_loss": val_wave_loss,
                    "val_edc_loss": val_edc_loss,
                    "val_early_loss": val_early_loss,
                    "val_effective_edc_weight": val_effective_edc_weight,
                    "args": vars(args),
                    "room_context_columns": room_context_columns,
                },
                checkpoint_path,
            )
            print(f"  -> Saved best model to {checkpoint_path}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_wave_loss": val_wave_loss,
                    "val_edc_loss": val_edc_loss,
                    "val_early_loss": val_early_loss,
                    "val_effective_edc_weight": val_effective_edc_weight,
                    "args": vars(args),
                    "room_context_columns": room_context_columns,
                },
                checkpoint_path,
            )

    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.6f} (epoch {best_epoch + 1})")
    print(f"Best validation wave loss: {best_val_wave_loss:.6f}")
    print(f"Best validation EDC loss: {best_val_edc_loss:.6f}")
    print(f"Best validation early loss: {best_val_early_loss:.6f}")
    print(f"Best validation effective EDC weight: {best_val_effective_edc_weight:.6f}")
    print()

    # Evaluate on test set
    print("Evaluating on test set...")
    checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if len(test_loader) > 0:
        test_loss, test_wave_loss, test_edc_loss, test_early_loss, test_effective_edc_weight = evaluate(
            model,
            test_loader,
            criterion,
            args.device,
            edc_loss_weight=args.edc_loss_weight,
            edc_criterion=edc_criterion,
            edc_auto_balance=args.edc_auto_balance,
            early_loss_weight=args.early_loss_weight,
            early_criterion=early_criterion,
            early_window_samples=early_window_samples,
        )
        print(f"Test Total Loss: {test_loss:.6f}")
        print(f"Test Wave Loss: {test_wave_loss:.6f}")
        print(f"Test EDC Loss: {test_edc_loss:.6f}")
        print(f"Test Early Loss: {test_early_loss:.6f}")
        print(f"Test effective EDC weight: {test_effective_edc_weight:.6f}")
    else:
        test_loss = best_val_loss
        test_wave_loss = best_val_wave_loss
        test_edc_loss = best_val_edc_loss
        test_early_loss = best_val_early_loss
        test_effective_edc_weight = best_val_effective_edc_weight
        print("Test set is empty. Using validation loss as test metric.")
    print()

    # Save training summary
    summary = {
        "best_epoch": best_epoch + 1,
        "best_val_loss": float(best_val_loss),
        "best_val_wave_loss": float(best_val_wave_loss),
        "best_val_edc_loss": float(best_val_edc_loss),
        "best_val_early_loss": float(best_val_early_loss),
        "best_val_effective_edc_weight": float(best_val_effective_edc_weight),
        "test_loss": float(test_loss),
        "test_wave_loss": float(test_wave_loss),
        "test_edc_loss": float(test_edc_loss),
        "test_early_loss": float(test_early_loss),
        "test_effective_edc_weight": float(test_effective_edc_weight),
        "hyperparameters": vars(args),
        "model_parameters": num_params,
    }
    summary_path = os.path.join(args.output_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved training summary to {summary_path}")


if __name__ == "__main__":
    main()
