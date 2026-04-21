"""
Inference script for estimating RIR from source-receiver coordinates using a trained model.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

from models.rir_network import RIRNetwork


def load_model(checkpoint_path: str, device: str) -> RIRNetwork:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct model from saved arguments
    args = checkpoint["args"]
    model = RIRNetwork(
        input_dim=6,
        output_dim=args["output_dim"],
        hidden_dim=args["hidden_dim"],
        num_hidden_layers=args["num_hidden_layers"],
        num_frequencies=args["num_frequencies"],
        room_context_dim=0,
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model


def predict_rir(
    model: RIRNetwork,
    src_pos: np.ndarray,
    rec_pos: np.ndarray,
    sr: int = 16000,
    device: str = "cuda",
) -> np.ndarray:
    """
    Predict RIR for given source and receiver positions.
    
    Args:
        model: Trained RIRNetwork model
        src_pos: Source position [x, y, z] in meters
        rec_pos: Receiver position [x, y, z] in meters
        sr: Sample rate in Hz
        device: Device to run inference on
        
    Returns:
        RIR waveform as numpy array
    """
    # Create coordinate tensor
    coords = np.concatenate([src_pos, rec_pos])
    coords = torch.from_numpy(coords).float().unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        rir = model(coords)
    
    # Convert to numpy
    rir = rir.squeeze(0).cpu().numpy()
    
    return rir


def main():
    parser = argparse.ArgumentParser(
        description="Predict RIR using trained model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--src-x",
        type=float,
        required=True,
        help="Source X coordinate in meters",
    )
    parser.add_argument(
        "--src-y",
        type=float,
        required=True,
        help="Source Y coordinate in meters",
    )
    parser.add_argument(
        "--src-z",
        type=float,
        required=True,
        help="Source Z coordinate in meters",
    )
    parser.add_argument(
        "--rec-x",
        type=float,
        required=True,
        help="Receiver X coordinate in meters",
    )
    parser.add_argument(
        "--rec-y",
        type=float,
        required=True,
        help="Receiver Y coordinate in meters",
    )
    parser.add_argument(
        "--rec-z",
        type=float,
        required=True,
        help="Receiver Z coordinate in meters",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predicted_rir.wav",
        help="Output file path for RIR wav",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Sample rate in Hz",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch predict from CSV file (requires --csv input)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="CSV file with columns: src_x, src_y, src_z, rec_x, rec_y, rec_z",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="predictions",
        help="Output directory for batch predictions",
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.device)
    print(f"Using device: {args.device}")
    print()

    if args.batch:
        # Batch prediction from CSV
        if not args.csv:
            print("Error: --csv required for batch prediction")
            return

        import pandas as pd

        print(f"Loading predictions from {args.csv}...")
        df = pd.read_csv(args.csv)
        
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"Predicting RIRs for {len(df)} samples...")
        for idx, row in df.iterrows():
            src_pos = np.array([row["src_x"], row["src_y"], row["src_z"]])
            rec_pos = np.array([row["rec_x"], row["rec_y"], row["rec_z"]])
            
            rir = predict_rir(model, src_pos, rec_pos, args.sr, args.device)
            
            output_path = os.path.join(args.output_dir, f"rir_{idx:04d}.wav")
            sf.write(output_path, rir, args.sr)
            
            if (idx + 1) % 10 == 0:
                print(f"  Saved {idx + 1} / {len(df)} predictions")
        
        print(f"Batch prediction complete! Saved to {args.output_dir}/")

    else:
        # Single prediction
        src_pos = np.array([args.src_x, args.src_y, args.src_z])
        rec_pos = np.array([args.rec_x, args.rec_y, args.rec_z])

        print("Predicting RIR...")
        print(f"Source position: ({args.src_x}, {args.src_y}, {args.src_z})")
        print(f"Receiver position: ({args.rec_x}, {args.rec_y}, {args.rec_z})")
        print()

        rir = predict_rir(model, src_pos, rec_pos, args.sr, args.device)

        print(f"Predicted RIR shape: {rir.shape}")
        print(f"RIR value range: [{rir.min():.4f}, {rir.max():.4f}]")
        print(f"RIR energy: {(rir ** 2).sum():.4f}")
        print()

        # Save RIR
        sf.write(args.output, rir, args.sr)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
