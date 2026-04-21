# RIR Prediction Training Script

This training script sets up and trains the `RIRNetwork` model to predict continuous Room Impulse Response (RIR) waveforms from source-receiver coordinates in the acoustic rooms dataset.

## Overview

The script:
1. Loads RIR data from `single_channel_ir_1/` directory with metadata from `metadata/`
2. Splits data into train/val/test sets (80/10/10) by room to avoid leakage
3. Creates a PyTorch dataset and dataloader for efficient batching
4. Trains the RIRNetwork model using MSE or L1 loss
5. Validates after each epoch and saves the best model
6. Evaluates on the test set
7. Saves checkpoints and training summary

## Quick Start

### Option 1: Testing with a Subset (Recommended for First Run)

For quick iteration and testing, use a subset of the dataset:

```bash
# Quick test: 2 rooms, up to 50 epochs
python train.py --max-rooms 2 --epochs 50 --output-dir checkpoints/test

# Fast test: 50 samples, 10 epochs
python train.py --max-samples 50 --epochs 10 --output-dir checkpoints/quick_test

# Medium test: 5 rooms, 20 epochs
python train.py --max-rooms 5 --epochs 20 --output-dir checkpoints/medium_test
```

### Option 2: Using the bash script (recommended)

```bash
chmod +x run_training.sh
./run_training.sh [batch_size] [epochs] [learning_rate] [device]
```

Examples:
```bash
./run_training.sh 16 50 0.001 cuda
./run_training.sh 32 100 0.0005 cuda
./run_training.sh 8 20 0.002 cpu
```

### Option 2: Direct Python command

```bash
python train.py \
    --ir-base single_channel_ir_1 \
    --meta-base metadata \
    --output-dir checkpoints \
    --batch-size 16 \
    --epochs 50 \
    --lr 0.001 \
    --device cuda
```

## Command-line Arguments

```
--ir-base           Path to single_channel_ir_1 directory (default: single_channel_ir_1)
--meta-base         Path to metadata directory (default: metadata)
--output-dir        Directory to save checkpoints (default: checkpoints)
--batch-size        Batch size for training (default: 16)
--epochs            Number of epochs to train (default: 50)
--lr                Learning rate (default: 1e-3)
--hidden-dim        Hidden dimension for MLP (default: 512)
--num-hidden-layers Number of hidden layers (default: 4)
--num-frequencies   Number of Fourier frequencies (default: 8)
--output-dim        Output RIR length in samples (default: 16000)
--device            Device to train on (default: cuda if available, else cpu)
--seed              Random seed (default: 42)
--loss-fn           Loss function: mse or l1 (default: mse)
--max-rooms         Maximum number of unique rooms to use (default: None = all)
--max-samples       Maximum number of samples to use (default: None = all)
```

### Dataset Limiting Parameters

Use `--max-rooms` or `--max-samples` for quick testing and iteration:

- **`--max-rooms N`**: Use only N random unique rooms from the dataset
  - Useful for: Pipeline validation, architecture testing, quick iterations
  - Example: `--max-rooms 5` trains on ~5-10% of the data

- **`--max-samples N`**: Use only N random samples total
  - Useful for: Memory-constrained environments, very quick tests
  - Example: `--max-samples 100` for minimal testing

Both parameters respect the 80/10/10 train/val/test split.

## Model Architecture

The `RIRNetwork` uses:
- **Input**: 6-dimensional coordinates [src_x, src_y, src_z, rec_x, rec_y, rec_z]
- **Encoding**: Fourier feature encoding with positional embeddings
- **Processing**: Multi-layer MLP with SiLU activations
- **Output**: RIR waveform (default 16,000 samples at 16 kHz ≈ 1 second)

## Output Structure

Training creates the following files in `--output-dir`:

```
checkpoints/
├── best_model.pt            # Best model checkpoint (lowest val loss)
├── checkpoint_epoch_10.pt   # Checkpoint every 10 epochs
├── checkpoint_epoch_20.pt
├── ...
└── training_summary.json    # Final metrics and hyperparameters
```

### training_summary.json example:
```json
{
  "best_epoch": 42,
  "best_val_loss": 0.003456,
  "test_loss": 0.003512,
  "model_parameters": 2097152,
  "hyperparameters": {
    "batch_size": 16,
    "epochs": 50,
    "lr": 0.001,
    ...
  }
}
```

## Data Format

### Input coordinates
- **src_x, src_y, src_z**: Source (speaker) position in meters
- **rec_x, rec_y, rec_z**: Receiver (microphone) position in meters

### Target RIR
- Waveform normalized to [-1, 1]
- Length: 16,000 samples (1 second at 16 kHz)
- Variable lengths in the dataset are padded to max batch length

## Training Tips

### GPU Memory
- Reduce `--batch-size` if running out of memory
- Start with 16 and increase if you have spare GPU memory

### Convergence
- Lower learning rate (e.g., 0.0005) for more stable training
- Increase `--num-hidden-layers` for larger model capacity
- Use `--loss-fn l1` for more robust loss (less sensitive to outliers)

### Model Size
- Increase `--hidden-dim` to 1024 for larger capacity
- Increase `--num-frequencies` to 16 for better frequency resolution

## Loading a Trained Model

```python
import torch
from models.rir_network import RIRNetwork

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pt')

# Create model
model = RIRNetwork(
    input_dim=6,
    output_dim=16000,
    hidden_dim=512,
    num_hidden_layers=4,
    num_frequencies=8,
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
with torch.no_grad():
    coords = torch.randn(1, 6)  # [src_x, src_y, src_z, rec_x, rec_y, rec_z]
    rir = model(coords)  # Shape: (1, 16000)
```

## Dataset Statistics

The script automatically prints dataset statistics:
```
Rooms  — train: 320, val: 40, test: 40
Samples — train: 12800, val: 1600, test: 1600
```

## Troubleshooting

- **"No such file or directory" errors**: Ensure you're running from the workspace root and data directories exist
- **CUDA out of memory**: Reduce `--batch-size`
- **Very slow training**: Check that GPU is being used (set `--device cpu` to verify CPU speed)
- **NaN losses**: Try lower learning rate or gradient clipping is enabled (already included)

## References

- Model: RIRNetwork with Fourier feature encoding
- Dataset: AcousticRooms with single-channel impulse responses
- Training: Cosine annealing LR schedule with MSE/L1 loss
