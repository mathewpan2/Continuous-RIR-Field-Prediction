# Focus on Early RIR Reconstruction

## Motivation

The original full-RIR reconstruction objective asks the model to predict an entire room impulse response, including direct sound, early reflections, and the late reverberant tail. That is a very broad target: the late tail depends on dense high-order reflections, room materials, scattering, and simulation details that are difficult to infer from only source and receiver coordinates.

Following the project feedback, this version narrows the research question to **early RIR reconstruction**. The early part of an RIR is more physically interpretable and more closely tied to room geometry: it includes direct sound and early reflections that encode source-receiver distance, nearby surfaces, and first-order reflection structure. This makes the project better scoped while still preserving an acoustically meaningful prediction problem.

## Target Definition

The early target is defined as the **fixed first 80 ms of the RIR from sample 0**. It is not aligned to the direct sound peak.

For the current Apartments RIR data:

- sample rate: 22050 Hz
- early window: 80 ms
- output length: `round(22050 * 80 / 1000) = 1764` samples

The full-RIR mode remains available for comparison, but the default training and export mode is now early reconstruction.

## Model Interface

The model inputs are unchanged:

- `rir_small`: 6D source and receiver coordinates
- `rir_depth`: 6D coordinates plus the receiver depth map
- `base_model`: 6D coordinate baseline through the existing LSTM-style baseline
- `traditional_way_baseline`: non-learning image-source baseline with direct path and first-order wall reflections

The main change is the output target:

- full RIR: `(batch, 22050)` for a 1-second waveform at 22050 Hz
- early RIR: `(batch, 1764)` for the first 80 ms at 22050 Hz

No larger architecture is introduced. The project is refocused by changing the prediction target, not by adding a bigger model.

## Traditional Geometry Baseline

The `traditional_way_baseline` provides a comparison point against the learned early-RIR models. It estimates a simple axis-aligned shoebox room from the available source and receiver coordinate ranges, then synthesizes an early impulse response from:

- the direct source-to-receiver path
- six first-order image-source reflections, one for each shoebox wall
- distance-based amplitude decay
- a fixed reflection gain

This baseline does not use training labels to learn parameters. It is intentionally simple: the available metadata does not include full wall geometry or material absorption, so the method should be treated as a traditional geometric approximation rather than a full acoustic simulator.

## Training

Train the recommended coordinate baseline:

```bash
python scripts/train_models.py --models rir_small --target-mode early --early-ms 80 --epochs 5
```

Train the depth-conditioned model:

```bash
python scripts/train_models.py --models rir_depth --target-mode early --early-ms 80 --epochs 5
```

Create the traditional geometry baseline checkpoint:

```bash
python scripts/train_models.py --models traditional_way_baseline --target-mode early --early-ms 80
```

Early checkpoints are named with the target window to avoid overwriting full-RIR runs:

```text
artifacts/checkpoints/rir_small_early80ms.pt
artifacts/checkpoints/rir_depth_early80ms.pt
artifacts/checkpoints/traditional_way_baseline_early80ms.pt
```

To run the older full-RIR objective:

```bash
python scripts/train_models.py --models rir_small --target-mode full --target-length 22050
```

## Exporting Predictions

Export early predictions on the held-out room split:

```bash
python scripts/export_predictions.py --models rir_small --target-mode early --early-ms 80
```

Export the traditional baseline:

```bash
python scripts/export_predictions.py --models traditional_way_baseline --target-mode early --early-ms 80
```

The result file includes prediction arrays and target metadata:

- `y_true`
- `y_pred`
- `sample_id`
- `method_name`
- `sample_rate`
- `output_dim`
- `target_mode`
- `early_ms`

Example output path:

```text
result/rir_small_early80ms_predictions.npz
```

## Evaluation

Recommended evaluation focuses on the early window:

- **Waveform MSE**: direct sample-wise error over the first 80 ms.
- **Early energy error**: compare `sum(rir[:N] ** 2)` between prediction and target.
- **Qualitative waveform plots**: overlay predicted and ground-truth early RIRs for selected test samples.

These metrics match the scoped objective: reconstructing the early acoustic structure rather than claiming exact full-room reverberation synthesis.

## Research Framing

This pivot makes the project easier to justify:

- Full RIR reconstruction can require large-scale simulation data, material labels, and high-capacity models.
- EDC mediation is useful for perceptual decay behavior, but it can be abstract when exact reconstruction is not necessary.
- Early RIR reconstruction is concrete, measurable, and connected to geometry-driven acoustics.
- The project can compare coordinate-only prediction against depth-conditioned prediction without turning into a big-model training effort.
