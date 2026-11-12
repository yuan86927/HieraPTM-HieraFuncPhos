# Task1: Complete Training and Evaluation Code for Eight PTM Heads

This directory provides the final HieraPTM model definitions, two-stage training, data loading, validation-based checkpoint selection, checkpoint recovery, and evaluation on the fixed test set. Each of the eight PTM heads is trained separately.

1. Sequence Backbone: train the CNN–BiGRU, gated ESM2 fusion, and center-aware Transformer.
2. HieraPTM: load the selected Sequence Backbone, freeze the sequence backbone and its classifier, and train the 3D GAT and cross-attention residual branch.

Training uses the released split tables for the eight PTM heads.

## File Organization

```text
train.py                  # Training command
evaluate.py               # Evaluation command
configs/task1.json        # Parameters and input paths for all eight heads
src/
  models.py               # Public model interface
  sequence_backbone.py    # Sequence Backbone
  hieraptm.py             # HieraPTM 3D branch and complete model
  backbones.py            # CNN, BiGRU, and center-aware Transformer
  esm_fusion.py           # ESM projection and gating
  graph_layers.py         # GAT layers
  dataset.py              # Final split tables and tensor loading
  data.py                 # Batch assembly, data loaders, and input checks
  esm_cache.py            # ESM feature shard reader
  graph_data.py           # Graph cache reader and validation helpers
  training.py             # Two-stage training and validation-based selection
  sequence_optimization.py # Sequence backbone training/validation per epoch
  structure_optimization.py # 3D branch optimizer, scheduler, and training per epoch
  evaluation.py           # Selected checkpoint loading, test inference, and metrics
  checkpoints.py          # Checkpoint saving and training recovery
  metrics.py              # AUROC, AUPRC, and ROC/PR curve data
  reporting.py            # Prediction tables and figures
  configuration.py       # Configuration loading and path resolution
  pipeline.py            # Feature resource management and workflow dispatch
  utils.py                # Shared training utilities
```

## Environment

Use Python 3.11 or later. First install a CUDA-enabled PyTorch build compatible with your hardware (torch 2.5 or later), then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

## Input Configuration

Edit `paths` in `configs/task1.json`. Relative paths are resolved against the directory containing this README. Absolute paths are also supported.

| Configuration field | Input |
|---|---|
| `split_directory` | Directory containing the eight final `<head>.tsv.gz` site tables |
| `esm_cache_root` | Root directory of frozen 640-dimensional ESM2 residue feature shards |
| `esm_cache_registry` | ESM shard registry TSV |
| `prepared_site_index` | TSV mapping sites to ESM windows |
| `prepared_protein_index` | TSV mapping proteins to full residue feature shards |
| `graph_cache_root` | Directory containing the prepared site-centered 3D subgraphs |
| `output_root` | Output directory for new training runs |

Configure the paths above for the released split tables and precomputed ESM2 and 3D graph features. For evaluation, use the checkpoints from the training run.

Use the released site tables with their original columns, including `sample_id`, `protein_id`, `ptm`, `head`, `sequence_61`, `label`, `valid_mask_61`, `split`, `fold_id`, center-site information, and provenance fields.

`fit`, `development`, and `relaxed_test` correspond to `fold_id=0`, `1`, and `2`, respectively. The approximate training, validation, and test proportions are 81%/9%/10%.

## Training and Evaluation

First check the eight split tables, then check feature file paths and shard availability:

```bash
python train.py --stage check-data
python train.py --stage check-inputs
```

Train all eight heads. Checkpoints are selected by validation AUROC, with AUPRC as the secondary criterion:

```bash
python train.py
```

To train a single head:

```bash
python train.py --head phosphorylation_st
```

After training, evaluate the selected checkpoints on the fixed test set:

```bash
python evaluate.py
```

To resume interrupted training, rerun the same training command. Use `python evaluate.py --head phosphorylation_st` to evaluate a single head.

By default, each head writes to `runs/task1/<head>/sequence_backbone/` and `runs/task1/<head>/HieraPTM/`. Outputs include `checkpoints/best.pt`, training logs, validation metrics, and the per-sample predictions, ROC/PR coordinates, and metrics generated during explicit test evaluation. The final HieraPTM checkpoint contains both the complete sequence backbone and the 3D branch.

## Fixed Training Parameters

| Parameter | Sequence Backbone | HieraPTM |
|---|---:|---:|
| Batch size | 64 | 64 |
| Maximum epochs | 80 | 30 |
| Early-stopping patience | 10 | 6 |
| Weight decay | 0.0001 | 0.0001 |
| Optimizer | AdamW | AdamW with parameter groups |
| Learning rate | 0.0003 | graph 0.00005 / cross 0.0001 / gate 0.0002 |
| Loss | Weighted BCE | Weighted BCE + 0.001 × residual-logit MSE |

Training uses FP16 AMP and gradient clipping at 1.0. HieraPTM uses 8% warmup followed by cosine decay, with a minimum learning-rate ratio of 0.1. See the source and configuration for model dimensions and complete settings. AUPRC is computed as average precision.

## Feature File Formats

The ESM registry must include at least `cache_shard_id`, `array_relative_path`, `unique_residues`, `array_sha256`, and `status` (`COMPLETE`). Each shard is a float16 `.npy` array of shape `[number_of_residues,640]`.

Site index columns: `sample_id`, `protein_id`, `center_position_1based`, `cache_shard_id`, `protein_shard_offset0`, `window_protein_start0`, `window_protein_end0_exclusive`, `window_shard_start0`, `window_shard_end0_exclusive`, `left_padding`, `right_padding`, and `valid_mask_61`.

Protein index columns: `protein_id`, `protein_length`, `cache_shard_id`, and `protein_shard_offset0`.

Use a prepared 3D graph cache at `graph_cache_root`. It contains `manifest.json`, `READY.flag`, `sample_index.tsv` with the columns `sample_id` and `row_id`, and the following arrays:

| File | Shape and dtype |
|---|---|
| `node_positions_1based.int32.npy` | `[N,33]` int32 |
| `node_centered_ca.float16.npy` | `[N,33,3]` float16 |
| `node_geometry.float16.npy` | `[N,33,8]` float16 |
| `node_mask.bool.npy` | `[N,33]` bool |

The 33 node slots contain the center and up to 32 neighbors.
