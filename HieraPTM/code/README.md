# Task1: Complete Training and Evaluation Code for Eight PTM Heads

This directory provides the final HieraPTM model definitions, two-stage training, data loading, validation-based checkpoint selection, checkpoint recovery, and evaluation on the fixed test set. Each of the eight PTM heads is trained separately.

1. Sequence Backbone: train the CNN–BiGRU, gated ESM2 fusion, and center-aware Transformer.
2. HieraPTM: load the selected Sequence Backbone, freeze the sequence backbone and its classifier, and train the 3D GAT and cross-attention residual branch.

The model architecture and training computations originate from the final frozen archive. The public names are Sequence Backbone and HieraPTM. Renaming preserves the original checkpoint parameter keys, tensor structure, and computations. `SOURCE_PROVENANCE.json` records the historical sources, name mappings, retained functions, and SHA256 hashes. Training uses the final split tables directly, without resampling or searching for development-set seeds.

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
  utils.py               # Random seeds, sampling, and file utilities
```

## Environment

Use Python 3.11 or later. First install a CUDA-enabled PyTorch build compatible with your hardware (torch 2.5 or later), then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

This package does not include a smoke-test entry point. The module split was checked using static syntax, import-reference, and source-structure checks only; no model or training code was executed during that refactoring. Hardware and software may differ from the historical training environment, so retraining is not guaranteed to produce bitwise-identical weights.

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

This delivery contains code and configuration only. Data tables, pretrained/final weights, and large feature caches are not copied into the code package; provide them through the paths above. Training consumes precomputed frozen features. The pipeline for generating feature caches from raw FASTA files and structures is not included in this delivery.

Site tables must retain the previously exported columns, including `sample_id`, `protein_id`, `ptm`, `head`, `sequence_61`, `label`, `valid_mask_61`, `split`, `fold_id`, center-site information, and provenance fields. The reader checks row counts, sample uniqueness, labels, window lengths, consistency between `split` and `fold_id`, and the SHA256 hash of the final split vector. If the export's `manifest.json` is present in the directory, the SHA256 hash of each complete table is also checked.

`fit`, `development`, and `relaxed_test` correspond to `fold_id=0`, `1`, and `2`, respectively; this field encodes the data role. The approximate training/validation/test proportions are 81%/9%/10%. Splitting is performed at the site level, and proteins may appear in multiple sets. The final development-set seed is 20260835 for Pho-ST, 20260831 for Ubi-K, and 20260830 for the other heads. The training seed is 20260830 for every head. Pho-ST and Ubi-K require tables with the final split overlays already merged; the hash checks reject the earlier splits.

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

Training does not compute outer-test predictions. After validation-based checkpoint selection is complete for all eight heads, evaluate the selected weights on the fixed test set:

```bash
python evaluate.py
```

`evaluate.py` first checks the training-completion markers and weights for every selected head and fails if any are missing. Evaluation independently loads the frozen weights without creating an optimizer or calling a training loop. Repeated runs reuse completed results with matching identities. To resume interrupted training, rerun the same training command. Checkpoints are rejected if their data, configuration, or upstream Sequence Backbone weights do not match. Use `python evaluate.py --head phosphorylation_st` to evaluate a single head.

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

The 3D graph directory retains the original training-cache convention: `manifest.json` with `status=READY_FOR_G1` (a historical cache-format name), `READY.flag`, `sample_index.tsv` containing `sample_id,row_id`, and the following arrays:

| File | Shape and dtype |
|---|---|
| `node_positions_1based.int32.npy` | `[N,33]` int32 |
| `node_centered_ca.float16.npy` | `[N,33,3]` float16 |
| `node_geometry.float16.npy` | `[N,33,8]` float16 |
| `node_mask.bool.npy` | `[N,33]` bool |

The 33 node slots contain the center and up to 32 neighbors. Input loading follows the original implementation, including missing-ESM masks, graph-node masks, and fallback to the sequence model when 3D information is unavailable.
