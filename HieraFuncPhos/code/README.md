# Task2: HieraPTM Final Model Training, Validation, and Test Code

This README corresponds to the Chinese review version, `README_zh-CN.md`.

This code package is intended for functional phosphorylation prediction and includes the final model training, validation, and test workflows for nine replicates, rep01–rep09.

## 1. Data Split Protocol

Each replicate uses one fixed training, validation, and test set:

```text
data/splits/
  rep01_train.parquet
  rep01_validation.parquet
  rep01_test.parquet
  ...
  rep09_train.parquet
  rep09_validation.parquet
  rep09_test.parquet
```

The complete dataset consists of 27 Parquet tables. Each table contains site identifiers, labels, sequence windows, and the final split assignment, and can be read independently.

The development set is the union of the training and validation sets; it is not saved as a separate, duplicated file.

## 2. Data Split Protocol

1. Divide each replicate into approximately 80% development data and 20% test data.
2. Within the 80% development set, divide the data into training and validation sets at a ratio of 4:1.

The training, validation, and test sets therefore account for approximately 64%, 16%, and 20% of all data, respectively. Retaining the original class-balanced allocation gives the following counts for each replicate:

| Dataset | Total sites | Functional positives | Unannotated background sites |
|---|---:|---:|---:|
| Development: training + validation | 14,410 | 7,205 | 7,205 |
| Training | 11,528 | 5,764 | 5,764 |
| Validation | 2,882 | 1,441 | 1,441 |
| Test | 3,604 | 1,802 | 1,802 |

Development/test membership is unchanged from the original data. For each replicate, one previously saved 4:1 training/validation assignment is used without resampling or choosing a split based on model performance.

Each replicate uses only one fixed training/validation/test assignment. No cross-validation assignment columns are provided, and no five-fold cross-validation is performed.

Splitting is performed at the site level. Within a replicate, site IDs and canonical physical-site keys do not overlap between the training, validation, and test sets. However, the same protein may appear in different sets, so the split is not protein-disjoint. Protein overlap counts are recorded in the data manifest.

## 3. Labels and Shared Sites Across Replicates

Each replicate contains 18,014 sites: 9,007 functional positives and 9,007 unannotated background sites.

Label 0 denotes unannotated background, not a site experimentally established to be nonfunctional.

All nine replicates share the same set of 9,007 positive sites, while their background sets are pairwise disjoint. Together, they contain 90,070 distinct sites: 9,007 positives and 81,063 unannotated background sites.

Because the replicates share positive sites, these nine replicates should not be treated as mutually independent external validation datasets.

## 4. Data Fields

| Field | Meaning |
|---|---|
| `site_id` | Original unique site identifier |
| `physical_site_key` | Original physical-site provenance key |
| `mmf_canonical_site_key` | Original canonical site key |
| `accession_base` | Protein accession used by the model pipeline |
| `sequence_lookup_accession` | Accession used to retrieve the original protein sequence |
| `sequence_length` | Length of the full original protein sequence |
| `sequence_sha256` | SHA256 of the full original protein sequence |
| `residue` | Central phosphoacceptor residue: S, T, or Y |
| `position_1based` | Position in the original protein, using one-based indexing |
| `mmf_label` | 1: functional positive; 0: unannotated background |
| `mmf_replicate` | Public replicate identifier, ranging from 1 to 9 |
| `mmf_split` | Split assignment: `train`, `validation`, or `test` |
| `mmf_split_seed` | Preserved original development/test split seed |
| `sequence31` | Original 31-residue sequence window; center index 15, using zero-based indexing |
| `sequence61` | Original 61-residue sequence window; center index 30, using zero-based indexing |

Sequences, labels, site positions, protein identifiers, and site IDs remain unchanged. Original row order is preserved within each exported subset.

## 5. Reading the Data

```python
import pandas as pd

replicate = 1
train = pd.read_parquet(f"data/splits/rep{replicate:02d}_train.parquet")
validation = pd.read_parquet(f"data/splits/rep{replicate:02d}_validation.parquet")
test = pd.read_parquet(f"data/splits/rep{replicate:02d}_test.parquet")
```

Reading the data requires pandas and a Parquet engine such as pyarrow.

`manifest.json` preserves the preceding data-preparation records, including replicate mappings, split counts, data provenance, data-preservation checks, and expected data-file hashes.

`FILE_MANIFEST_SHA256.tsv` records the files currently present in the code directory, their sizes, and their SHA256 hashes. It does not imply that external data, feature caches, or model weights are included in this directory.

No model training or inference was performed while preparing this code package. Full protein sequences, structure files, feature caches, and model weights are handled as separate resources.

## 6. Final Model Code Structure

This package retains the final two-stage HieraPTM workflow:

1. Train the functional prediction LoRA model using residue-matched ST/Y branches.
2. Freeze the selected functional prediction model and its functional classification head, then train the dynamics adapter.

Each replicate uses only one fixed training/validation/test assignment. No cross-validation runners, alternative baseline-model training entry points, or multi-model prediction averaging workflows are provided.

```text
train.py                    # Two-stage training and validation-based checkpoint selection
evaluate.py                 # Evaluate the selected final model on validation or test data
config.json                 # File paths, runtime settings, and final hyperparameters
requirements.txt
hieraptm/
  config.py                 # Public/source replicate mapping and source-weight checks
  data.py                   # Fixed-split loading, cache alignment, and tensor conversion
  train_lora.py             # Functional LoRA training and checkpoint selection
  train_dynamics.py         # Dynamics training and selection with the functional model frozen
  anchors.py                # Representation extraction from the frozen functional model
  pipeline.py               # Training orchestration
  evaluation.py             # Final predictions and STY/ST/Y/S/T metrics
  lora_math.py               # Original optimizer groups, warmup, and per-epoch computation
  dynamic_math.py            # Original dynamics losses and numerical safeguards
  normalization.py          # Training-only normalization of Motion/Response/Network features
  feature_names.py          # Feature names
  metrics.py                # Evaluation metrics
  utils.py                  # General utilities
  models/                   # Backbone, structure encoder, LoRA, and dynamics modules
CODE_PROVENANCE.json         # Archived source filenames and checksums
STATIC_CHECKS.json           # Static-check records; no model execution tests
```

Each feature-cache directory must contain `index.parquet`, `audit.json`, and the NumPy arrays listed below. The number of sites, N, is 18,014; the shapes below exclude the leading site dimension N.

| Array names | Shape excluding the site dimension | Storage dtype |
|---|---|---|
| `token`, `attention` | 61 | uint8, bool |
| `esm150` | 61 × 640 | float16 |
| `positions`, `node_mask` | 33 | int32, bool |
| `coordinates`, `geometry`, `node_esm` | 33 × 3, 33 × 8, 33 × 640 | float16 |
| `global` | 2560 | float16 |
| `local_esm`, `remote_esm` | 25 × 2560, 48 × 2560 | float16 |
| `site_quality` | 5 | float32 |
| `node`, `quality` | 73 × 28, 73 × 4 | float32 |
| `view_mask`, `mask` | 73 × 3, 73 | bool |
| `local_pair`, `cross` | 25 × 25 × 8, 25 × 48 × 8 | float16 |
| `cross_reverse_prs`, `center_pair` | 25 × 48, 48 × 8 | float16 |

### 6.1 Training and Validation

Run the following commands from the code-package directory. Relative paths in `config.json` are resolved relative to the directory containing that configuration file. Public replicate numbers used on the command line range from 1 to 9.

Install the dependencies and train rep01:

```bash
pip install -r requirements.txt
python train.py --config config.json --rep 1
```

Train all nine replicates sequentially:

```bash
python train.py --config config.json --rep 1 2 3 4 5 6 7 8 9
```

### 6.2 Final Validation or Test Evaluation

```bash
# Evaluate the validation set for rep01
python evaluate.py --config config.json --rep 1 --split validation

# Evaluate the test set for rep01
python evaluate.py --config config.json --rep 1 --split test

# Evaluate the test sets for all nine replicates sequentially
python evaluate.py --config config.json --rep 1 2 3 4 5 6 7 8 9 --split test
```
