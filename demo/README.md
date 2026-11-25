# HieraPTM demo

The demo runs HieraPTM on two experimentally established human phosphorylation
sites from the released fixed test set. `demo_features.npz` contains the prepared
sequence, ESM-2, and 3D graph tensors for these two examples.

## Run the demo

Keep `README.md`, `demo.py`, and `demo_features.npz` together under `demo/` at
the repository root. Use Python 3.11 or later and install the dependencies as
described in the [repository README](../readme.md). The demo runs on a CPU.

The script loads the released checkpoints from
`HieraPTM/weights/phosphorylation_st/best.pt` and
`HieraPTM/weights/phosphorylation_y/best.pt`.

Run from the repository root.

```bash
python demo/demo.py
```

To run only the ERK1 example, use `python demo/demo.py --head phosphorylation_y`.
To change the output location, pass `--output path/to/predictions.tsv`.

## Expected output

The default command prints two scores and writes `demo_predictions.tsv` in the
current working directory. The TSV includes the sequence window, UniProt
accession, sample ID, test-set label, computed score, archived score, and
absolute difference for each site.

| Human phosphosite | UniProt accession | Released test-set label | Archived HieraPTM score | Experimental evidence |
|---|---|---:|---:|---|
| AKT1 Ser473 | P31749 | 1 | 0.99905223 | [Sarbassov et al., 2005](https://pubmed.ncbi.nlm.nih.gov/15718470/) |
| ERK1 Tyr204 | P27361 | 1 | 0.98250419 | [Kinoshita et al., 2008](https://pubmed.ncbi.nlm.nih.gov/18983981/) |

Label `1` denotes an annotated positive. The script checks that each computed
score agrees with the archived score within an absolute tolerance of `1e-5`.

## Measured runtime

The default command finished in about 10 seconds including Python startup and
imports, with about 3 seconds reported by the script itself. This measurement
used an Intel Core i7-14650HX, Windows 11, Python 3.13.12, and
PyTorch 2.10.0+cu126 in CPU mode.
