"""Join published fixed roles to an immutable, site-indexed feature cache.

Cache labels/legacy assignments never determine a split. The 27 public split
tables are the sole source of labels and train/validation/test membership.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .normalization import DynamicsNormalization
from .utils import read, sha


# Array names, trailing shapes, and storage dtypes from the frozen final cache.
ARRAYS = {
    "token": ((61,), "uint8"), "attention": ((61,), "bool"),
    "esm150": ((61, 640), "float16"), "positions": ((33,), "int32"),
    "coordinates": ((33, 3), "float16"), "geometry": ((33, 8), "float16"),
    "node_mask": ((33,), "bool"), "node_esm": ((33, 640), "float16"),
    "global": ((2560,), "float16"), "local_esm": ((25, 2560), "float16"),
    "remote_esm": ((48, 2560), "float16"), "site_quality": ((5,), "float32"),
    "node": ((73, 28), "float32"), "view_mask": ((73, 3), "bool"),
    "quality": ((73, 4), "float32"), "mask": ((73,), "bool"),
    "local_pair": ((25, 25, 8), "float16"), "cross": ((25, 48, 8), "float16"),
    "cross_reverse_prs": ((25, 48), "float16"), "center_pair": ((48, 8), "float16"),
}


class FeatureCache:
    def __init__(self, directory, split_root, replicate):
        self.directory = Path(directory)
        audit = read(self.directory / "audit.json")
        if sha(self.directory / "index.parquet") != audit["index_sha256"]:
            raise RuntimeError("Feature-cache index checksum differs")
        cached = pd.read_parquet(self.directory / "index.parquet")
        cached["site_id"] = cached.site_id.astype(str)
        if cached.site_id.duplicated().any() or not np.array_equal(cached.row_id, np.arange(len(cached))):
            raise RuntimeError("Expected unique cache site IDs and contiguous row_id in array order")

        parts, self.split_hashes = [], {}
        for role in ("train", "validation", "test"):
            path = Path(split_root) / f"rep{replicate:02d}_{role}.parquet"
            frame = pd.read_parquet(path)
            frame["site_id"] = frame.site_id.astype(str)
            if not frame.mmf_split.eq(role).all() or not frame.mmf_replicate.eq(replicate).all():
                raise RuntimeError(f"Incorrect published role or replicate: {path}")
            if not set(frame.mmf_label.unique()) <= {0, 1}:
                raise RuntimeError("Labels must be binary")
            if set(frame.mmf_label.unique()) != {0, 1}:
                raise RuntimeError(f"Both label classes required: {path}")
            parts.append(frame)
            self.split_hashes[role] = sha(path)
        published = pd.concat(parts, ignore_index=True)
        if len(published) != 18014 or published.site_id.duplicated().any():
            raise RuntimeError("Expected 18,014 unique sites across the three fixed roles")
        if published.physical_site_key.duplicated().any() or published.mmf_canonical_site_key.duplicated().any():
            raise RuntimeError("Physical/canonical sites overlap between fixed roles")
        if set(published.site_id) != set(cached.site_id):
            raise RuntimeError("Cache site membership differs from this public replicate")
        aligned = published.set_index("site_id").loc[cached.site_id].reset_index()
        # Validate metadata if present; never inherit cache labels or assignments.
        for source, public in (("label", "mmf_label"), ("residue", "residue"),
                               ("position_1based", "position_1based"),
                               ("protein_id", "accession_base"), ("sequence_sha256", "sequence_sha256")):
            if source in cached and not np.array_equal(cached[source].astype(str), aligned[public].astype(str)):
                raise RuntimeError(f"Cache metadata differs from public data: {source}")
        aligned["row_id"] = np.arange(len(aligned), dtype=np.int64)
        aligned["label"] = aligned.mmf_label.astype(np.float32)
        aligned["protein_id"] = aligned.accession_base.astype(str)
        aligned["residue_id"] = aligned.residue.map({"S": 0, "T": 1, "Y": 2})
        if aligned.residue_id.isna().any():
            raise RuntimeError("Only S/T/Y residues are supported")
        self.index = aligned
        self.roles = {role: np.flatnonzero(aligned.mmf_split.to_numpy() == role)
                      for role in ("train", "validation", "test")}
        self.arrays = {}
        for name, (shape, dtype) in ARRAYS.items():
            path = self.directory / (name + ".npy")
            expected_hash = audit["array_sha256"].get(path.name)
            if expected_hash is None or sha(path) != expected_hash:
                raise RuntimeError(f"Missing or changed feature-cache checksum: {path.name}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.shape != (len(aligned), *shape) or array.dtype != np.dtype(dtype):
                raise RuntimeError(f"Invalid cache shape/dtype: {name}: {array.shape}/{array.dtype}")
            self.arrays[name] = array
        self.audit_sha256 = sha(self.directory / "audit.json")


class SiteDataset(Dataset):
    def __init__(self, cache, ids):
        self.cache = cache
        self.ids = np.asarray(ids, dtype=np.int64)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        row_id = int(self.ids[item])
        row = self.cache.index.iloc[row_id]
        def tensor(name, dtype):
            return torch.from_numpy(np.array(self.cache.arrays[name][row_id], dtype=dtype, copy=True))
        attention = tensor("attention", np.bool_)
        return {
            "site_id": str(row.site_id), "row_id": row_id,
            "token_ids": tensor("token", np.int64), "attention_mask": attention,
            "esm_present_mask": attention.clone(), "esm_embeddings": tensor("esm150", np.float16),
            "node_esm": tensor("node_esm", np.float16), "node_positions": tensor("positions", np.int64),
            "node_coordinates": tensor("coordinates", np.float32),
            "node_geometry": tensor("geometry", np.float32), "node_mask": tensor("node_mask", np.bool_),
            "global_embedding": tensor("global", np.float16),
            "residue_id": torch.tensor(int(row.residue_id), dtype=torch.long),
            "label": torch.tensor(float(row.label), dtype=torch.float32),
        }


def make_loader(cache, ids, args, shuffle, seed, *, workers=None):
    count = args.workers if workers is None else workers
    options = dict(batch_size=args.batch_size, shuffle=shuffle, num_workers=count,
                   pin_memory=str(args.device).startswith("cuda"), drop_last=False,
                   generator=torch.Generator().manual_seed(seed))
    if count:
        options.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(SiteDataset(cache, ids), **options)


class DynamicFeatures(DynamicsNormalization):
    def __init__(self, cache):
        self.arrays = {key: cache.arrays[key] for key in
                       ("node", "view_mask", "quality", "mask", "local_pair", "cross", "cross_reverse_prs", "center_pair")}
        self.local_esm = cache.arrays["local_esm"]
        self.remote_esm = cache.arrays["remote_esm"]
        self.site_quality = cache.arrays["site_quality"]
