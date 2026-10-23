"""
Galaxy10 DECaLS (17,736 images, 256×256, 10 classes) as a row dataset like the HF ones
used for pre-training: ds[i] → {"image": PIL, "label": int}.

Split: stratified 80/20 with seed 0 ("train" / "test"; "validation" is the test split).
The paper (VISReg, Wu et al. 2606.02572) does not give one; the same split serves
pre-training (train only) and the in-domain linear probe of downstream_eval.py.

The h5 stores images gzip-chunked as (555, 16, 16, 1), so a single-image read
decompresses ~770 chunks; they are unpacked once into an uncompressed .npy next to
it and memory-mapped.
"""

import os

import numpy as np
from PIL import Image

ROOT = "/project/khanhnt/datasets/downstream"


def _cache(h5_path, npy_path):
    import h5py
    with h5py.File(h5_path, "r") as f:
        y = np.asarray(f["ans"]).astype(np.int64)
        if not os.path.exists(npy_path):
            d = f["images"]
            tmp = npy_path + ".tmp"
            out = np.lib.format.open_memmap(tmp, "w+", np.uint8, d.shape)
            step = d.chunks[0] if d.chunks else 1024
            for s in range(0, len(d), step):
                out[s:s + step] = d[s:s + step]
            out.flush()
            del out
            os.replace(tmp, npy_path)
    return y


class Galaxy10Rows:
    def __init__(self, split, root=ROOT, seed=0, test_frac=0.2):
        h5 = os.path.join(root, "galaxy10", "Galaxy10_DECals.h5")
        self.npy, self.x = h5.replace(".h5", "_images.npy"), None
        y = _cache(h5, self.npy)
        rng = np.random.RandomState(seed)
        test = np.zeros(len(y), bool)
        for c in np.unique(y):
            idx = rng.permutation(np.flatnonzero(y == c))
            test[idx[:int(round(test_frac * len(idx)))]] = True
        self.idx = np.flatnonzero(~test if split == "train" else test)
        self.y = y

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        if self.x is None:              # opened lazily: one memmap per worker
            self.x = np.load(self.npy, mmap_mode="r")
        j = self.idx[i]
        return {"image": Image.fromarray(np.ascontiguousarray(self.x[j])), "label": int(self.y[j])}
