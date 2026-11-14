"""Download and prepare the two empirical datasets used in this paper.

1. UCI Gas Sensor Array Drift (Vergara et al., 2012)
   https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset
   10 batches, 16 MOX sensors, 6 gases, 13,910 measurements over 36 months.

2. Corn NIR (Eigenvector Research data sets; Cargill corn)
   80 samples measured on three NIR instruments (m5, mp5, mp6),
   700 wavelengths (1100-2498 nm), four reference properties.

The raw files are mirrored into data/raw/ and parsed into compressed npz.
Every artefact is hashed into data/manifest.json.

Usage:
    python experiments/download_data.py
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import DATA, sha256

RAW = DATA / "raw"
RAW.mkdir(parents=True, exist_ok=True)


def _download(url: str, dest: Path, attempts: int = 8, min_bytes: int = 1000) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        try:
            if dest.exists() and dest.stat().st_size >= min_bytes:
                return dest
            tmp = dest.with_suffix(dest.suffix + ".part")
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if chunk:
                            f.write(chunk)
            if tmp.stat().st_size < min_bytes:
                raise IOError(f"file too small: {tmp.stat().st_size}")
            tmp.replace(dest)
            return dest
        except Exception as exc:
            print(f"  attempt {attempt}/{attempts} failed for {url}: {exc}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"could not download {url}")


# ---------------------------------------------------------------------------
# UCI drift
# ---------------------------------------------------------------------------

def parse_uci_batch(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse one UCI batch file.

    Format per line:
        <gas>;<concentration> 1:<value> 2:<value> ... 128:<value>
    Returns X (n,128), gas (n,), concentration (n,).
    """
    X, gas, conc = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            head, *feats = line.split()
            g, c = head.split(";")
            gas.append(int(float(g)))
            conc.append(float(c))
            vals = np.empty(128, dtype=np.float64)
            for tok in feats:
                idx, val = tok.split(":")
                vals[int(idx) - 1] = float(val)
            X.append(vals)
    return np.asarray(X), np.asarray(gas), np.asarray(conc)


def prepare_uci() -> dict:
    base = "https://raw.githubusercontent.com/HakanKARASU/Gas-Sensor-Array-Drift-Dataset/main/batch{i}.dat"
    batches = []
    for i in range(1, 11):
        dest = RAW / "uci" / f"batch{i}.dat"
        if not dest.exists():
            print(f"  downloading UCI batch {i}")
            _download(base.format(i=i), dest, min_bytes=100_000)
        X, g, c = parse_uci_batch(dest)
        batches.append((X, g, c))
        print(f"    batch {i}: {X.shape}")

    X = np.vstack([b[0] for b in batches])
    gas = np.concatenate([b[1] for b in batches])
    conc = np.concatenate([b[2] for b in batches])
    batch = np.concatenate([np.full(len(b[1]), i + 1) for i, b in enumerate(batches)])

    # The 16 sensor resistances are feature indices 1, 9, ..., 121 (1-based),
    # i.e. columns 0, 8, ..., 120 (0-based): the first feature of each sensor.
    sensor_cols = np.arange(0, 128, 8)
    R = X[:, sensor_cols]

    np.savez_compressed(
        DATA / "uci_drift.npz",
        X=X.astype(np.float32),
        R=R.astype(np.float64),
        gas=gas.astype(np.int8),
        concentration=conc.astype(np.float32),
        batch=batch.astype(np.int8),
    )
    return {
        "n": int(len(gas)),
        "n_features": int(X.shape[1]),
        "n_sensors": int(R.shape[1]),
        "n_batches": 10,
        "gases": sorted(np.unique(gas).tolist()),
        "batch_sizes": [int(np.sum(batch == b)) for b in range(1, 11)],
    }


# ---------------------------------------------------------------------------
# Corn NIR
# ---------------------------------------------------------------------------

def _unwrap(x):
    while isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        x = x.flat[0]
    return x


def prepare_corn() -> dict:
    from scipy.io import loadmat

    dest = RAW / "corn" / "corn.mat"
    if not dest.exists():
        z = RAW / "corn" / "corn.zip"
        print("  downloading corn.mat")
        _download(
            "https://eigenvector.com/wp-content/uploads/2019/06/corn.mat_.zip",
            z,
            min_bytes=100_000,
        )
        with zipfile.ZipFile(z) as zf:
            member = [m for m in zf.namelist() if m.endswith("corn.mat")][0]
            with zf.open(member) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)

    d = loadmat(dest)
    out = {}
    for name in ["m5spec", "mp5spec", "mp6spec"]:
        s = _unwrap(d[name])
        out[name] = np.asarray(_unwrap(s["data"]), dtype=np.float64)
        axis = None
        if s["axisscale"].size:
            axarr = _unwrap(s["axisscale"])
            if isinstance(axarr, np.ndarray):
                for idx in np.ndindex(axarr.shape):
                    cand = _unwrap(axarr[idx])
                    if np.size(cand) == out[name].shape[1]:
                        axis = cand
                        break
        if axis is not None:
            out[name + "_wavelengths"] = np.asarray(axis, dtype=float).ravel()

    pv = _unwrap(d["propvals"])
    propvals = np.asarray(_unwrap(pv["data"]), dtype=np.float64)
    raw_labels = np.asarray(_unwrap(pv["label"])).ravel()
    labels = []
    for x in raw_labels:
        xu = _unwrap(x)
        if isinstance(xu, np.ndarray):
            for v in xu.ravel():
                s = str(v).strip()
                if s and s not in {"[]", "['']"}:
                    labels.append(s)
        elif str(xu).strip():
            labels.append(str(xu).strip())

    np.savez_compressed(
        DATA / "corn.npz",
        m5=out["m5spec"],
        mp5=out["mp5spec"],
        mp6=out["mp6spec"],
        wavelengths=out["m5spec_wavelengths"],
        propvals=propvals,
        property_labels=np.array(labels),
    )
    return {
        "n_samples": int(out["m5spec"].shape[0]),
        "n_wavelengths": int(out["m5spec"].shape[1]),
        "instruments": ["m5", "mp5", "mp6"],
        "wavelength_range_nm": [float(out["m5spec_wavelengths"][0]), float(out["m5spec_wavelengths"][-1])],
        "properties": labels,
    }


def main():
    manifest = {}
    print("Preparing UCI drift ...")
    manifest["uci_drift"] = prepare_uci()
    print("Preparing Corn NIR ...")
    manifest["corn"] = prepare_corn()

    hashes = {}
    for p in sorted(DATA.rglob("*")):
        if p.is_file() and p.suffix in {".npz", ".dat", ".mat"}:
            hashes[str(p.relative_to(DATA))] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    manifest["files"] = hashes
    with open(DATA / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
