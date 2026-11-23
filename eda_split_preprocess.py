#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eda_split_preprocess.py  -  Kelompok 5, Project Kecerdasan Buatan (TAP-1: DINOv3 daun kopi)

Satu skrip untuk persiapan data:
  1. Inventaris citra dan penyelarasan label ke healthy / rust / other
  2. Statistik kualitas citra (resolusi, format, ukuran berkas, kecerahan, ketajaman, citra rusak)
  3. Duplikat persis (MD5) dan hampir identik (dHash, jarak Hamming <= 3) -> group_id
  4. Pembagian data S1 (in-domain), S2 (LODO), S3 (label terbatas), S4 (keparahan)
  5. Tabel dan grafik untuk laporan

Contoh pakai (dijalankan dari folder repo):
  python eda_split_preprocess.py --inventaris                    # cek pemetaan label saja (cepat)
  python eda_split_preprocess.py --limit 50 --out outputs_uji    # uji di subset kecil
  python eda_split_preprocess.py                                 # penuh
"""
import argparse
import hashlib
import io
import math
import random
import re
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    sys.exit(f"Paket belum terpasang: {e}. Pasang numpy, pandas, pillow, dan matplotlib.")

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASSES = ["healthy", "rust", "other"]
BINARY = ["healthy", "rust"]
DOMAIN_ORDER = ["kenya", "ecuador", "brazil", "peru"]

# nama sumber: (domain/negara, folder di bawah --root)
SOURCES = {
    "kenya_jmuben": ("kenya", "raw/kenya_jmuben"),
    "kenya_jmuben2": ("kenya", "raw/kenya_jmuben2"),
    "ecuador_rocole": ("ecuador", "processed/ecuador_rocole"),
    "peru": ("peru", "raw/peru"),
    "brazil_bracol": ("brazil", "processed/brazil_bracol"),  # hanya dipakai kalau ada labels.csv
}

OTHER_KEYS = ("miner", "cercospora", "cerscospora", "cerc", "cersc", "phoma", "spider", "mite", "spot",
              "brown", "blight", "other")  # "cerscospora" = salah ketik nama folder di JMuBEN


# ---------------------------------------------------------------- label
def map_label(text):
    t = str(text).lower().replace("-", "_").replace(" ", "_")
    if "healthy" in t or "sehat" in t:
        return "healthy"
    if "rust" in t or "karat" in t:
        return "rust"
    if any(k in t for k in OTHER_KEYS):
        return "other"
    return None


def label_from_parts(parts):
    """Cari label dari nama folder (dari yang terdalam), lalu dari nama berkas."""
    cands = list(reversed(parts[:-1])) + [Path(parts[-1]).stem]
    for c in cands:
        lab = map_label(c)
        if lab:
            return lab, c
    return "unknown", cands[0]


def parse_severity(sev, raw):
    if sev is not None:
        try:
            v = float(sev)
            if not math.isnan(v):
                return v
        except (TypeError, ValueError):
            pass
    m = re.search(r"level[_ ]?(\d)", str(raw).lower())
    return float(m.group(1)) if m else float("nan")


def orig_split(parts):
    for p in parts[:-1]:
        q = p.lower()
        if q in ("train", "training"):
            return "train"
        if q in ("val", "valid", "validation"):
            return "val"
        if q in ("test", "testing"):
            return "test"
    return ""


def scan_source(source, domain, folder, limit, rng):
    if not folder.exists():
        print(f"  [lewati] {source}: folder {folder} tidak ada")
        return []
    table = {}
    lc = folder / "labels.csv"
    if lc.exists():
        t = pd.read_csv(lc, dtype=str)
        has_sev = "severity" in t.columns
        for _, r in t.iterrows():
            table[str(r["file"]).replace("\\", "/")] = (str(r["label"]), r["severity"] if has_sev else None)
    elif source == "brazil_bracol":
        print(f"  [lewati] {source}: tidak ada labels.csv, jadi tidak dipakai")
        return []
    rows = []
    for p in sorted(folder.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMG_EXT:
            continue
        if p.name.startswith("._") or "__MACOSX" in p.parts:
            continue
        rel = p.relative_to(folder)
        key = rel.as_posix()
        if key in table:
            raw, sev = table[key]
            label = map_label(raw) or "unknown"
        else:
            label, raw = label_from_parts(rel.parts)
            sev = None
        rows.append(dict(uid=f"{source}:{key}", source=source, domain=domain, path=str(p),
                         raw_label=raw, label=label, severity=parse_severity(sev, raw),
                         orig_split=orig_split(rel.parts)))
    if limit > 0:
        by = {}
        for r in rows:
            by.setdefault(r["raw_label"], []).append(r)
        rows = []
        for k in sorted(by):
            lst = by[k]
            rng.shuffle(lst)
            rows += lst[:limit]
    print(f"  {source}: {len(rows)} citra")
    return rows


# ---------------------------------------------------------------- statistik citra
def image_stats(path):
    rec = dict(width=0, height=0, format="", filesize=0, brightness=float("nan"),
               sharpness=float("nan"), corrupt=0, md5="", dhash="0" * 16)
    try:
        data = Path(path).read_bytes()
        rec["filesize"] = len(data)
        rec["md5"] = hashlib.md5(data).hexdigest()
        with Image.open(io.BytesIO(data)) as im:
            rec["format"] = im.format or ""
            rec["width"], rec["height"] = im.size
            im.load()
            g = im.convert("L")
            a = np.asarray(g.resize((9, 8), Image.LANCZOS), dtype=np.int16)
            bits = (a[:, 1:] > a[:, :-1]).flatten()
            rec["dhash"] = format(int("".join("1" if b else "0" for b in bits), 2), "016x")
            g.thumbnail((512, 512))  # ketajaman dihitung pada ukuran seragam (sisi terpanjang <= 512)
            x = np.asarray(g, dtype=np.float32)
            rec["brightness"] = float(x.mean())
            if min(x.shape) >= 3:
                lap = -4 * x[1:-1, 1:-1] + x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:]
                rec["sharpness"] = float(lap.var())
    except Exception:
        rec["corrupt"] = 1
    return rec


# ---------------------------------------------------------------- duplikat dan grup
_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount64(x):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    return _POP8[np.ascontiguousarray(x).view(np.uint8).reshape(-1, 8)].sum(axis=1)


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def add_groups(df, thr):
    """group_id per domain: duplikat persis (MD5) dan hampir identik (dHash <= thr) satu grup."""
    df["group_id"] = ""
    for domain, sub in df.groupby("domain"):
        ok = sub[sub["corrupt"] == 0]
        n = len(ok)
        if n == 0:
            continue
        idx = ok.index.to_numpy()
        uf = UF(n)
        first = {}
        for i, m in enumerate(ok["md5"].tolist()):
            if m in first:
                uf.union(i, first[m])
            else:
                first[m] = i
        h = np.array([int(x, 16) for x in ok["dhash"].tolist()], dtype=np.uint64)
        for c in range(4):  # 4 potongan 16 bit: jarak <= 3 pasti menyisakan >= 1 potongan yang sama
            chunk = ((h >> np.uint64(16 * c)) & np.uint64(0xFFFF)).astype(np.int64)
            order = np.argsort(chunk, kind="stable")
            sc = chunk[order]
            starts = np.flatnonzero(np.r_[True, sc[1:] != sc[:-1]])
            ends = np.r_[starts[1:], n]
            for s, e in zip(starts, ends):
                if e - s < 2:
                    continue
                mem = order[s:e]
                hv = h[mem]
                for k in range(len(mem) - 1):
                    d = popcount64(hv[k] ^ hv[k + 1:])
                    for j in np.flatnonzero(d <= thr):
                        uf.union(int(mem[k]), int(mem[k + 1 + j]))
        roots = [uf.find(i) for i in range(n)]
        df.loc[idx, "group_id"] = [f"{domain}_{r:06d}" for r in roots]
    has = df["group_id"] != ""
    df["group_size"] = 0
    df.loc[has, "group_size"] = df[has].groupby("group_id")["uid"].transform("size")
    nlab = df[has].groupby("group_id")["label"].agg(lambda s: len(set(x for x in s if x != "unknown")))
    conflict = set(nlab[nlab > 1].index)
    df["label_conflict"] = df["group_id"].isin(conflict)
    return df


# ---------------------------------------------------------------- pembagian data
def assign_groups(groups, ratios, names, rng):
    """Greedy: grup diacak, tiap grup masuk ke split yang paling kekurangan (menurut jumlah citra)."""
    order = list(groups)
    rng.shuffle(order)
    total = sum(s for _, s in order)
    target = [r * total for r in ratios]
    cur = [0] * len(ratios)
    out = {}
    for gid, size in order:
        k = max(range(len(ratios)), key=lambda i: target[i] - cur[i])
        out[gid] = names[k]
        cur[k] += size
    return out


def split_by_group(sub, ratios, names, rng):
    sizes = sub.groupby("group_id").size()
    assign = assign_groups(sorted(sizes.items()), ratios, names, rng)
    return sub["group_id"].map(assign)


def make_s1(elig, seed):
    rows = []
    for domain, sd in elig.groupby("domain"):
        for cls in BINARY:
            sub = sd[sd["label"] == cls]
            if len(sub) == 0:
                continue
            sp = split_by_group(sub, [0.70, 0.15, 0.15], ["train", "val", "test"], random.Random(seed))
            rows += [("S1", domain, seed, u, s) for u, s in zip(sub["uid"], sp)]
    return rows


def make_s2(elig, seed, cap):
    """LODO. Mengembalikan baris split dan dict fold -> DataFrame train/val (dipakai S3)."""
    rows, folds = [], {}
    domains = [d for d in DOMAIN_ORDER if d in set(elig["domain"])
               and all((elig[(elig.domain == d) & (elig.label == c)].shape[0] > 0) for c in BINARY)]
    if len(domains) < 3:
        print(f"  PERHATIAN: hanya {len(domains)} domain punya healthy dan rust; LODO kurang dari 3 sumber latih.")
    for held in domains:
        test = elig[(elig.domain == held) & elig.label.isin(BINARY)]
        rows += [("S2", held, seed, u, "test") for u in test["uid"]]
        parts = []
        for d in domains:
            if d == held:
                continue
            for cls in BINARY:
                sub = elig[(elig.domain == d) & (elig.label == cls)]
                if len(sub) > cap:
                    sub = sub.sample(n=cap, random_state=seed)
                sp = split_by_group(sub, [0.85, 0.15], ["train", "val"], random.Random(seed))
                parts.append(sub.assign(split=sp.values))
        pool = pd.concat(parts)
        rows += [("S2", held, seed, u, s) for u, s in zip(pool["uid"], pool["split"])]
        folds[held] = pool
    return rows, folds


def make_s3(folds, seeds, pcts, min_train, min_val):
    rows = []
    for held, pool in folds.items():
        for pct in pcts:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                for (dom, cls, split), sub in pool.groupby(["domain", "label", "split"]):
                    sub = sub.sort_values("uid")
                    n = len(sub)
                    mn = min_train if split == "train" else min_val
                    k = min(n, max(mn, int(math.ceil(pct / 100 * n))))
                    pick = rng.choice(n, size=k, replace=False)
                    rows += [(f"S3_{pct}pct", held, seed, u, split) for u in sub["uid"].iloc[pick]]
    return rows


def make_s4(df, seed):
    """Keparahan: 70/15/15 distratifikasi menurut tingkat keparahan, per domain yang punya label keparahan."""
    rows = []
    ok = df[(df.corrupt == 0) & df.severity.notna() & (~df.label_conflict)]
    for domain, sd in ok.groupby("domain"):
        if len(sd) < 20:
            continue
        for lvl, sub in sd.groupby("severity"):
            sp = split_by_group(sub, [0.70, 0.15, 0.15], ["train", "val", "test"], random.Random(seed))
            rows += [(f"S4_{domain}", f"level{int(lvl)}", seed, u, s) for u, s in zip(sub["uid"], sp)]
    return rows


def verify_no_leak(splits, df):
    g = df.set_index("uid")["group_id"]
    sp = splits.assign(group_id=splits["uid"].map(g))
    bad = 0
    for _, s in sp.groupby(["scenario", "fold", "seed"]):
        bad += int((s.groupby("group_id")["split"].nunique() > 1).sum())
    return bad


# ---------------------------------------------------------------- laporan
def tabel_statistik(df, key):
    rows = []
    for k, s in df.groupby(key):
        ok = s[s.corrupt == 0]
        g = ok[ok.group_id != ""]
        rows.append({key: k, "total": len(s),
                     "healthy": int((s.label == "healthy").sum()), "rust": int((s.label == "rust").sum()),
                     "other": int((s.label == "other").sum()), "unknown": int((s.label == "unknown").sum()),
                     "rusak": int(s.corrupt.sum()), "citra_duplikat": int(len(g) - g.group_id.nunique()),
                     "median_lebar": ok.width.median(), "median_tinggi": ok.height.median(),
                     "median_kecerahan": round(ok.brightness.median(), 1),
                     "median_ketajaman": round(ok.sharpness.median(), 1)})
    return pd.DataFrame(rows)


def tabel_sumber_kelas(df):
    """Per sumber x kelas: untuk melihat apakah satu kelas hanya berasal dari satu sumber (risiko jalan pintas)."""
    ok = df[df.corrupt == 0]
    t = ok.groupby(["domain", "source", "label"]).agg(
        jumlah=("uid", "size"), median_lebar=("width", "median"), median_tinggi=("height", "median"),
        median_ukuran_berkas_kb=("filesize", lambda s: round(s.median() / 1024, 1)),
        median_kecerahan=("brightness", lambda s: round(s.median(), 1)),
        median_ketajaman=("sharpness", lambda s: round(s.median(), 1))).reset_index()
    return t


def simpan_grup_duplikat(df, path):
    """Daftar grup berisi lebih dari satu citra, untuk diperiksa manual (apakah benar duplikat)."""
    g = df[(df.group_id != "") & (df.group_size > 1)].copy()
    g["n_md5_unik"] = g.groupby("group_id")["md5"].transform("nunique")
    cols = ["domain", "group_id", "group_size", "n_md5_unik", "label", "source", "orig_split", "path"]
    g.sort_values(["domain", "group_id", "path"])[cols].to_csv(path, index=False)
    return g.group_id.nunique()


def mapping_report(df):
    lines = ["PEMETAAN LABEL (sumber | label asli -> label seragam | jumlah)"]
    for src, sub in df.groupby("source"):
        c = sub.groupby(["raw_label", "label"]).size().sort_values(ascending=False).head(40)
        for (raw, lab), n in c.items():
            lines.append(f"{src} | {raw} -> {lab} | {n}")
        nu = int((sub.label == "unknown").sum())
        if nu:
            lines.append(f"{src} | PERHATIAN: {nu} citra berlabel unknown (periksa nama folder)")
    return "\n".join(lines)


def plot_distribution(df, path):
    doms = [d for d in DOMAIN_ORDER if d in set(df.domain)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("white")
    w = 0.26
    colors = {"healthy": "#4C9F70", "rust": "#E07B39", "other": "#5B7DB1"}
    for i, cls in enumerate(CLASSES):
        vals = [int(((df.domain == d) & (df.label == cls)).sum()) for d in doms]
        pos = [j + (i - 1) * w for j in range(len(doms))]
        bars = ax.bar(pos, [max(v, 0.8) for v in vals], w, label=cls, color=colors[cls])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, max(v, 0.8) * 1.1, str(v), ha="center", fontsize=7)
    ax.set_yscale("log")
    top = max(int(((df.domain == d) & (df.label == c)).sum()) for d in doms for c in CLASSES)
    ax.set_ylim(0.7, max(top, 1) * 3)
    ax.set_xticks(range(len(doms)))
    ax.set_xticklabels(doms)
    ax.set_ylabel("Jumlah citra (skala log)")
    ax.set_title("Distribusi kelas per dataset")
    ax.legend()
    ax.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def plot_samples(df, path, seed, per=2):
    doms = [d for d in DOMAIN_ORDER if d in set(df.domain)]
    ok = df[df.corrupt == 0]
    fig, axes = plt.subplots(len(doms), per * len(CLASSES), figsize=(2.0 * per * len(CLASSES), 2.1 * len(doms)),
                             squeeze=False)
    fig.patch.set_facecolor("white")
    for r, d in enumerate(doms):
        for ci, cls in enumerate(CLASSES):
            sub = ok[(ok.domain == d) & (ok.label == cls)]
            pick = sub.sample(n=min(per, len(sub)), random_state=seed) if len(sub) else sub
            for k in range(per):
                ax = axes[r][ci * per + k]
                ax.axis("off")
                if k < len(pick):
                    try:
                        with Image.open(pick.iloc[k]["path"]) as im:
                            im = im.convert("RGB")
                            im.thumbnail((256, 256))
                            ax.imshow(im)
                    except Exception:
                        ax.text(0.5, 0.5, "gagal dibuka", ha="center", va="center", fontsize=7)
                elif k == 0:
                    ax.text(0.5, 0.5, "tidak ada", ha="center", va="center", fontsize=8)
                if r == 0 and k == 0:
                    ax.set_title(cls, loc="left", fontsize=10)
                if ci == 0 and k == 0:
                    ax.text(-0.08, 0.5, d, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def orig_split_leak(df):
    lines = []
    sub = df[(df.group_id != "") & (df.orig_split != "")]
    for domain, s in sub.groupby("domain"):
        n = s.groupby("group_id")["orig_split"].nunique()
        multi = n[n > 1]
        lines.append(f"  {domain}: {len(multi)} dari {len(n)} grup melintasi split bawaan dataset "
                     f"({int(s.group_id.isin(multi.index).sum())} citra)")
    return lines


def build_transforms(train=True, size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Alur pra-pemrosesan (dipakai di notebook pelatihan). Untuk DINOv3 pakai mean/std dari AutoImageProcessor."""
    from torchvision import transforms as T
    steps = [T.Resize((size, size))]  # resize langsung tanpa crop
    if train:
        steps += [T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.5), T.RandomRotation(20),
                  T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02)]
    steps += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(steps)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="EDA, pemetaan label, duplikat, dan pembagian data (TAP-1)")
    ap.add_argument("--root", default="datasets", help="folder datasets (berisi raw/ dan processed/)")
    ap.add_argument("--out", default="outputs", help="folder keluaran")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cap", type=int, default=600, help="batas citra per dataset x kelas di data latih S2")
    ap.add_argument("--limit", type=int, default=0, help="uji cepat: maks citra per label asli per sumber")
    ap.add_argument("--sources", default="", help="pisahkan koma, mis. kenya_jmuben,peru (default semua)")
    ap.add_argument("--dhash-thr", type=int, default=3)
    ap.add_argument("--inventaris", action="store_true", help="hanya cek pemetaan label, tanpa membaca isi citra")
    ap.add_argument("--reuse", action="store_true", help="pakai manifest.csv yang sudah ada")
    ap.add_argument("--min-train", type=int, default=5)
    ap.add_argument("--min-val", type=int, default=2)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(a.root)
    rng = random.Random(a.seed)
    man_path = out / "manifest.csv"

    if a.reuse and man_path.exists():
        df = pd.read_csv(man_path, dtype={"md5": str, "dhash": str})
        for c in ("orig_split", "raw_label", "label", "group_id", "dhash"):
            df[c] = df[c].fillna("")
        df["label_conflict"] = df["label_conflict"].astype(bool)
        print(f"Memakai manifest yang ada: {len(df)} citra")
    else:
        want = [s for s in a.sources.split(",") if s] or list(SOURCES)
        print("Memindai folder:")
        rows = []
        for name in want:
            dom, folder = SOURCES[name]
            rows += scan_source(name, dom, root / folder, a.limit, rng)
        if not rows:
            sys.exit("Tidak ada citra ditemukan. Jalankan dari folder repo dan cek --root.")
        df = pd.DataFrame(rows)
        rep = mapping_report(df)
        (out / "mapping_report.txt").write_text(rep, encoding="utf-8")
        print("\n" + rep)
        if a.inventaris:
            print(f"\nTotal {len(df)} citra. Laporan pemetaan: {out / 'mapping_report.txt'}")
            return
        print(f"\nMembaca statistik {len(df)} citra ...")
        stats = []
        for i, p in enumerate(df["path"], 1):
            stats.append(image_stats(p))
            if i % 2000 == 0:
                print(f"  {i}/{len(df)}")
        df = pd.concat([df, pd.DataFrame(stats)], axis=1)
        print("Mencari duplikat dan membentuk grup ...")
        df = add_groups(df, a.dhash_thr)
        df.to_csv(man_path, index=False)
        print(f"Tersimpan: {man_path}")

    # --- tabel
    t_dom = tabel_statistik(df, "domain")
    t_src = tabel_statistik(df, "source")
    t_dom.to_csv(out / "tabel2_statistik_per_dataset.csv", index=False)
    t_src.to_csv(out / "tabel2_statistik_per_sumber.csv", index=False)
    print("\nTABEL 2 (per dataset/negara):")
    print(t_dom.to_string(index=False))
    t_sk = tabel_sumber_kelas(df)
    t_sk.to_csv(out / "tabel2_per_sumber_dan_kelas.csv", index=False)
    print("\nPER SUMBER DAN KELAS (cek apakah satu kelas hanya dari satu sumber):")
    print(t_sk.to_string(index=False))
    n_dupgrup = simpan_grup_duplikat(df, out / "grup_duplikat.csv")
    print(f"\nGrup duplikat/hampir identik (>1 citra): {n_dupgrup} -> {out / 'grup_duplikat.csv'}")

    # --- grafik
    plot_distribution(df, out / "gambar3_distribusi_kelas.png")
    plot_samples(df, out / "gambar4_contoh_citra.png", a.seed)

    # --- pembagian data
    elig = df[(df.corrupt == 0) & df.label.isin(BINARY) & (~df.label_conflict)]
    n_conf = int(df.label_conflict.sum())
    print(f"\nCitra layak split (healthy/rust, tidak rusak, tanpa konflik label): {len(elig)} dari {len(df)}")
    rows = make_s1(elig, a.seed)
    r2, folds = make_s2(elig, a.seed, a.cap)
    rows += r2
    rows += make_s3(folds, [42, 43, 44], [1, 5, 10], a.min_train, a.min_val)
    rows += make_s4(df, a.seed)
    splits = pd.DataFrame(rows, columns=["scenario", "fold", "seed", "uid", "split"])
    splits.to_csv(out / "splits.csv", index=False)

    summ = (splits.groupby(["scenario", "fold", "seed", "split"]).size().unstack("split", fill_value=0)
            .reset_index())
    for c in ("train", "val", "test"):
        if c not in summ:
            summ[c] = 0
    s2test = summ[summ.scenario == "S2"].set_index("fold")["test"].to_dict()
    is3 = summ.scenario.str.startswith("S3")
    summ.loc[is3, "test"] = summ.loc[is3, "fold"].map(s2test)  # uji S3 = uji S2 pada fold yang sama
    summ = summ[["scenario", "fold", "seed", "train", "val", "test"]]
    summ.to_csv(out / "tabel3_ukuran_split.csv", index=False)
    print("\nTABEL 3 (ukuran split; uji S3 = uji S2 fold yang sama):")
    print(summ.to_string(index=False))

    bad = verify_no_leak(splits, df)
    print(f"\nPemeriksaan kebocoran grup antar split: {bad} grup bocor (harus 0)")

    # --- ringkasan temuan
    ok = df[df.corrupt == 0]
    lines = ["RINGKASAN TEMUAN EDA", f"Total citra: {len(df)} | rusak: {int(df.corrupt.sum())}",
             f"Grup dengan label bertentangan (dikeluarkan dari split): "
             f"{df[df.label_conflict].group_id.nunique()} grup, {n_conf} citra"]
    for d, s in ok.groupby("domain"):
        g = s[s.group_id != ""]
        dup_md5 = int(len(g) - g.md5.nunique())
        lines.append(f"{d}: {len(s)} citra | duplikat persis (MD5): {dup_md5} | "
                     f"citra berlebih akibat duplikat/hampir identik: {int(len(g) - g.group_id.nunique())}")
    q = ok.sharpness.quantile(0.05)
    lines.append(f"Ketajaman (varians Laplacian) kuantil 5%: {q:.1f}; citra di bawahnya {int((ok.sharpness < q).sum())}")
    ol = orig_split_leak(df)
    if ol:
        lines.append("Grup yang melintasi split bawaan dataset (indikasi kebocoran pada split asli):")
        lines += ol
    lines.append(f"Kebocoran grup pada split buatan skrip: {bad} (harus 0)")
    (out / "ringkasan_eda.txt").write_text("\n".join(lines) + "\n\n" + mapping_report(df), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nSelesai. Semua keluaran ada di: {out}/")


if __name__ == "__main__":
    main()
