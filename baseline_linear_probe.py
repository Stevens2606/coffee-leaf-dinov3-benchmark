"""Baseline Progress 3: fitur terkunci + kepala linear (logistic regression).

Alur:
  1. Ekstraksi fitur sekali per citra unik (kunci = MD5) dengan model terkunci. Disimpan di outputs/fitur/.
  2. Latih kepala linear (healthy vs rust) untuk tiap skenario di outputs/splits.csv (S1, S2, S3).
  3. Catat kurva loss per epoch, metrik uji (macro-F1, akurasi), dan konfigurasi.

Model: efficientnet_b0, convnext_tiny (timm), dinov3_vits16 (transformers, perlu akses Hugging Face).

Pakai (dari folder repo, di server):
  python -u baseline_linear_probe.py --limit 64        # uji cepat ekstraksi, tidak menyimpan
  python -u baseline_linear_probe.py                   # jalan penuh
  python -u baseline_linear_probe.py --models efficientnet_b0   # satu model saja
"""
import argparse
import json
import os
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = {
    "efficientnet_b0": ("timm", "timm/efficientnet_b0.ra_in1k"),
    "convnext_tiny": ("timm", "timm/convnext_tiny.fb_in22k_ft_in1k"),
    "dinov3_vits16": ("hf", "facebook/dinov3-vits16-pretrain-lvd1689m"),
}
SIZE = 224
LABEL_ID = {"healthy": 0, "rust": 1}


# ------------------------------------------------------------------ ekstraksi fitur
def citra_unik(man, spl):
    """Satu citra per MD5 untuk semua uid yang muncul di splits.csv."""
    sub = man[man["uid"].isin(set(spl["uid"]))]
    return sub.drop_duplicates("md5")[["md5", "path"]].reset_index(drop=True)


def bangun_model(name):
    import torch  # noqa: F401
    kind, repo = MODELS[name]
    if kind == "timm":
        import timm
        m = timm.create_model("hf_hub:" + repo, pretrained=True, num_classes=0)
        mean, std = m.pretrained_cfg["mean"], m.pretrained_cfg["std"]

        def fwd(x):
            return m(x)
    else:
        from transformers import AutoModel
        m = AutoModel.from_pretrained(repo)
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        def fwd(x):
            return m(pixel_values=x).pooler_output
    m.eval()
    return fwd, np.array(mean, dtype=np.float32), np.array(std, dtype=np.float32)


def ekstrak(name, rep, simpan, workers, batch):
    import torch
    from PIL import Image

    fwd, mean, std = bangun_model(name)

    class Data(torch.utils.data.Dataset):
        def __len__(self):
            return len(rep)

        def __getitem__(self, i):
            im = Image.open(rep["path"].iloc[i])
            if im.format == "JPEG":
                im.draft("RGB", (SIZE * 2, SIZE * 2))
            im = im.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
            a = (np.asarray(im, dtype=np.float32) / 255.0 - mean) / std
            return torch.from_numpy(a.transpose(2, 0, 1).copy())

    torch.set_num_threads(os.cpu_count() or 4)
    dl = torch.utils.data.DataLoader(Data(), batch_size=batch, shuffle=False, num_workers=workers)
    feats, t0 = [], time.time()
    with torch.inference_mode():
        for k, x in enumerate(dl, 1):
            feats.append(fwd(x).float().cpu().numpy())
            if k % 10 == 0 or k == len(dl):
                n = min(k * batch, len(rep))
                print(f"  {name}: {n}/{len(rep)} citra, {n / (time.time() - t0):.1f} citra/detik", flush=True)
    X = np.concatenate(feats).astype(np.float32)
    if simpan is not None:
        simpan.parent.mkdir(parents=True, exist_ok=True)
        np.savez(simpan, md5=rep["md5"].to_numpy().astype(str), X=X)
        print(f"  tersimpan: {simpan} {X.shape}", flush=True)
    return X


# ------------------------------------------------------------------ kepala linear (numpy)
def macro_f1(y, p):
    f = []
    for c in (0, 1):
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        d = 2 * tp + fp + fn
        f.append(2 * tp / d if d > 0 else 0.0)
    return float(np.mean(f))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def ce(y, p):
    e = 1e-7
    return float(-(y * np.log(p + e) + (1 - y) * np.log(1 - p + e)).mean())


def latih_kepala(Xtr, ytr, Xva, yva, epochs, lr, l2):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, V = (Xtr - mu) / sd, (Xva - mu) / sd
    n, d = A.shape
    cw = np.array([n / (2 * max(1, int((ytr == 0).sum()))), n / (2 * max(1, int((ytr == 1).sum())))], dtype=np.float32)
    sw = cw[ytr]
    swn = float(sw.sum())
    y = ytr.astype(np.float32)
    w, b = np.zeros(d, dtype=np.float32), 0.0
    mw, vw, mb, vb = np.zeros(d, np.float32), np.zeros(d, np.float32), 0.0, 0.0
    b1, b2, eps = 0.9, 0.999, 1e-8
    best, best_score, kurva = None, None, []
    for ep in range(1, epochs + 1):
        p = sigmoid(A @ w + b)
        tl = float(-(sw * (y * np.log(p + 1e-7) + (1 - y) * np.log(1 - p + 1e-7))).sum() / swn) + 0.5 * l2 * float(w @ w)
        gw = A.T @ (sw * (p - y)) / swn + l2 * w
        gb = float((sw * (p - y)).sum() / swn)
        mw, vw = b1 * mw + (1 - b1) * gw, b2 * vw + (1 - b2) * gw * gw
        mb, vb = b1 * mb + (1 - b1) * gb, b2 * vb + (1 - b2) * gb * gb
        w = w - lr * (mw / (1 - b1 ** ep)) / (np.sqrt(vw / (1 - b2 ** ep)) + eps)
        b = b - lr * (mb / (1 - b1 ** ep)) / (np.sqrt(vb / (1 - b2 ** ep)) + eps)
        pv = sigmoid(V @ w + b) if len(V) else np.array([])
        if len(V):
            vl, pred = ce(yva.astype(np.float32), pv), (pv >= 0.5).astype(int)
            va, vf = float((pred == yva).mean()), macro_f1(yva, pred)
        else:
            vl, va, vf = float("nan"), float("nan"), 0.0
        kurva.append(dict(epoch=ep, train_loss=tl, val_loss=vl, val_acc=va, val_f1=vf))
        score = (vf, -vl if len(V) else 0.0)
        if best_score is None or score > best_score:
            best_score, best = score, (w.copy(), float(b), ep)
    return best, mu, sd, kurva


def prediksi(X, w, b, mu, sd):
    return (sigmoid(((X - mu) / sd) @ w + b) >= 0.5).astype(int)


def evaluasi(name, fitur, man, spl, a):
    idx = {m: i for i, m in enumerate(fitur["md5"])}
    X = fitur["X"]
    info = man.set_index("uid")[["md5", "label"]]
    hasil, kurva_semua = [], []
    # S3 tidak menyimpan baris uji: ujinya sama dengan uji S2 pada fold yang sama
    uji_s2 = {f: gg for f, gg in spl[(spl["scenario"] == "S2") & (spl["split"] == "test")].groupby("fold")}
    for (skenario, fold, seed), g in spl.groupby(["scenario", "fold", "seed"], sort=True):
        if not (skenario.startswith("S1") or skenario.startswith("S2") or skenario.startswith("S3")):
            continue
        d = g.join(info, on="uid")
        d = d[d["md5"].isin(idx)]
        parts = {s: d[d["split"] == s] for s in ("train", "val", "test")}
        if skenario.startswith("S3") and len(parts["test"]) == 0 and fold in uji_s2:
            t = uji_s2[fold].join(info, on="uid")
            parts["test"] = t[t["md5"].isin(idx)]
        if len(parts["train"]) == 0 or len(parts["test"]) == 0 or parts["train"]["label"].nunique() < 2:
            print(f"  lewati {skenario}/{fold}/{seed}: data tidak lengkap")
            continue

        def ambil(df):
            return X[[idx[m] for m in df["md5"]]], df["label"].map(LABEL_ID).to_numpy()

        Xtr, ytr = ambil(parts["train"])
        Xva, yva = ambil(parts["val"])
        (w, b, ep), mu, sd, kurva = latih_kepala(Xtr, ytr, Xva, yva, a.epochs, a.lr, a.l2)
        Xte, yte = ambil(parts["test"])
        pte = prediksi(Xte, w, b, mu, sd)
        ded = parts["test"].drop_duplicates("md5")
        Xd, yd = ambil(ded)
        pd_ = prediksi(Xd, w, b, mu, sd)
        hasil.append(dict(model=name, skenario=skenario, fold=fold, seed=seed, n_train=len(Xtr), n_val=len(Xva),
                          n_test=len(Xte), n_test_unik=len(Xd), epoch_terbaik=ep,
                          macro_f1=macro_f1(yte, pte), acc=float((pte == yte).mean()),
                          macro_f1_unik=macro_f1(yd, pd_), acc_unik=float((pd_ == yd).mean())))
        for k in kurva:
            kurva_semua.append(dict(model=name, skenario=skenario, fold=fold, seed=seed, **k))
    return pd.DataFrame(hasil), pd.DataFrame(kurva_semua)


# ------------------------------------------------------------------ laporan
def ringkas(hasil):
    g = hasil.groupby(["model", "skenario", "fold"])
    r = g.agg(n_seed=("seed", "nunique"), n_train=("n_train", "mean"), n_test=("n_test", "first"),
              n_test_unik=("n_test_unik", "first"),
              macro_f1_rata2=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
              macro_f1_unik_rata2=("macro_f1_unik", "mean"), acc_rata2=("acc", "mean")).reset_index()
    return r.round(4)


def gambar_kurva(kurva, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = kurva[(kurva["skenario"] == "S2") & (kurva["seed"] == kurva["seed"].min())]
    if k.empty:
        return
    folds = sorted(k["fold"].unique())
    fig, ax = plt.subplots(2, len(folds), figsize=(4.6 * len(folds), 6.2), facecolor="white", squeeze=False)
    warna = {"efficientnet_b0": "#1f77b4", "convnext_tiny": "#2ca02c", "dinov3_vits16": "#d62728"}
    for j, f in enumerate(folds):
        for m, s in k[k["fold"] == f].groupby("model"):
            c = warna.get(m, "#555555")
            ax[0][j].plot(s["epoch"], s["train_loss"], "--", color=c, label=f"{m} latih")
            ax[0][j].plot(s["epoch"], s["val_loss"], "-", color=c, label=f"{m} val")
            ax[1][j].plot(s["epoch"], s["val_acc"], "-", color=c, label=m)
        ax[0][j].set_title(f"S2, uji: {f}")
        ax[0][j].set_ylabel("loss")
        ax[1][j].set_ylabel("akurasi validasi")
        ax[1][j].set_xlabel("epoch")
        for r in (0, 1):
            ax[r][j].set_facecolor("white")
            ax[r][j].grid(alpha=0.3)
    ax[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")


def versi(p):
    try:
        return metadata.version(p)
    except Exception:
        return "tidak terpasang"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="uji cepat: ekstrak N citra saja, tidak menyimpan")
    a = ap.parse_args()

    out = Path(a.out)
    man = pd.read_csv(out / "manifest.csv", dtype=str, keep_default_na=False, low_memory=False)
    spl = pd.read_csv(out / "splits.csv", dtype={"fold": str, "seed": int})
    rep = citra_unik(man, spl)
    print(f"Citra unik yang perlu fitur: {len(rep)} (dari {spl['uid'].nunique()} uid di splits.csv)")
    nama = [m for m in a.models.split(",") if m]

    if a.limit:
        for m in nama:
            try:
                t0 = time.time()
                X = ekstrak(m, rep.head(a.limit), None, a.workers, a.batch)
                print(f"UJI OK {m}: bentuk fitur {X.shape}, {time.time() - t0:.0f} detik, "
                      f"perkiraan penuh {(time.time() - t0) / a.limit * len(rep) / 60:.0f} menit (kasar)")
            except Exception as e:
                print(f"UJI GAGAL {m}: {type(e).__name__}: {str(e)[:200]}")
        return

    fdir = out / "fitur"
    for m in nama:
        f = fdir / f"{m}.npz"
        if f.exists():
            print(f"{m}: fitur sudah ada, dipakai ulang")
            continue
        try:
            print(f"{m}: mulai ekstraksi {len(rep)} citra", flush=True)
            ekstrak(m, rep, f, a.workers, a.batch)
        except Exception as e:
            print(f"{m}: DILEWATI ({type(e).__name__}: {str(e)[:200]})")

    bdir = out / "baseline"
    bdir.mkdir(parents=True, exist_ok=True)
    hasil_semua, kurva_semua = [], []
    for m in nama:
        f = fdir / f"{m}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=False)
        print(f"{m}: evaluasi skenario ...", flush=True)
        h, k = evaluasi(m, {"md5": z["md5"], "X": z["X"]}, man, spl, a)
        hasil_semua.append(h)
        kurva_semua.append(k)
    if not hasil_semua:
        sys.exit("Tidak ada hasil (tidak ada fitur yang berhasil dibuat).")
    hasil, kurva = pd.concat(hasil_semua), pd.concat(kurva_semua)
    hasil.to_csv(bdir / "hasil_per_run.csv", index=False)
    kurva.to_csv(bdir / "kurva_loss.csv", index=False)
    tabel = ringkas(hasil)
    tabel.to_csv(bdir / "tabel_metrik_baseline.csv", index=False)
    gambar_kurva(kurva, bdir / "gambar_kurva_baseline.png")
    cfg = dict(task="healthy vs rust (biner)", model=MODELS, ukuran_masukan=SIZE, praproses="resize 224x224 bicubic, normalisasi sesuai model, tanpa augmentasi",
               kepala="logistic regression linear, bobot kelas seimbang, Adam full-batch, fitur distandardisasi dari data latih",
               epochs=a.epochs, lr=a.lr, l2=a.l2, pemilihan_epoch="macro-F1 validasi tertinggi",
               seed_split="42 (S1, S2), 42/43/44 (S3)", metrik="macro-F1 (utama), akurasi; versi 'unik' = satu citra per MD5",
               versi={p: versi(p) for p in ("numpy", "pandas", "torch", "timm", "transformers")})
    (bdir / "konfigurasi_baseline.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nTABEL METRIK BASELINE")
    print(tabel.to_string(index=False))
    print(f"\nSelesai. Keluaran di {bdir}/")


if __name__ == "__main__":
    main()
