"""Hitung ulang group_id di outputs/manifest.csv tanpa membaca ulang citra.

Perbaikan: pengelompokan lama (single-linkage) bisa merangkai foto yang berbeda menjadi satu
grup raksasa (A mirip B, B mirip C, ...). Di sini komponen yang "melebar" (jarak terjauh antar
anggota > 2 x ambang) dipecah dengan pengelompokan pemimpin: tiap anggota harus dekat
(jarak <= ambang) dengan pemimpin grupnya.

Pakai:  python regroup.py --manifest outputs/manifest.csv --thr 3
Setelah itu jalankan:  python eda_split_preprocess.py --root datasets --out outputs --reuse
"""
import argparse
import numpy as np
import pandas as pd


def popcount(x):
    x = np.asarray(x, dtype=np.uint64)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.int32)
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1).astype(np.int32)


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a


def cluster_domain(hashes, thr):
    """hashes: array uint64 (satu per MD5 unik). Kembalikan nomor grup per elemen."""
    n = len(hashes)
    H = np.zeros((n, n), dtype=np.int8)
    for i in range(n - 1):
        d = popcount(hashes[i] ^ hashes[i + 1:])
        H[i, i + 1:] = d
        H[i + 1:, i] = d
    uf = UF(n)
    ii, jj = np.nonzero(np.triu(H <= thr, k=1))
    for i, j in zip(ii.tolist(), jj.tolist()):
        uf.union(i, j)
    comp = np.array([uf.find(i) for i in range(n)])
    out = np.full(n, -1, dtype=np.int64)
    nxt = 0
    for c in np.unique(comp):
        mem = np.flatnonzero(comp == c)
        sub = H[np.ix_(mem, mem)]
        if len(mem) == 1 or sub.max() <= 2 * thr:
            out[mem] = nxt
            nxt += 1
            continue
        left = np.ones(len(mem), dtype=bool)  # pengelompokan pemimpin di dalam komponen
        for k in range(len(mem)):
            if not left[k]:
                continue
            take = left & (sub[k] <= thr)
            take[k] = True
            out[mem[take]] = nxt
            left &= ~take
            nxt += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="outputs/manifest.csv")
    ap.add_argument("--thr", type=int, default=3)
    a = ap.parse_args()

    df = pd.read_csv(a.manifest, dtype={"md5": str, "dhash": str}, keep_default_na=False)
    df["corrupt"] = df["corrupt"].astype(int)
    df["group_id"] = ""
    for domain, sub in df.groupby("domain"):
        ok = sub[sub["corrupt"] == 0]
        rep = ok.drop_duplicates("md5")
        h = np.array([int(x, 16) for x in rep["dhash"]], dtype=np.uint64)
        gid = cluster_domain(h, a.thr)
        m = dict(zip(rep["md5"], gid))
        df.loc[ok.index, "group_id"] = [f"{domain}_{m[x]:06d}" for x in ok["md5"]]
        print(f"{domain}: {len(ok)} citra, {len(rep)} MD5 unik, {len(set(gid))} grup")

    has = df["group_id"] != ""
    df["group_size"] = 0
    df.loc[has, "group_size"] = df[has].groupby("group_id")["uid"].transform("size")
    nlab = df[has].groupby("group_id")["label"].agg(lambda s: len(set(x for x in s if x != "unknown")))
    conflict = set(nlab[nlab > 1].index)
    df["label_conflict"] = df["group_id"].isin(conflict)
    df.to_csv(a.manifest, index=False)

    print("\nGrup terbesar (citra):", df[has].group_id.value_counts().head(5).to_dict())
    print("Grup berlabel bertentangan:", len(conflict), "grup,", int(df.label_conflict.sum()), "citra")
    print(df[df.label_conflict].groupby("source").size().to_string() if conflict else "tidak ada")
    print("Tersimpan:", a.manifest)


if __name__ == "__main__":
    main()
