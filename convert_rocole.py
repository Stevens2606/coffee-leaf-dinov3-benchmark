import pyarrow.parquet as pq
from pathlib import Path
import csv

NAMES = ["healthy", "red_spider_mite", "rust_level_1",
         "rust_level_2", "rust_level_3", "rust_level_4"]
SRC = Path("datasets/raw/ecuador_rocole")
OUT = Path("datasets/processed/ecuador_rocole")
OUT.mkdir(parents=True, exist_ok=True)

rows_csv, n = [], 0
for f in sorted(SRC.glob("train-*.parquet")):
    pf = pq.ParquetFile(f)
    for batch in pf.iter_batches(batch_size=32, columns=["image", "objects"]):
        for r in batch.to_pylist():
            cats = r["objects"]["categories"]
            label = NAMES[cats[0]]
            mixed = len(set(cats)) > 1
            ext = Path(r["image"]["path"] or "x.jpg").suffix or ".jpg"
            d = OUT / label
            d.mkdir(exist_ok=True)
            name = f"rocole_{n:05d}{ext}"
            (d / name).write_bytes(r["image"]["bytes"])
            rows_csv.append([f"{label}/{name}", label, mixed])
            n += 1

with open(OUT / "labels.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["file", "label", "mixed_classes"])
    w.writerows(rows_csv)
print("selesai:", n, "gambar")
