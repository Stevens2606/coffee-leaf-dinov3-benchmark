from pathlib import Path
from unrar.cffi import rarfile

rar = "datasets/raw/peru/Coffee leaf dataset by phytosanitary class/Coffe Leaf.rar"
dest = Path("datasets/raw/peru/citra")
ganti = {"Sanas": "Sanas_healthy", "Roya": "Roya_rust", "Ojo_Gallo": "Ojo_Gallo_other"}

rf = rarfile.RarFile(rar)
n = 0
for name in rf.namelist():
    if name.endswith("/") or "/" not in name:
        continue
    kelas, berkas = name.split("/", 1)
    if kelas not in ganti or not berkas.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    out = dest / ganti[kelas] / Path(berkas).name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(rf.read(name))
    n += 1
    if n % 200 == 0:
        print(n, flush=True)
print("selesai:", n, "citra", flush=True)
