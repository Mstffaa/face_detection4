"""
Crop wajah dengan detektor MediaPipe & YuNet lewat library DeepFace
(YuNet.py dan MediaPipe.py dari dosen adalah backend milik DeepFace).

Persiapan:
    pip install deepface mediapipe opencv-python
    (bobot model YuNet otomatis terdownload saat pertama kali dipakai)

Hasil:
    hasil_mediapipe/orang_01/img3_wajah01.jpg, ...
    hasil_yunet/orang_01/img3_wajah01.jpg, ...
    log_deteksi.csv
"""
import os

# Parameter bawaan backend dosen. Ubah di sini kalau mau bereksperimen.
os.environ.setdefault("yunet_score_threshold", "0.9")            # YuNet.py  (default 0.9)
os.environ.setdefault("MEDIAPIPE_MIN_DETECTION_CONFIDENCE", "0.7")  # MediaPipe.py (default 0.7)
os.environ.setdefault("MEDIAPIPE_MODEL_SELECTION", "0")          # 0 = jarak dekat, 1 = jarak jauh
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import csv
import time
import cv2
import numpy as np
from deepface import DeepFace

INPUT_DIR = "dataset_wajah"
SEMUA = {"mediapipe": "hasil_mediapipe", "yunet": "hasil_yunet"}
# Opsional: jalankan satu model saja, contoh (PowerShell): $env:BACKENDS="yunet"
PILIH = os.environ.get("BACKENDS", "mediapipe,yunet").split(",")
OUT = {k: v for k, v in SEMUA.items() if k in PILIH}
BORDER = 0.30   # margin tambahan di sekeliling gambar sebelum deteksi (crop manual biasanya mepet)
PADDING = 0.10  # margin tambahan di sekitar kotak wajah saat crop
EXT = (".jpg", ".jpeg", ".png")


def baca(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def simpan(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img)
    if ok:
        buf.tofile(path)


def crop(img, x, y, w, h):
    H, W = img.shape[:2]
    px, py = int(w * PADDING), int(h * PADDING)
    x1, y1 = max(0, x - px), max(0, y - py)
    x2, y2 = min(W, x + w + px), min(H, y + h + py)
    return img[y1:y2, x1:x2]


def deteksi(img, backend):
    """Kembalikan list kotak (x, y, w, h, confidence) dari backend DeepFace."""
    try:
        hasil = DeepFace.extract_faces(
            img_path=img, detector_backend=backend,
            enforce_detection=True, align=False)
    except ValueError as e:
        # DeepFace melempar ValueError saat tidak ada wajah. Error lain
        # (misalnya model gagal terdownload) tidak boleh dianggap "0 wajah".
        if "could not be detected" in str(e):
            return []
        raise
    kotak = []
    for r in hasil:
        a = r["facial_area"]
        kotak.append((a["x"], a["y"], a["w"], a["h"], r.get("confidence", 0)))
    return kotak


def main():
    baris = []
    for root, _, files in os.walk(INPUT_DIR):
        rel = os.path.relpath(root, INPUT_DIR)
        for fn in sorted(files):
            if not fn.lower().endswith(EXT):
                continue
            img = baca(os.path.join(root, fn))
            if img is None:
                continue
            H, W = img.shape[:2]
            by, bx = int(H * BORDER), int(W * BORDER)
            img = cv2.copyMakeBorder(img, by, by, bx, bx, cv2.BORDER_REPLICATE)
            nama = os.path.splitext(fn)[0]

            hasil = {}
            for backend, out in OUT.items():
                t0 = time.perf_counter()
                kotak = deteksi(img, backend)
                dt = (time.perf_counter() - t0) * 1000
                for i, (x, y, w, h, _) in enumerate(kotak):
                    c = crop(img, x, y, w, h)
                    if c.size:
                        simpan(os.path.join(out, rel, f"{nama}_wajah{i+1:02d}.jpg"), c)
                conf = max((k[4] for k in kotak), default=0)
                hasil[backend] = (len(kotak), dt, conf)

            m = hasil.get("mediapipe", (0, 0, 0))
            y = hasil.get("yunet", (0, 0, 0))
            baris.append([rel, fn, m[0], f"{m[2]:.2f}", f"{m[1]:.1f}",
                          y[0], f"{y[2]:.2f}", f"{y[1]:.1f}"])
            print(f"{rel}/{fn:25s} MediaPipe={m[0]}  YuNet={y[0]}")

    with open("log_deteksi.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["folder", "file", "mp_wajah", "mp_conf", "mp_ms",
                    "yunet_wajah", "yunet_conf", "yunet_ms"])
        w.writerows(baris)

    n = len(baris)
    print(f"\nTotal gambar : {n}")
    if "mediapipe" in OUT:
        print(f"Gagal deteksi MediaPipe: {sum(b[2]==0 for b in baris)} dari {n}")
    if "yunet" in OUT:
        print(f"Gagal deteksi YuNet    : {sum(b[5]==0 for b in baris)} dari {n}")
    print("Detail di log_deteksi.csv")


if __name__ == "__main__":
    main()
