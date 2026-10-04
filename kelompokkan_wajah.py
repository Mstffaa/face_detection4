"""
Mengelompokkan foto di dataset_wajah/_tidak_diketahui ke folder orang_XX.

Cara kerja:
1. Tiap foto diubah jadi embedding wajah (ArcFace via DeepFace, wajah dideteksi MediaPipe).
2. Foto yang jaraknya ke salah satu orang_01..orang_09 < AMBANG -> masuk folder orang itu.
3. Sisanya dikelompokkan sesama mereka (hierarchical clustering) -> orang_10, orang_11, ...
4. Foto tanpa wajah terdeteksi / berisi banyak wajah -> _perlu_dicek (jangan ditebak).

Mode crop (untuk memakai hasil crop YuNet, semua wajah ikut terproses):
    Command Prompt:  set INPUT_DIR=hasil_yunet
                     set OUTPUT_DIR=hasil_yunet_rapi
                     set SUDAH_CROP=1
                     python kelompokkan_wajah.py

Folder asli TIDAK diubah. Hasil disalin ke dataset_wajah_rapi/ + pengelompokan.csv.
Selalu cek hasilnya dengan mata sendiri sebelum dipakai!

Persiapan: pip install deepface tf-keras "mediapipe==0.10.21" opencv-python scipy
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MEDIAPIPE_MODEL_SELECTION", "1")
os.environ.setdefault("MEDIAPIPE_MIN_DETECTION_CONFIDENCE", "0.5")

import csv
import shutil
import collections
import cv2
import numpy as np
from deepface import DeepFace
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Bisa diubah lewat environment variable (lihat petunjuk di atas)
INPUT_DIR = os.environ.get("INPUT_DIR", "dataset_wajah")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "dataset_wajah_rapi")
# SUDAH_CROP=1 -> foto input sudah berupa crop wajah (mis. hasil_yunet), jadi tidak dideteksi ulang
SUDAH_CROP = os.environ.get("SUDAH_CROP", "0") == "1"
UNKNOWN = "_tidak_diketahui"
MODEL = "ArcFace"
AMBANG = 0.60       # jarak cosine maks agar dianggap orang yang sama (default DeepFace ArcFace = 0.68)
RAGU = 0.52         # di atas ini (tapi < AMBANG) ditandai "ragu"
EXT = (".jpg", ".jpeg", ".png")


def baca(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def jarak(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def embedding(path):
    """Return (embedding, status). status: ok | tanpa_deteksi | banyak_wajah"""
    img = baca(path)
    if img is None:
        return None, "gagal_baca"
    if SUDAH_CROP:
        r = DeepFace.represent(img_path=img, model_name=MODEL, detector_backend="skip",
                               enforce_detection=False)
        return np.array(r[0]["embedding"]), "ok"
    try:
        r = DeepFace.represent(img_path=img, model_name=MODEL, detector_backend="mediapipe",
                               enforce_detection=True, align=True)
    except ValueError as e:
        if "could not be detected" not in str(e):
            raise
        r = DeepFace.represent(img_path=img, model_name=MODEL, detector_backend="skip",
                               enforce_detection=False)
        return np.array(r[0]["embedding"]), "tanpa_deteksi"
    if len(r) > 1:
        return np.array(r[0]["embedding"]), "banyak_wajah"
    return np.array(r[0]["embedding"]), "ok"


def main():
    # 1. embedding semua foto
    data = {}  # path -> (folder, emb, status)
    for folder in sorted(os.listdir(INPUT_DIR)):
        fp = os.path.join(INPUT_DIR, folder)
        if not os.path.isdir(fp):
            continue
        for fn in sorted(os.listdir(fp)):
            if fn.lower().endswith(EXT):
                p = os.path.join(fp, fn)
                emb, st = embedding(p)
                if emb is not None:
                    data[p] = (folder, emb, st)
                print(f"{folder}/{fn}: {st}")

    known = collections.defaultdict(list)
    for p, (folder, emb, st) in data.items():
        if folder != UNKNOWN:
            known[folder].append(emb)

    hasil = {}  # path -> (folder_tujuan, jarak, keterangan)
    sisa = []
    for p, (folder, emb, st) in data.items():
        if folder != UNKNOWN:
            hasil[p] = (folder, 0.0, "asli")
        elif st != "ok":
            hasil[p] = ("_perlu_dicek", -1, st)
        else:
            dist = {k: min(jarak(emb, e) for e in es) for k, es in known.items()}
            k, d = min(dist.items(), key=lambda x: x[1])
            if d < AMBANG:
                hasil[p] = (k, d, "ragu" if d > RAGU else "cocok_folder_lama")
            else:
                sisa.append(p)

    # 2. cluster sisanya -> orang baru
    nomor = len(known)
    if len(sisa) == 1:
        nomor += 1
        hasil[sisa[0]] = (f"orang_{nomor:02d}", 0.0, "sendirian")
    elif len(sisa) > 1:
        E = np.array([data[p][1] for p in sisa])
        D = np.array([[jarak(a, b) for b in E] for a in E])
        np.fill_diagonal(D, 0)
        Z = linkage(squareform(D, checks=False), method="average")
        lab = fcluster(Z, t=AMBANG, criterion="distance")
        ukuran = collections.Counter(lab)
        peta = {}
        for l in sorted(set(lab), key=lambda l: -ukuran[l]):
            nomor += 1
            peta[l] = f"orang_{nomor:02d}"
        for p, l in zip(sisa, lab):
            hasil[p] = (peta[l], 0.0, "grup_baru" if ukuran[l] > 1 else "sendirian")

    # 3. salin ke folder baru + laporan
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    baris = []
    for p, (tujuan, d, ket) in sorted(hasil.items(), key=lambda x: (x[1][0], x[0])):
        os.makedirs(os.path.join(OUTPUT_DIR, tujuan), exist_ok=True)
        shutil.copy2(p, os.path.join(OUTPUT_DIR, tujuan, os.path.basename(p)))
        baris.append([os.path.basename(p), data[p][0], tujuan, f"{d:.2f}" if d >= 0 else "", ket])
    with open("pengelompokan.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "folder_asal", "folder_tujuan", "jarak", "keterangan"])
        w.writerows(baris)

    print("\nRingkasan folder tujuan:")
    for k, v in sorted(collections.Counter(b[2] for b in baris).items()):
        print(f"  {k}: {v} foto")


if __name__ == "__main__":
    main()
