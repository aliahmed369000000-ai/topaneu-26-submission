"""
معايرة Threshold لتعظيم المقاييس الرسمية (Precision / Recall / MCC).

الاستخدام بعد التدريب:
    python calibrate_threshold.py --data-root /path/to/topaneu --models-dir ./models

يقرأ أوزان الـ folds، يحسب احتمالات على Validation (أو كل البيانات)،
ثم يبحث عن أفضل threshold عام (أو لكل فئة) يعظم متوسط (P+R+MCC)/3 أو MCC.
يحفظ النتيجة في models/thresholds.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import TopAneuNet
from dataset_split import build_label_matrix, make_folds, evaluable_classes
from vessel_preprocessing import load_and_resize_case
from train import TopAneuDataset, FLIP_MAP, N_CLASSES


def precision_recall_mcc(y_true: np.ndarray, y_pred: np.ndarray):
    """y_true, y_pred: binary arrays لنفس الفئة."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    rec = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tn * tp) - (fn * fp)) / denom if denom > 0 else np.nan
    return prec, rec, mcc


def evaluate_at_threshold(probs: np.ndarray, labels: np.ndarray, thr: float):
    preds = (probs >= thr).astype(np.int32)
    precs, recs, mccs = [], [], []
    for c in range(labels.shape[1]):
        if labels[:, c].sum() == 0 and preds[:, c].sum() == 0:
            continue  # كلاهما صفر → غير معرف
        p, r, m = precision_recall_mcc(labels[:, c], preds[:, c])
        if not np.isnan(p):
            precs.append(p)
        if not np.isnan(r):
            recs.append(r)
        if not np.isnan(m):
            mccs.append(m)
    mean_p = float(np.nanmean(precs)) if precs else float("nan")
    mean_r = float(np.nanmean(recs)) if recs else float("nan")
    mean_m = float(np.nanmean(mccs)) if mccs else float("nan")
    score = np.nanmean([mean_p, mean_r, mean_m])
    return mean_p, mean_r, mean_m, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--models-dir", type=str, default="./models")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    images_dir = data_root / "images"
    vessel_masks_dir = data_root / "vessel_masks"
    location_jsons_dir = data_root / "location_jsons"
    location_mapping = json.load(open(data_root / "location_mapping.json"))

    case_ids = sorted(p.stem.replace(".nii", "") for p in images_dir.glob("*.nii.gz"))
    Y = build_label_matrix(str(location_jsons_dir), location_mapping, case_ids, n_classes=N_CLASSES)
    folds = make_folds(case_ids, Y, n_splits=args.n_splits, seed=args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # جمع احتمالات Validation من كل fold
    all_probs = []
    all_labels = []

    for fold_idx, fold in enumerate(folds):
        weight_path = Path(args.models_dir) / f"topaneu_task1_fold{fold_idx}.pt"
        if not weight_path.is_file():
            print(f"تخطي fold {fold_idx}: لا يوجد {weight_path}")
            continue

        model = torch.load(str(weight_path), map_location=device, weights_only=False)
        model.eval()

        val_ids = fold["val"]
        val_idx = [case_ids.index(c) for c in val_ids]
        Y_val = Y[val_idx]

        ds = TopAneuDataset(val_ids, images_dir, vessel_masks_dir, Y_val)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                probs = torch.sigmoid(model(xb)).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(yb.numpy())

    if not all_probs:
        print("لا توجد احتمالات. تأكد من وجود الأوزان.")
        return

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print(f"عينات Validation مجمّعة: {probs.shape[0]}")

    # بحث شبكي عن أفضل threshold عام
    best_thr = 0.5
    best_score = -1.0
    results = []
    for thr in np.arange(0.15, 0.85, 0.025):
        p, r, m, score = evaluate_at_threshold(probs, labels, thr)
        results.append({"threshold": float(thr), "precision": p, "recall": r, "mcc": m, "score": score})
        if score > best_score:
            best_score = score
            best_thr = float(thr)

    print(f"\nأفضل threshold عام: {best_thr:.3f}  (score={best_score:.4f})")
    for row in results:
        marker = " <-- best" if abs(row["threshold"] - best_thr) < 1e-6 else ""
        print(
            f"  thr={row['threshold']:.3f}  P={row['precision']:.3f}  "
            f"R={row['recall']:.3f}  MCC={row['mcc']:.3f}  score={row['score']:.3f}{marker}"
        )

    out = {
        "global_threshold": best_thr,
        "best_score": best_score,
        "search_results": results,
    }
    out_path = Path(args.models_dir) / "thresholds.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nحُفظ في {out_path}")
    print("حدّث DECISION_THRESHOLD في inference.py بهذا الرقم.")


if __name__ == "__main__":
    main()
