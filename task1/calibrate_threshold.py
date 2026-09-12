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

from model import TopAneuNet, get_device
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


def evaluate_class_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float):
    y_pred = (y_score >= thr).astype(np.int32)
    return precision_recall_mcc(y_true, y_pred)


def calibrate_per_class(probs: np.ndarray, labels: np.ndarray, min_positives: int = 1,
                         fallback_threshold: float = 0.5):
    """
    يبحث عن أفضل threshold مستقل لكل فئة من الـ52 (بدل threshold عام واحد).

    لماذا لكل فئة؟ الفئات الشائعة (مثل ICA terminus, MCA M1) والفئات
    النادرة جدًا لها ديناميكيات مختلفة تمامًا؛ threshold عام واحد يجبر
    توازنًا خاطئًا بينهما، وهو مصدر رئيسي لـ False Positives الكثيرة
    (راجع نقاش استراتيجية تقليل False Positives في جلسة التصميم).

    ⚠️ تصحيح مهم (كان threshold=0.99 سابقًا، أُلغي): الفئات بعدد حالات
    إيجابية أقل من min_positives تستخدم الآن fallback_threshold (threshold
    العام المُعاير على كل البيانات، من evaluate_at_threshold/الحلقة في
    main()) بدل قيمة شبه-حظر (0.99). اختبار فعلي كشف أن 0.99 على فولد
    صغير (30 حالة تحقق) يرفض كل الفئات الـ52 دفعة واحدة (لا فئة لها 3+
    حالات إيجابية) وينتج MCC=1.000 مضلِّلًا مع Recall=0.067 فقط -- النموذج
    يتوقف عن التنبؤ بأي شيء تقريبًا، وهذا ليس نجاحًا حقيقيًا (راجع الشرح
    الكامل في جلسة التصميم لماذا MCC مثالي هنا لا يعني نموذجًا جيدًا).
    الفئات النادرة تستحق فرصة تنبؤ معقولة (threshold العام المُختبر على
    كامل البيانات) بدل تعطيلها بالكامل.
    """
    n_classes = labels.shape[1]
    per_class_thr = []
    per_class_info = []

    for c in range(n_classes):
        y_true = labels[:, c]
        y_score = probs[:, c]
        n_pos = int(y_true.sum())

        if n_pos < min_positives:
            per_class_thr.append(fallback_threshold)
            per_class_info.append({
                "class_idx": c, "n_positives": n_pos, "threshold": fallback_threshold,
                "reason": f"إشارة غير كافية ({n_pos} < {min_positives}) — استُخدم "
                          f"threshold العام ({fallback_threshold:.3f}) بدل رفض كامل",
            })
            continue

        best_thr_c, best_mcc_c = 0.5, -1.0
        for thr in np.arange(0.15, 0.90, 0.025):
            _, _, mcc = evaluate_class_at_threshold(y_true, y_score, thr)
            if not np.isnan(mcc) and mcc > best_mcc_c:
                best_mcc_c = mcc
                best_thr_c = float(thr)

        per_class_thr.append(best_thr_c)
        p, r, m = evaluate_class_at_threshold(y_true, y_score, best_thr_c)
        per_class_info.append({
            "class_idx": c, "n_positives": n_pos, "threshold": best_thr_c,
            "precision": float(p) if not np.isnan(p) else None,
            "recall": float(r) if not np.isnan(r) else None,
            "mcc": float(m) if not np.isnan(m) else None,
        })

    return per_class_thr, per_class_info


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

    device = get_device()

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

    # معايرة لكل فئة على حدة (الاستراتيجية الأساسية لتقليل False Positives)
    per_class_thr, per_class_info = calibrate_per_class(
        probs, labels, min_positives=1, fallback_threshold=best_thr
    )
    n_rejected = sum(1 for info in per_class_info if info.get("reason"))
    print(f"\nمعايرة لكل فئة: {n_rejected}/{len(per_class_info)} فئة رُفضت "
          f"(إشارة غير كافية، threshold=0.99)")

    # تقييم الأداء الكلي باستخدام thresholds لكل فئة (بدل العام) للمقارنة
    preds_per_class = np.zeros_like(labels, dtype=np.int32)
    for c in range(labels.shape[1]):
        preds_per_class[:, c] = (probs[:, c] >= per_class_thr[c]).astype(np.int32)
    precs, recs, mccs = [], [], []
    for c in range(labels.shape[1]):
        if labels[:, c].sum() == 0 and preds_per_class[:, c].sum() == 0:
            continue
        p, r, m = precision_recall_mcc(labels[:, c], preds_per_class[:, c])
        if not np.isnan(p): precs.append(p)
        if not np.isnan(r): recs.append(r)
        if not np.isnan(m): mccs.append(m)
    pc_score = float(np.nanmean([np.nanmean(precs), np.nanmean(recs), np.nanmean(mccs)]))
    print(f"مقارنة: threshold عام={best_score:.4f}  |  threshold لكل فئة={pc_score:.4f}")

    out = {
        "global_threshold": best_thr,
        "global_score": best_score,
        "per_class_thresholds": per_class_thr,
        "per_class_score": pc_score,
        "per_class_info": per_class_info,
        "search_results": results,
    }
    out_path = Path(args.models_dir) / "thresholds.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nحُفظ في {out_path}")
    print("inference.py يقرأ per_class_thresholds تلقائيًا إذا وُجد الملف "
          "(fallback إلى DECISION_THRESHOLD العام إذا لم يوجد).")


if __name__ == "__main__":
    main()
