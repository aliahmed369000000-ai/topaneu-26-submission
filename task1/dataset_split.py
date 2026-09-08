"""
استراتيجية Validation لبيانات قليلة (415 حالة) و52 فئة معظمها نادر:
K-Fold عادي عشوائي قد يُنتج فولد بدون أي حالة إيجابية لفئة نادرة معينة،
مما يكسر حساب AUC/F1 لتلك الفئة. الحل: Multilabel Stratified K-Fold،
الذي يوازن توزيع كل فئة على حدة عبر الفولدات قدر الإمكان.

يتطلب: pip install iterative-stratification --break-system-packages
"""

import json
from pathlib import Path

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def build_label_matrix(
    location_jsons_dir: str,
    location_mapping: dict,
    case_ids: list,
    n_classes: int = 52,
) -> np.ndarray:
    """
    يبني مصفوفة (N, 52) من ملفات location_jsons/*.json.
    location_mapping: القاموس المحمّل من location_mapping.json (القيم 1..52).
    الفهرسة المستخدمة: index = value - 1 (مؤكدة سابقًا من الملف الرسمي).
    """
    Y = np.zeros((len(case_ids), n_classes), dtype=np.int32)
    for i, cid in enumerate(case_ids):
        path = Path(location_jsons_dir) / f"{cid}.json"
        with open(path) as f:
            locs = json.load(f)  # قائمة أسماء الفئات الموجودة في هذه الحالة
        for name in locs:
            idx = location_mapping["labels"][name] - 1
            Y[i, idx] = 1
    return Y


def make_folds(case_ids: list, Y: np.ndarray, n_splits: int = 5, seed: int = 42) -> list:
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X_dummy = np.zeros((len(case_ids), 1))
    folds = []
    for train_idx, val_idx in mskf.split(X_dummy, Y):
        folds.append(
            {
                "train": [case_ids[i] for i in train_idx],
                "val": [case_ids[i] for i in val_idx],
            }
        )
    return folds


def evaluable_classes(Y_subset: np.ndarray) -> list:
    """
    الفئات القابلة للتقييم فقط: التي تحتوي حالة إيجابية واحدة على الأقل
    في المجموعة المُعطاة (لتجنّب AUC غير مُعرّف رياضيًا لفئة بدون أي Positive).
    """
    return [c for c in range(Y_subset.shape[1]) if Y_subset[:, c].sum() > 0]


if __name__ == "__main__":
    # مثال استخدام — عدّل المسارات لتطابق بيئتك
    import sys

    location_mapping = json.load(open("location_mapping.json"))
    case_ids = [p.stem.replace(".nii", "") for p in Path("images").glob("*.nii.gz")]

    Y = build_label_matrix("location_jsons", location_mapping, case_ids)
    folds = make_folds(case_ids, Y, n_splits=5)

    for i, fold in enumerate(folds):
        val_idx = [case_ids.index(c) for c in fold["val"]]
        n_eval = len(evaluable_classes(Y[val_idx]))
        print(f"Fold {i}: train={len(fold['train'])} val={len(fold['val'])} evaluable_classes={n_eval}/52")
