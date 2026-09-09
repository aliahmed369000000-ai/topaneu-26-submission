"""
حلقة تدريب كاملة لـTask 1: تدمج model.py مع dataset_split.py
(K-Fold Multilabel Stratified) و vessel_preprocessing.py (المعالجة المسبقة)
الموجودين فعليًا في هذا المستودع.

الاستخدام المتوقع:
    python train.py --data-root /path/to/topaneu

يفترض بنية مجلد كما هي موثّقة في docs/CHALLENGE_REFERENCE.md:
    topaneu/
    ├── images/            *.nii.gz
    ├── vessel_masks/      *.nii.gz (نفس اسم الملف في images/)
    ├── location_jsons/    *.json (نفس case_id)
    └── location_mapping.json  (نسخة موجودة أيضًا في docs/ هنا)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import TopAneuNet, count_parameters
from dataset_split import build_label_matrix, make_folds, evaluable_classes
from vessel_preprocessing import load_and_resize_case

N_CLASSES = 52


def compute_pos_weight(labels: np.ndarray, min_w: float = 1.0, max_w: float = 15.0, smoothing: float = 1.0):
    """صيغة pos_weight مع Laplace Smoothing (يمنع القسمة على صفر للفئات
    بدون أي حالة إيجابية) و Cap (يمنع عدم استقرار التدريب من فئة نادرة
    جدًا) — مُختبرة سابقًا على بيانات اصطناعية، راجع جلسة التصميم."""
    labels_t = torch.from_numpy(labels).float()
    n = labels_t.shape[0]
    pos = labels_t.sum(dim=0)
    neg = n - pos
    w = (neg + smoothing) / (pos + smoothing)
    return w.clamp(min=min_w, max=max_w)


class TopAneuDataset(Dataset):
    def __init__(self, case_ids, images_dir, vessel_masks_dir, Y, augment_flip=False, flip_map=None):
        self.case_ids = case_ids
        self.images_dir = Path(images_dir)
        self.vessel_masks_dir = Path(vessel_masks_dir)
        self.Y = Y
        self.augment_flip = augment_flip
        self.flip_map = flip_map
        if augment_flip:
            assert flip_map is not None, "augment_flip=True يتطلب تمرير flip_map"

    def __len__(self):
        n = len(self.case_ids)
        return n * 2 if self.augment_flip else n

    def __getitem__(self, idx):
        do_flip = self.augment_flip and idx >= len(self.case_ids)
        real_idx = idx % len(self.case_ids)
        cid = self.case_ids[real_idx]

        image_path = self.images_dir / f"{cid}_0000.nii.gz"
        if not image_path.is_file():
            image_path = self.images_dir / f"{cid}.nii.gz"
        vessel_path = self.vessel_masks_dir / f"{cid}.nii.gz"
        volume = load_and_resize_case(str(image_path), str(vessel_path))  # (2,128,128,128)
        labels = self.Y[real_idx].copy().astype(np.float32)

        volume_t = torch.from_numpy(volume)
        labels_t = torch.from_numpy(labels)

        if do_flip:
            volume_t = torch.flip(volume_t, dims=[-1])  # محور 0 الأصلي = آخر بُعد بعد stack
            flipped_labels = torch.zeros_like(labels_t)
            for i in range(N_CLASSES):
                flipped_labels[self.flip_map.get(i, i)] = labels_t[i]
            labels_t = flipped_labels

        return volume_t, labels_t


FLIP_MAP = {
    0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 8: 9, 9: 8, 10: 11, 11: 10,
    12: 13, 13: 12, 14: 15, 15: 14, 17: 18, 18: 17, 19: 20, 20: 19,
    21: 22, 22: 21, 23: 24, 24: 23, 25: 26, 26: 25, 27: 28, 28: 27,
    29: 30, 30: 29, 31: 32, 32: 31, 33: 34, 34: 33, 36: 37, 37: 36,
    38: 39, 39: 38, 40: 41, 41: 40, 42: 43, 43: 42, 44: 45, 45: 44,
    46: 47, 47: 46, 48: 49, 49: 48, 50: 51, 51: 50,
    # 6, 7, 16, 35 (BA trunk, VA-BA junction, BA tip, Acom complex) = NA، تبقى كما هي
}


def auc_manual(y_true: np.ndarray, y_score: np.ndarray):
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            probs = torch.sigmoid(model(xb)).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(yb.numpy())
    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    aucs = []
    for c in evaluable_classes(labels):
        a = auc_manual(labels[:, c], probs[:, c])
        if a is not None:
            aucs.append(a)
    mean_auc = float(np.mean(aucs)) if aucs else float("nan")
    return mean_auc, len(aucs)


def train_one_fold(fold_idx, fold, Y, case_ids, args, device):
    train_ids = fold["train"]
    val_ids = fold["val"]
    train_idx = [case_ids.index(c) for c in train_ids]
    val_idx = [case_ids.index(c) for c in val_ids]

    Y_train = Y[train_idx]
    Y_val = Y[val_idx]

    train_ds = TopAneuDataset(
        train_ids, args.images_dir, args.vessel_masks_dir, Y_train,
        augment_flip=True, flip_map=FLIP_MAP,
    )
    val_ds = TopAneuDataset(val_ids, args.images_dir, args.vessel_masks_dir, Y_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = TopAneuNet(in_channels=2, feature_dim=512).to(device)
    pos_weight = compute_pos_weight(Y_train).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_auc = -1.0
    best_path = Path(args.output_dir) / f"topaneu_task1_fold{fold_idx}.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            # Gradient Clipping: احتياط ضروري — لوحظ فعليًا أثناء الاختبار أن
            # التدرجات قد تكون ضخمة جدًا (مليارات) مع batch صغير (BatchNorm
            # غير مستقر على عينتين) وأوزان عشوائية في بداية التدريب.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            total_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)

        mean_auc, n_eval = evaluate(model, val_loader, device)
        print(
            f"[fold {fold_idx}] epoch {epoch + 1}/{args.epochs} "
            f"loss={total_loss / max(n_seen, 1):.4f} val_AUC={mean_auc:.3f} ({n_eval} فئة قابلة للتقييم)"
        )

        if mean_auc > best_auc:
            best_auc = mean_auc
            torch.save(model, best_path)  # يُحفظ النموذج كاملًا (ليس state_dict فقط)
            # ليتوافق مباشرة مع torch.load(...) في inference.py

    print(f"[fold {fold_idx}] أفضل AUC: {best_auc:.3f} — محفوظ في {best_path}")
    return best_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./models")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    args.images_dir = data_root / "images"
    args.vessel_masks_dir = data_root / "vessel_masks"
    location_jsons_dir = data_root / "location_jsons"
    location_mapping_path = data_root / "location_mapping.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"الجهاز: {device}")

    location_mapping = json.load(open(location_mapping_path))
    def _cid(path):
        name = path.name
        if name.endswith(".nii.gz"): name = name[:-7]
        elif name.endswith(".nii"): name = name[:-4]
        if name.endswith("_0000"): name = name[:-5]
        return name
    case_ids = sorted({_cid(p) for p in args.images_dir.glob("*.nii.gz")})
    print(f"عدد الحالات: {len(case_ids)}")

    Y = build_label_matrix(str(location_jsons_dir), location_mapping, case_ids, n_classes=N_CLASSES)
    folds = make_folds(case_ids, Y, n_splits=args.n_splits, seed=args.seed)

    model_check = TopAneuNet(in_channels=2, feature_dim=512)
    print(f"معاملات النموذج: {count_parameters(model_check):,}")

    fold_aucs = []
    for i, fold in enumerate(folds):
        auc = train_one_fold(i, fold, Y, case_ids, args, device)
        fold_aucs.append(auc)

    print("\n=== ملخص K-Fold ===")
    print(f"AUC لكل فولد: {[f'{a:.3f}' for a in fold_aucs]}")
    print(f"متوسط ± انحراف: {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")


if __name__ == "__main__":
    main()
