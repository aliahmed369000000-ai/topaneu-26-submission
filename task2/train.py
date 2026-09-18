"""
حلقة تدريب Task 2: تجزئة كثيفة متعددة الفئات (53: خلفية+52 موقعًا).
K-Fold + CrossEntropyLoss (موزونة) + Dice Loss مُركّبة + Flip Augmentation
+ Gradient Clipping (احتياط، نفس درس Task 1 مع batch صغير).

الاستخدام:
    python train.py --data-root /path/to/topaneu --output-dir ./models --epochs 30
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import UNet3D, count_parameters, N_CLASSES_TASK2
from dataset_split import build_label_matrix, make_folds
from vessel_preprocessing import load_and_resize_case_seg, flip_location_mask, FLIP_MAP


def get_device() -> torch.device:
    """نفس منطق task1/model.py's get_device() بالضبط (فحص فعلي بدل
    is_available() الذي لا يكتشف أعطال P100). نسخة مستقلة عمدًا (حاويتا
    Docker منفصلتان، راجع تعليق FLIP_MAP في vessel_preprocessing.py)."""
    import os
    if os.environ.get("TOPANEU_FORCE_CPU", "0") == "1":
        print("FORCE CPU via TOPANEU_FORCE_CPU")
        return torch.device("cpu")
    if torch.cuda.is_available():
        try:
            x = torch.zeros(1, device="cuda")
            del x
            torch.cuda.empty_cache()
            return torch.device("cuda")
        except Exception as e:
            print(f"CUDA unusable ({e}); falling back to CPU")
            return torch.device("cpu")
    return torch.device("cpu")


class Task2Dataset(Dataset):
    def __init__(self, case_ids, images_dir, vessel_masks_dir, location_masks_dir,
                 target_shape=(128, 128, 128), augment_flip=False):
        self.case_ids = case_ids
        self.images_dir = Path(images_dir)
        self.vessel_masks_dir = Path(vessel_masks_dir)
        self.location_masks_dir = Path(location_masks_dir)
        self.target_shape = target_shape
        self.augment_flip = augment_flip

    def __len__(self):
        return len(self.case_ids) * (2 if self.augment_flip else 1)

    def __getitem__(self, idx):
        do_flip = self.augment_flip and idx >= len(self.case_ids)
        cid = self.case_ids[idx % len(self.case_ids)]

        volume, target = load_and_resize_case_seg(
            str(self.images_dir / f"{cid}.nii.gz"),
            str(self.vessel_masks_dir / f"{cid}.nii.gz"),
            str(self.location_masks_dir / f"{cid}.nii.gz"),
            target_shape=self.target_shape,
        )
        if do_flip:
            volume = volume[:, :, :, ::-1].copy()
            target = flip_location_mask(target)

        return torch.from_numpy(volume), torch.from_numpy(target)


def compute_class_weights(case_ids, location_masks_dir, n_classes=N_CLASSES_TASK2,
                           min_w=1.0, max_w=50.0):
    """وزن لكل فئة (بما فيها الخلفية) لـCrossEntropyLoss -- عدم توازن
    فوكسل أشد بكثير من عدم توازن Task 1 على مستوى الحالة (الخلفية تهيمن
    على 99%+ من الفوكسلات في أي حجم طبي). نحسب من عينة عشوائية من الحالات
    فقط (لا كل الحالات، توفيرًا للوقت -- عدد الفوكسلات ضخم أصلًا)."""
    import nibabel as nib
    counts = np.zeros(n_classes, dtype=np.int64)
    sample = case_ids[:min(len(case_ids), 30)]
    for cid in sample:
        path = Path(location_masks_dir) / f"{cid}.nii.gz"
        if not path.exists():
            continue
        data = nib.load(str(path)).get_fdata()
        vals, cnts = np.unique(np.round(data).astype(np.int64), return_counts=True)
        for v, c in zip(vals, cnts):
            if 0 <= v < n_classes:
                counts[v] += c
    counts = counts.astype(np.float64)
    counts[counts == 0] = 1.0  # تفادي القسمة على صفر لفئة غائبة عن العينة
    total = counts.sum()
    weights = total / (n_classes * counts)
    weights = np.clip(weights, min_w, max_w)
    return torch.from_numpy(weights.astype(np.float32))


def dice_loss_multiclass(logits: torch.Tensor, target: torch.Tensor, n_classes: int,
                          smooth: float = 1.0) -> torch.Tensor:
    """Dice Loss ناعمة متعددة الفئات (مكمّلة لـCrossEntropy، شائعة جدًا
    في تجزئة طبية بسبب عدم توازن الفوكسل الشديد -- CrossEntropy وحدها
    قد لا تُحسّن Recall على الفئات النادرة جدًا بما يكفي)."""
    probs = F.softmax(logits, dim=1)
    target_onehot = F.one_hot(target, num_classes=n_classes).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    intersection = (probs * target_onehot).sum(dims)
    union = probs.sum(dims) + target_onehot.sum(dims)
    dice_per_class = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice_per_class.mean()


def evaluate_dice(model, loader, device, n_classes=N_CLASSES_TASK2):
    """متوسط Dice لكل فئة موجودة فعليًا في دفعة التحقق (تجاهل الفئات
    الغائبة كليًا، بنفس منطق evaluable_classes في Task 1)."""
    model.eval()
    dice_sums = np.zeros(n_classes)
    dice_counts = np.zeros(n_classes)
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            for c in range(n_classes):
                true_c = (yb == c)
                pred_c = (preds == c)
                if true_c.sum() == 0 and pred_c.sum() == 0:
                    continue
                inter = (true_c & pred_c).sum().item()
                denom = true_c.sum().item() + pred_c.sum().item()
                dice = (2 * inter / denom) if denom > 0 else 0.0
                dice_sums[c] += dice
                dice_counts[c] += 1
    valid = dice_counts > 0
    mean_dice = (dice_sums[valid] / dice_counts[valid]).mean() if valid.any() else float("nan")
    return mean_dice, int(valid.sum())


def train_one_fold(fold_idx, fold, args, device, class_weights):
    train_ids = fold["train"]
    val_ids = fold["val"]

    train_ds = Task2Dataset(train_ids, args.images_dir, args.vessel_masks_dir,
                             args.location_masks_dir, target_shape=args.target_shape,
                             augment_flip=True)
    val_ds = Task2Dataset(val_ids, args.images_dir, args.vessel_masks_dir,
                           args.location_masks_dir, target_shape=args.target_shape)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = UNet3D().to(device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_dice = -1.0
    best_path = Path(args.output_dir) / f"topaneu_task2_fold{fold_idx}.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss, n_seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = ce_loss_fn(logits, yb) + dice_loss_multiclass(logits, yb, N_CLASSES_TASK2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            total_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)

        mean_dice, n_eval = evaluate_dice(model, val_loader, device)
        print(f"[fold {fold_idx}] epoch {epoch + 1}/{args.epochs} "
              f"loss={total_loss / max(n_seen, 1):.4f} val_Dice={mean_dice:.3f} "
              f"({n_eval} فئة قابلة للتقييم)")

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(model, best_path)

    print(f"[fold {fold_idx}] أفضل Dice: {best_dice:.3f} — محفوظ في {best_path}")
    return best_dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./models")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)  # تجزئة 3D أثقل بكثير من Task 1
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-shape", type=int, nargs=3, default=(128, 128, 128))
    args = parser.parse_args()
    args.target_shape = tuple(args.target_shape)

    data_root = Path(args.data_root)
    args.images_dir = data_root / "images"
    args.vessel_masks_dir = data_root / "vessel_masks"
    args.location_masks_dir = data_root / "location_masks"
    location_jsons_dir = data_root / "location_jsons"
    location_mapping_path = data_root / "location_mapping.json"

    device = get_device()
    print(f"الجهاز: {device}")

    location_mapping = json.load(open(location_mapping_path))
    case_ids = sorted(p.stem.replace(".nii", "") for p in args.images_dir.glob("*.nii.gz"))
    print(f"عدد الحالات: {len(case_ids)}")

    Y = build_label_matrix(str(location_jsons_dir), location_mapping, case_ids, n_classes=52)
    folds = make_folds(case_ids, Y, n_splits=args.n_splits, seed=args.seed)

    model_check = UNet3D()
    print(f"معاملات النموذج: {count_parameters(model_check):,}")

    print("حساب أوزان الفئات (عينة من الحالات، لتوفير الوقت)...")
    class_weights = compute_class_weights(case_ids, args.location_masks_dir)
    print(f"أوزان الفئات: min={class_weights.min():.2f} max={class_weights.max():.2f}")

    fold_dices = []
    for i, fold in enumerate(folds):
        dice = train_one_fold(i, fold, args, device, class_weights)
        fold_dices.append(dice)

    print("\n=== ملخص K-Fold ===")
    print(f"Dice لكل فولد: {[f'{d:.3f}' for d in fold_dices]}")
    print(f"متوسط ± انحراف: {np.mean(fold_dices):.3f} ± {np.std(fold_dices):.3f}")


if __name__ == "__main__":
    main()
