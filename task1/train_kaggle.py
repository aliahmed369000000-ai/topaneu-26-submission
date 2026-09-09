"""
تدريب TopAneu Task 1 على Kaggle مع Checkpoint كامل + استئناف تلقائي.

الميزات:
- حفظ checkpoint كل epoch (model + optimizer + epoch + best_auc + fold)
- استئناف تلقائي إذا وُجد checkpoint
- حفظ أفضل نموذج لكل fold
- نسخ الأوزان النهائية إلى /kaggle/working/models
- متوافق مع مسارات Kaggle الشائعة

الاستخدام في Notebook:
    !python /kaggle/input/topaneu-code/task1/train_kaggle.py \\
        --data-root /kaggle/input/topaneu-data \\
        --output-dir /kaggle/working/models \\
        --epochs 30 --batch-size 2
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# استيراد من نفس المجلد
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import TopAneuNet, count_parameters
from dataset_split import build_label_matrix, make_folds, evaluable_classes
from vessel_preprocessing import load_and_resize_case

N_CLASSES = 52

FLIP_MAP = {
    0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 8: 9, 9: 8, 10: 11, 11: 10,
    12: 13, 13: 12, 14: 15, 15: 14, 17: 18, 18: 17, 19: 20, 20: 19,
    21: 22, 22: 21, 23: 24, 24: 23, 25: 26, 26: 25, 27: 28, 28: 27,
    29: 30, 30: 29, 31: 32, 32: 31, 33: 34, 34: 33, 36: 37, 37: 36,
    38: 39, 39: 38, 40: 41, 41: 40, 42: 43, 43: 42, 44: 45, 45: 44,
    46: 47, 47: 46, 48: 49, 49: 48, 50: 51, 51: 50,
}


def compute_pos_weight(labels: np.ndarray, min_w: float = 1.0, max_w: float = 15.0, smoothing: float = 1.0):
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
            assert flip_map is not None

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
        if vessel_path.is_file():
            volume = load_and_resize_case(str(image_path), str(vessel_path))
        else:
            # fallback: intensity only + zero vessel channel
            import numpy as np, nibabel as nib
            from scipy.ndimage import zoom
            img = nib.load(str(image_path)).get_fdata().astype(np.float32)
            zf = [t/s for t,s in zip((128,128,128), img.shape)]
            img = zoom(img, zf, order=1)
            nz = img[img>0]
            if nz.size:
                img = (img - nz.mean()) / (nz.std()+1e-8)
            volume = np.stack([img, np.zeros_like(img)], 0).astype(np.float32)
        labels = self.Y[real_idx].copy().astype(np.float32)

        volume_t = torch.from_numpy(volume)
        labels_t = torch.from_numpy(labels)

        if do_flip:
            volume_t = torch.flip(volume_t, dims=[-1])
            flipped_labels = torch.zeros_like(labels_t)
            for i in range(N_CLASSES):
                flipped_labels[self.flip_map.get(i, i)] = labels_t[i]
            labels_t = flipped_labels

        return volume_t, labels_t


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


def save_checkpoint(path, model, optimizer, epoch, best_auc, fold_idx, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_auc": best_auc,
        "fold_idx": fold_idx,
    }
    if extra:
        state.update(extra)
    torch.save(state, path)
    # نسخة احتياطية فورية
    bak = path.with_suffix(".pt.bak")
    shutil.copy2(path, bak)
    print(f"  ✓ checkpoint → {path} (epoch={epoch}, best_auc={best_auc:.4f})")


def load_checkpoint(path, model, optimizer, device):
    path = Path(path)
    if not path.is_file():
        return 0, -1.0
    print(f"  ↻ استئناف من {path}")
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("epoch", 0)) + 1, float(ckpt.get("best_auc", -1.0))


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

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    model = TopAneuNet(in_channels=2, feature_dim=512).to(device)
    pos_weight = compute_pos_weight(Y_train).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"checkpoint_fold{fold_idx}.pt"
    best_path = out_dir / f"topaneu_task1_fold{fold_idx}.pt"

    start_epoch, best_auc = load_checkpoint(ckpt_path, model, opt, device)
    if best_path.is_file() and best_auc < 0:
        # تحميل أفضل نموذج سابق إن وُجد
        try:
            best_model = torch.load(str(best_path), map_location=device, weights_only=False)
            if hasattr(best_model, "state_dict"):
                model.load_state_dict(best_model.state_dict())
            print(f"  ↻ حُمّل أفضل نموذج سابق من {best_path}")
        except Exception as e:
            print(f"  ⚠ فشل تحميل best model: {e}")

    print(f"\n===== Fold {fold_idx} | start_epoch={start_epoch} | best_auc={best_auc:.4f} =====")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            total_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)

        mean_auc, n_eval = evaluate(model, val_loader, device)
        avg_loss = total_loss / max(n_seen, 1)
        print(
            f"[fold {fold_idx}] epoch {epoch + 1}/{args.epochs} "
            f"loss={avg_loss:.4f} val_AUC={mean_auc:.4f} ({n_eval} classes)"
        )

        # حفظ checkpoint كل epoch (لا يضيع التقدم)
        save_checkpoint(ckpt_path, model, opt, epoch, best_auc, fold_idx)

        if mean_auc > best_auc:
            best_auc = mean_auc
            # حفظ النموذج الكامل (متوافق مع inference.py)
            torch.save(model, best_path)
            # تحديث best_auc في الـ checkpoint أيضًا
            save_checkpoint(ckpt_path, model, opt, epoch, best_auc, fold_idx)
            print(f"  ★ أفضل نموذج جديد → {best_path} (AUC={best_auc:.4f})")

    print(f"[fold {fold_idx}] انتهى | أفضل AUC: {best_auc:.4f}")
    return best_auc


def find_data_root(explicit: str = None) -> Path:
    """البحث التلقائي عن مجلد البيانات على Kaggle."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend([
        Path("/kaggle/input/topaneu-data"),
        Path("/kaggle/input/topaneu"),
        Path("/kaggle/input/topaneu-2026"),
        Path("/kaggle/input/topaneu26"),
        Path("./topaneu"),
        Path("/kaggle/working/topaneu"),
    ])
    for p in candidates:
        if (p / "images").is_dir() and (p / "location_jsons").is_dir():
            return p
        # أحيانًا يكون هناك مجلد فرعي واحد
        if p.is_dir():
            for sub in p.iterdir():
                if sub.is_dir() and (sub / "images").is_dir():
                    return sub
    raise FileNotFoundError(
        "لم يُعثر على بيانات TopAneu. ارفعها كـ Kaggle Dataset "
        "باسم topaneu-data (يجب أن تحتوي images/ و location_jsons/ و vessel_masks/)."
    )


def main():
    import sys as _sys
    _log_path = Path("/kaggle/working/train.log")
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data); s.flush()
                except Exception:
                    pass
        def flush(self):
            for s in self.streams:
                try: s.flush()
                except Exception: pass
    try:
        _sys.stdout = _Tee(_sys.__stdout__, open(_log_path, "a", buffering=1))
        _sys.stderr = _Tee(_sys.__stderr__, open(_log_path, "a", buffering=1))
        print("TRAIN_LOG", _log_path)
    except Exception as _e:
        print("tee failed", _e)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/models")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-fold", type=int, default=0, help="ابدأ من fold معين (للاستئناف)")
    args = parser.parse_args()

    data_root = find_data_root(args.data_root)
    args.images_dir = data_root / "images"
    args.vessel_masks_dir = data_root / "vessel_masks"
    location_jsons_dir = data_root / "location_jsons"

    # location_mapping
    mapping_candidates = [
        data_root / "location_mapping.json",
        Path(__file__).parent.parent / "docs" / "location_mapping.json",
        Path("/kaggle/input/topaneu-code/docs/location_mapping.json"),
    ]
    location_mapping_path = None
    for mc in mapping_candidates:
        if mc.is_file():
            location_mapping_path = mc
            break
    if location_mapping_path is None:
        raise FileNotFoundError("location_mapping.json غير موجود")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"الجهاز: {device}")
    print(f"البيانات: {data_root}")
    print(f"الإخراج: {args.output_dir}")

    location_mapping = json.load(open(location_mapping_path))
    def case_id_from_image(path: Path) -> str:
        # images: topaneu_center1_mr_001_0000.nii.gz -> topaneu_center1_mr_001
        # jsons/masks: topaneu_center1_mr_001.json (no _0000)
        name = path.name
        if name.endswith(".nii.gz"):
            name = name[:-7]
        elif name.endswith(".nii"):
            name = name[:-4]
        if name.endswith("_0000"):
            name = name[:-5]
        return name

    case_ids = sorted({case_id_from_image(p) for p in args.images_dir.glob("*.nii.gz")})
    print(f"عدد الحالات: {len(case_ids)}")
    if len(case_ids) == 0:
        raise RuntimeError("لا توجد صور .nii.gz في images/")

    # verify image path resolution uses _0000 suffix
    sample = case_ids[0]
    img_candidates = [
        args.images_dir / f"{sample}_0000.nii.gz",
        args.images_dir / f"{sample}.nii.gz",
    ]
    print(f"مثال case_id={sample}")
    for c in img_candidates:
        print(f"  exists {c.name}: {c.is_file()}")

    Y = build_label_matrix(str(location_jsons_dir), location_mapping, case_ids, n_classes=N_CLASSES)
    folds = make_folds(case_ids, Y, n_splits=args.n_splits, seed=args.seed)

    model_check = TopAneuNet(in_channels=2, feature_dim=512)
    print(f"معاملات النموذج: {count_parameters(model_check):,}")

    fold_aucs = []
    for i, fold in enumerate(folds):
        if i < args.start_fold:
            print(f"تخطي fold {i} (--start-fold={args.start_fold})")
            # حاول قراءة AUC السابق
            best_path = Path(args.output_dir) / f"topaneu_task1_fold{i}.pt"
            if best_path.is_file():
                fold_aucs.append(float("nan"))  # غير معروف بدون إعادة تقييم
            continue
        auc = train_one_fold(i, fold, Y, case_ids, args, device)
        fold_aucs.append(auc)

    print("\n=== ملخص K-Fold ===")
    valid = [a for a in fold_aucs if not (isinstance(a, float) and np.isnan(a))]
    print(f"AUC لكل فولد: {[f'{a:.4f}' if not (isinstance(a, float) and np.isnan(a)) else 'N/A' for a in fold_aucs]}")
    if valid:
        print(f"متوسط ± انحراف: {np.mean(valid):.4f} ± {np.std(valid):.4f}")

    # نسخ أفضل النماذج إلى مكان واضح
    out = Path(args.output_dir)
    print(f"\nالأوزان المحفوظة في: {out}")
    for p in sorted(out.glob("topaneu_task1_fold*.pt")):
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.name} ({size_mb:.1f} MB)")

    # ملف جاهزية
    ready = {
        "folds_trained": len(valid),
        "mean_auc": float(np.mean(valid)) if valid else None,
        "model_files": [p.name for p in sorted(out.glob("topaneu_task1_fold*.pt"))],
    }
    with open(out / "training_summary.json", "w") as f:
        json.dump(ready, f, indent=2)
    print("تم حفظ training_summary.json")
    print("\n✅ انتهى التدريب. نزّل مجلد models/ أو ادفعه كـ Kaggle Dataset.")


if __name__ == "__main__":
    main()
