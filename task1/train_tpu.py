"""
تدريب TopAneu Task1 على Kaggle TPU عبر PyTorch XLA.
يستأنف من checkpoint_fold*.pt إن وُجد.
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

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    HAS_XLA = True
except Exception as e:
    HAS_XLA = False
    print("torch_xla not available:", e)

from dataset_split import build_label_matrix, make_folds
from model import TopAneuNet, count_parameters
from vessel_preprocessing import load_and_resize_case, FLIP_MAP


class TopAneuDataset(Dataset):
    def __init__(self, case_ids, Y, images_dir, vessel_masks_dir, location_jsons_dir, augment_flip=False):
        self.case_ids = list(case_ids)
        self.Y = Y
        self.images_dir = Path(images_dir)
        self.vessel_masks_dir = Path(vessel_masks_dir)
        self.location_jsons_dir = Path(location_jsons_dir)
        self.augment_flip = augment_flip
        self.n = len(self.case_ids) * (2 if augment_flip else 1)

    def __len__(self):
        return self.n

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
            import nibabel as nib
            from scipy.ndimage import zoom
            img = nib.load(str(image_path)).get_fdata().astype(np.float32)
            zf = [tgt / src for tgt, src in zip((128, 128, 128), img.shape)]
            img = zoom(img, zf, order=1)
            nz = img[img > 0]
            if nz.size:
                img = (img - nz.mean()) / (nz.std() + 1e-8)
            volume = np.stack([img, np.zeros_like(img)], 0).astype(np.float32)

        labels = self.Y[real_idx].copy().astype(np.float32)
        if do_flip:
            volume = np.ascontiguousarray(volume[:, :, :, ::-1])
            flipped = labels.copy()
            # FLIP_MAP من vessel_preprocessing.py هي 0-based بالفعل (تطابق
            # ترقيم مصفوفة labels مباشرة) -- لا حاجة لأي طرح/إضافة. الطرح
            # السابق (src-1, dst-1) كان خطأً يُنتج إزاحة خاطئة صامتة لو
            # طُبّق على خريطة 0-based (كما هي الحال هنا بعد توحيد المصدر).
            for src, dst in FLIP_MAP.items():
                flipped[dst] = labels[src]
            labels = flipped

        return torch.from_numpy(volume), torch.from_numpy(labels)


def compute_pos_weight(Y_train: np.ndarray, min_w: float = 1.0, max_w: float = 15.0,
                        smoothing: float = 1.0) -> torch.Tensor:
    """موحّدة الآن مع train.py بالضبط (نفس صيغة Laplace Smoothing + نفس
    الحد الأقصى 15.0 -- كانت 50.0 بدون Smoothing سابقًا، ديناميكية تدريب
    مختلفة عن باقي التجارب دون مبرر)."""
    pos = Y_train.sum(axis=0).astype(np.float32)
    neg = Y_train.shape[0] - pos
    w = (neg + smoothing) / (pos + smoothing)
    w = np.clip(w, min_w, max_w)
    return torch.from_numpy(w)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        if HAS_XLA:
            xm.mark_step()
        all_logits.append(logits.float().cpu())
        all_y.append(yb)
    logits = torch.cat(all_logits, 0).numpy()
    y = torch.cat(all_y, 0).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    aucs = []
    for c in range(y.shape[1]):
        if y[:, c].sum() == 0 or y[:, c].sum() == len(y):
            continue
        try:
            from sklearn.metrics import roc_auc_score
            aucs.append(roc_auc_score(y[:, c], probs[:, c]))
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else 0.0, len(aucs)


def save_ckpt(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    if path.exists():
        shutil.copy(path, str(path) + ".bak")
    tmp.replace(path)


def train_one_fold(fold_idx, fold, Y, case_ids, args, device):
    id_to_i = {cid: i for i, cid in enumerate(case_ids)}
    train_ids, val_ids = fold["train"], fold["val"]
    Y_train = Y[np.array([id_to_i[c] for c in train_ids])]
    Y_val = Y[np.array([id_to_i[c] for c in val_ids])]

    train_ds = TopAneuDataset(
        train_ids, Y_train, args.images_dir, args.vessel_masks_dir, args.location_jsons_dir, augment_flip=True
    )
    val_ds = TopAneuDataset(
        val_ids, Y_val, args.images_dir, args.vessel_masks_dir, args.location_jsons_dir, augment_flip=False
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = TopAneuNet().to(device)
    pos_weight = compute_pos_weight(Y_train).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    ckpt_path = Path(args.output_dir) / f"checkpoint_fold{fold_idx}.pt"
    best_path = Path(args.output_dir) / f"topaneu_task1_fold{fold_idx}.pt"
    start_epoch, best_auc = 0, -1.0

    if ckpt_path.is_file():
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            best_auc = float(ckpt.get("best_auc", -1.0))
            print(f"  resume fold {fold_idx} from epoch {start_epoch}, best_auc={best_auc:.4f}")
        except Exception as e:
            print(f"  checkpoint load failed: {e}")

    print(f"===== Fold {fold_idx} | start_epoch={start_epoch} | best_auc={best_auc:.4f} | device={device} =====")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        # ParallelLoader improves TPU throughput
        if HAS_XLA:
            loader_epoch = pl.MpDeviceLoader(train_loader, device)
        else:
            loader_epoch = train_loader

        for xb, yb in loader_epoch:
            if not HAS_XLA:
                xb = xb.to(device)
                yb = yb.to(device)
            else:
                yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if HAS_XLA:
                xm.optimizer_step(optimizer)
                xm.mark_step()
            else:
                optimizer.step()
            total_loss += float(loss.detach().cpu())
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if HAS_XLA:
            val_loader_x = pl.MpDeviceLoader(val_loader, device)
            auc, n_cls = evaluate(model, val_loader_x, device)
        else:
            auc, n_cls = evaluate(model, val_loader, device)

        print(f"[fold {fold_idx}] epoch {epoch+1}/{args.epochs} loss={avg_loss:.4f} val_AUC={auc:.4f} ({n_cls} classes)")

        payload = {
            "epoch": epoch,
            "best_auc": best_auc,
            "model": {k: v.cpu() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
        }
        if auc > best_auc:
            best_auc = auc
            payload["best_auc"] = best_auc
            # مهم: نحفظ كائن النموذج الكامل (torch.save(model, path))، تمامًا
            # كما يفعل train.py -- لا قاموس state_dict. inference.py و
            # calibrate_threshold.py يحمّلان عبر torch.load() ثم يستدعيان
            # .eval() مباشرة على الناتج؛ قاموس عادي ليس له .eval() فيفشل
            # الاستدلال فورًا لو استُخدمت أوزان مُدرّبة بالصيغة القديمة.
            model_cpu = TopAneuNet()
            model_cpu.load_state_dict(payload["model"])
            save_ckpt(best_path, model_cpu)
            print(f"  ★ best → {best_path} (AUC={best_auc:.4f})")
        save_ckpt(ckpt_path, payload)
        print(f"  ✓ checkpoint → {ckpt_path} (epoch={epoch})")

    return best_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/models")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--start-fold", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    root = Path(args.data_root)
    args.images_dir = root / "images"
    args.vessel_masks_dir = root / "vessel_masks"
    args.location_jsons_dir = root / "location_jsons"
    mapping_path = root / "location_mapping.json"
    if not mapping_path.is_file():
        mapping_path = Path("/kaggle/working/topaneu-code/docs/location_mapping.json")
    location_mapping = json.load(open(mapping_path))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if HAS_XLA:
        device = xm.xla_device()
        print("XLA device:", device)
    else:
        device = torch.device("cpu")
        print("Fallback CPU (no XLA)")

    def case_id_from_image(path: Path) -> str:
        name = path.name
        if name.endswith(".nii.gz"):
            name = name[:-7]
        elif name.endswith(".nii"):
            name = name[:-4]
        if name.endswith("_0000"):
            name = name[:-5]
        return name

    case_ids = sorted({case_id_from_image(p) for p in args.images_dir.glob("*.nii.gz")})
    print(f"cases={len(case_ids)}")
    Y = build_label_matrix(str(args.location_jsons_dir), location_mapping, case_ids)
    folds = make_folds(case_ids, Y, n_splits=args.n_splits)
    print(f"params={count_parameters(TopAneuNet()):,}")

    results = []
    for i in range(args.start_fold, len(folds)):
        auc = train_one_fold(i, folds[i], Y, case_ids, args, device)
        results.append(auc)
        print(f"Fold {i} done best_auc={auc:.4f}")
    if results:
        print(f"Mean AUC={float(np.mean(results)):.4f}")


if __name__ == "__main__":
    main()
