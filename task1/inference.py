"""
inference.py متوافق 100% مع القالب الرسمي لـ TopAneu-26
(https://github.com/Bangulli/TopAneu-26/templates/task1)

main.py الرسمي يستدعي:
  - infer_ct(img: sitk.Image) -> list[int]
  - infer_mr(img: sitk.Image) -> list[int]

الإخراج: قائمة أعداد صحيحة (1-based) مثل [34, 35]
تُكتب مباشرة كـ JSON array في detected-aneurysm-locations.json

ملاحظات مهمة:
- لا يوجد vessel mask في وقت الاستدلال (فقط الصورة).
- النموذج الحالي مدرب على قناتين (Intensity + Vessel). نستخدم قناة وعاء صفرية
  كحل عملي مؤقت. لأداء أفضل: أعد التدريب بقناة واحدة أو ولّد mask داخليًا.
- يدعم Ensemble من عدة folds + Test-Time Flip إذا وُجدت الأوزان.
"""

from pathlib import Path
import numpy as np
import torch
import SimpleITK as sitk
from scipy.ndimage import zoom

# ---------------------------------------------------------------------------
# إعدادات قابلة للتعديل
# ---------------------------------------------------------------------------
# في GC: الأوزان تُستخرج إلى /opt/ml/model
# في الاختبار المحلي: المجلد ./model يُركب هناك
MODEL_DIR = Path("/opt/ml/model")
# أسماء ملفات الأوزان المتوقعة (من train.py: topaneu_task1_fold{i}.pt)
FOLD_WEIGHTS = [
    "topaneu_task1_fold0.pt",
    "topaneu_task1_fold1.pt",
    "topaneu_task1_fold2.pt",
    "topaneu_task1_fold3.pt",
    "topaneu_task1_fold4.pt",
]
# fallback إذا وُجد نموذج واحد نهائي
FINAL_WEIGHT = "topaneu_task1_final.pt"

# threshold عام (يُفضّل معايرته لاحقًا على Validation لتعظيم MCC)
DECISION_THRESHOLD = 0.45

# هل نستخدم Test-Time Flip (يتطلب flip_map)
USE_TTA_FLIP = True

TARGET_SHAPE = (128, 128, 128)

# flip_map الرسمي (indices 0-based)
FLIP_MAP = {
    0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4,
    8: 9, 9: 8, 10: 11, 11: 10, 12: 13, 13: 12,
    14: 15, 15: 14, 17: 18, 18: 17, 19: 20, 20: 19,
    21: 22, 22: 21, 23: 24, 24: 23, 25: 26, 26: 25,
    27: 28, 28: 27, 29: 30, 30: 29, 31: 32, 32: 31,
    33: 34, 34: 33, 36: 37, 37: 36, 38: 39, 39: 38,
    40: 41, 41: 40, 42: 43, 43: 42, 44: 45, 45: 44,
    46: 47, 47: 46, 48: 49, 49: 48, 50: 51, 51: 50,
}

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODELS = None  # list of models (ensemble)


def _load_models_once():
    """تحميل كل النماذج المتوفرة مرة واحدة (Ensemble)."""
    global _MODELS
    if _MODELS is not None:
        return _MODELS

    models = []
    # جرّب الـ folds أولاً
    for name in FOLD_WEIGHTS:
        path = MODEL_DIR / name
        if path.is_file():
            m = torch.load(str(path), map_location=_DEVICE, weights_only=False)
            m.eval()
            models.append(m)

    # fallback: نموذج نهائي واحد
    if not models:
        path = MODEL_DIR / FINAL_WEIGHT
        if path.is_file():
            m = torch.load(str(path), map_location=_DEVICE, weights_only=False)
            m.eval()
            models.append(m)

    if not models:
        # محاولة إضافية من المسار النسبي (للتطوير المحلي)
        local_dir = Path("./models")
        for name in FOLD_WEIGHTS + [FINAL_WEIGHT]:
            path = local_dir / name
            if path.is_file():
                m = torch.load(str(path), map_location=_DEVICE, weights_only=False)
                m.eval()
                models.append(m)

    if not models:
        raise FileNotFoundError(
            f"لم يُعثر على أي أوزان في {MODEL_DIR} أو ./models. "
            "شغّل train.py أولاً وضع الملفات .pt في مجلد models/"
        )

    _MODELS = models
    return _MODELS


def _sitk_to_numpy(img: sitk.Image) -> np.ndarray:
    """تحويل SimpleITK Image إلى numpy (Z,Y,X) ثم نتعامل معه كـ (D,H,W)."""
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    return arr


def _preprocess(intensity: np.ndarray, vessel: np.ndarray = None) -> np.ndarray:
    """
    Resize إلى 128³ + z-score على intensity + قناة vessel (binary أو أصفار).
    الإخراج: (2, 128, 128, 128)
    """
    zoom_factors = [t / s for t, s in zip(TARGET_SHAPE, intensity.shape)]
    image_resized = zoom(intensity, zoom_factors, order=1)

    if vessel is not None:
        vessel_resized = zoom(vessel, zoom_factors, order=0)
        vessel_binary = (vessel_resized > 0).astype(np.float32)
    else:
        # لا يوجد vessel mask في الاستدلال → قناة صفرية
        vessel_binary = np.zeros(TARGET_SHAPE, dtype=np.float32)

    nonzero = image_resized[image_resized > 0]
    if nonzero.size > 0:
        mean = nonzero.mean()
        std = nonzero.std() + 1e-8
        image_resized = (image_resized - mean) / std

    return np.stack([image_resized, vessel_binary], axis=0).astype(np.float32)


def _predict_probs(volume: np.ndarray) -> np.ndarray:
    """
    volume: (2, 128, 128, 128)
    يرجع متوسط الاحتمالات عبر الـ Ensemble: (52,)
    """
    models = _load_models_once()
    tensor = torch.from_numpy(volume).unsqueeze(0).to(_DEVICE)  # (1,2,128,128,128)

    all_probs = []
    with torch.no_grad():
        for model in models:
            logits = model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy().squeeze()  # (52,)
            all_probs.append(probs)

            if USE_TTA_FLIP:
                # Flip على المحور الأيسر-أيمن (آخر بُعد بعد الـ stack هو axis الأصلي 0)
                flipped = torch.flip(tensor, dims=[-1])
                logits_f = model(flipped)
                probs_f = torch.sigmoid(logits_f).cpu().numpy().squeeze()
                # أعد ترتيب الاحتمالات حسب flip_map
                probs_mapped = np.zeros_like(probs_f)
                for i in range(52):
                    probs_mapped[FLIP_MAP.get(i, i)] = probs_f[i]
                all_probs.append(probs_mapped)

    return np.mean(all_probs, axis=0)


def _probs_to_locations(probs: np.ndarray, threshold: float = DECISION_THRESHOLD) -> list:
    """تحويل احتمالات (52,) إلى قائمة قيم 1-based."""
    return [int(idx + 1) for idx, p in enumerate(probs) if p >= threshold]


def infer_ct(img: sitk.Image) -> list:
    """Predicts aneurysm locations in CTA images. Official signature."""
    intensity = _sitk_to_numpy(img)
    volume = _preprocess(intensity, vessel=None)
    probs = _predict_probs(volume)
    return _probs_to_locations(probs)


def infer_mr(img: sitk.Image) -> list:
    """Predicts aneurysm locations in MRA images. Official signature."""
    # نفس المسار (النموذج modality-agnostic حاليًا)
    return infer_ct(img)
