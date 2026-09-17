"""
معالجة حالة واحدة لـTask 2: تحميل الصورة + قناع الأوعية (نفس مدخل
Task 1 بالضبط) + قناع المواقع (location_mask) كهدف تجزئة.

فرق جوهري عن location_jsons في Task 1: هنا الهدف قناع كثيف (Dense) بنفس
أبعاد الصورة، لا قائمة أرقام. كل فوكسل قيمته = فئة موقعه (0=خلفية،
1..52=مواقع، نفس جدول location_mapping.json). يُعامَل معاملة تصنيفية
بالكامل (Nearest-Neighbor حصرًا عند Resize، لا يجوز أي استيفاء آخر).

الإخراج:
  volume: (2, *target_shape) — نفس صيغة Task 1 (intensity + vessel binary)
  target: (*target_shape,) — قيم صحيحة 0..52 (segmentation ground truth)
"""

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

# نسخة FLIP_MAP الخاصة بـTask 2 (مطابقة حرفيًا لـtask1/vessel_preprocessing.py).
# لا استيراد متقاطع من task1/ عمدًا: Task 1 وTask 2 حاويتا Docker منفصلتان
# تمامًا وقت النشر (لن يتشاركا نفس عملية بايثون أبدًا)، ومحاولة استيراد
# متقاطع محليًا عبر sys.path.insert تُسبب تعارض أسماء حقيقي (task1/model.py
# يُخفي task2/model.py عند استيراد "import model" لاحقًا في نفس السكريبت
# -- اكتُشف هذا فعليًا أثناء الاختبار). إن غيّرت هذه القيم في task1، حدّثها
# هنا يدويًا أيضًا.
FLIP_MAP = {
    0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 8: 9, 9: 8, 10: 11, 11: 10,
    12: 13, 13: 12, 14: 15, 15: 14, 17: 18, 18: 17, 19: 20, 20: 19,
    21: 22, 22: 21, 23: 24, 24: 23, 25: 26, 26: 25, 27: 28, 28: 27,
    29: 30, 30: 29, 31: 32, 32: 31, 33: 34, 34: 33, 36: 37, 37: 36,
    38: 39, 39: 38, 40: 41, 41: 40, 42: 43, 43: 42, 44: 45, 45: 44,
    46: 47, 47: 46, 48: 49, 49: 48, 50: 51, 51: 50,
}


def load_and_resize_case_seg(
    image_path: str,
    vessel_mask_path: str,
    location_mask_path: str,
    target_shape: tuple = (128, 128, 128),
):
    img_nii = nib.load(image_path)
    image = img_nii.get_fdata().astype(np.float32)

    vessel_nii = nib.load(vessel_mask_path)
    vessel = vessel_nii.get_fdata()

    loc_nii = nib.load(location_mask_path)
    loc = loc_nii.get_fdata()

    zoom_factors = [t / s for t, s in zip(target_shape, image.shape)]

    image_resized = zoom(image, zoom_factors, order=1)          # مستمرة: خطي
    vessel_resized = zoom(vessel, zoom_factors, order=0)         # تصنيفية: NN
    loc_resized = zoom(loc, zoom_factors, order=0)                # تصنيفية: NN حصرًا

    vessel_binary = (vessel_resized > 0).astype(np.float32)

    nonzero = image_resized[image_resized > 0]
    if nonzero.size > 0:
        mean, std = nonzero.mean(), nonzero.std() + 1e-8
        image_resized = (image_resized - mean) / std

    volume = np.stack([image_resized, vessel_binary], axis=0).astype(np.float32)
    target = np.round(loc_resized).astype(np.int64)
    target = np.clip(target, 0, 52)  # احتياط: أي قيمة خارج المدى المتوقع تُقصّ للخلفية/أقصى فئة

    assert volume.shape == (2, *target_shape), f"شكل غير متوقع (volume): {volume.shape}"
    assert target.shape == tuple(target_shape), f"شكل غير متوقع (target): {target.shape}"
    return volume, target


def flip_location_mask(target: np.ndarray) -> np.ndarray:
    """قلب قناع المواقع أفقيًا (محور W) + إعادة تسمية القيم عبر FLIP_MAP.
    قيم target هنا 1-based (0=خلفية)، بينما FLIP_MAP 0-based -- التحويل:
    value -> (value-1) -> FLIP_MAP.get(., .) -> +1، مع إبقاء 0 كما هو."""
    flipped_spatial = np.ascontiguousarray(target[:, :, ::-1])
    remapped = flipped_spatial.copy()
    for c0, mapped0 in FLIP_MAP.items():
        remapped[flipped_spatial == c0 + 1] = mapped0 + 1
    return remapped


if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) != 4:
        print("Usage: python vessel_preprocessing.py <image.nii.gz> <vessel_mask.nii.gz> <location_mask.nii.gz>")
        _sys.exit(1)

    vol, tgt = load_and_resize_case_seg(_sys.argv[1], _sys.argv[2], _sys.argv[3])
    print("volume shape:", vol.shape)
    print("target shape:", tgt.shape, "unique classes present:", np.unique(tgt))
