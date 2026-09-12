"""
معمارية Task 1 الكاملة — Backbone v1 (Conv3D) + رأس التصنيف المخصص
(13 طبقة + Skip Connections)، كما تم بناؤها والتحقق منها تجريبيًا في
جلسة التصميم (راجع docs/CHALLENGE_REFERENCE.md لكل التفاصيل والقرارات).

قرارات مهمة مأخوذة من التجارب التشخيصية السابقة (لا تُغيَّر بدون سبب موثّق):
  - Skip Connections إلزامية: بدونها الرأس (13 طبقة) فشل تمامًا (AUC~0.50)
    حتى مع 500 عينة اصطناعية — إثبات أنها مشكلة تدرج، لا نقص بيانات.
  - feature_dim=512 (بدل 2048 الأصلي): تقليل معاملات الرأس ~93.6%
    (وصلات التخطي هي أكبر مستهلك للمعاملات؛ تصغير feature_dim يقلصها
    كلها دفعة واحدة، بخلاف تقليل عدد الطبقات فقط).
  - Backbone يقبل قناتين دخل (Intensity + Vessel Mask Binary).
"""

import os

import torch
import torch.nn as nn


class Backbone3D_v1(nn.Module):
    """طبقة Conv واحدة لكل مرحلة، 5 مراحل، بدون أوزان مُدرّبة مسبقًا
    (تم استبعاد Transfer Learning عمدًا — تدريب من الصفر، قرار المستخدم)."""

    def __init__(self, in_channels: int = 2, out_features: int = 512):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(), nn.MaxPool3d(2)
        )
        self.stage2 = nn.Sequential(
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d(2)
        )
        self.stage3 = nn.Sequential(
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(), nn.MaxPool3d(2)
        )
        self.stage4 = nn.Sequential(
            nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(), nn.MaxPool3d(2)
        )
        self.stage5 = nn.Sequential(
            nn.Conv3d(128, out_features, 3, padding=1), nn.BatchNorm3d(out_features), nn.ReLU(), nn.MaxPool3d(2)
        )
        self.gap = nn.AdaptiveAvgPool3d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return self.gap(x).flatten(1)  # (batch, out_features)


class ResidualHead(nn.Module):
    """رأس التصنيف: تسلسل المستخدم بالضبط (مُصغّر نسبيًا لـfeature_dim=512
    بدل 2048 الأصلي، بنفس الشكل النسبي المتذبذب)، مع Skip Connection من
    متجه الميزات الأصلي لكل طبقة عبر Linear Projection بدون Bias.

    التسلسل الأصلي (feature_dim=2048) الذي كتبه المستخدم، للمرجعية:
      2048,1024,716,630,430,590,656,235,392,327,364,304,117,52
    التسلسل المُستخدم فعليًا هنا (مُصغّر بنسبة 512/2048 = 0.25،
    مع تقريب لأقرب عدد صحيح، وإبقاء آخر رقم=52 كما هو لأنه عدد الفئات
    الثابت):
      512,256,179,158,108,148,164,59,98,82,91,76,29,52
    """

    DEFAULT_WIDTHS = [512, 256, 179, 158, 108, 148, 164, 59, 98, 82, 91, 76, 29, 52]

    def __init__(self, widths: list = None, dropout: float = 0.15):
        super().__init__()
        widths = widths or self.DEFAULT_WIDTHS
        self.widths = widths
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skip_proj = nn.ModuleList()
        for i in range(len(widths) - 1):
            self.layers.append(nn.Linear(widths[i], widths[i + 1]))
            if i < len(widths) - 2:  # لا BatchNorm/ReLU/Dropout بعد الطبقة الأخيرة (logits خام)
                self.norms.append(nn.LayerNorm(widths[i + 1]))
            self.skip_proj.append(nn.Linear(widths[0], widths[i + 1], bias=False))
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for i, layer in enumerate(self.layers):
            out = layer(x) + self.skip_proj[i](x0)
            if i < len(self.layers) - 1:
                out = self.drop(self.act(self.norms[i](out)))
            x = out
        return x  # logits خام (52,) — استخدم BCEWithLogitsLoss أو torch.sigmoid عند الاستدلال


# خريطة المجموعات التشريحية الخمس (مأخوذة مباشرة من ترقيم location_mapping.json
# الرسمي: الرقم قبل النقطة في كل تسمية، مثل "1.4 BA trunk" أو "R-3.7 ICA
# C7-terminus" -- 5 مناطق وعائية كبرى: 1=فقري قاعدي (17 فئة)، 2=مخيّة خلفية
# (4 فئات)، 3=سباتي داخلي (14 فئة)، 4=مخيّة أمامية+Acom (9 فئات)، 5=مخيّة
# وسطى (8 فئات). مفتاح تجميعي رسمي، لا تخمين.
GROUP_MAP = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0,
    11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0,
    17: 1, 18: 1, 19: 1, 20: 1,
    21: 2, 22: 2, 23: 2, 24: 2, 25: 2, 26: 2, 27: 2, 28: 2, 29: 2, 30: 2,
    31: 2, 32: 2, 33: 2, 34: 2,
    35: 3, 36: 3, 37: 3, 38: 3, 39: 3, 40: 3, 41: 3, 42: 3, 43: 3,
    44: 4, 45: 4, 46: 4, 47: 4, 48: 4, 49: 4, 50: 4, 51: 4,
}
N_GROUPS = 5


class TopAneuNet(nn.Module):
    """الشبكة الكاملة: Backbone3D_v1 + ResidualHead + رأس مساعد اختياري
    (Auxiliary Group Head) يتنبأ بالمنطقة الوعائية الكبرى (5 مجموعات بدل
    52 فئة مفردة). الهدف: إشارة تدريب أكثف بكثير لكل فئة نادرة (تُجمع
    حالاتها القليلة مع فئات مجاورة تشريحيًا في نفس المجموعة)، تُحسّن
    تمثيل الـBackbone المشترك عبر Multi-task Learning -- دون تغيير شكل
    الإخراج الرسمي المطلوب (52، عبر self.head فقط). الرأس المساعد
    يُستخدم فقط أثناء التدريب (خسارة إضافية)؛ يمكن تجاهله بالكامل وقت
    الاستدلال الفعلي (raw_logits فقط تُستخدم في inference.py)."""

    def __init__(self, in_channels: int = 2, feature_dim: int = 512, head_widths: list = None,
                 use_aux_group_head: bool = True):
        super().__init__()
        widths = head_widths or ResidualHead.DEFAULT_WIDTHS
        assert widths[0] == feature_dim, (
            f"أول رقم في head_widths ({widths[0]}) يجب أن يطابق feature_dim ({feature_dim})"
        )
        self.backbone = Backbone3D_v1(in_channels=in_channels, out_features=feature_dim)
        self.head = ResidualHead(widths)
        self.use_aux_group_head = use_aux_group_head
        if use_aux_group_head:
            self.aux_group_head = nn.Linear(feature_dim, N_GROUPS)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        feat = self.backbone(x)
        main_out = self.head(feat)
        if return_aux and self.use_aux_group_head:
            aux_out = self.aux_group_head(feat)
            return main_out, aux_out
        return main_out

    @staticmethod
    def labels_to_group_labels(y: torch.Tensor) -> torch.Tensor:
        """يحوّل مصفوفة تسميات (batch, 52) إلى (batch, 5) على مستوى
        المجموعة: المجموعة إيجابية إذا كانت أي فئة ضمنها إيجابية."""
        batch = y.shape[0]
        group_y = torch.zeros(batch, N_GROUPS, dtype=y.dtype, device=y.device)
        for c, g in GROUP_MAP.items():
            group_y[:, g] = torch.maximum(group_y[:, g], y[:, c])
        return group_y


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_device() -> torch.device:
    """اختيار الجهاز بأمان (مصدر وحيد، يستورده train.py/inference.py/
    calibrate_threshold.py/train_kaggle.py -- كان مكررًا بصيغتين مختلفتين:
    train_kaggle.py وحده كان يفحص فعليًا (يحاول تخصيص تنسور على cuda)،
    بينما البقية تكتفي بـtorch.cuda.is_available() الذي *لا* يكتشف أعطالًا
    مثل P100 (sm_60) مع إصدارات PyTorch الحديثة التي أسقطت دعم هذه
    المعمارية -- is_available() يتحقق فقط من وجود السائق (driver)، لا من
    قابلية تشغيل العمليات فعليًا على الجهاز؛ الفشل الحقيقي يحدث لاحقًا
    عند أول عملية حقيقية على الـGPU، بشكل مفاجئ ومربك.

    يدعم أيضًا متغير البيئة TOPANEU_FORCE_CPU=1 لتعطيل GPU يدويًا (مفيد
    لتفادي عطل معروف بدون انتظار اكتشافه في كل مرة)."""
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


if __name__ == "__main__":
    net = TopAneuNet()

    dummy = torch.randn(1, 2, 128, 128, 128)
    net.eval()
    with torch.no_grad():
        out = net(dummy)
        out_with_aux, aux_out = net(dummy, return_aux=True)
    assert out.shape == (1, 52), f"شكل غير متوقع: {out.shape}"
    assert aux_out.shape == (1, N_GROUPS), f"شكل الرأس المساعد غير متوقع: {aux_out.shape}"
    assert torch.allclose(out, out_with_aux), "الإخراج الرئيسي يجب أن يبقى مطابقًا بوجود/غياب return_aux"

    n_backbone = count_parameters(net.backbone)
    n_head = count_parameters(net.head)
    n_aux = count_parameters(net.aux_group_head) if net.use_aux_group_head else 0
    n_total = n_backbone + n_head + n_aux

    print(f"شكل الإخراج الرئيسي: {tuple(out.shape)} (متوقع: (1, 52)) ✓")
    print(f"شكل الرأس المساعد: {tuple(aux_out.shape)} (متوقع: (1, {N_GROUPS})) ✓")
    print(f"معاملات Backbone: {n_backbone:,}")
    print(f"معاملات الرأس: {n_head:,}")
    print(f"معاملات الرأس المساعد: {n_aux:,}")
    print(f"الإجمالي: {n_total:,} ({n_total * 4 / 1e6:.1f} MB, fp32)")
