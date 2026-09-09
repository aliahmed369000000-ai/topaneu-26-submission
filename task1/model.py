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
                self.norms.append(nn.BatchNorm1d(widths[i + 1]))
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


class TopAneuNet(nn.Module):
    """الشبكة الكاملة: Backbone3D_v1 + ResidualHead."""

    def __init__(self, in_channels: int = 2, feature_dim: int = 512, head_widths: list = None):
        super().__init__()
        widths = head_widths or ResidualHead.DEFAULT_WIDTHS
        assert widths[0] == feature_dim, (
            f"أول رقم في head_widths ({widths[0]}) يجب أن يطابق feature_dim ({feature_dim})"
        )
        self.backbone = Backbone3D_v1(in_channels=in_channels, out_features=feature_dim)
        self.head = ResidualHead(widths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    net = TopAneuNet()

    dummy = torch.randn(1, 2, 128, 128, 128)
    net.eval()
    with torch.no_grad():
        out = net(dummy)
    assert out.shape == (1, 52), f"شكل غير متوقع: {out.shape}"

    n_backbone = count_parameters(net.backbone)
    n_head = count_parameters(net.head)
    n_total = n_backbone + n_head

    print(f"شكل الإخراج: {tuple(out.shape)} (متوقع: (1, 52)) ✓")
    print(f"معاملات Backbone: {n_backbone:,}")
    print(f"معاملات الرأس: {n_head:,}")
    print(f"الإجمالي: {n_total:,} ({n_total * 4 / 1e6:.1f} MB, fp32)")
