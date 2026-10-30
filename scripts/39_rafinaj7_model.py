"""Çok tipli rafinaj ağı — 7 landmark için tek model.

TİP PAYLAŞIMI
-------------
7 nokta var ama 4 GÖRSEL TİP:

    tip 0  koltuk altı   <- nokta 0, 1   (ayna simetrik, flip ile paylaşılır)
    tip 1  c7            <- nokta 2
    tip 2  omurga        <- nokta 3, 4, 5 (üçü de oluk üzerinde, aynı doku)
    tip 3  sakrum        <- nokta 6

Paylaşım veri verimliliği için: omurga_1/2/3 aynı görsel özelliği arıyor
(oluk), ayrı kanal ayırmak 3 kat az veriyle 3 kat fazla parametre demek.
Koltuk altı da sol/sağ ayna olduğu için tek tip + flip augmentasyonu.

ÇIKTI BİÇİMİ
------------
forward() TÜM tipler için koordinat döndürür: (B, 4, 2). Çağıran taraf
hangi tipi istediğini seçer. Grafik statik kaldığı için TFLite'a temiz
çevrilir — tip'i girdi yapsaydık dinamik dallanma olurdu.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# nokta indeksi -> gorsel tip
NOKTA_TIPI = [0, 0, 1, 2, 2, 2, 3]
TIP_ADI = ["koltuk_alti", "c7", "omurga", "sakrum"]
TIP_SAYISI = len(TIP_ADI)


class Rafinaj7(nn.Module):
    """MobileNetV3-small gövde + tip başına heatmap kanalı + soft-argmax.

    Girdi : (B, 3, 128, 128)
    Çıktı : koordinat (B, TIP_SAYISI, 2) kırpım pikselinde
            heatmap   (B, TIP_SAYISI, 64, 64)
    """

    def __init__(self, heatmap_boyut=64, pretrained=True):
        super().__init__()
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

        agirlik = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        govde = mobilenet_v3_small(weights=agirlik).features
        self.govde = nn.Sequential(*list(govde.children())[:9])   # stride 16, 48 kanal

        # 8x8 -> 16 -> 32 -> 64   (kernel 4, stride 2, padding 1 = tam 2x, hizalama kaymaz)
        self.head = nn.Sequential(
            nn.ConvTranspose2d(48, 96, 4, 2, 1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, TIP_SAYISI, 1),          # tip basina bir heatmap
        )
        self.hb = heatmap_boyut
        lin = torch.linspace(0, 1, heatmap_boyut)
        self.register_buffer("izgara_y", lin.view(-1, 1).expand(heatmap_boyut, heatmap_boyut).contiguous())
        self.register_buffer("izgara_x", lin.view(1, -1).expand(heatmap_boyut, heatmap_boyut).contiguous())

    def forward(self, x):
        girdi = x.shape[-1]
        h = self.head(self.govde(x))                       # (B, T, 64, 64)
        B, T = h.shape[0], h.shape[1]
        duz = h.reshape(B * T, -1)
        # sicaklik 10: tepeyi keskinlestirir, alt-piksel cozunurlugu artirir
        ol = F.softmax(duz * 10.0, dim=1).reshape(B, T, self.hb, self.hb)
        x_n = (ol * self.izgara_x).sum(dim=(2, 3))
        y_n = (ol * self.izgara_y).sum(dim=(2, 3))
        koord = torch.stack([x_n, y_n], dim=2) * (girdi - 1)   # (B, T, 2)
        return koord, h


def gauss_hedef(koord, boyut, girdi_boyut, sigma=1.5):
    """(B,2) koordinattan (B,1,boyut,boyut) Gauss heatmap."""
    B = koord.shape[0]
    olcek = (boyut - 1) / (girdi_boyut - 1)
    cx, cy = koord[:, 0] * olcek, koord[:, 1] * olcek
    ax = torch.arange(boyut, device=koord.device, dtype=torch.float32)
    gx = torch.exp(-((ax.view(1, -1) - cx.view(-1, 1)) ** 2) / (2 * sigma ** 2))
    gy = torch.exp(-((ax.view(1, -1) - cy.view(-1, 1)) ** 2) / (2 * sigma ** 2))
    return (gy.view(B, boyut, 1) * gx.view(B, 1, boyut)).unsqueeze(1)


class Rafinaj7Kayip(nn.Module):
    """Sadece ORNEGIN TIPINE ait kanaldan kayip alir; digerleri serbest kalir."""

    def __init__(self, heatmap_agirlik=1.0, girdi_boyut=128, heatmap_boyut=64):
        super().__init__()
        self.hw = heatmap_agirlik
        self.gb = girdi_boyut
        self.hb = heatmap_boyut

    def forward(self, koord_hepsi, hm_hepsi, hedef_koord, tip):
        B = koord_hepsi.shape[0]
        ix = torch.arange(B, device=koord_hepsi.device)
        koord = koord_hepsi[ix, tip]                 # (B,2)
        hm = hm_hepsi[ix, tip].unsqueeze(1)          # (B,1,H,W)
        koord_kayip = F.l1_loss(koord, hedef_koord)
        hedef_hm = gauss_hedef(hedef_koord, self.hb, self.gb)
        hm_kayip = F.mse_loss(torch.sigmoid(hm), hedef_hm)
        return koord_kayip + self.hw * hm_kayip * 100.0, koord_kayip


if __name__ == "__main__":
    m = Rafinaj7(pretrained=False)
    k, h = m(torch.randn(2, 3, 128, 128))
    p = sum(t.numel() for t in m.parameters())
    print(f"koordinat {tuple(k.shape)}   heatmap {tuple(h.shape)}")
    print(f"parametre: {p:,} ({p/1e6:.2f}M)   tipler: {TIP_ADI}")
    print(f"nokta->tip: {NOKTA_TIPI}")
