"""Asama 3 - Hassas rafinaj agi (heatmap + soft-argmax).

NEDEN
-----
YOLO-pose noktalari stride 8/16/32 izgarasindan DOGRUDAN koordinat olarak
regresyon ediyor. Olculdu: train=val=test=~24 px. Ezberleme yok, yani sorun
veri miktari degil TEMSIL BICIMI. Etiketler 2.45 px tutarli, hedef ogrenilebilir.

COZUM
-----
Iki asamali (top-down) yaklasim:
  Asama 1: YOLO modeli kaba konumu verir (~24 px hata) -> kirpma penceresi
  Asama 2: bu ag, 128x128 kirpim uzerinde HEATMAP uretir ve SOFT-ARGMAX ile
           alt-piksel koordinat cikarir.

Kazanc iki yerden:
  - kirpma buyutmesi: nokta artik karenin %20'sini kaplayan bir bolgede
  - soft-argmax: argmax'in aksine tam sayi piksele yuvarlamaz, alt-piksel verir

TARAF-BAGIMSIZ
--------------
Sol ve sag koltuk alti birbirinin ayna goruntusu. Tek model egitip rastgele
yatay flip uyguluyoruz: veri iki katina cikiyor. (Bu, TAM GORUNTUDE yasak
olan flip'in kirpim seviyesinde neden serbest oldugunu gosterir - burada
"hangi taraf" bilgisi tasinmiyor, sadece "kose nerede" soruluyor.)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RafinajAgi(nn.Module):
    """MobileNetV3-small govde + deconv head -> tek kanal heatmap.

    Girdi  : (B, 3, 128, 128) kirpim
    Cikti  : (B, 2) koordinat [x, y], kirpim pikseli cinsinden, alt-piksel
             ayrica (B, 1, 64, 64) heatmap (egitimde kayip icin)
    """

    def __init__(self, heatmap_boyut=64, pretrained=True):
        super().__init__()
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

        agirlik = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        govde = mobilenet_v3_small(weights=agirlik).features
        # features[:9] -> stride 16, 48 kanal. 128x128 girdi icin 8x8.
        self.govde = nn.Sequential(*list(govde.children())[:9])
        kanal = 48

        # deconv head: 8x8 -> 16x16 -> 32x32 -> 64x64
        # kernel_size=4, stride=2, padding=1 -> tam 2x buyutme (hizalama kaymaz)
        self.head = nn.Sequential(
            nn.ConvTranspose2d(kanal, 96, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            # kernel 3 + padding 1 = "same" (senin sordugun ayar: 3 ile 1 gider, 3 degil)
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        self.heatmap_boyut = heatmap_boyut

        # soft-argmax icin sabit koordinat izgarasi.
        # contiguous() SART: expand() stride'i 0 olan bir gorunum uretir, ayni
        # bellek hucresi defalarca gosterilir. Boyle bir tampona
        # load_state_dict kopyalama yapamiyor ve CPU'da model YUKLENMIYOR
        # ("more than one element ... refers to a single memory location").
        # CUDA'da fark edilmiyordu cunku .to("cuda") tamponu zaten
        # bitisiklestiriyor. Degerler ayni, yalnizca depolama gercek.
        lin = torch.linspace(0, 1, heatmap_boyut)
        self.register_buffer("izgara_y", lin.view(-1, 1)
                             .expand(heatmap_boyut, heatmap_boyut).contiguous())
        self.register_buffer("izgara_x", lin.view(1, -1)
                             .expand(heatmap_boyut, heatmap_boyut).contiguous())

    def forward(self, x):
        girdi_boyut = x.shape[-1]
        h = self.head(self.govde(x))                    # (B,1,64,64)

        # --- soft-argmax ---
        B = h.shape[0]
        duz = h.view(B, -1)
        # sicaklik: keskinlestirir, alt-piksel cozunurlugu artirir
        olasilik = F.softmax(duz * 10.0, dim=1).view(B, self.heatmap_boyut,
                                                     self.heatmap_boyut)
        x_norm = (olasilik * self.izgara_x).sum(dim=(1, 2))
        y_norm = (olasilik * self.izgara_y).sum(dim=(1, 2))
        koordinat = torch.stack([x_norm, y_norm], dim=1) * (girdi_boyut - 1)
        return koordinat, h


def gauss_hedef(koord, boyut, girdi_boyut, sigma=1.5):
    """Koordinattan Gauss heatmap hedefi uret (kayip stabilizasyonu icin)."""
    B = koord.shape[0]
    olcek = (boyut - 1) / (girdi_boyut - 1)
    cx = koord[:, 0] * olcek
    cy = koord[:, 1] * olcek
    ax = torch.arange(boyut, device=koord.device, dtype=torch.float32)
    gx = torch.exp(-((ax.view(1, -1) - cx.view(-1, 1)) ** 2) / (2 * sigma ** 2))
    gy = torch.exp(-((ax.view(1, -1) - cy.view(-1, 1)) ** 2) / (2 * sigma ** 2))
    h = gy.view(B, boyut, 1) * gx.view(B, 1, boyut)
    return h.unsqueeze(1)


class RafinajKayip(nn.Module):
    """Koordinat L1 (asil hedef) + heatmap MSE (stabilizasyon).

    Saf soft-argmax kaybi egitimin basinda kararsiz olabilir; Gauss heatmap
    terimi aga 'tek tepe uret' baskisi yapar.
    """

    def __init__(self, heatmap_agirlik=1.0, girdi_boyut=128, heatmap_boyut=64):
        super().__init__()
        self.hw = heatmap_agirlik
        self.girdi_boyut = girdi_boyut
        self.heatmap_boyut = heatmap_boyut

    def forward(self, tahmin_koord, tahmin_hm, gercek_koord):
        koord_kayip = F.l1_loss(tahmin_koord, gercek_koord)
        hedef = gauss_hedef(gercek_koord, self.heatmap_boyut, self.girdi_boyut)
        hm_kayip = F.mse_loss(torch.sigmoid(tahmin_hm), hedef)
        return koord_kayip + self.hw * hm_kayip * 100.0, koord_kayip


if __name__ == "__main__":
    m = RafinajAgi(pretrained=False)
    x = torch.randn(2, 3, 128, 128)
    k, h = m(x)
    p = sum(t.numel() for t in m.parameters())
    print(f"koordinat {tuple(k.shape)}  heatmap {tuple(h.shape)}")
    print(f"parametre: {p:,}  ({p/1e6:.2f}M)")
