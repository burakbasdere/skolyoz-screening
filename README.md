# Skolyoz — telefonla radyasyonsuz sırt asimetrisi taraması

**Tek bir sırt fotoğrafından, tamamen telefonun içinde, gövde asimetrisi için bir
tarama göstergesi üretir.** Yedi anatomik nokta tespit edilir; bunlardan
**aksiller eğim açısı** ve **koronal denge** hesaplanır. Kameranın eğik tutulmasından
doğan hata telefonun kendi IMU sensörüyle giderilir. Bu ölçümlerden ayrıca **Cobb açısı
ölçümü** de yapılır; uygulama bunu "Cobb tahmini ~X° (±3,5°)" olarak gösterir. Bu özellik
**test aşamasındadır** ve radyografiyle doğrulanmamıştır.
**Sistem şu an test aşamasındadır, son kullanıcı ürünü değildir.**

🇬🇧 [English README](README.en.md) (dokümanların çoğu Türkçedir)

![Android](https://img.shields.io/badge/Android-min%20SDK%2026-3DDC84)
![TFLite](https://img.shields.io/badge/TFLite-float16-FF6F00)
![On-device](https://img.shields.io/badge/inference-on--device-0A7EA4)

<table>
<tr>
<td align="center" valign="top"><img src="docs/img/uygulama/ekran-1.jpg" width="250" alt="Uygulama ekranı: 7,2° sağ omuz yüksek, koronal denge 2,5° sola, yedi nokta ve omurga çizgisi"></td>
<td align="center" valign="top"><img src="docs/img/uygulama/ekran-2.jpg" width="250" alt="Uygulama ekranı: 7,3° sol omuz yüksek, koronal denge 4,0° sağa"></td>
<td align="center" valign="top"><img src="docs/img/uygulama/ekran-3.jpg" width="250" alt="Uygulama ekranı: 7,4° sağ omuz yüksek, koronal denge 2,8° sola"></td>
</tr>
</table>

<sub>Uygulama gerçek bir telefonda çalışırken (Android, tamamen cihaz üstünde). Kamera, bir monitörde gösterilen sırt fotoğraflarına
tutulmuş; karşıda canlı bir kişi yok. Ekrandaki **"Cobb tahmini" satırı**, sistemin yaptığı Cobb açısı ölçümüdür. **Bu özellik
test aşamasındadır:** radyografi eşleşmeli veriyle doğrulanmadı, bu yüzden radyografik Cobb ölçümünün yerine geçmez
([docs/METRICS.md](docs/METRICS.md#7-bilinen-boşluklar), `cobb_estimate`). Alttaki milisaniye değeri
o çekimde uygulamanın kendi sayacıdır; kontrollü bir gecikme ölçümü değildir. Ekrandaki fotoğrafların kaynağı ve kullanım izni:
[docs/DATA_AND_LICENSING.md](docs/DATA_AND_LICENSING.md) §1.</sub>

> [!WARNING]
> **Bu bir araştırma yazılımı ve tarama yardımcısıdır; tanı aracı değildir.**
> **Test aşamasındadır, son kullanıcı ürünü değildir.** Yüzey asimetrisini (aksiller eğim,
> koronal denge) ve bunlardan bir Cobb açısı ölçümü yapar; Cobb ölçümü radyografiyle
> doğrulanmamıştır ve radyografik Cobb açısının yerine geçmez. Hastalar üzerinde
> doğrulanmamış ve tıbbi cihaz olarak onaylanmamıştır. Klinik karar vermek için
> kullanılmamalıdır.

## Nasıl çalışır

```
Fotoğraf + eşzamanlı IMU yerçekimi örneği
   │  640×640'a yeniden boyutlandırma (eğitimdeki germe birebir tekrarlanır)
   ▼
Aşama 1 — YOLO11n-pose (ince ayar): 7 landmark'ın kaba konumu
   ▼
Aşama 2 — MobileNetV3-small + ısı haritası + soft-argmax, landmark başına bir kez:
          kaba nokta çevresinden 128×128 kırpım → alt-piksel konum
   ▼
Orijinal çözünürlüğe geri dönüşüm; ölçüm saf geometri
   ├─ aksiller eğim   = iki koltuk altı noktasının atan2(Δy, |Δx|) değeri − IMU roll
   ├─ koronal denge   = C7'nin sakrumdan geçen şakül hattına yatay sapması
   └─ Cobb ölçümü     = bu ikisinin ve omurga noktası sapmasının doğrusal birleşimi (test aşaması, doğrulanmamış)
   ▼
Kapılar: |pitch| > 8° çekimi reddeder · düşük güvenli ısı haritaları reddedilir
```

İki aşama neden: koordinatı doğrudan regrese eden tek ağ, eğitim/doğrulama/test
kümelerinde aynı ≈24 px hatada takıldı; sınır veri miktarı değil, temsil biçimiydi.

## Bu depoda ne var

Bu depo bir **kataloğdur**: ürünün görünümünü ve performans verilerini gösterir. Kaynak kod,
test, eğitim verisi, model ağırlığı ya da uygulama paketi **içermez**.

| Sayfa | İçerik |
|---|---|
| [docs/METRICS.md](docs/METRICS.md) | Doğruluk, dağıtım, gecikme, dayanıklılık ve **bilinen boşluklar**; her sayının durumu ve kaynağı |
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | Modeller, kullanım amacı, kapsam dışı kullanım, sınırlar ve riskler |
| [docs/ORNEKLER.md](docs/ORNEKLER.md) | Çizilmiş figür ve bir gerçek fotoğraf üzerinde örnek çıktılar |
| [docs/DATA_AND_LICENSING.md](docs/DATA_AND_LICENSING.md) | Veri kaynağı, gizlilik, görsellerin izin durumu, lisans |
| [`metrics/`](metrics) | Ham performans verisi: `results.json` ve kaydedilmiş benchmark koşuları |

## Başlıca metrikler

| Metrik | Değer | Kanıt |
|---|---|---|
| Aksiller eğim hatası, medyan / p90 | 0.271° / 1.237° | raporlanan |
| Koronal denge hatası, medyan / p90 | 1.292° / 3.385° | raporlanan |
| Aksiller landmark hatası, medyan (etiket gürültü tabanı) | 3.045 px (2.45 px) | raporlanan |
| Tarama duyarlılığı, eğim ≥ 3° (ayrılmış küme, n = 50) | 93.8 % (15/16; %95 GA 71.7–98.9 %) | raporlanan |
| Model boyutu, iki aşama, float16 | 6.44 MB | ölçülen |
| Parametre sayısı, Aşama 1 / Aşama 2 | 2.69 M / 0.41 M | ölçülen |
| Model ileri geçişi, 1 × Aşama 1 + 7 × Aşama 2 (TFLite, CPU, medyan / p90) | 70.8 ms / 75.7 ms — 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz | ölçülen |
| Cobb açısı ölçümü | Yapılır; test aşamasında, radyografiyle doğrulanmadı | doğrulanmamış |
| Açık boşluklar | 6 madde — bkz. [docs/METRICS.md](docs/METRICS.md#7-bilinen-boşluklar) | — |

### Grafikler

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/ablation-dark.svg">
  <img alt="Yalnızca Aşama 1 ile Aşama 1 + 2 için keypoint ve açı hatası çubuk grafiği" src="docs/img/ablation.svg" width="760">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/angle-dark.svg">
  <img alt="Aksiller eğim ve koronal denge için medyan hata, p90 hata ve sinyal büyüklüğü" src="docs/img/angle.svg" width="760">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/decision-dark.svg">
  <img alt="3, 5 ve 7 derece eşiklerinde duyarlılık ve doğruluk, %95 Wilson aralıklarıyla" src="docs/img/decision.svg" width="760">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/latency-dark.svg">
  <img alt="Aşama 1, Aşama 2 ve tam zincir için TFLite ve PyTorch gecikme çubuk grafiği" src="docs/img/latency.svg" width="760">
</picture>

Bu sayıları alıntılamadan önce [docs/METRICS.md](docs/METRICS.md) dosyasını okuyun: her
sayının ne anlama geldiğini, nasıl elde edildiğini, ne kadar belirsiz olduğunu ve
**neyin ölçülmediğini** anlatır. Kısaca: doğruluk rakamları 337 görüntülük, web kaynaklı küçük
bir kümede yapılmış çevrimdışı deneylerdir; cihaz üstü ve telefon fotoğrafı doğruluğu açık
sorudur; Cobb ölçümü doğrulanmamıştır. Her sayının durumu ve kaynağı
[`metrics/results.json`](metrics/results.json) içinde kayıtlıdır.

## Bilinen sınırlar

1. **Alan kayması.** Eğitim görüntüleri web kaynaklı; telefon fotoğrafları farklı bir dağılımdan geliyor.
   Aynı sahnenin art arda üç çekimi 1.25°, 3.37° ve 6.73° verdi. Güven kapısı kararsız ölçümü
   yanlış açı göstermek yerine reddeder ama nedeni çözmez — çözüm telefonla toplanmış eğitim verisidir.
2. **Omurga eğrilik profili yok.** Yüzeydeki orta hat sapması (gövde boyunun ≈%1.4'ü) etiket
   gürültüsünün altında kalıyor; bu varsayılmadı, ölçülerek gösterildi.
3. **IMU roll düzeltmesinin işareti doğrulanmadı.** Çizim ve formül birbirini tutmuyor; belgelerde
   cihazda doğrulandığına dair kayıt yok ([docs/METRICS.md](docs/METRICS.md#7-bilinen-boşluklar),
   `imu_roll_sign`). Telefon eğik tutulduğunda düzeltme hatayı artırıyor olabilir.
4. **Cobb doğrulaması yok.** Radyografi eşleşmeli veri yok. Uygulama Cobb açısı ölçümü de yapar
   (test aşaması); bu ölçüm kalibre edilmemiş katsayılar kullanır —
   [docs/METRICS.md](docs/METRICS.md#7-bilinen-boşluklar) içindeki `cobb_estimate` maddesine bakın.

## Gizlilik, veri ve lisans

Çıkarım cihaz üzerinde çalışır; uygulama internet erişimi istemez, yalnızca kamera izni ister
ve yedeklemeyi kapatır. Hasta fotoğrafları bu depoya eklenmez; yukarıdaki ekran fotoğrafları ve
referans fotoğraf istisnadır. Eğitim verisinin kaynağı ve görsellerin izin durumu
[docs/DATA_AND_LICENSING.md](docs/DATA_AND_LICENSING.md) içinde.

**Lisans: henüz seçilmedi.** Bu depodaki içerik tüm hakları saklıdır. Uygulamanın kendisi ve
kaynak kodu bu depoda yayımlanmaz.
