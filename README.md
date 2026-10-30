# Skolyoz — telefonla radyasyonsuz sırt asimetrisi taraması

**Tek bir sırt fotoğrafından, tamamen telefonun içinde, gövde asimetrisi için bir
tarama göstergesi üretir.** Yedi anatomik nokta tespit edilir; bunlardan
**aksiller eğim açısı** ve **koronal denge** hesaplanır. Kameranın eğik tutulmasından
doğan hata telefonun kendi IMU sensörüyle giderilir.

🇬🇧 [English README](README.en.md) (dokümanların çoğu Türkçedir)

![Android](https://img.shields.io/badge/Android-min%20SDK%2026-3DDC84)
![TFLite](https://img.shields.io/badge/TFLite-float16-FF6F00)
![On-device](https://img.shields.io/badge/inference-on--device-0A7EA4)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)

> [!WARNING]
> **Bu bir araştırma yazılımı ve tarama yardımcısıdır; tanı aracı değildir.**
> Ölçtüğü şey *yüzey asimetrisidir*, radyografik **Cobb açısı değildir**; hastalar
> üzerinde doğrulanmamış ve tıbbi cihaz olarak onaylanmamıştır. Klinik karar
> vermek için kullanılmamalıdır.

## Nasıl çalışır

```
Fotoğraf (CameraX) + eşzamanlı IMU yerçekimi örneği
   │  640×640'a yeniden boyutlandırma (eğitimdeki germe birebir tekrarlanır)
   ▼
Aşama 1 — YOLO11n-pose (ince ayar): 7 landmark'ın kaba konumu
   ▼
Aşama 2 — MobileNetV3-small + ısı haritası + soft-argmax, landmark başına bir kez:
          kaba nokta çevresinden 128×128 kırpım → alt-piksel konum
   ▼
Orijinal çözünürlüğe geri dönüşüm; ölçüm saf geometri
   ├─ aksiller eğim   = iki koltuk altı noktasının atan2(Δy, |Δx|) değeri − IMU roll
   └─ koronal denge   = C7'nin sakrumdan geçen şakül hattına yatay sapması
   ▼
Kapılar: |pitch| > 8° çekimi reddeder · düşük güvenli ısı haritaları reddedilir
```

İki aşama neden: koordinatı doğrudan regrese eden tek ağ, eğitim/doğrulama/test
kümelerinde aynı ≈24 px hatada takıldı; sınır veri miktarı değil, temsil biçimiydi.
Ayrıntı: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Örnek çıktılar

Aşağıdakiler, gönderilen gerçek TFLite modellerinin **çizilmiş (sentetik) bir sırt figürü**
üzerindeki çıktılarıdır; hiçbir görüntü gerçek bir kişiye ait değildir. Bunlar başarım kanıtı
değil, boru hattının ne yaptığının gösterimidir: modeller gerçek fotoğraflarla eğitildi ve çizilmiş
figürde nokta hatası belirgin (5° çizilen eğim 3.3° bulundu). Sayılar ve tüm noktalar:
[docs/ORNEKLER.md](docs/ORNEKLER.md).

<table>
<tr>
<td align="center"><img src="docs/img/ornekler/duz.png" width="250" alt="Simetrik figür: aksiller eğim +0.33°"><br><sub><b>Simetrik figür</b><br>eğim +0.33°, kapı geçti</sub></td>
<td align="center"><img src="docs/img/ornekler/asimetri.png" width="250" alt="Sağ omuz yüksek figür: aksiller eğim +3.34°"><br><sub><b>Sağ omuz yüksek</b><br>çizimde 5°, ölçülen +3.34°</sub></td>
<td align="center"><img src="docs/img/ornekler/bulanik.png" width="250" alt="Bulanık çekim: güven kapısı reddeder"><br><sub><b>Bulanık çekim</b><br>kapı reddeder (entropi 3.74)</sub></td>
</tr>
<tr>
<td align="center"><img src="docs/img/ornekler/pitch.png" width="250" alt="Telefon 11 derece yatık: ölçüm geçersiz"><br><sub><b>Telefon 11° yatık</b><br>deklanşör kilitli</sub></td>
<td align="center"><img src="docs/img/ornekler/kisi-yok.png" width="250" alt="Boş kare: kişi bulunamadı"><br><sub><b>Karede kimse yok</b><br>ölçüm yapılmaz</sub></td>
<td></td>
</tr>
</table>

Yeniden üretmek için model dosyaları gerekir: `python -m skolyoz_metrics demo --stage1 asama1.tflite --stage2 asama2.tflite`.

## Depoda neler var

Bu depo yalnızca bir sonuç sayfası değil; uygulamanın, araştırma hattının ve ölçüm
altyapısının tüm kaynak kodunu içerir. İçermedikleri: hasta görüntüleri ve eğitilmiş
model ağırlıkları (bilerek; bkz. [docs/DATA_AND_LICENSING.md](docs/DATA_AND_LICENSING.md)).

| Klasör | İçerik |
|---|---|
| [`android/`](android) | Kotlin uygulaması: CameraX çekimi, IMU yerçekimi enterpolasyonu, TFLite çıkarım zinciri, ölçüm matematiği, kılavuz çizim; 42 JVM birim testi + 4 cihaz testi |
| [`scripts/`](scripts) | Numaralı araştırma hattı `00` → `52`: veri temizleme, etiket gürültüsü ölçümü, eğitim, değerlendirme, ONNX/TFLite dışa aktarma, Kotlin altın verisi üretimi |
| [`skolyoz_metrics/`](skolyoz_metrics) | Bakımı yapılan Python kodu: metrik doğrulama/üretim ve veri gerektirmeyen benchmark'lar |
| [`tests/`](tests) | pytest paketi (Python referansı ↔ Kotlin altın veri eşdeğerliği dahil) |
| [`metrics/`](metrics) | Metriklerin tek kaynağı (`results.json`, şemalar) ve kaydedilmiş benchmark koşuları |
| [`docs/`](docs) | Mimari, teknoloji yığını, benchmark protokolü, model kartı, veri/lisans notları, **üretilen** metrik sayfası |
| [`docs/tr/`](docs/tr) | Orijinal Türkçe teknik rapor ve çalışma notları |
| [`requirements/`](requirements) | Bağımlılık setleri: `dev`, `bench`, `train`, `export` |
| [`.github/`](.github) | CI iş akışları, Dependabot, issue / PR şablonları |

## Hızlı başlangıç

**Testleri ve metrik denetimini çalıştır** — GPU ve özel veri gerekmez:

```bash
pip install -r requirements/dev.txt
python -m skolyoz_metrics check     # şema + tutarlılık + üretilmiş dokümanlar senkron mu
pytest
ruff check .
```

**Android uygulamasını derle** (JDK 17; `JAVA_HOME` Android Studio JBR'sini göstermeli):

```bash
cd android
./gradlew :app:testDebugUnitTest     # JVM testleri; cihaz ve model dosyası gerekmez
./gradlew :app:assembleDebug
```

Eğitilmiş model dosyaları **dağıtılmaz**. Uygulamayı çalıştırmak için `asama1.tflite` ve
`asama2.tflite` dosyalarını `android/app/src/main/assets/` altına koyun.

**Model mimarilerini ölç** — rastgele ağırlık, sentetik girdi, hasta verisi yok:

```bash
pip install -r requirements/bench.txt
python -m skolyoz_metrics bench arch
python -m skolyoz_metrics bench tflite --stage1 asama1.tflite --stage2 asama2.tflite
python -m skolyoz_metrics render    # dokümanları yenile
```

Neyin ölçülüp neyin ölçülmediği: [docs/BENCHMARKING.md](docs/BENCHMARKING.md).

**Araştırmayı yeniden üretmek** (`scripts/00`–`52`) özel görüntü kümesi ve CUDA'lı bir GPU
gerektirir; bkz. [requirements/train.txt](requirements/train.txt) ve [docs/tr/](docs/tr).

## Teknoloji

| Katman | Altyapı |
|---|---|
| Mobil uygulama | Kotlin 2.0, Jetpack Compose + Material 3, CameraX 1.4, TensorFlow Lite 2.16 (float16, 4 iş parçacığı), Android min SDK 26 |
| Modeller | Ultralytics YOLO11n-pose (Aşama 1), torchvision MobileNetV3-small + özel ısı haritası başlığı (Aşama 2); PyTorch ile eğitildi |
| Dışa aktarma | PyTorch → ONNX → `onnx2tf` → TFLite float16 |
| Metrik ve benchmark | Python paketi [`skolyoz_metrics`](skolyoz_metrics): şema denetimli sonuçlar, üretilmiş dokümanlar, veri gerektirmeyen gecikme ölçümleri |
| Kalite kapıları | GitHub Actions: Ruff, pytest, metrik sapma denetimi, Gradle birim testleri |

Sürümler ve nerede sabitlendikleri: [docs/TECH_STACK.md](docs/TECH_STACK.md).

## Başlıca metrikler

<!-- headline:begin -->
| Metrik | Değer | Kanıt |
|---|---|---|
| Aksiller eğim hatası, medyan / p90 | 0.271° / 1.237° | raporlanan |
| Koronal denge hatası, medyan / p90 | 1.292° / 3.385° | raporlanan |
| Aksiller landmark hatası, medyan (etiket gürültü tabanı) | 3.045 px (2.45 px) | raporlanan |
| Tarama duyarlılığı, eğim ≥ 3° (ayrılmış küme, n = 50) | 93.8 % (15/16; %95 GA 71.7–98.9 %) | raporlanan |
| Model boyutu, iki aşama, float16 | 6.44 MB | ölçülen |
| Parametre sayısı, Aşama 1 / Aşama 2 | 2.69 M / 0.41 M | ölçülen |
| Model ileri geçişi, 1 × Aşama 1 + 7 × Aşama 2 (TFLite, CPU, medyan / p90) | 70.8 ms / 75.7 ms — 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz | ölçülen |
| Kotlin ↔ Python eşdeğerliği (altın veri) | 42 JVM + 4 cihaz testi; raporlanan fark 0.000 px / 0.0000° | raporlanan |
| Açık boşluklar | 6 madde — bkz. [docs/METRICS.md](docs/METRICS.md#8-bilinen-boşluklar) | — |
<!-- headline:end -->

### Grafikler

<!-- charts:begin -->
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
<!-- charts:end -->

Bu sayıları alıntılamadan önce [docs/METRICS.md](docs/METRICS.md) dosyasını okuyun: her
sayının ne anlama geldiğini, nasıl elde edildiğini, ne kadar belirsiz olduğunu ve
**neyin ölçülmediğini** anlatır. Kısaca: doğruluk rakamları 337 görüntülük, web kaynaklı küçük
bir kümede yapılmış çevrimdışı deneylerdir; cihaz üstü ve telefon fotoğrafı doğruluğu açık
sorudur; uygulamada gösterilen Cobb tahmini doğrulanmamıştır.

Yukarıdaki tablodaki her sayı [`metrics/results.json`](metrics/results.json) ve
[`metrics/benchmarks/`](metrics/benchmarks) dosyalarından üretilir; ikisi ayrışırsa CI başarısız olur.

## Bilinen sınırlar

1. **Alan kayması.** Eğitim görüntüleri web kaynaklı; telefon fotoğrafları farklı bir dağılımdan geliyor.
   Aynı sahnenin art arda üç çekimi 1.25°, 3.37° ve 6.73° verdi. Güven kapısı kararsız ölçümü
   yanlış açı göstermek yerine reddeder ama nedeni çözmez — çözüm telefonla toplanmış eğitim verisidir.
2. **Omurga eğrilik profili yok.** Yüzeydeki orta hat sapması (gövde boyunun ≈%1.4'ü) etiket
   gürültüsünün altında kalıyor; bu varsayılmadı, ölçülerek gösterildi.
3. **IMU roll düzeltmesinin işareti doğrulanmadı.** Çizim ve formül birbirini tutmuyor; belgelerde
   cihazda doğrulandığına dair kayıt yok ([docs/METRICS.md](docs/METRICS.md#8-bilinen-boşluklar),
   `imu_roll_sign`). Telefon eğik tutulduğunda düzeltme hatayı artırıyor olabilir.
4. **Cobb doğrulaması yok.** Radyografi eşleşmeli veri yok; burada hiçbir şey Cobb açısı tahmin etmez.
   Uygulamadaki Cobb tahmini kalibre edilmemiş katsayılar kullanır —
   [docs/METRICS.md](docs/METRICS.md#8-bilinen-boşluklar) içindeki `cobb_estimate` maddesine bakın.

## Gizlilik, veri ve lisans

Çıkarım cihaz üzerinde çalışır; uygulama `INTERNET` izni istemez, yalnızca `CAMERA` ister ve
yedeklemeyi kapatır. Hasta fotoğrafları asla commit edilmez (`.gitignore`). Eğitim verisinin
kaynağı, üçüncü taraf lisansları ve Ultralytics'in AGPL-3.0 sonuçları
[docs/DATA_AND_LICENSING.md](docs/DATA_AND_LICENSING.md) içinde.

**Lisans: henüz seçilmedi.** `LICENSE` dosyası eklenene kadar kod "tüm hakları saklıdır" durumundadır.
Seçim Ultralytics bağımlılığı nedeniyle kısıtlıdır; ilgili dokümana bakın.

## Katkı

[CONTRIBUTING.md](CONTRIBUTING.md) dosyasına bakın. Önemli tek kural: bir sayıya dokunan
her değişiklik `metrics/results.json` üzerinden, durum ve kaynak belirtilerek yapılır.
