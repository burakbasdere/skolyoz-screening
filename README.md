# Skolyoz — telefonla radyasyonsuz sırt asimetrisi taraması

**Tek bir sırt fotoğrafından, tamamen telefonun içinde, gövde asimetrisi için bir
tarama göstergesi üretir.** Yedi anatomik nokta tespit edilir; bunlardan
**aksiller eğim açısı** ve **koronal denge** hesaplanır. Kameranın eğik tutulmasından
doğan hata telefonun kendi IMU sensörüyle giderilir.

🇬🇧 [English README](README.md)

![Android](https://img.shields.io/badge/Android-min%20SDK%2026-3DDC84)
![TFLite](https://img.shields.io/badge/TFLite-float16-FF6F00)
![On-device](https://img.shields.io/badge/inference-on--device-0A7EA4)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)

> [!WARNING]
> **Bu bir araştırma yazılımı ve tarama yardımcısıdır; tanı aracı değildir.**
> Ölçtüğü şey *yüzey asimetrisidir*, radyografik **Cobb açısı değildir**; hastalar
> üzerinde doğrulanmamış ve tıbbi cihaz olarak onaylanmamıştır. Klinik karar
> vermek için kullanılmamalıdır.

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
| Açık boşluklar | 5 madde — bkz. [docs/METRICS.md](docs/METRICS.md#8-known-gaps) | — |
<!-- headline:end -->

Bu sayıları alıntılamadan önce [docs/METRICS.md](docs/METRICS.md) dosyasını okuyun
(İngilizce): her sayının ne anlama geldiğini, nasıl elde edildiğini, ne kadar
belirsiz olduğunu ve *Known gaps* bölümünde **neyin ölçülmediğini** anlatır. Kısaca:
doğruluk rakamları 337 görüntülük, web kaynaklı küçük bir kümede yapılmış çevrimdışı
deneylerdir; cihaz üstü ve telefon fotoğrafı doğruluğu açık sorudur; uygulamada
gösterilen Cobb tahmini doğrulanmamıştır.

Yukarıdaki tablodaki her sayı [`metrics/results.json`](metrics/results.json) ve
[`metrics/benchmarks/`](metrics/benchmarks) dosyalarından üretilir; ikisi ayrışırsa CI başarısız olur.

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

## Teknoloji

| Katman | Altyapı |
|---|---|
| Mobil uygulama | Kotlin 2.0, Jetpack Compose + Material 3, CameraX 1.4, TensorFlow Lite 2.16 (float16, 4 iş parçacığı), Android min SDK 26 |
| Modeller | Ultralytics YOLO11n-pose (Aşama 1), torchvision MobileNetV3-small + özel ısı haritası başlığı (Aşama 2); PyTorch ile eğitildi |
| Dışa aktarma | PyTorch → ONNX → `onnx2tf` → TFLite float16 |
| Metrik ve benchmark | Python paketi [`skolyoz_metrics`](skolyoz_metrics): şema denetimli sonuçlar, üretilmiş dokümanlar, veri gerektirmeyen gecikme ölçümleri |
| Kalite kapıları | GitHub Actions: Ruff, pytest (Python↔Kotlin altın veri eşdeğerliği dahil), metrik sapma denetimi, Gradle birim testleri |

Sürümler ve nerede sabitlendikleri: [docs/TECH_STACK.md](docs/TECH_STACK.md).

## Hızlı başlangıç

**Metrikleri denetle ve testleri çalıştır** — GPU ve özel veri gerekmez:

```bash
pip install -r requirements/dev.txt
python -m skolyoz_metrics check     # şema + tutarlılık + üretilmiş dokümanlar senkron mu
pytest
ruff check .
```

**Model mimarilerini ölç** — rastgele ağırlık, sentetik girdi:

```bash
pip install -r requirements/bench.txt
python -m skolyoz_metrics bench arch
python -m skolyoz_metrics bench tflite --stage1 asama1.tflite --stage2 asama2.tflite
python -m skolyoz_metrics render    # dokümanları yenile
```

**Android uygulamasını derle** (JDK 17; `JAVA_HOME` Android Studio JBR'sini göstermeli):

```bash
cd android
./gradlew :app:testDebugUnitTest     # JVM testleri; cihaz ve model dosyası gerekmez
./gradlew :app:assembleDebug
```

Eğitilmiş model dosyaları **dağıtılmaz** ([docs/DATA_AND_LICENSING.md](docs/DATA_AND_LICENSING.md)).
Uygulamayı çalıştırmak için `asama1.tflite` ve `asama2.tflite` dosyalarını
`android/app/src/main/assets/` altına koyun.

**Araştırmayı yeniden üretmek** (`scripts/00`–`52`) özel görüntü kümesi ve CUDA'lı bir GPU
gerektirir. Türkçe araştırma notları: [docs/tr/](docs/tr).

## Bilinen sınırlar

1. **Alan kayması.** Eğitim görüntüleri web kaynaklı; telefon fotoğrafları farklı bir dağılımdan geliyor.
   Aynı sahnenin art arda üç çekimi 1.25°, 3.37° ve 6.73° verdi. Güven kapısı kararsız ölçümü
   yanlış açı göstermek yerine reddeder ama nedeni çözmez — çözüm telefonla toplanmış eğitim verisidir.
2. **Omurga eğrilik profili yok.** Yüzeydeki orta hat sapması (gövde boyunun ≈%1.4'ü) etiket
   gürültüsünün altında kalıyor; bu varsayılmadı, ölçülerek gösterildi.
3. **Cobb doğrulaması yok.** Radyografi eşleşmeli veri yok; burada hiçbir şey Cobb açısı tahmin etmez.
   Uygulamadaki Cobb tahmini kalibre edilmemiş katsayılar kullanır —
   [docs/METRICS.md](docs/METRICS.md#8-known-gaps) içindeki `cobb_estimate` maddesine bakın.

## Gizlilik, veri ve lisans

Çıkarım cihaz üzerinde çalışır; uygulamanın manifestosu yalnızca `CAMERA` iznini ister
(`INTERNET` yok) ve yedeklemeyi kapatır. Hasta fotoğrafları asla commit edilmez (`.gitignore`).
Eğitim verisinin kaynağı, üçüncü taraf lisansları ve Ultralytics'in AGPL-3.0 sonuçları
[docs/DATA_AND_LICENSING.md](docs/DATA_AND_LICENSING.md) içinde.

**Lisans: henüz seçilmedi.** `LICENSE` dosyası eklenene kadar kod "tüm hakları saklıdır" durumundadır.
Seçim Ultralytics bağımlılığı nedeniyle kısıtlıdır; ilgili dokümana bakın.

## Katkı

[CONTRIBUTING.md](CONTRIBUTING.md) dosyasına bakın. Önemli tek kural: bir sayıya dokunan
her değişiklik `metrics/results.json` üzerinden, durum ve kaynak belirtilerek yapılır.
