# Teknoloji yığını

Sürümler, hafızadan değil, son sütunda adı geçen build dosyalarından alınmıştır. Bir
sabitleme orada değişirse burada da değiştirin (daha iyisi: yalnızca orada değiştirin).

## Android uygulaması

| Konu | Seçim | Nerede sabitli |
|---|---|---|
| Dil | Kotlin 2.0.21, JVM hedefi 17 | `android/gradle/libs.versions.toml`, `app/build.gradle.kts` |
| Derleme | Android Gradle Plugin 8.7.3, Gradle 8.10.2 (wrapper) | `libs.versions.toml`, `gradle-wrapper.properties` |
| SDK seviyeleri | min 26 (Android 8.0), compile / target 35 | `app/build.gradle.kts` |
| Arayüz | Jetpack Compose (BOM 2024.12.01), Material 3, Activity-Compose 1.9.3, Lifecycle 2.8.7 | `libs.versions.toml` |
| Kamera | CameraX 1.4.1 (`core`, `camera2`, `lifecycle`, `view`); sıfır deklanşör gecikmeli çekim; dikey yön kilitli | `libs.versions.toml`, `CekimEkrani.kt` |
| Yön algılama | `SensorManager` `TYPE_GRAVITY`; halka tampon + karenin donanım zaman damgasında doğrusal enterpolasyon | `sensor/` |
| Çıkarım | TensorFlow Lite 2.16.1, 4 CPU iş parçacığı, modeller bellek eşlemeli (`noCompress += tflite`) | `libs.versions.toml`, `OlcumZinciri.kt` |
| Release ABI'leri | `arm64-v8a`, `armeabi-v7a` (debug, emülatör için evrensel kalır) | `app/build.gradle.kts` |
| Birim testleri | JUnit 4.13.2; altın veri CSV'leri; JVM ön işleme zamanlama testi | `app/src/test` |
| Cihaz testleri | `androidx.test` (ext-junit 1.2.1, runner 1.6.2) — telefonda çalışır, CI'da çalışmaz | `app/src/androidTest` |
| İzinler | Yalnızca `CAMERA`, `INTERNET` yok (birleştirilmiş manifestte AndroidX'in uygulamaya özel `DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION` izni de var); kamera ve ivmeölçer zorunlu özellik; yedekleme kapalı | `AndroidManifest.xml` |

## Modeller ve eğitim

| Konu | Seçim |
|---|---|
| Aşama 1 | Ultralytics YOLO11n-pose, 640 × 640'ta 7 keypoint için ince ayar |
| Aşama 2 | torchvision MobileNetV3-small (ilk 9 özellik bloğu, stride 16) + 3 transpoze konvolüsyon + landmark tipi başına 64 × 64 ısı haritası, soft-argmax okuması |
| Çerçeve | PyTorch (CUDA; RTX 50xx / Blackwell için `cu128` wheel'leri gerekir) |
| Etiketleme | Roboflow (keypoint dışa aktarımı); `scripts/00`–`12` ile temizlendi, yinelemeler ayıklandı |
| Doğrulama | Grup duyarlı 5 katlı çapraz doğrulama (`scripts/45`, `46`) |
| Dışa aktarma | PyTorch → ONNX (`scripts/22`, `51`) → `onnx2tf` → TFLite float16 (`scripts/25`, `52`) |
| Kuantizasyon | float16 kullanılıyor; tam tamsayı INT8 denendi ve Aşama 1 için çalışmıyor (bkz. METRICS §2.1) |

Python bağımlılık setleri: [`requirements/`](../requirements) — `dev`, `bench`, `train`,
`export`. Dışa aktarma kendi sanal ortamında çalışır; çünkü TensorFlow ve `onnx2tf`,
eğitim yığınıyla çakışan NumPy / protobuf sürümleri ister.

## Metrikler, benchmark'lar ve kalite kapıları

| Konu | Seçim |
|---|---|
| Paket | `skolyoz_metrics` — çalışma zamanında yalnızca standart kütüphane (`bench` ek bağımlılıkları torch, ultralytics ve bir TFLite çalışma zamanı getirir) |
| Tek kaynak | `metrics/results.json`, `metrics/results.schema.json` ile; benchmark koşuları `metrics/benchmark.schema.json` ile doğrulanır |
| Üretilen dokümanlar | `docs/METRICS.md` ve iki README'deki başlıca metrik tablosu (`python -m skolyoz_metrics render`) |
| İstatistik | Doğrusal enterpolasyonlu yüzdelikler; oranlar için Wilson skor aralıkları |
| Lint / test | Ruff (E, F, I, UP, B) ve pytest; araştırma betikleri lint dışındadır |
| CI | GitHub Actions: `ci.yml` (lint, testler, metrik denetimi, benchmark duman koşusu), `android.yml` (wrapper doğrulama, Gradle testleri, debug derleme) |
| Bağımlılık güncellemeleri | Dependabot: Actions ve Gradle, aylık ve gruplu; büyük sürümler elle |

## Benchmark sayılarının üretildiği araçlar

`torch 2.5.1`, `ultralytics 8.4.38`, `ai-edge-litert 2.3.0`, Windows 11 üzerinde
Python 3.11; tam makine ve iş parçacığı ayarları `metrics/benchmarks/` içindeki her
dosyaya yazılıdır.
