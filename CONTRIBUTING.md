# Katkı rehberi

İlginiz için teşekkürler. Bu, tıbbi bir yönü olan araştırma yazılımıdır; aşağıdaki kurallar
sizi yavaşlatmak için değil, projeyi dürüst ve güvenli tutmak için var.

## Temel kurallar

1. **Asla hasta verisi yok** — fotoğraf, onlardan türetilmiş etiket ya da onlarla eğitilmiş
   ağırlık eklemeyin. Bkz. [docs/DATA_AND_LICENSING.md](docs/DATA_AND_LICENSING.md).
2. **Sayılar `metrics/` içinde yaşar.** Değişikliğiniz bir metrik ekliyor ya da değiştiriyorsa
   `metrics/results.json` dosyasını düzenleyin (her bölümde `status` ve `source` olmalı) ya da
   yeni bir benchmark dosyası ekleyin, sonra `python -m skolyoz_metrics render` çalıştırın.
   Bir sonucu README'ye elle yazmayın: CI hepsini yeniden üretir ve fark varsa başarısız olur.
   Grafikler (`docs/img/`) de aynı yolla üretilir.
3. **Bir sayının ne olmadığını yazın.** Ayrılmış küme sonucu çapraz doğrulama sonucu değildir;
   masaüstü zamanlaması telefon zamanlaması değildir. Ölçemiyorsanız atlamak yerine `gaps`
   içine bir madde ekleyin.
4. **Yeni arayüzde ya da dokümanda Cobb açısı belirtmeyin.** Projede radyografi eşleşmeli veri yok.

## Kurulum

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements/dev.txt
python -m skolyoz_metrics check && pytest && ruff check .
```

Daha ağır, isteğe bağlı: benchmark'lar ve mimari testleri için `requirements/bench.txt`
(torch / ultralytics yoksa bu testler kendilerini atlar).

Android (JDK 17; Android SDK platform 35):

```bash
cd android && ./gradlew :app:testDebugUnitTest
```

## Ölçüm matematiğini değiştirmek

Kotlin kodu `scripts/19_kilavuz_cizim.py` dosyasının portudur. Önce Python referansını
değiştirin, altın veriyi yeniden üretin (`scripts/23_altin_veri.py`, `27`, `29`), sonra
Kotlin testlerini geçirin. Hem `tests/test_golden_parity.py` hem de Gradle testleri yeşil olmalı.

## Stil

- Python: Ruff (`ruff check .`), satır uzunluğu 100, genel fonksiyonlarda tür ipuçları.
  `scripts/` deneylerin donmuş kaydıdır, lint edilmez; hata düzeltin ama yeniden biçimlendirmeyin.
- Kotlin: resmî stil (`kotlin.code.style=official`).
- Commit'ler: emir kipi, her biri tek mantıksal değişiklik ("p99'u gecikme özetine ekle").
- Pull request'ler: hangi metriklerin değiştiğini yazın ve yeniden üretilen dosyaları ekleyin.

## Sorun bildirmek

Issue şablonlarını kullanın. Hasta verisini ilgilendiren herhangi bir şey için onu **eklemeyin**
— bkz. [SECURITY.md](SECURITY.md).
