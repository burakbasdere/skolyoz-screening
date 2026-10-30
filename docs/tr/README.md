# Türkçe araştırma notları

Bu klasör projenin orijinal çalışma belgelerini içerir. Bunlar **tarihsel kayıttır**;
güncel, kaynağı izlenebilir sayılar için [../METRICS.md](../METRICS.md) (İngilizce,
`metrics/results.json`'dan üretilir) esas alınmalıdır.

| Belge | İçerik |
|---|---|
| [TEKNIK_DOKUMAN.md](TEKNIK_DOKUMAN.md) | Yöntem, bulgular, olumsuz sonuçlar, kaynakça |
| [MOBIL_YAPILACAKLAR.md](MOBIL_YAPILACAKLAR.md) | Mobil taraf, konvansiyonlar, tuzaklar, cihaz üstü boru hattı, faz notları |
| [ETIKETLEME_REHBERI.md](ETIKETLEME_REHBERI.md) | Etiketleme kuralları ve bulgular |

## Sonradan fark edilen düzeltmeler

Yeni depo hazırlanırken belgeler birbirleriyle ve kodla karşılaştırıldı. Aşağıdaki
noktalar bu belgelerde farklı yazıyor; `METRICS.md` doğru kabul ettiğimiz halini gösterir.

| Belge / yer | Yazılan | Düzeltme ve gerekçe |
|---|---|---|
| `TEKNIK_DOKUMAN` §6.2 karar tablosu (3°/5° eşikleri) ve eski README | 5 katlı çapraz doğrulamadan | Aynı sayılar `MOBIL_YAPILACAKLAR` §2'de **n = 50 test kümesi** altında geçiyor; "16 pozitif vakanın 15'i" 337 görüntülük bir değerlendirmeden çıkamaz. Ayrılmış n = 50 sonucu olarak sınıflandırıldı. Doğrulamak için `scripts/20` yeniden çalıştırılmalı. |
| `TEKNIK_DOKUMAN` §6.1–6.2, eski README "doğrudan cihaz üstü başarım" | Landmark / açı hataları uçtan uca ve cihaz başarımı gibi sunuluyor | CV (`scripts/45`) Aşama 2'yi, etiket çevresinde **±35 px yapay sapmalı** (`JITTER = 35`) kırpımlarla ölçüyor; gerçek Aşama 1 çıktısı ve TFLite float16 kaybı içermiyor. Float16 maliyeti yalnızca 2 noktalı modellerde ölçüldü. |
| Eski README: "2.66 M parametre · 7 landmark" | Aşama 1 | 2.66 M **2 noktalı** modelin değeri. 7 landmark'lı model **2 688 252** parametre (Aşama 2: 405 340). |
| Eski README: "46 birim testi" | | 42 JVM birim testi + 4 cihaz (instrumented) testi. Hepsi 2026-10-31'de çalıştırıldı: 42/42 geçti. |
| Eski README: model "6.4 MB", APK "~32 MB" | | Modeller 5 575 676 + 860 284 bayt = 6.44 MB. APK: release 20.19 MB, debug 33.58 MB (universal). |
| `scripts/31` çıktısı "~100–200 ms/cihaz" | | İki noktalı zincir içindi. Şimdiki zincir Aşama 2'yi **7 kez** çağırıyor; cihaz üstü gecikme hâlâ ölçülmedi. |
| `MOBIL_YAPILACAKLAR` §1, §7; `MainActivity.kt` yorumu: "arayüzde Cobb yazılmaz" | | Android arayüzü `Cobb tahmini: ~X° (±3.5°)` gösteriyor (`KilavuzCizer.kt`, `Olcum.kt` `Kalibrasyon`, `scripts/48`). Katsayılar (1.25 / 1.10 / 2.40) ve ±3.5° hiçbir veriye uydurulmadı. `METRICS.md`'de `cobb_estimate` olarak **doğrulanmamış** işaretli. |
