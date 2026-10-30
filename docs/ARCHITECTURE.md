# Mimari

Bu sayfa, bir fotoğrafın nasıl bir sayıya dönüştüğünü ve hangi kararların ölçüme
dayanılarak verildiğini anlatır. Sayılar bilerek burada **tekrarlanmaz**; doğru tutulacak
tek yer [METRICS.md](METRICS.md) olsun diye orada durur. Araştırmanın tam anlatımı
[tr/TEKNIK_DOKUMAN.md](tr/TEKNIK_DOKUMAN.md) ve
[tr/MOBIL_YAPILACAKLAR.md](tr/MOBIL_YAPILACAKLAR.md) dosyalarındadır.

## Ne ölçülüyor

| Büyüklük | Tanım |
|---|---|
| **Aksiller eğim** | Sol ve sağ koltuk altı noktalarını birleştiren doğrunun *yerçekimi yatayına* göre açısı: `atan2(Δy, \|Δx\|) − roll`. Pozitif değer, kişinin sağ omzunun yüksek olduğunu gösterir. |
| **Koronal denge** | C7 (vertebra prominens) noktasının, sakrumdan indirilen şakül hattına yatay sapmasının gövde boyuna göre açısı — klinik muayenedeki C7 şakül testinin fotoğraftaki karşılığı. |

İkisi de Cobb açısı değildir. Cobb açısı radyografide tanımlanır; bunlar yüzey
göstergeleridir. Kompanse eğriler omuzları düz bırakabilir, bacak boyu ya da kas
asimetrisi de eğri olmadan omuzları eğebilir.

## Boru hattı

```
CameraX fotoğrafı + deklanşör zaman damgasında örneklenen yerçekimi vektörü
  └─ pitch kapısı: |pitch| > 8° → çekim reddedilir (perspektif açıyı bozar)
640×640'a yeniden boyutlandırma (germe, letterbox değil)      ← eğitimle aynı olmalı
Aşama 1  YOLO11n-pose  → 7 kaba landmark (+ güven)
Aşama 2  MobileNetV3-small + ısı haritası başlığı, landmark başına bir kez
         128×128 kırpım (kenarda siyah dolgu) → 64×64 ısı haritası → soft-argmax
güven kapısı: ısı haritası entropisi ve tepe-ağırlık merkezi uzaklığı
landmark'ları ORİJİNAL çözünürlüğe geri çevir
ölçüm (saf geometri) + IMU roll düzeltmesi
```

### Yedi landmark ve dört görsel tip

| İndeks | Landmark | Aşama 2 tipi |
|---|---|---|
| 0 / 1 | sağ / sol koltuk altı | 0 — koltuk altı (ayna simetrik, ortak) |
| 2 | C7 | 1 |
| 3 / 4 / 5 | üç omurga noktası | 2 — omurga (aynı doku, ortak) |
| 6 | sakrum | 3 |

Benzer görünen landmark'lar arasında ağırlık paylaşmak her tipe daha çok eğitim sinyali
verir: Aşama 2 ağı *tip* başına bir ısı haritası kanalı üretir ve kayıp yalnızca örneğin
kendi tipinin kanalından hesaplanır. Aşama 1 çıktısı `(1, 26, 8400)`'dür: 4 kutu değeri,
1 güven ve 7 × (x, y, görünürlük); değerler zaten piksele çözülmüştür, bu yüzden uygulamada
anchor ya da stride mantığı yoktur ve (karede tek kişi beklendiği için) NMS yerine en
yüksek güvenli aday alınır.

## Neden iki aşama

Koordinatı doğrudan regrese eden tek bir YOLO-pose ağı, eğitim, doğrulama ve test
kümelerinde aynı hatayı verdi. Aşırı öğrenme değildi; stride 8/16/32 çıktı ızgarası
istenen hassasiyette yerelleştirme yapamıyor. Kaba noktanın çevresini kırpıp ısı haritası
ile soft-argmax kullanmak, kaba bir regresyon problemini ince bir probleme çeviriyor.
Ölçülen kazanç: [METRICS.md §1.4](METRICS.md).

## Zaman kaybettiren tuzaklar (her biri bir ölçüm ya da testle yakalandı)

1. **Aşama 1'de yatay flip ve döndürme augmentasyonu yok.** Sağ/sol ve yönelim sinyalin
   kendisidir. (Aşama 2 yalnızca ayna simetrik koltuk altı tipinde kırpımları çevirir.)
2. **Sağ/sol'u asla duruş modelinden alma.** COCO duruş modelleri arkadan çekimde anatomik
   tarafı yalnızca ≈%12 doğru atar — rastgele değil, sistematik ters. Kural konumsaldır:
   kişinin sağı karede sağdadır.
3. **Eğitimdeki boyutlandırmayı birebir tekrarla.** Veri, kenar yumuşatmayla 640×640'a
   gerilmişti. `Bitmap.createScaledBitmap` aliasing bırakıp 0.29° hata ekledi; bu yüzden
   Pillow bilineer algoritması Kotlin'e portlandı (`OnIsleme.kucultBilinear`) ve
   Pillow'a karşı ±1 piksel doğrulandı.
4. **Açıyı orijinal çözünürlükte hesapla.** Germe açıları bozar; 4:3 bir fotoğrafta
   640×640 uzayında ölçmek ≈%25 hata yapar.
5. **Kanal sırası ve normalizasyon aşamaya göre farklı.** Ultralytics NumPy dizilerinde BGR
   varsayar; Aşama 1 `/255`, Aşama 2 ImageNet ortalama/std alır.
6. **`mAP` bu görevde anlamsız.** Ultralytics, COCO dışı keypoint'lerde OKS sigmalarını
   `1/nkpt` alır; 40 px hata bile 0.995 puan aldı. Piksel ve derece hatasına bakın.
7. **Kısmi etiketli landmark'larla eğitirken `kobj=0`.** Etiketsiz ama görünür noktalar
   `v=0` işaretlenir; görünürlük başlığı "yok" olarak öğrenirdi.
8. **IMU örneğini enterpole et, en yakınını seçme.** 19 ms'lik sensör periyodunda 30 °/s
   dönen bir el, kaçınılabilir ≈0.3° hata ekler.
9. **Dağınık ısı haritasında soft-argmax kendinden emin biçimde yanlış çıkar.** Güven
   kapısı bu sessiz hataları görünür reddedişe çevirir.

## Kotlin ve Python'u aynı tutmak

Ölçüm kodu, [`scripts/19_kilavuz_cizim.py`](../scripts/19_kilavuz_cizim.py) dosyasının
bilinçli, satır satır portudur. `android/app/src/test/resources/*.csv` içindeki beklenen
değerler elle yazılmaz, *Python referansı çalıştırılarak üretilir*
(`scripts/23_altin_veri.py`, `27`, `29`); böylece iki taraf aynı yanlış işaret
konvansiyonunu paylaşamaz. CI bunu iki uçtan denetler:

- Gradle, Kotlin testlerini CSV'lere karşı çalıştırır;
- `tests/test_golden_parity.py` aynı satırları Python referansından yeniden türetir.

## Neyin nerede olduğu

| Konu | Yol |
|---|---|
| Ölçüm matematiği (referans) | `scripts/19_kilavuz_cizim.py` |
| Ölçüm matematiği (port) | `android/app/src/main/java/com/skolyoz/olcum/olcum/Olcum.kt` |
| Cihaz üstü zincir | `android/.../cikarim/OlcumZinciri.kt`, `Asama1Coz.kt`, `OnIsleme.kt` |
| IMU halka tamponu ve enterpolasyon | `android/.../sensor/OrnekHalkasi.kt`, `YercekimiTakipci.kt` |
| Aşama 2 ağı | `scripts/39_rafinaj7_model.py` |
| Metrikler ve benchmark'lar | `skolyoz_metrics/`, `metrics/` |
