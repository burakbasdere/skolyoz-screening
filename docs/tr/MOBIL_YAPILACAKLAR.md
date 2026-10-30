# Skolyoz Ölçüm Uygulaması — Mobil Taraf Devir Dokümanı

> **Not.** Bu, orijinal çalışma belgesidir ve tarihsel bir kayıttır (2 noktalı
> dönemin sayıları dahil). Güncel sayılar [../METRICS.md](../METRICS.md) içindedir;
> sonradan düzeltilen noktalar [README.md](README.md) dosyasında listelenmiştir.

> Bu belge, projenin **görüntü işleme tarafı bittikten sonra** mobil geliştirmeye
> geçmek için yazıldı. Yeni bir oturum bu dosyayı okuyarak sıfırdan bağlam
> kurabilmeli. Python tarafı bu repoda `scripts/` altında duruyor.

---

## 1. Ürün ne yapıyor — ve ne yapmıyor

Kullanıcı, hastanın **arkadan, ayakta** fotoğrafını telefonla çeker. Uygulama
iki koltuk altı noktasını tespit eder, aralarındaki çizginin **yerçekimi
yatayına göre** eğimini hesaplar ve ekranda kılavuz çizgilerle gösterir.

### Kritik terminoloji uyarısı

Ölçtüğümüz şey **aksiller eğim açısı**dır — **Cobb açısı değildir.**

Cobb açısı tanımı gereği röntgende, en çok eğimli üst omurun üst plağı ile alt
omurun alt plağı arasındaki açıdır. Bizim ölçtüğümüz yüzey asimetrisidir.
Aralarında korelasyon vardır ama eşitlik yoktur:

- Kompanse lomber eğride omuzlar düz kalabilir → gerçek Cobb 30°, bizim açı ~2°
- Bacak boyu farkı / kas asimetrisinde → gerçek Cobb 0°, bizim açı 6°

**Elimizde röntgenle eşleşmiş Cobb ground truth yok**, dolayısıyla Cobb tahmini
iddiası doğrulanamaz. Arayüzde "Cobb açısı" yazılmamalı. Uygulamada
**"Tanı aracı değildir · tarama amaçlıdır"** ibaresi zorunlu.

İleride röntgen eşleşmeli veri toplanırsa, `cobb = f(aksiller_açı)` kalibrasyon
katmanı tek noktadan eklenebilecek şekilde tasarlanmalı — ölçüm çekirdeğinin
geri kalanı değişmemeli.

---

## 2. Mevcut durum — görüntü işleme tarafı

### Veri

Roboflow projesi `education-ovbua/bladeco v1`. Ham export 1119 dosyaydı ama:

| Bulgu | Sonuç |
|---|---|
| Rotasyon kopyaları (aynı foto 0/90/180/270°) | 1119 dosya → **441 benzersiz** |
| Sağ/sol etiket çelişkisi | kümelerin **%38.8**'inde |
| 640×640 **Stretch** resize | en-boy oranı bozulmuş, **kalıcı hasar** |
| Anotasyon tipi | keypoint değil, **bounding box** (nokta = kutu merkezi) |
| Kullanılamaz (tek taraf / çift kutu) | 74 görüntü |

Temizlik sonrası: **337 görüntü**, `data/final/` altında 236/51/50 olarak
train/val/test'e bölünmüş. Bölme **grup bazlı** (benzerlik 0.85 eşiğiyle
gruplanıp grup bölünmeden tek split'e verildi) — aksi halde aynı kişinin
kareleri train ve val'e dağılıp skoru şişiriyordu.

**Etiket gürültüsü ölçüldü:** kopya kümelerindeki bağımsız etiketlerden,
2.45 px / 0.435°. Ortalama alınarak 0.232°'ye düşürüldü ve 233/337 görüntü
ortalanmış etiket taşıyor. Bu, modelin inebileceği fiziksel tabandır.

### Modeller

İki aşamalı (top-down) boru hattı:

| | Model | Girdi | Çıktı | Parametre |
|---|---|---|---|---|
| **Aşama 1** | YOLO11n-pose (fine-tune) | 640×640 | 2 kaba nokta | 2.66 M |
| **Aşama 2** | MobileNetV3-small + deconv heatmap | 128×128 kırpım | 1 alt-piksel nokta | 0.41 M |

Aşama 2, heatmap üretip **soft-argmax** ile alt-piksel koordinat çıkarır.
Tek model iki taraf için de kullanılır (kırpım seviyesinde sol/sağ ayrımı yok,
eğitimde rastgele yatay flip uygulandı).

**Neden iki aşama:** tek başına YOLO-pose train/val/test'te aynı ~24 px hata
verdi. Ezberleme yoktu, yani sorun veri miktarı değil temsil biçimiydi —
stride 8/16/32 ızgarasından doğrudan koordinat regresyonu bu hassasiyete
yetmiyor. Heatmap + kırpma büyütmesi 6.9× kazandırdı.

### Ölçülen doğruluk (test, n=50)

| Metrik | Aşama 1 | **Aşama 1+2** | Taban |
|---|---|---|---|
| Keypoint hatası (medyan) | 24.35 px | **3.55 px** | 2.45 px |
| Keypoint hatası (ortalama) | 27.37 px | 7.83 px | — |
| Açı hatası (medyan) | 2.094° | **0.380°** | ~0.23° |
| Açı hatası (p90) | 4.742° | 1.706° | — |
| PCK@5px | %5 | **%72** | — |
| PCK@10px | %18 | **%90** | — |

Karar metrikleri:

| Karar | accuracy | makro-F1 | pozitif recall |
|---|---|---|---|
| Hangi omuz yüksek (3 sınıf) | 0.860 | 0.847 | — |
| Tarama \|açı\| ≥ 3° | 0.880 | 0.870 | 0.938 |
| Tarama \|açı\| ≥ 5° | 0.920 | 0.834 | 0.833 |
| Tarama \|açı\| ≥ 7° | 1.000 | 1.000 | 1.000 |

> ⚠️ **7°'deki 1.000 anlamsız** — test setinde o eşiğin üstünde sadece 4 vaka
> var. 5°'de 6 vaka. Test seti toplam 50 görüntü, güven aralıkları çok geniş.
> Bu sayıları pazarlama malzemesi yapma.

> ⚠️ Ortalama hata medyanın iki katı — noktaların **%10'u 10 px'in ötesinde**.
> Bir hata kuyruğu var, çoğunlukla eğilmiş/atipik pozlarda.

### Dosyalar

```
data/final/            temiz veri seti (GIT'E GİRMİYOR — hasta fotoğrafı)
runs/baseline/weights/best.pt    Aşama 1 ağırlığı
runs/rafinaj/best.pt             Aşama 2 ağırlığı
runs/export/                     ONNX + TFLite çıktıları (GIT'E GİRMİYOR)
scripts/17_rafinaj_model.py      Aşama 2 mimarisi
scripts/19_kilavuz_cizim.py      ölçüm matematiği + overlay (PORTLANDI)
scripts/20_tam_degerlendirme.py  tam boru hattı + metrikler (PyTorch)
scripts/22_onnx_export.py        her iki aşama → ONNX + doğrulama
scripts/23_altin_veri.py         Kotlin birim testleri için altın veri üretir
scripts/24_temsilci_veri.py      INT8 kalibrasyon veri seti (NHWC!)
scripts/25_tflite_donusum.py     ONNX → TFLite  (.venv-tflite içinde koşar)
scripts/26_tflite_degerlendirme.py  TFLite metrikleri (.venv-tflite içinde)
android/                         Android uygulaması (Faz 2 iskeleti)
```

---

## 3. Cihaz üstünde çalışacak boru hattı

```
   ┌─ CameraX: fotoğraf çek
   │  └─ AYNI ANDA: SensorManager'dan yerçekimi vektörü örnekle
   │                (timestamp'leri eşleştir!)
   ▼
[1] Ön işleme: orijinal foto (ör. 3000×4000) → 640×640'a GER (stretch)
   ▼
[2] Aşama 1 TFLite → 2 kaba nokta (640×640 uzayında)
   ▼
[3] Her nokta için 128×128 kırpım al → Aşama 2 TFLite → alt-piksel nokta
   ▼
[4] Noktaları ORİJİNAL çözünürlüğe geri map'le
   ▼
[5] Açıyı ORİJİNAL uzayda hesapla + IMU roll düzeltmesi
   ▼
[6] Pitch kontrolü: |pitch| > 8° ise ölçümü REDDET
   ▼
[7] Overlay çiz: sarı çizgi + kesikli yerçekimi yatayı + açı yayı + okuma
```

### Adım 1 hakkında — en ince nokta

Eğitim verisi 640×640'a **stretch** edilmişti (letterbox değil). Model bu
dağılımı gördü. Bu yüzden çıkarımda da **aynı stretch uygulanmalı**, letterbox
değil — yoksa dağılım dışına çıkarsın.

Ama **açıyı 640×640 uzayında hesaplama.** Stretch en-boy oranını bozduğu için
oradaki açı gerçek açı değildir. Doğru sıra:

1. Fotoğrafı 640×640'a ger, modeli çalıştır
2. Çıkan noktaları orijinal `(W, H)` çözünürlüğüne geri ölçekle:
   `x_orj = x_640 * W / 640`, `y_orj = y_640 * H / 640`
3. Açıyı **orijinal koordinatlarda** hesapla

Bu adımı atlarsan 4:3 bir fotoğrafta ~%25 açı hatası yaparsın.

---

## 4. Ölçüm matematiği (Kotlin'e portlanacak)

Referans uygulama: [`scripts/19_kilavuz_cizim.py`](../../scripts/19_kilavuz_cizim.py).
Saf geometridir, görüntü işleme içermez, birebir çevrilebilir.

```kotlin
// Telefonun roll açısı — IMU yerçekimi vektöründen
fun rollAcisi(gx: Float, gy: Float): Double =
    Math.toDegrees(atan2(gx.toDouble(), gy.toDouble()))

// Aksiller eğim açısı. Pozitif = kişinin SAĞ omzu daha yüksek.
// sag/sol: orijinal görüntü pikselinde, y aşağı doğru artar.
fun aksillerAci(sag: PointF, sol: PointF, rollDerece: Double): Double {
    val dx = (sol.x - sag.x).toDouble()
    val dy = (sol.y - sag.y).toDouble()
    if (abs(dx) < 1e-9 && abs(dy) < 1e-9) return 0.0
    val ham = Math.toDegrees(atan2(dy, abs(dx)))
    return ham - rollDerece      // görüntü yatayını gerçek yataya çevir
}

// Telefon öne/arkaya yatıkken perspektif açıyı bozar → çekimi reddet
fun olcumGecerliMi(pitchDerece: Double, esik: Double = 8.0) =
    abs(pitchDerece) <= esik
```

### Konvansiyonlar — bunları değiştirme

| | |
|---|---|
| Çekim yönü | **Arkadan** (hastanın sırtı kameraya dönük) |
| Nokta sırası | model çıktısında **index 0 = sağ**, **index 1 = sol** |
| Sağ/sol kuralı | arkadan çekimde kişinin sağı **karede sağdadır** |
| Açı işareti | **pozitif = sağ omuz yüksek** |
| Koordinat | y **aşağı** doğru artar (görüntü konvansiyonu) |

---

## 5. Kritik tuzaklar — bunlar bize zaman kaybettirdi

### 5.1 Yatay flip YASAK

Sağ/sol koltuk altı **anlam taşıyor**. Yatay flip etiketi ters çevirir.
Eğitimde `fliplr=0.0` kullanıldı. Mobil tarafta da hiçbir aşamada aynalama
yapma. **Ön kamera görüntüyü aynalar** — arka kamera kullan, ya da aynalamayı
açıkça geri al.

### 5.2 Rotasyon augmentasyonu YASAK

Ölçtüğümüz şey oryantasyonun kendisi. Rotasyon augmentasyonu modele
"oryantasyonu yoksay" diye öğretir — yani cevabı yok eder. Veri setindeki
2.54× şişme tam olarak bu hatadan geliyordu.

### 5.3 Pose modelleri arkadan çekimde sol/sağı ters biliyor

COCO pose modelleri ağırlıklı olarak öne bakan insanla eğitilmiş. Ölçüldü:
arkadan çekimde YOLO11-pose'un anatomik sol/sağ ataması **sadece %11.7 doğru**
— rastgele değil, sistematik ters. Sol/sağ'ı **asla pose modelinden alma**,
konum kuralından al (karede sağdaki = kişinin sağı).

### 5.4 RGB / BGR

Ultralytics'e numpy dizisi verirsen **BGR varsayar** (OpenCV geleneği). PIL'den
gelen RGB diziyi doğrudan vermek keypoint hatasını 24 px'den 40 px'e çıkardı.
Android tarafında da Bitmap → tensör dönüşümünde kanal sırasını modelin
beklediğiyle eşleştir ve **birim testle doğrula**.

### 5.5 mAP bu görevde anlamsız

Ultralytics COCO dışı keypoint sayılarında OKS sigmasını `1/nkpt` alıyor —
2 nokta için 0.5, COCO'nun 0.025–0.1 aralığının ~10 katı. Metrik o kadar
hoşgörülü ki 40 px hatayla bile `mAP50-95 = 0.995` veriyordu. **Piksel hatası
ve derece cinsinden açı hatasına bak**, mAP'a değil.

### 5.6 Normalizasyon

Aşama 2 ImageNet normalizasyonu bekliyor:
`mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`, girdi `[0,1]`
aralığına bölünmüş. Aşama 1 (YOLO) ise sadece `/255`. İkisini karıştırma.

---

## 6. Yapılacaklar

### Faz 1 — Model dönüşümü ve doğrulama ✅ BİTTİ

- [x] **Aşama 1'i TFLite'a çevir.** `scripts/22_onnx_export.py` → ONNX,
      `scripts/25_tflite_donusum.py` → TFLite (onnx2tf).
- [x] **Aşama 2'yi TFLite'a çevir.** Soft-argmax **sorunsuz dönüştü** — ne
      ONNX'te ne TFLite'ta sorun çıktı. Heatmap yedeği de üretildi ama
      gerekmedi: ikisi arasında ölçülebilir fark yok (0.417° vs 0.416°).
      Tek çıktılı `koord` varyantı kullanılıyor, Kotlin'de elle argmax yok.
- [x] **Quantization kaybını ölç.** `scripts/26_tflite_degerlendirme.py`,
      aynı test setinde (n=50), aynı metriklerle.
- [x] Model boyutlarını ölç. **6.40 MB** — hedefin altında.

#### Sonuçlar (test seti, n=50)

| Varyant | MB | px medyan | **açı medyan** | omuz acc |
|---|---|---|---|---|
| **float16 ← SEÇİLEN** | **6.40** | 3.44 | **0.417°** | 0.860 |
| dynamic_range_quant | 3.59 | 3.80 | 0.463° | 0.820 |
| float32 | 12.57 | 3.45 | 0.412° | 0.860 |
| full_integer_quant (INT8) | 3.58 | — | — | — |
| INT8 + int16 aktivasyon | 3.79 | 35.10 | 1.644° | 0.420 |
| PyTorch (referans) | — | 3.55 | 0.380° | 0.860 |

**Neden float16:** pratikte kayıpsız (float32'den farkı 0.005°) ve boyut
hedefinin altında. `dynamic_range` daha küçük ve açı kriterini geçiyor ama
ürünün asıl karar metriğini bozuyor: hangi omzun yüksek olduğu doğruluğu
0.860 → 0.820. Boyut sıkıntısı yokken bu takas mantıksız.

#### INT8 neden kullanılamıyor

`full_integer_quant` **50/50 görüntüde hiç tespit üretmedi.** Sebep ölçüldü:

```
cikti quantization: olcek=3.5909  sifir=-75
```

YOLO'nun tek çıktı tensörü hem kutu koordinatlarını (0–640) hem güven
değerini (0–1) taşıyor. INT8'de o tensöre **tek bir ölçek** atanıyor;
koordinatları temsil edebilmek için ölçek ~3.6 seçiliyor ve 0–1 aralığındaki
güven yalnızca 0.3 kuantizasyon adımına düşüyor — hiçbir aday 0.25 eşiğini
geçemiyor.

INT8 istenirse tek yol modelin çıktı başını ayırmak (koordinatlar ve güven
ayrı tensörler, ayrı ölçekler). Şu an gerek yok: float16 zaten hedefleri
karşılıyor.

#### Yol boyunca çıkan üç sessiz hata

Üçü de hata vermeden yanlış sonuç üretiyordu, kayıt için:

1. **`17_rafinaj_model.py` CPU'da yüklenmiyordu.** `expand()` ile üretilen
   izgara tamponlarının stride'ı 0; `load_state_dict` kopyalayamıyor.
   CUDA'da fark edilmiyordu çünkü `.to("cuda")` tamponu bitişikleştiriyor.
   Mobil dönüşüm CPU'da yapıldığı için burada patladı. `.contiguous()` eklendi.
2. **INT8 dönüşümü sessizce düşüyordu.** Kalibrasyon verisi NCHW'ydi;
   onnx2tf onu ONNX'e değil dönüştürülmüş NHWC TFLite modeline besliyor
   (`128 % 3 = 2` hatası). `non_verbose=True` hatayı yutuyordu, sadece float
   dosyalar üretilmişti. Dosya listesine bakıp "INT8 hazır" varsaymak
   quantization kaybını sıfır sanmamıza yol açardı.
3. **`19_kilavuz_cizim.py`'de pitch fonksiyonu yoktu.** Doküman pitch'i
   IMU'dan gelen hazır bir değer varsayıyordu; Kotlin portunun
   karşılaştırılacağı referans yoktu. `pitch_acisi()` eklendi.

#### Üretilen dosyalar

```
android/app/src/main/assets/asama1.tflite   5.52 MB  (git'e GİRMİYOR)
android/app/src/main/assets/asama2.tflite   0.88 MB  (git'e GİRMİYOR)
```

### Faz 2 — Android iskelet ✅ BİTTİ

Proje `android/` altında. Gerçek cihazda doğrulandı: **Redmi Note 9 Pro,
Android 12 (API 31)**.

- [x] Kotlin projesi, min SDK 26, CameraX. (TFLite bağımlılığı Faz 3'te
      eklenecek — modeller `app/src/main/assets/` içinde hazır bekliyor.)
- [x] **CameraX `ImageCapture`** — arka kamera zorunlu (`DEFAULT_BACK_CAMERA`),
      aynalama yok.
- [x] **SensorManager `TYPE_GRAVITY`** — cihazda donanımsal olarak var,
      ivmeölçer yedeğine düşülmedi.
- [x] Canlı seviye göstergesi: yapay ufuk (roll) + dikey cetvel (pitch).
- [x] `|pitch| > 8°` iken deklanşör kilitli.

#### Eşleştirmede yapılan değişiklik: en yakın örnek → enterpolasyon

Doküman "deklanşör anındaki örneği timestamp'e göre eşleştir" diyordu. İlk
uygulama en yakın örneği seçiyordu. Gerçek cihazda ölçüldü: sensör periyodu
**19.1 ms**, deklanşörle en yakın örnek arası sapma **2.3 ms ve 7.6 ms**.

En yakın örneği seçmek ±9.5 ms'lik **yapısal** hata bırakıyor. Deklanşöre
basarken el ~30°/s dönebiliyor; 10 ms'de bu 0.3° eder — boru hattının açı
hatası medyanı 0.380°, etiket gürültüsü tabanı 0.23°. Yani zincirin en hassas
halkasıyla aynı büyüklükte hata ekleniyordu.

Çözüm: deklanşörü **kuşatan iki örnek arasında doğrusal enterpolasyon**.
Hata örnekleme periyoduna değil, yerçekimi vektörünün iki örnek arasındaki
doğrusallıktan sapmasına iniyor. Birim testle ölçüldü: 30°/s dönüşte
0.30° → <0.01°.

İki durumda enterpolasyon yapılmıyor ve ölçüm reddediliyor: örnekleme boşluğu
80 ms'i aşarsa (uygulama arka plandaydı), ve `onPause`'da tampon siliniyor.

#### Deklanşör gecikmesi

Gerçek cihazda ölçüldü ve kademe kademe düşürüldü:

| Değişiklik | Toplam |
|---|---|
| Başlangıç (MAXIMIZE_QUALITY, ana thread'de JPEG çözme) | ~1100 ms |
| JPEG çözme arka plana + `inSampleSize` ile küçülterek çözme | ~950 ms |
| `CAPTURE_MODE_ZERO_SHUTTER_LAG` | **588 ms** |

Son ölçümün dağılımı: `basış→yakalama −204 ms` (negatif — ZSL tampondan
basıştan önceki kareyi veriyor), `yakalama→teslim 674 ms` (JPEG kodlama),
`çözme 118 ms`.

Çekim çözünürlüğü 12 MP → ~3 MP düşürüldü; JPEG kodlama süresini azaltması
bekleniyor ama **bu ölçüm doğrulanmadı**. Ölçüme etkisi yok: Aşama 1 girdisi
640×640, Aşama 2 o uzayda 128×128 kırpım alıyor.

> Deklanşör gecikmesi ölçüm doğruluğunu bozmuyor: yerçekimi karenin kendi
> donanım damgasına göre enterpole ediliyor. Gecikme yalnızca kullanım hissi
> meselesi.

#### Seri çekim (burst) — yazıldı, kapalı

`SERI_KARE_SAYISI` sabiti şu an **1**. Çok kareli çekim + **medyan** (ortalama
değil — hata kuyruğunu elemek için) hata kuyruğunu bastırmak için cazip, ama
kare başına ~600 ms gecikme var ve medyan yalnızca **bağımsız** gürültüyü
bastırır. Arka arkaya çekilen kareler neredeyse aynı görüntüdür; model üçünde
de aynı hatayı yapabilir.

Kazanç telefonda ölçülemez (ground truth yok). Ölçüm Python'da, 50 görüntülük
test setinde yapılmalı. Olumluysa tek sabiti değiştirmek yeterli — çekim ve
birleştirme yapısı hazır.

### Faz 3 — Çıkarım zinciri ✅ BİTTİ

- [x] Bitmap → 640×640 stretch → Aşama 1 → 2 kaba nokta
- [x] Her nokta etrafından 128×128 kırpım (kenarda **siyah** padding — PIL'in
      davranışı; Aşama 2 o dağılımla eğitildi)
- [x] Aşama 2 → alt-piksel nokta → orijinal çözünürlüğe geri map
- [x] `aksillerAci()` + roll düzeltmesi
- [x] **Birim testler.** Gerçek telefon fotoğrafında, aynı 640×640 tensörle:
      **0.000 px / 0.0000°**. Şart ≤0.05° idi.

Zincirin saf parçaları `Bitmap` yerine `IntArray` üzerinde çalışıyor; §5'teki
tuzakların çoğu (kanal sırası, iki aşamanın farklı normalizasyonu, kırpım
kaydırması ve yuvarlaması) orada yaşıyor ve düz JVM testiyle kapsanıyor.
38 test, beklenen değerlerin tamamı Python'dan üretiliyor.

#### Küçültme kendi başına bir hata kalemiydi

`Bitmap.createScaledBitmap` düz bilineer örnekleme yapıp **aliasing**
bırakıyor. PIL küçültürken üçgen filtrenin desteğini ölçek kadar genişletiyor.
Gerçek fotoğrafta ölçüldü: aradaki fark açıya **0.29°** yansıyordu — hata
bütçesinin yarısı kadar, sırf ölçekleme algoritmasından.

Pillow'un algoritması Kotlin'e portlandı (`OnIsleme.kucultBilinear`), PIL'e
karşı ±1 piksel doğrulandı. Hangisinin doğru olduğu keyfi değil: eğitim
verisi Roboflow'da antialiasing'li olarak 640×640'a gerildi, model o dağılımı
gördü.

### Faz 3.5 — Güven kapısı ⚠️ YENİ, ÜRÜN AÇISINDAN KRİTİK

Gerçek telefon fotoğraflarında aynı sahnenin üç karesi **1.25°, 3.37°, 6.73°**
verdi. Kök neden arandı:

| Kontrol | Sonuç |
|---|---|
| Kotlin ↔ Python, aynı 640 girdi | 0.000 px — kod doğru |
| Küçültme ↔ PIL | max 1/255 — kod doğru |
| **Aşama 1** kararlılığı | 0.4–2.5 px — sağlam |
| **Aşama 2** kararlılığı | **16.8 px'e kadar** — kararsız |

Sebep: soft-argmax dağılımın **ağırlık merkezini** alıyor. Dağılım dağınıksa
merkez, girdideki 1/255'lik farkla bile onlarca piksel kayıyor. Model bu
durumda hata vermiyor — kendinden emin görünen bir sayı üretiyor.

Aşama 2 heatmap entropisi:

| | test+val (n=202) | telefon (n=6) |
|---|---|---|
| Entropi | p50 1.698 · p95 2.043 · p99 2.908 | 1.88 … 6.92 |
| Tepe–merkez uzaklığı | p50 6.57 · p95 11.16 px | 5.72 … 51.64 px |

Telefon ölçümleri doğrulanmış dağılımın dışında. Bu, §7'de öngörülen
**domain gap**.

**Yapılan:** Aşama 2 heatmap varyantına geçti (doğruluk aynı: 0.416° vs
0.417°), soft-argmax Kotlin'e taşındı, iki ölçütlü kapı eklendi:

```
entropi > 2.5           -> dağınık heatmap
tepe uzaklığı > 15 px   -> soft-argmax gerçek tepenin üstünde değil
```

İkinci ölçüt şart: iki **keskin** tepeli heatmap'te entropi düşük çıkıyor
(~0.72) ama soft-argmax ikisinin ortasını veriyor ve o nokta hiçbir tepenin
üstünde değil. Entropinin kör noktası bu; birim testi yakaladı.

Ölçülen maliyet: test+val'de 101 ölçümün 5'i reddediliyor (**%5.0**), kararsız
telefon ölçümlerinde **3/3**.

> Kapı modeli İYİLEŞTİRMİYOR. Yaptığı tek şey sessizce yanlış olanı görünür
> biçimde reddedilen hâle çevirmek. Asıl çözüm §7'deki birinci madde:
> telefon kamerasıyla çekilmiş eğitim verisi.

**Açık:** kapı gerçek çekimle henüz denenmedi.

### Faz 4 — Kılavuz görselleştirme

`scripts/19_kilavuz_cizim.py::kilavuz_ciz` fonksiyonunu Canvas'a çevir:

- [ ] Sarı koltuk altı çizgisi + R (kırmızı) / L (mavi) nokta işaretleri
- [ ] **Kesikli yerçekimi yatayı** — görüntünün yatayı değil, roll kadar döndürülmüş
- [ ] Orta noktada açı yayı (yataydan çizgiye)
- [ ] Okuma paneli: `"7.5° sağ omuz yüksek"` + altında
      `"aksiller eğim · IMU roll +0.0° düzeltildi"`
- [ ] Pitch reddi durumunda kırmızı panel: `"ÖLÇÜM GEÇERSİZ"`
- [ ] Alt şerit: `"Tanı aracı değildir · tarama amaçlıdır"`

Referans görsel: `runs/kilavuz/urun_ciktisi.jpg`

### Faz 5 — Ürün katmanı

- [ ] Ölçüm kaydı (yerel DB), tarih + açı + fotoğraf
- [ ] İlerleme grafiği (aynı hastanın zaman içindeki değişimi)
- [ ] **Gizlilik**: fotoğraflar cihazdan çıkmayacak. Sunucu yok, analytics'e
      görsel gönderme yok. Bu tıbbi veri.
- [ ] Kalibrasyon katmanı için boş kanca (`cobb = f(aksillerAci)`) — ileride
      röntgen eşleşmeli veri gelirse tek noktadan takılsın.

---

## 7. Bilinen sınırlar

| Sınır | Detay |
|---|---|
| **Stretch hasarı kalıcı** | Orijinal fotoğraflar kayıp; eğitim verisinin en-boy oranı geri getirilemez |
| **Cobb ground truth yok** | Çıktı aksiller eğim açısı olarak kalmalı |
| **Test seti küçük** | 50 görüntü; 5°+ eşiklerde 4–6 pozitif vaka |
| **Domain gap** | Eğitim setinde stok fotoğraflar, gri tonlama klinik görüntüler, illüstrasyonlar var. Ürün sadece renkli telefon fotoğrafı görecek |
| **Hata kuyruğu** | Noktaların %10'u >10 px sapıyor, çoğunlukla eğilmiş/atipik pozlarda |
| **Tek poz** | Sadece ayakta, arkadan çekim. Öne eğilme (Adam testi) desteklenmiyor |

### Doğruluğu artırmak istersen — etki sırasına göre

1. **Yeni veri topla**: telefonla, ayakta, arkadan, tam çözünürlük, **stretch
   yok**. 200–300 temiz görüntü mevcut setin tamamından değerlidir.
2. **Röntgen eşleşmeli veri**: Cobb kalibrasyonunu açar, ürünü tarama
   aracından ölçüm aracına taşır.
3. **Gerçek keypoint etiketi**: mevcut noktalar kutu merkezi. Ölçüldü, tutarlı
   (2.45 px), yani acil değil — ama tavan burası.

---

## 8. Bu repoyu kullanmak

```bash
# Ortam (RTX 50xx / Blackwell için cu128 ŞART, düz pip install torch çalışmaz)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics
```

```bash
# Tam boru hattını test setinde değerlendir
python scripts/20_tam_degerlendirme.py
```

```bash
# Ürün çıktısı görsellerini üret
python scripts/21_urun_ciktisi.py
```

> `data/` ve `runs/` **git'e girmiyor** (`.gitignore`). Hasta fotoğrafları
> kişisel sağlık verisidir ve bu depo herkese açıktır. Model ağırlıklarını
> mobil projeye elle kopyala, commit etme.
