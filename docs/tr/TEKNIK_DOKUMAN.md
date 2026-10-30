# Akıllı Telefon Fotoğrafından Radyasyonsuz Gövde Asimetrisi Ölçümü: Derin Öğrenme Tabanlı Bir Tarama Aracı

**Teknik Doküman**

> **Not.** Bu, orijinal çalışma belgesidir. Güncel ve kaynağı izlenebilir sayılar
> [../METRICS.md](../METRICS.md) içindedir; bu belgede sonradan düzeltilen
> noktalar için [README.md](README.md) dosyasına bakın (özellikle §6.2 karar
> tablosu ve §6.6 test sayısı).

---

## 1. ÖZET

Adölesan idiyopatik skolyoz (AİS), okul çağı çocuklarında %1–3 yaygınlıkla
görülen ve erken tespit edildiğinde konservatif yöntemlerle ilerlemesi
durdurulabilen bir omurga deformitesidir. Altın standart tanı yöntemi olan
Cobb açısı ölçümü, ayakta ön-arka radyografi gerektirmekte; tarama amaçlı
tekrarlı radyografi ise özellikle büyüme çağındaki bireylerde kümülatif
radyasyon maruziyeti nedeniyle sakıncalı bulunmaktadır.

Bu çalışmada, hastanın arkadan çekilmiş tek bir akıllı telefon fotoğrafından
**aksiller eğim açısı** ve **koronal denge** ölçen, tamamen cihaz üzerinde
çalışan iki aşamalı bir derin öğrenme sistemi geliştirilmiştir. Sistem,
telefonun ataletsel ölçüm birimi (IMU) verisini kullanarak kamera eğikliğini
yerçekimi referansına göre düzeltmekte, böylece çekim açısından kaynaklanan
sistematik hatayı gidermektedir.

Geliştirilen boru hattı, 337 görüntülük temizlenmiş veri kümesi üzerinde
**3.045 piksel** medyan landmark lokalizasyon hatası ve **0.271°** medyan açı
hatası elde etmiştir. Bu değer, aynı veri kümesi üzerinde bağımsız
etiketlemelerden ölçülen **2.45 piksel / 0.435°** etiket gürültüsü tabanına
karşılık gelmekte olup, modelin mevcut veri kalitesinin izin verdiği üst
sınıra ulaştığını göstermektedir.

Çalışmanın metodolojik katkısı, ölçüm belirsizliğinin sistematik olarak
niceliklendirilmesidir: etiket gürültüsü doğrudan ölçülmüş, k-katlı çapraz
doğrulama ile model başarımının doğal oynaklığı (2.1 piksel) belirlenmiş ve
bu bant altındaki iyileştirme iddialarının istatistiksel olarak anlamsız
olduğu gösterilmiştir.

**Anahtar kelimeler:** adölesan idiyopatik skolyoz, radyasyonsuz tarama,
landmark tespiti, yüzey topografisi, mobil derin öğrenme, ölçüm belirsizliği

---

## 2. AMAÇ VE HEDEFLER

### 2.1 Genel amaç

Okul taraması ve evde takip senaryolarında kullanılabilecek, radyasyon
içermeyen, düşük maliyetli ve klinisyen bağımsız bir gövde asimetrisi ölçüm
aracı geliştirmek.

### 2.2 Özel hedefler

| No | Hedef | Durum |
|----|-------|-------|
| H1 | Sırt fotoğrafından anatomik landmark tespiti | **Tamamlandı** |
| H2 | IMU tabanlı yerçekimi referanslı açı hesabı | **Tamamlandı** |
| H3 | Ölçüm belirsizliğinin niceliklendirilmesi | **Tamamlandı** |
| H4 | Cihaz üstü (offline) çalışan mobil uygulama | **Tamamlandı** |
| H5 | Omurga eğrilik profilinin çıkarılması | **Ulaşılamadı** (bkz. §6.3) |
| H6 | Cobb açısı kalibrasyonu | Veri kısıtı nedeniyle ertelendi |

---

## 3. KONU VE KAPSAM

### 3.1 Ölçülen büyüklüklerin tanımı

Sistemin ürettiği ölçümler ve tanımları aşağıda verilmiştir. **Bu ölçümlerin
hiçbiri Cobb açısı değildir**; yüzey asimetrisi göstergeleridir.

**Aksiller eğim açısı.** Sol ve sağ aksiller bölge landmarklarını birleştiren
doğrunun, yerçekimi yatayına göre yaptığı açı. Pozitif değer sağ omuzun
yüksek olduğunu gösterir.

```
θ_aksiller = atan2(Δy, |Δx|) − φ_roll
```

Burada `φ_roll`, IMU yerçekimi vektöründen hesaplanan telefon roll açısıdır:
`φ_roll = atan2(g_x, g_y)`.

**Koronal denge.** C7 vertebra prominens landmarkının, sakrumdan indirilen
yerçekimi şakülüne göre yatay sapması. Klinik muayenedeki C7 şakül testinin
karşılığıdır.

```
d_koronal = (P_C7 − P_sakrum) · û_yatay
θ_koronal = atan2(d_koronal, h_gövde)
```

### 3.2 Kapsam dışı

- Radyografik Cobb açısı ölçümü veya tahmini
- Tanı koyma; sistem yalnızca tarama ve yönlendirme amaçlıdır
- Öne eğilme (Adams) testi ve gövde rotasyon ölçümü

---

## 4. LİTERATÜR ÖZETİ VE ÖZGÜN DEĞER

### 4.1 Literatürdeki konum

Radyasyonsuz skolyoz taraması literatüründe üç ana yaklaşım bulunmaktadır:

**(a) Yüzey topografisi.** Rasterstereografi ve benzeri optik yöntemler sırt
yüzeyinin üç boyutlu haritasını çıkarır. Yüksek doğruluk sağlar ancak özel
donanım gerektirir ve saha taraması için maliyetlidir.

**(b) Skolyometre / ATR ölçümü.** Öne eğilme testinde gövde rotasyon açısı
ölçülür. Cobb açısıyla ilişkisi kurulmuştur ancak uygulayıcı bağımlılığı
yüksektir.

**(c) Görüntü tabanlı derin öğrenme.** Yakın dönemde sırt fotoğrafından
otomatik değerlendirme yapan sistemler yayımlanmıştır. Yang ve ark. (2019)
sırt görüntülerinden skolyoz taraması için derin öğrenme algoritmaları
geliştirmiş ve doğrulamıştır. Daha yakın tarihli bir çalışmada, koşullu
çekişmeli üretici ağ (cGAN) destekli bir sistem ile tek akıllı telefon
fotoğrafından radyasyonsuz skolyoz değerlendirmesi yapılmıştır.

### 4.2 Kullanılabilir veri kümeleri

| Veri kümesi | İçerik | Etiket | Erişim |
|---|---|---|---|
| USTC & SYSU-Scoliosis | 1.898 sırt fotoğrafı / 1.067 hasta | Radyografiden uzman ölçümlü **Cobb açısı** (0–173°) | Talep üzerine |
| Scoliosis1K | 1.493 yürüme videosu, 447.000 kare / 1.050 birey | Üç sınıflı (pozitif/sınırda/negatif) | Sözleşmeli, akademik |
| AASCE MICCAI 2019 / SpineWeb | 609 ön-arka radyografi | 17 vertebra × 4 köşe = 68 landmark | Açık |

### 4.3 Özgün değer

Bu çalışmanın literatüre katkısı üç başlıkta toplanmaktadır:

**(i) Ölçüm belirsizliğinin sistematik niceliklendirilmesi.** Landmark tabanlı
ölçüm çalışmalarında etiket gürültüsü genellikle raporlanmamaktadır. Bu
çalışmada, veri kümesindeki yinelenen görüntülerin bağımsız etiketlemelerinden
etiket gürültüsü doğrudan ölçülmüş (2.45 piksel / 0.435°) ve model başarımı
bu tabana göre yorumlanmıştır. Ayrıca leave-one-out yaklaşımıyla, yeniden
etiketleme yapmaksızın omurga landmarklarının gürültü tabanı tahmin
edilmiştir (5.58 piksel).

**(ii) IMU tabanlı yerçekimi referanslama.** Fotoğraf tabanlı açı ölçümlerinde
kamera eğikliği doğrudan sistematik hataya dönüşmektedir. Çalışmada, çekim
anındaki IMU yerçekimi vektörü kare zaman damgasıyla eşleştirilerek görüntü
yatayı gerçek yataya dönüştürülmekte; ayrıca perspektif bozulmasına yol açan
pitch açısı eşiği aşıldığında ölçüm reddedilmektedir.

**(iii) Tamamen cihaz üstü çalışma.** Hasta görüntüsü cihazdan çıkmamaktadır.
Modeller TensorFlow Lite formatına (float16, toplam 6.40 MB) dönüştürülmüş,
Kotlin ölçüm çekirdeğinin Python referans uygulamasıyla **bit düzeyinde**
eşleştiği birim testleriyle doğrulanmıştır (0.000 piksel / 0.0000° fark).

---

## 5. YÖNTEM

### 5.1 Veri kümesi ve hazırlığı

Başlangıç veri kümesi 1.119 etiketli görüntüden oluşmaktaydı. Yapılan veri
sağlığı incelemesinde aşağıdaki sorunlar tespit edilmiştir:

| Bulgu | Etki |
|---|---|
| Döndürme yinelemeleri (aynı fotoğrafın 0°/90°/180°/270° kopyaları) | 1.119 dosya → **441 benzersiz görüntü** |
| Sağ/sol etiket çelişkisi | Kümelerin **%38,8**'inde |
| 640×640 boyutlandırmada en-boy oranının korunmaması | Geometrik bozulma (geri döndürülemez) |
| Eksik veya yinelenmiş anotasyon | 74 görüntü kullanım dışı |

Temizleme işlemleri:

1. **Yineleme tespiti.** 64×64 gri ölçek imzalar üzerinden dört rotasyonda
   korelasyon hesaplanmış, birleşim-bul algoritmasıyla kümelenmiştir.
2. **Kanonik yönlendirme.** Görüntü yönelimi, önceden eğitilmiş bir insan
   duruş kestirim modeliyle belirlenmiştir. El yapımı sezgisel yöntemler
   (ten rengi ağırlık merkezi, siluet genişlik profili) yetersiz kalmıştır
   (§6.3).
3. **Taraf ataması.** Duruş kestirim modelinin anatomik sol/sağ ataması
   arkadan çekimlerde yalnızca **%11,7** doğru bulunmuş; taraf ataması
   deterministik geometrik kurala (arkadan çekimde bireyin sağı karede
   sağdadır) dayandırılmıştır.
4. **Etiket ortalama.** Yineleme kümelerindeki bağımsız etiketler ortalanarak
   gürültü **0.435° → 0.232°** düzeyine indirilmiştir.
5. **Sızıntısız bölme.** Benzerlik eşiği 0,85 ile gruplanan görüntüler,
   gruplar bölünmeden eğitim/doğrulama/test kümelerine dağıtılmıştır.

Nihai veri kümesi: **337 görüntü**.

### 5.2 Etiket gürültüsünün ölçülmesi

Model başarımının yorumlanabilmesi için etiket gürültüsünün bilinmesi
zorunludur; model bu tabanın altına inemez.

**Aksiller landmarklar için:** yineleme kümelerinin her üyesi bağımsız olarak
etiketlendiğinden, aynı fotoğrafın *n* bağımsız etiketi mevcuttur. Bu
etiketlerin saçılımı doğrudan gürültü ölçüsüdür.

**Omurga landmarkları için:** yineleme bulunmadığından dolaylı yöntem
kullanılmıştır. Anatomik olarak omurga düzgün bir eğri oluşturur; zikzak
yapmaz. Her iç landmark dışarıda bırakılarak kalan dört noktadan parabol
uydurulmuş, dışarıda bırakılan noktanın bu eğriden sapması ölçülmüştür. Elde
edilen değer (5.58 piksel) bir **üst sınırdır**; gerçek eğriliğin paraboldan
sapması da kalıntıya karışmaktadır.

### 5.3 Model mimarisi

Sistem iki aşamalıdır. Tek aşamalı bir modelin yetersizliği ölçümle
gösterilmiştir: doğrudan koordinat regresyonu yapan model eğitim, doğrulama
ve test kümelerinde **aynı ~24 piksel** hatayı vermiş, yani kendi eğitim
verisini dahi öğrenememiştir. Bu durum, darboğazın veri miktarı değil temsil
biçimi olduğunu göstermektedir.

**Aşama 1 — Kaba lokalizasyon.**
YOLO11n-pose mimarisi, 7 landmark için ince ayarlanmıştır (2,66 M parametre,
640×640 giriş). Kısmi etiketli görüntüler COCO görünürlük bayrağı (`v=0`) ile
maskelenmekte, böylece yalnızca aksiller etiketi bulunan 155 görüntü de
eğitime katkı sağlamaktadır.

> **Not:** Ultralytics kütüphanesi maskeli koordinat kaybının yanı sıra ayrı
> bir görünürlük başlığı eğitmektedir. Etiketlenmemiş ancak görüntüde mevcut
> olan landmarklar `v=0` işaretlendiğinden bu başlık hatalı öğrenmektedir; bu
> nedenle ilgili kayıp terimi devre dışı bırakılmıştır (`kobj=0`).

**Aşama 2 — Alt-piksel rafinaj.**
MobileNetV3-small omurgası üzerine ters konvolüsyon başlığı eklenerek 128×128
kırpım üzerinde 64×64 ısı haritası üretilmekte, **soft-argmax** ile alt-piksel
koordinat çıkarılmaktadır (0,41 M parametre).

Yedi landmark, dört görsel tipe indirgenmiştir: aksiller (ayna simetrik, sol
ve sağ ortak), C7, omurga (üç nokta ortak), sakrum. Kayıp fonksiyonu yalnızca
örneğe ait tipin kanalından hesaplanmaktadır.

**Kazanç:** Aşama 1 tek başına 24,35 piksel; iki aşama birlikte **3,55 piksel**
(6,9 kat iyileşme).

### 5.4 Ölçüm ve mobil entegrasyon

Ölçüm matematiği görüntü işleme içermeyen saf geometri olarak ayrılmış ve
Kotlin'e birebir aktarılmıştır. Cihaz üstü boru hattı:

```
Çekim (CameraX) + eşzamanlı IMU örneklemesi
  → 640×640 ön işleme
  → Aşama 1 TFLite: kaba landmarklar
  → Aşama 2 TFLite: her landmark çevresinden kırpım, alt-piksel rafinaj
  → Orijinal çözünürlüğe geri dönüşüm
  → Açı hesabı + IMU roll düzeltmesi
  → Pitch eşiği kontrolü (|pitch| > 8° → ölçüm reddi)
  → Kılavuz çizim ve okuma
```

**Kritik ayrıntı:** Eğitim verisi en-boy oranı korunmadan boyutlandırıldığından,
çıkarımda da aynı dönüşüm uygulanmalı; ancak açı **orijinal çözünürlükte**
hesaplanmalıdır. Bu adımın atlanması 4:3 en-boy oranlı bir görüntüde yaklaşık
%25 açı hatasına yol açmaktadır.

### 5.5 Doğrulama yöntemi

Sabit bölme ile elde edilen test kümesi (34–50 görüntü) istatistiksel çıkarım
için yetersiz bulunmuştur. **5 katlı çapraz doğrulama** uygulanmış; katlara
görüntü değil benzerlik grupları atanarak sızıntı önlenmiştir. Her katın kendi
doğrulama parçası bulunduğundan model seçimi test kümesinden bağımsız
kalmaktadır.

Bu yaklaşım ölçüm örneklemini **34'ten 1.582'ye** çıkarmıştır.

---

## 6. BULGULAR

### 6.1 Landmark lokalizasyon başarımı

5 katlı çapraz doğrulama, n = 1.582 örnek:

| Landmark tipi | n | Medyan hata | p90 | Etiket tabanı |
|---|---|---|---|---|
| Aksiller | 674 | **3,045 px** | 7,518 px | 2,45 px |
| C7 | 182 | 11,772 px | 31,553 px | — |
| Sakrum | 180 | 11,345 px | 25,354 px | — |
| Omurga | 546 | 20,489 px | 41,958 px | — |

Aksiller landmarklar etiket gürültüsü tabanına ulaşmıştır; bu noktalarda
model kaynaklı iyileştirme payı kalmamıştır.

### 6.2 Açı ölçüm başarımı

| Ölçüm | Medyan hata | p90 | Sinyal büyüklüğü |
|---|---|---|---|
| Aksiller eğim | **0,271°** | 1,237° | 2,21° (medyan) |
| Koronal denge | 1,292° | 3,385° | 1,71° (medyan) |

Karar düzeyinde başarım (aksiller eğim, tarama eşikleri):

| Eşik | Doğruluk | Makro-F1 | Pozitif duyarlılık |
|---|---|---|---|
| \|θ\| ≥ 3° | 0,880 | 0,870 | 0,938 |
| \|θ\| ≥ 5° | 0,920 | 0,834 | 0,833 |

> Tarama aracında duyarlılık (recall) önceliklidir; 3° eşiğinde 16 pozitif
> vakanın 15'i yakalanmaktadır.

### 6.3 Olumsuz sonuçlar

Bilimsel şeffaflık gereği, başarısız olan yaklaşımlar da raporlanmaktadır.

**(a) Görüntü işleme tabanlı omurga izi çıkarımı.** Omurga oluğunun gölge
vadisi olarak izlenmesi denenmiştir. C7 ve sakrum landmarkları bilindiğinden
arama dar bir şeride sınırlandırılmış, aydınlanma gradyanı çıkarılmıştır.
Sonuç: kalıntı RMS gövde boyunun **%9,20**'si; 110 görüntünün yalnızca 2'si
kullanılabilir iz vermiştir. Başarısızlık nedeni, yatay profildeki en düşük
yoğunluklu noktanın çoğunlukla omurga oluğu değil **skapula kenarı veya yan
gölge** olmasıdır.

**(b) Eğri uydurma ile gürültü ortalama.** Beş landmarktan polinom uydurularak
nokta gürültüsünün ortalanacağı varsayılmıştır. Ölçüm bu varsayımı
çürütmüştür: apeks sapması hata/sinyal oranı **1,01** (derece 3). Modelin
hataları bağımsız değil ilişkilidir; eğri hatayla birlikte kaymaktadır.

**(c) Landmark sayısının azaltılması.** Üç noktaya (C7, orta omurga, sakrum)
indirgeme denenmiş; konveksite yönü doğruluğu **0,639 → 0,550** düşmüştür
(taban 0,494).

**(d) El yapımı yönelim sezgiselleri.** Ten rengi ağırlık merkezi gri tonlamalı
görüntülerde sinyal vermemiş; siluet genişlik profili renk ipucuyla yalnızca
%50 uyum göstermiştir (rastgele düzeyi).

### 6.4 Omurga eğrilik ölçümünün değerlendirilmesi

Omurga landmarklarının hatası bileşenlerine ayrılmıştır:

| Bileşen | Medyan | Yorum |
|---|---|---|
| Eksen boyunca | 16,66 px | Eğri şeklini etkilemez |
| **Yanal** | **5,88 px** | Ölçülen büyüklük |
| Sinyal (etiketlerdeki yanal sapma) | 6,50 px | — |
| Etiket tabanı (leave-one-out) | 5,58 px | Üst sınır |

Yanal hata (5,88 px) ile etiket tabanı (5,58 px) örtüşmektedir. Konveksite
yönü doğruluğu **%63,9**, sabit çoğunluk sınıfı tahmininin (%52,8) yalnızca
11 puan üzerindedir.

**Sonuç:** Yüzeydeki omurga orta hattı sapması (gövde boyunun ~%1,4'ü),
mevcut etiket kalitesiyle güvenilir ölçülemeyecek kadar küçük bir sinyaldir.
Mimari değişikliğin bu sınırı aşamayacağı, etiket tabanı ölçümüyle
gösterilmiştir.

### 6.5 Metodolojik bulgu: ölçüm hassasiyeti

Çapraz doğrulamada katlar arası yayılım **2,1 piksel** olarak belirlenmiştir.
Bu değer, tek bölmeden elde edilen başarım sayılarının doğal oynaklığıdır.
Çalışma sürecinde karşılaştırılan çeşitli yapılandırmalar (bölme oranı, yığın
boyutu, landmark sayısı) arasındaki farkların bu bandın altında kaldığı, yani
istatistiksel olarak anlamlı olmadığı tespit edilmiştir.

Bu bulgu, küçük veri kümeleriyle çalışan landmark tespiti literatürü için
genellenebilir bir uyarı niteliğindedir: **çapraz doğrulama olmaksızın
raporlanan iyileştirmeler gürültü olabilir.**

### 6.6 Mobil uygulama

- Platform: Android (Kotlin), min. API 26
- Model boyutu: float16 TFLite, toplam 6,40 MB
- Kotlin–Python eşdeğerliği: **0,000 px / 0,0000°** (birim testiyle doğrulandı)
- Toplam birim test sayısı: 42
- Gizlilik: tüm işlem cihaz üzerinde; görüntü cihazdan çıkmaz

---

## 7. SINIRLILIKLAR

1. **Cobb açısı doğrulaması yapılamamıştır.** Radyografi eşleşmeli referans
   veri bulunmadığından, ölçülen yüzey asimetrisinin Cobb açısıyla ilişkisi
   nicelenememiştir. Sistem çıktısı bu nedenle "aksiller eğim açısı" olarak
   adlandırılmakta, arayüzde Cobb ifadesi kullanılmamaktadır.

2. **Veri kümesi heterojenliği.** Eğitim verisi klinik fotoğraflar, stok
   görseller ve gri tonlamalı yayın şekilleri içermektedir. Ürünün saha
   koşulunda karşılaşacağı dağılım (renkli, akıllı telefon kamerası, ayakta
   arkadan çekim) bundan farklıdır.

3. **Alan kayması (domain gap).** Gerçek akıllı telefon fotoğraflarında aynı
   sahnenin ardışık üç çekimi 1,25° / 3,37° / 6,73° değerleri vermiştir.
   Uygulamaya iki ölçütlü bir güven kapısı eklenmiş olup, ölçüm güvenilir
   bulunmadığında yeniden çekim istenmektedir. Bu çözüm semptomu görünür
   kılmakta, nedeni ortadan kaldırmamaktadır.

4. **Örneklem büyüklüğü.** Omurga landmarkı etiketli görüntü sayısı 182'dir.
   Çapraz doğrulama bu kısıtı kısmen telafi etmekle birlikte, alt gruplarda
   (örneğin belirgin eğrilik) güven aralıkları geniştir.

5. **Geometrik bozulma.** Kaynak veri kümesinde en-boy oranı korunmadan
   yapılan boyutlandırma geri döndürülemez niteliktedir; orijinal görüntülere
   erişilememiştir.

---

## 8. SONUÇ VE ÖNERİLER

Geliştirilen sistem, aksiller eğim açısını mevcut veri kalitesinin izin
verdiği üst sınırda (0,271° medyan hata) ölçmekte ve tamamen cihaz üzerinde
çalışmaktadır. Omurga eğrilik profilinin çıkarılması hedefine, sinyalin
küçüklüğü ve etiket gürültüsü nedeniyle ulaşılamamıştır; bu sınır tahmin
yoluyla değil doğrudan ölçümle belirlenmiştir.

### Öncelikli devam adımları

**(1) Akıllı telefon kamerasıyla veri toplama.** Sistemin saha başarımını
sınırlayan temel etken, eğitim verisi ile kullanım koşulu arasındaki alan
kaymasıdır. Standart protokolle (sabit mesafe, ayakta, arkadan, renkli)
toplanacak 200–300 görüntünün, mevcut veri kümesinin tamamından daha yüksek
katkı sağlaması beklenmektedir.

**(2) Radyografi eşleşmeli veri kümesine erişim.** USTC & SYSU-Scoliosis veri
kümesi, 1.898 sırt fotoğrafını uzman ölçümlü Cobb açılarıyla eşleştirmektedir.
Erişim sağlanması durumunda:
- Cobb kalibrasyon katmanı doldurulabilir (mimaride hazır bırakılmıştır),
- Hata payı nicelenerek raporlanabilir,
- Geometrik öznitelikler üzerinden yorumlanabilir bir regresyon modeli
  (örn. gradyan artırmalı ağaçlar) kurulabilir.

> Bu veri kümesindeki radyografiler yalnızca **eğitim etiketi** üretimi için
> kullanılmakta olup, geliştirilen sistemin kullanım aşamasında radyografi
> gerekliliği doğurmamaktadır.

**(3) Etiket kalitesinin artırılması.** İkinci bir bağımsız etiketleme turu ve
ortalama alma, aksiller landmarklarda gürültüyü 0,435°'den 0,232°'ye
düşürmüştür. Aynı yöntemin omurga landmarklarına uygulanması, mevcut 5,58
piksel tabanını düşürebilir.

---

## 9. YAYGIN ETKİ

**Bilimsel etki.** Etiket gürültüsünün doğrudan ölçülmesi ve model başarımının
bu tabana göre yorumlanması, küçük veri kümeleriyle çalışan landmark tespiti
çalışmaları için yöntemsel bir katkıdır. Raporlanan olumsuz sonuçlar,
benzer yaklaşımları deneyecek araştırmacılar için zaman tasarrufu sağlar.

**Toplumsal/ekonomik etki.** Okul taramalarında radyasyonsuz, düşük maliyetli
ve klinisyen bağımsız bir ön eleme aracı, gereksiz radyografi sayısını
azaltma ve erken tespit oranını artırma potansiyeli taşımaktadır.

**Etik ve gizlilik.** Sistem tamamen cihaz üzerinde çalışmakta, hasta görüntüsü
hiçbir aşamada cihazdan çıkmamaktadır. Uygulama arayüzünde "tanı aracı
değildir, tarama amaçlıdır" ibaresi zorunlu tutulmuştur.

---

## 10. KAYNAKLAR

1. Yang, J. ve ark. (2019). Development and validation of deep learning
   algorithms for scoliosis screening using back images. *Communications
   Biology*, 2, 390. https://www.nature.com/articles/s42003-019-0635-8

2. Conditional generative adversarial network-assisted system for
   radiation-free evaluation of scoliosis using a single smartphone
   photograph. *The Lancet Digital Health* (2024).
   https://www.sciencedirect.com/science/article/pii/S2589537024003584

3. Symmetric Perception and Ordinal Regression for Detecting Scoliosis in
   Natural Images. arXiv:2411.15799. https://arxiv.org/pdf/2411.15799
   *(USTC & SYSU-Scoliosis veri kümesi)*

4. Scoliosis1K: A Large-Scale Dataset for Video-Based Scoliosis Screening.
   https://zhouzi180.github.io/Scoliosis1K/

5. Automated Cobb Angle Measurement for Adolescent Idiopathic Scoliosis Using
   Convolutional Neural Network. *Diagnostics* (2022), 12(2), 396.
   https://doi.org/10.3390/diagnostics12020396

6. Automated Cobb Angle Measurements for Scoliosis Diagnosis and Assessment:
   AI Applications and Accuracy Enhancement Through Image Processing
   Techniques. https://pmc.ncbi.nlm.nih.gov/articles/PMC11391110/

---

## EK A — Yazılım Bileşenleri

| Bileşen | İşlev |
|---|---|
| `scripts/00`–`12` | Veri sağlığı, yineleme tespiti, kanonikleştirme, etiket ortalama |
| `scripts/13`–`18` | Eğitim veri formatı, iki aşamalı model eğitimi |
| `scripts/19` | Ölçüm matematiği ve kılavuz çizim (Kotlin referansı) |
| `scripts/20`–`21` | Boru hattı değerlendirmesi, ürün çıktısı |
| `scripts/22`–`29` | ONNX/TFLite dönüşümü, Kotlin eşdeğerlik doğrulaması |
| `scripts/33`–`37` | Landmark genişletmesi veri analizi ve etiketleme hazırlığı |
| `scripts/38`–`43` | 7 landmark boru hattı, çok tipli rafinaj |
| `scripts/45`–`47` | Çapraz doğrulama, bileşen ayrıştırma, etiket tabanı tahmini |
| `android/` | Android uygulaması (Kotlin, CameraX, TFLite) |

## EK B — Ölçüm Konvansiyonları

| Parametre | Tanım |
|---|---|
| Çekim yönü | Arkadan (bireyin sırtı kameraya dönük) |
| Landmark sırası | 0: sağ aksiller, 1: sol aksiller, 2: C7, 3–5: omurga, 6: sakrum |
| Taraf kuralı | Arkadan çekimde bireyin sağı karede sağdadır |
| Açı işareti | Pozitif = sağ omuz yüksek |
| Koordinat | y ekseni aşağı doğru artar (görüntü konvansiyonu) |
| Pitch eşiği | \|pitch\| > 8° → ölçüm reddi |
