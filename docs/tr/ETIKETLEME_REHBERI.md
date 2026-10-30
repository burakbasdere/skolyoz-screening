# Etiketleme Rehberi — 7 Landmark

Modeli 2 noktadan (koltuk altı) **7 noktaya** genişletiyoruz. Bu belge
etiketleme sırasında bakılacak tek referanstır.

Görsel referans: `runs/omurga/ETIKET_ORNEGI.jpg`
(git'e girmez — `scripts/35_etiket_ornegi.py` ile yeniden üretilir)

---

## 1. Neden genişletiyoruz

Mevcut ölçüm iki koltuk altı arasındaki eğim — yani **omuz asimetrisi**.
Tek başına eksik: hastanın omuzları düz olup gövdesi yana kaymış olabilir,
ya da tersi.

Eklenen noktalar iki yeni şey veriyor:

| Nokta | Ne ölçer |
|---|---|
| `c7` + `sakrum` | **Koronal denge** — C7'nin sakrumdan çekilen şakülden yatay sapması. Skolyoz muayenesinde C7'den sarkıtılan iplik ölçümünün karşılığı. |
| `omurga_1..3` | **Omurga eğrisi** — apeks konumu ve konveksite yönü ("sağa konveks torakal eğrilik") |

Uygulamada kılavuz artık iki eksenli: kesikli **yatay** (koltuk altı referansı)
ve kesikli **dikey** şakül (koronal denge referansı). İkisi de IMU'dan gelen
yerçekimi çerçevesinden türer, telefon eğik tutulunca birlikte dönerler.

---

## 2. Sınıf adları (Roboflow)

```
sag koltuk    sol koltuk    c7    omurga_1    omurga_2    omurga_3    sakrum
```

| Sınıf | Nereye |
|---|---|
| `sag koltuk` | **Mevcut etiket, yeniden çizme.** Kişinin sağı = karede sağ (arkadan çekim) |
| `sol koltuk` | Mevcut etiket, yeniden çizme |
| `c7` | Boyun dibindeki kemik çıkıntı (vertebra prominens) |
| `omurga_1` | Omurga oluğu üzerinde, **üst** bölge |
| `omurga_2` | Omurga oluğu üzerinde, **orta** bölge |
| `omurga_3` | Omurga oluğu üzerinde, **alt** bölge |
| `sakrum` | Kalça yarığının üst ucu |

---

## 3. Kurallar

### Sıra önemli
`omurga_1` en üstte (C7'ye yakın), `omurga_3` en altta (sakruma yakın).
Karıştırırsan eğri ters çıkar — **kod bunu yakalayamaz**, sessizce yanlış
sonuç üretir.

### Yükseklik serbest
Tam %25 / %50 / %75 olmak zorunda değil. Sonradan (yükseklik, yanal sapma)
çiftlerine eğri uyduruyoruz, istediğimiz yükseklikte değerlendiriyoruz.
Gereken tek şey noktaların C7 ile sakrum arasına **makul şekilde yayılmış**
olması.

### Oluk görünmüyorsa
Gövdenin iki kenarından eşit uzaklıktaki hattı işaretle. Tahmin etmekten
çekinme — **tutarlı ol, kusursuz olma.** Etiket gürültüsü geçen turda
2.45 px ölçüldü ve model tam o tabana oturdu; tutarlı bir tahmin, boş
bırakılmış bir noktadan iyidir.

Şüphelendiğin karede **parlaklık/kontrastı aç** — oluk çoğu zaman ortaya
çıkar. Referans görselindeki noktalar da öyle okundu (CLAHE + yerel kontrast).

### Yapma
- **C7–sakrum düz çizgisine hizalama.** Eğriliği yok eder; ölçmeye
  çalıştığımız şey tam olarak o sapma.
- **Omurları sayma.** T6/T12 gibi seviyeleri fotoğraftan seçemezsin,
  denemeye değmez.
- **Kürek kemiği kenarını oluk sanma.** Sırtta oluktan daha koyu şeritler
  var; OpenCV denemesinin çuvallama sebebi tam olarak buydu (bkz. §5).
- **Koltuk altı noktalarının tanımını değiştirme.** Mevcut 337 etiket kolun
  dış hattında; kıvrıma kaydırırsan iki farklı tanım karışır.

### Alt sırt görünmüyorsa
Sakrum seçilemiyorsa o görüntüyü atla. C7'siz de olmaz. Yarım etiket,
yanlış etiketten iyi değildir — ikisi de kullanılamıyor.

---

## 4. Bu turda yapılacaklar

Kaynak: `aag.v1i.coco.zip` (122 görüntü, hepsi mevcut 337'nin alt kümesi)

| İş | Adet |
|---|---|
| Hatalı etiketleri düzelt | 12 |
| Mevcut 110 görüntüye 3 omurga noktası ekle | 110 |
| Etiketlenmemiş "iyi" görüntüleri 7 noktayla etiketle | 61 |

### Düzeltilecek 12 görüntü

| Sorun | Dosyalar |
|---|---|
| `sakrum` eksik + `c7` mükerrer | `train_0012`, `train_0316` |
| `c7` eksik | `train_0227`, `train_0228`, `train_0230`, `train_0254`, `train_0255` |
| `c7` eksik + `sakrum` mükerrer | `test_0084`, `train_0211`, `train_0269`, `train_0298`, `val_0018` |

### Roboflow export ayarları — bunlar bizi iki kez yaktı

- **Augmentasyon kapalı.** Özellikle **rotasyon** ve **yatay flip**. Flip
  sağ/sol koltuk altını takas eder; rotasyon ölçtüğümüz sinyali yok eder.
  İlk turda veri seti 2.54× şişmiş ve etiketlerin %38.8'inde sağ/sol
  çelişkisi oluşmuştu.
- **Resize 640×640.** Görüntüler zaten kare, stretch zararsız. Başka boyut
  mevcut modellerle uyumsuz olur.
- **Export: COCO.** (Roboflow bu projeleri bounding box olarak veriyor;
  nokta = kutu merkezi. Sorun değil, kutular sıkı çizildiği sürece.)

---

## 5. Kayda geçen negatif sonuç: OpenCV ile omurga izi

Ara noktaları elle etiketlemek yerine omurga oluğunu otomatik izlemeyi
denedik (`scripts/34_omurga_izi_deney.py`). C7 ve sakrum bilindiği için
arama dar bir şeride hapsedildi, aydınlanma gradyanı çıkarıldı.

**Başarısız.** 110 görüntüde:

| | |
|---|---|
| Kalıntı RMS medyanı | **%9.20** (gövde boyunun) |
| Düzgün iz sayılabilecek (<%2) | **2 / 110** |

Sebep: yatay profildeki en karanlık nokta çoğunlukla omurga oluğu değil,
**gövdenin yan gölgesi veya kürek kemiği kenarı**. Oluğun kontrastı
rakiplerinden zayıf ve ışık koşuluna göre tamamen kayboluyor.

Bandı daraltmak en kötü sapmaları engeller ama dar bantta gürültü izlemeye
dönüşür. Sinyal güvenilir şekilde orada değil. **Elle etiketleme bu yüzden
gerekli** — model, ışıktan bağımsız olarak öğrenir.

---

## 6. Etiketleme setini hazırlama

```bash
python scripts/37_etiket_seti_hazirla.py
```

`data/etiket_seti/` altında dört klasör üretir. Görüntülere **kontrast işlemi**
uygulanır (CLAHE + unsharp) — omurga oluğu ham fotoğrafta zor seçiliyor,
işlenmiş halde net görünüyor. Kare 640×640 aynı kaldığı için **etiket
koordinatları orijinale birebir oturur**: etiketleme işlenmiş kopyada yapılır,
eğitim temiz orijinalle.

| Klasör | Adet | Ne yapılacak |
|---|---|---|
| `1_oncelik_kanitli` | 110 | C7+sakrum etiketi var — 7 noktayı sıfırdan çiz |
| `2_duzeltilecek` | 12 | Etiketi bozuktu — 7 noktayı temiz çiz |
| `3_yeni_iyi` | 61 | Hiç etiketlenmemiş, alt+üst sırt görünüyor |
| `4_sinirda` | 138 | Sakrum genelde kadrajda yok — **sakrumu atla**, gerisini çiz |

Öncelik **1 → 2 → 3** (183 görüntü, test seti ~28).

Grup 4 sakrumsuz olduğu için koronal denge *ölçümüne* katkı vermez, ama
görünürlük maskesi sayesinde C7 ve omurga noktaları için tam değerli
eğitim verisidir.

### Roboflow'a yükleme — baştan etiketleme YOK

```bash
python scripts/41_roboflow_yukleme.py            # grup 1-3, keypoint formatı
python scripts/41_roboflow_yukleme.py --gruplar 1,2,3,4
```

`data/roboflow_yukleme/` içine işlenmiş görüntüleri **ve mevcut etiketleri
içeren** bir COCO Keypoints dosyası yazar. Koltuk altı / C7 / sakrum hazır
gelir; sadece `omurga_1..3` çizilir — 7 nokta yerine 3.

**Proje tipi: Keypoint Detection** (Object Detection değil). Üç sebep:

1. Tıklamak sürüklemekten hızlı — 183 görüntü × 3 nokta
2. İskelet sabit olduğu için **`omurga_1/2/3` sıra hatası mimari olarak
   imkânsız** hale gelir (kod bu hatayı yakalayamıyor, sessizce yanlış
   sonuç üretiyordu)
3. Nokta gerçekten nokta olur; kutu merkezi yaklaşımı devre dışı kalır

Adımlar:

1. Roboflow → yeni proje → **Keypoint Detection**
2. Upload → `data/roboflow_yukleme/` içindeki **tüm dosyalar** birlikte
   (jpg'ler + `_annotations.coco.json`)
3. İskelet sırası: `sag koltuk → sol koltuk → c7 → omurga_1 → omurga_2 →
   omurga_3 → sakrum`
4. Her görüntüde sadece omurga noktalarını ekle
5. Export: **COCO**, augmentasyon kapalı, resize 640×640

> ⚠️ **Koltuk altı noktalarına dokunma.** Onlar ortalanmış etiketler —
> gürültüsü 0.435° yerine 0.232°. Yeniden çizilirse bu kazanç kaybolur.

`scripts/38` her iki formatı da (Keypoints ve sınıf-başına-bbox) okuyor,
hangisi geldiğini kendisi anlıyor.

### Gürültü tabanı — küçük bir ek iş

Ön doldurma sayesinde C7/sakrum yeniden çizilmiyor, dolayısıyla o noktalar
için **test-retest gürültü tabanını** ölçemiyoruz. Koltuk altında bu 2.45 px'ti
ve modelin nereye kadar inebileceğini o belirlemişti.

Ucuz çözüm: etiketleme bitince **25 görüntüde** C7 ve sakrum'u sıfırdan
yeniden çiz (ayrı bir Roboflow projesinde, ön doldurma olmadan). İki turu
karşılaştırıp tabanı çıkarırız — 110 yerine 25, aynı bilgi.

---

## 7. Etiketleme sonrası: mimari

Boru hattı **yazıldı ve test edildi**; etiketler gelince tak-çalıştır.

### Nokta sırası — değiştirme

```
0 sag koltuk   1 sol koltuk   2 c7
3 omurga_1     4 omurga_2     5 omurga_3   6 sakrum
```

0 ve 1 mevcut sırayı korur; eski etiketler ve Android tarafı bozulmaz.

### Aşama 1 — `scripts/38_yolo7_format.py`

YOLO-pose 2 yerine 7 keypoint. Eksik landmark `v=0` yazılır; Ultralytics'in
pose kaybı bunları maskeler (`utils/loss.py`: `kpt_mask = gt_kpt[..., 2] != 0`,
satır 801). Böylece koltuk altı için 337, yeni noktalar için ~183 görüntü
kullanılır — tek model.

> ⚠️ **Eğitimde `kobj=0.0` şart.** Ultralytics ayrı bir görünürlük başlığı
> eğitiyor (`kpts_obj_loss`). Etiketlenmemiş ama aslında GÖRÜNEN noktaları
> `v=0` yaptığımız için bu başlık "o nokta yok" diye yanlış öğrenir.

```bash
python scripts/38_yolo7_format.py --export <roboflow_coco.json>
```

Sınıf adları `TAKMA_AD` sözlüğünden eşlenir; tanımadığı ad görürse uyarır.

### Aşama 2 — `scripts/39_rafinaj7_model.py` + `40_rafinaj7_egit.py`

7 nokta ama **4 görsel tip**: `koltuk_alti` (0,1 — ayna simetrik),
`c7` (2), `omurga` (3,4,5 — üçü de aynı doku), `sakrum` (6). Tip paylaşımı
veri verimliliği için: omurga_1/2/3 aynı görsel özelliği arıyor, ayrı kanal
ayırmak 3 kat az veriyle 3 kat fazla parametre demek.

Model tüm tipler için koordinat döndürür `(B, 4, 2)`; çağıran seçer. Grafik
statik kaldığı için TFLite'a temiz çevrilir. Kayıp yalnızca örneğin tipine
ait kanaldan alınır.

**Parametre: 0.41 M** — tek tipli sürümden sadece 99 fazla, pratikte bedava.

```bash
python scripts/40_rafinaj7_egit.py --epochs 200 --batch 32
```

Duman testi (10 epoch, omurga etiketleri henüz yokken):
`koltuk_alti 4.53 px · c7 20.60 px · sakrum 12.45 px` — hata veri miktarıyla
orantılı (472 / 84 / 90 örnek), beklenen tablo.

### Test seti uyarısı

122 görüntünün orijinal split'i **train 92 / val 14 / test 16**. Hepsi
etiketlenirse test ~22 olur. Bu **çok küçük** — 16–22 görüntüyle ölçülen
doğruluğun güven aralığı çok geniş. Geçen turda 7° eşiğindeki `F1 = 1.000`
sadece 4 vakadan geliyordu; aynı tuzağa düşmeyelim. Sonuçları raporlarken
belirsizliği açıkça yazacağız.

**Split korunmalı.** 122 görüntünün hepsi mevcut 337'nin alt kümesi;
orijinal train/val/test ataması aynen kullanılmalı. Aşama 1'in train'indeki
bir görüntü yeni testimize düşerse ölçtüğümüz doğruluk sahte çıkar.
