# Model kartı

Sayılar burada tekrarlanmaz; bkz. [METRICS.md](METRICS.md). Aşağıdaki her iddia, o sayfanın
desteklediği kadarla sınırlıdır.

## Modeller

| | Aşama 1 | Aşama 2 |
|---|---|---|
| Rol | 7 landmark'ın kaba lokalizasyonu | Her landmark'ın alt-piksel rafinajı |
| Mimari | Ultralytics YOLO11n-pose, ince ayarlı (7 keypoint) | MobileNetV3-small gövde + transpoze konvolüsyon başlığı, ısı haritası + soft-argmax |
| Girdi | 640 × 640 RGB, 0–1 aralığında | 128 × 128 RGB kırpım, ImageNet ortalama/std |
| Dağıtım biçimi | TFLite float16 | TFLite float16 |
| Ağırlıklar | dağıtılmaz ([DATA_AND_LICENSING.md](DATA_AND_LICENSING.md)) | dağıtılmaz |

## Kullanım amacı

- Araştırma, eğitim ve pilot çalışmalarda **gövde asimetrisi için tarama yardımcısı**:
  dik duran, **arkadan** çekilmiş, telefon dik tutulmuş bir kişinin fotoğrafı; bir hekime
  yönlendirmenin değerli olup olmayabileceğine dair ipucu vermek için.
- Çıktılar *göstergelerdir* (aksiller eğim, koronal denge ve test aşamasındaki Cobb ölçümü);
  zorunlu "tanı aracı değildir" ibaresiyle birlikte gösterilir.
- Sistem **test aşamasındadır**; son kullanıcı ürünü değildir.

## Kapsam dışı — şunun için kullanmayın

- **Tanı, şiddet derecelendirmesi ya da tedavi kararı.**
- **Cobb açısının klinik ya da güvenilir bir ölçüsü olarak kullanım.** Uygulama Cobb açısı
  ölçümü de yapar (test aşaması); projede radyografi eşleşmeli veri olmadığı için bu ölçüm
  doğrulanmamıştır (`cobb_estimate` boşluğuna bakın).
- Ön ya da yan görünüm, öne eğilme (Adams) testi, video veya ön kamera (görüntüyü aynalar,
  sağ ve sol yer değiştirir).
- Eğitim verisinde bulunmayabilecek herhangi bir popülasyon. Yaş, cinsiyet ve ten rengi dağılımı
  belgelenmemiştir (aşağıya bakın), dolayısıyla kapsandığı bilinen bir popülasyon yoktur.

## Eğitim ve değerlendirme verisi

- Web kaynaklı bir Roboflow dışa aktarımından temizlenmiş 337 görüntü (1.119 dosya → 441
  benzersiz → 337 kullanılabilir); en-boy oranı korunmadan 640 × 640'a boyutlandırılmış.
  Karışımda klinik fotoğraflar, stok görseller ve yayınlardan gri tonlamalı şekiller var.
  Orijinallere erişilemiyor.
- Yinelemeler (aynı fotoğrafın döndürülmüş kopyaları) ve sağ/sol etiket çelişkileri
  ayıklandı ya da onarıldı; etiket gürültüsünü azaltmak için bağımsız etiketlerin ortalaması alındı.
- Değerlendirme: benzetilmiş kaba noktalarla Aşama 2'nin grup duyarlı 5 katlı çapraz
  doğrulaması ve 50 görüntülük tek bir ayrılmış küme. İki protokol ve sınırları METRICS §1'de.

## Bilinen sınırlar

- **Alan kayması.** Hedef girdi — amaçlanan protokolle çekilmiş renkli telefon fotoğrafları —
  modelin eğitildiği ya da değerlendirildiği şey değil. Gayriresmî bir denemede aynı sahnenin
  art arda üç çekimi çok farklı açılar verdi; güven kapısı bu yüzden var.
- 7 landmark'lı sistem için **uçtan uca ya da cihaz üstü doğruluk yok**.
- **Omurga landmark'ları zayıf** ve eğri profili üretilmiyor; sınır etiket gürültüsüdür ve
  ölçümle gösterilmiştir.
- **Küçük veri.** Omurga etiketi 182 görüntüde var; tarama kararı istatistikleri 4–16 pozitif
  vakaya dayanıyor.
- **Alt grup analizi yok.** Kaynaklarda demografi, ten rengi, vücut yapısı ya da yaş bilgisi
  yok; bu gruplar arasındaki başarım bilinmiyor. Adalet iddiası desteksizdir.

## Riskler ve önlemler

| Risk | Projedeki önlem | Hâlâ açık |
|---|---|---|
| Yanlış güvence ("normal görünüyor") ile bakımın gecikmesi | Zorunlu ibare; duyarlılık öncelikli metrik | Hastalar üzerinde doğrulanmadı; duyarlılık GA'sı geniş |
| Yanlış alarm | Çıktı, klinik derece değil, veri kümesi dağılımındaki konumla birlikte tarama göstergesi olarak etiketlenir | Hedef popülasyondaki yaygınlık bilinmiyor |
| Kararsız ısı haritasından sessizce yanlış açı | İki ölçütlü güven kapısı çekimi reddeder | Kapı gerçek bir çekim kampanyasında denenmedi |
| Eğik tutulan telefondan yanlış açı | IMU roll düzeltmesi, 8°'de pitch kapısı | Sensör kalitesine bağlı |
| Sağlık verisinin ifşası | Cihaz üstü çıkarım, internet izni yok, yedekleme kapalı, görüntüler depoya eklenmez | Debug derlemeleri teşhis görüntülerini diske yazar — gerçek kişilerde asla debug derlemesi kullanmayın |
| Abartılı iddia | Durumlu ve boşluk kayıtlı, üretilen metrik sayfası | — |
