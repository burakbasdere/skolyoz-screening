# Veri, gizlilik ve lisans

> Bu sayfa, projeyi yayımlamadan önce nelerin bilindiğini ve nelerin karara bağlanması
> gerektiğini kaydeder. Mühendislik notlarıdır, hukuki danışmanlık değildir.

## 1. Hasta verisi depoya asla girmez

Sırt fotoğrafları kişisel sağlık verisidir. `.gitignore`; `data/`, `datasets/`, tüm
raster görüntü türlerini ve DICOM dosyalarını, `runs/` klasörünü, her model biçimini
(`*.pt`, `*.pth`, `*.onnx`, `*.tflite`) ve paketleri (`*.apk`, `*.aab`) dışarıda tutar.
Tek istisna dokümantasyon şekilleridir: `docs/` altındaki PNG'lere izin verilir, bu yüzden
oraya eklediğiniz her görselin **gerçek bir kişi içermediğini kontrol edin**. (Bu depodaki
`docs/img/` grafikleri JSON'dan üretilen SVG'lerdir; görüntü içermez.)

**İstisnalar: iki küçük gerçek fotoğraf kümesi.** `.gitignore` bunları tek tek dosya adıyla
serbest bırakır. İkisi de gerçek kişilerin sırt fotoğraflarını içerir; yüz görünmez. **Kaynakları,
lisansları ve çekilen kişilerin (ya da velilerinin) yayımlama rızası projede kayıtlı değil.**

- `docs/img/uygulama/ekran-1.jpg`, `ekran-2.jpg`, `ekran-3.jpg` (README'lerin başında):
  uygulamanın gerçek bir telefonda çalışırken fotoğrafları. Kamera, bir monitörde gösterilen
  sırt fotoğraflarına tutulmuş. Ekranlardaki "Cobb tahmini" satırı kalibre edilmemiş bir yer
  tutucudur (`cobb_estimate` boşluğu).
- `docs/img/referans/skolyoz.webp`: arkadan çekilmiş tek bir referans fotoğrafı. Yalnızca
  `docs/ORNEKLER.md`'nin son bölümünde ve modellerin onun üzerine çizdiği çıktıda
  (`docs/img/ornekler/referans.png`) kullanılır. (`*.webp` aksi halde dışarıda tutulur.)

Depo herkese açık olduğundan her biri için kaynak + lisans ya da yazılı rıza belgeleyin.
Belgelenemeyeni silin ve onu gösteren yerleri kaldırın: README'lerin başındaki tablo ya da
`skolyoz_metrics/demo.py` içindeki `referans` örneği (ardından `demo`nun yeniden çalıştırılması).
Dosya bir kez geçmişe girdikten sonra silmek yeterli değildir (sonraki paragraf).

Bir hasta görüntüsü commit edilirse bunu sızıntı sayın: sonraki bir commit'te silmek geçmişten
kaldırmaz. Geçmişi yeniden yazın ve bağlantıları geçersiz kılın. Bkz. [SECURITY.md](../SECURITY.md).

## 2. Eğitim verisinin kaynağı

- Kaynak: bir Roboflow dışa aktarımı (`education-ovbua/bladeco v1`), 1.119 dosya. Görüntüler
  web kaynaklıdır: klinik fotoğraflar, stok görseller ve yayınlardan gri tonlamalı şekiller
  ([tr/TEKNIK_DOKUMAN.md §7](tr/TEKNIK_DOKUMAN.md)).
- **Tek tek görüntülerin lisansı projede hiçbir yerde kayıtlı değil.** Görüntüleri yeniden
  dağıtmayın; bu görüntülerle eğitilmiş ağırlıkların da kısıtlar devralabileceğini, durum
  netleşene kadar varsayın.
- Orijinal fotoğraflar kayıp; yalnızca gerilmiş 640 × 640 sürümler kaldı, bu yüzden kaynak
  dosyalardan geri izlenemez.

## 3. Üçüncü taraf bileşenler

| Bileşen | Nerede | Lisans (güvenmeden önce kaynağından doğrulayın) |
|---|---|---|
| Ultralytics YOLO11 (kod **ve** önceden eğitilmiş ağırlıklar) | Aşama 1 eğitimi, birçok `scripts/`, `skolyoz_metrics bench arch` | **AGPL-3.0**; kapalı kullanım için Ultralytics Enterprise lisansı var |
| PyTorch, torchvision | Aşama 2 eğitimi ve benchmark'lar | BSD tarzı |
| MobileNetV3-small ImageNet ağırlıkları (torchvision) | Aşama 2 başlangıcı | Kod BSD tarzı; ağırlıklar ImageNet üzerinde eğitildi ve ImageNet'in koşulları ticari kullanımı kısıtlar — kontrol edin |
| TensorFlow Lite, AndroidX (Compose, CameraX, …), Kotlin | Android uygulaması | Apache-2.0 |
| onnx2tf | Dışa aktarma | MIT |
| ai-edge-litert | TFLite benchmark | Apache-2.0 |

## 4. AGPL sorusu — yayımlamadan önce karar verin

İnce ayarlı Aşama 1 modeli Ultralytics YOLO11'den türetilmiştir ve Android uygulaması onu
içinde taşır. Ultralytics'in koşullarına göre bu, modeli ve onu gönderen uygulamayı
AGPL-3.0'a tabi kılar: uygulamayı alan kullanıcılar ilgili kaynak koda erişebilmelidir.
`import ultralytics` yapan araştırma betikleri de aynı durumdadır.

| Seçenek | Sonuç |
|---|---|
| **Her şeyi AGPL-3.0 ile yayımlamak** | En basit ve tutarlı. Herkes kullanabilir ve değiştirebilir; ancak türev işler (ağ hizmetleri dahil) açık kalmalıdır. Açık bir araştırma projesine uyar. |
| Bölmek: özgün kod (`android/` kaynakları, `skolyoz_metrics/`, dokümanlar) için izin verici lisans, Ultralytics'i içe aktaran her şey ve ağırlıklar için AGPL | Daha hassas, ama gönderilen APK AGPL ağırlıklarını içerdiği için *dağıtılan uygulama* yine AGPL olur. |
| Ultralytics Enterprise lisansı almak ya da Aşama 1'i izin verici lisanslı bir dedektörle yeniden eğitmek | Lisansı serbest bırakır; maliyet ya da mühendislik emeği gerektirir. Aşama 1'in yeniden değerlendirilmesi gerekir. |

**Öneri:** ticari bir ürün planlanmıyorsa AGPL-3.0 ile yayımlayın ve `LICENSE` dosyasını
depoyu yayımlayan commit'e ekleyin. Henüz lisans seçilmedi; seçilene kadar kod "tüm hakları
saklıdır" durumundadır ve başkaları onu yasal olarak yeniden kullanamaz.

## 5. Ağırlıklar ve APK'ler

Önceki depo bir debug APK ve imzasız bir release APK commit etmişti. Bunlar `asama1.tflite` ve
`asama2.tflite` dosyalarını içerir; yani README "ağırlıklar git'e girmez" dese de ağırlıklar fiilen
yayımlanmıştı. Bu depo ikili dosyaları dışarıda bırakır. İnsanların uygulamayı denemesini
isterseniz, 2. ve 4. bölümleri çözdükten **sonra** bir derlemeyi GitHub Release'e ekleyin ve
SHA-256 değerini `metrics/results.json` içine (`artifacts`) kaydedin.

## 6. Mevzuat

Tıbbi amaçlı yazılım, ücretsiz bile olsa düzenlemeye tabi bir tıbbi cihaz olabilir (örneğin AB
MDR, FDA kuralları ya da Türkiye'de TİTCK çerçevesi). Bu proje kendini araştırma yazılımı ve
tarama yardımcısı olarak konumlandırır ve kullanıcının gördüğü her çıktıda bunu söyler. Sağlık
amaçlı olarak kamuya dağıtacaksanız önce mevzuat danışmanlığı alın.
