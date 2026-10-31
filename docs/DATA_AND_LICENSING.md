# Veri, gizlilik ve lisans

> Bu sayfa, bu katalogdaki görsellerin ve verilerin nereden geldiğini ve nelerin karara
> bağlanması gerektiğini kaydeder. Mühendislik notlarıdır, hukuki danışmanlık değildir.

## 1. Hasta verisi depoya asla girmez

Sırt fotoğrafları kişisel sağlık verisidir. Bu depo kaynak kod, eğitim verisi, etiket, model
ağırlığı ya da uygulama paketi **içermez**; `.gitignore` her raster görüntü türünü
(`*.jpg`, `*.png`, `*.webp`, …), DICOM dosyalarını, model biçimlerini ve paketleri dışarıda
tutar. Tek istisnalar aşağıdaki dosyalardır; bunlar tek tek adıyla serbest bırakılmıştır ve
`docs/` altındaki grafikler (JSON'dan üretilmiş SVG'ler, görüntü içermez) ile çizilmiş
sentetik figür çıktılarıdır.

**İstisnalar: iki küçük gerçek fotoğraf kümesi.** İkisi de gerçek kişilerin sırt fotoğraflarını
içerir; yüz görünmez. **Kaynakları, lisansları ve çekilen kişilerin (ya da velilerinin)
yayımlama rızası projede kayıtlı değil.**

- `docs/img/uygulama/ekran-1.jpg`, `ekran-2.jpg`, `ekran-3.jpg` (README'lerin başında):
  uygulamanın gerçek bir telefonda çalışırken fotoğrafları. Kamera, bir monitörde gösterilen
  sırt fotoğraflarına tutulmuş. Ekranlardaki "Cobb tahmini" satırı uygulamanın Cobb
  ölçümüdür (test aşaması, doğrulanmamış; `cobb_estimate` boşluğu).
- `docs/img/referans/skolyoz.webp`: arkadan çekilmiş tek bir referans fotoğrafı. Yalnızca
  `docs/ORNEKLER.md`'nin son bölümünde ve modellerin onun üzerine çizdiği çıktıda
  (`docs/img/ornekler/referans.png`) kullanılır.

Depo herkese açık olduğundan her biri için kaynak + lisans ya da yazılı rıza belgeleyin.
Belgelenemeyeni silin ve onu gösteren yerleri (README'lerin başındaki tablo ya da
`docs/ORNEKLER.md`'nin son bölümü) kaldırın. Dosya bir kez geçmişe girdikten sonra silmek
yeterli değildir (sonraki paragraf).

Bir hasta görüntüsü commit edilirse bunu sızıntı sayın: sonraki bir commit'te silmek geçmişten
kaldırmaz. Geçmişi yeniden yazın ve bağlantıları geçersiz kılın. Bkz. [SECURITY.md](../SECURITY.md).

## 2. Eğitim verisinin kaynağı

- Kaynak: bir Roboflow dışa aktarımı (`education-ovbua/bladeco v1`), 1.119 dosya. Görüntüler
  web kaynaklıdır: klinik fotoğraflar, stok görseller ve yayınlardan gri tonlamalı şekiller.
- **Tek tek görüntülerin lisansı projede hiçbir yerde kayıtlı değil.** Görüntüleri yeniden
  dağıtmayın; bu görüntülerle eğitilmiş ağırlıkların da kısıtlar devralabileceğini, durum
  netleşene kadar varsayın.
- Orijinal fotoğraflar kayıp; yalnızca gerilmiş 640 × 640 sürümler kaldı, bu yüzden kaynak
  dosyalardan geri izlenemez.

## 3. Lisans

Bu depodaki içerik (metinler, grafikler, görseller, ölçüm verileri) için **lisans henüz
seçilmedi**; seçilene kadar tüm hakları saklıdır ve başkaları içeriği yasal olarak yeniden
kullanamaz. Uygulamanın kendisi bu depoda yayımlanmaz.

## 4. Mevzuat

Tıbbi amaçlı yazılım, ücretsiz bile olsa düzenlemeye tabi bir tıbbi cihaz olabilir (örneğin AB
MDR, FDA kuralları ya da Türkiye'de TİTCK çerçevesi). Bu proje kendini araştırma yazılımı ve
tarama yardımcısı olarak konumlandırır, test aşamasındadır ve son kullanıcı ürünü değildir.
Sağlık amaçlı olarak kamuya dağıtacaksanız önce mevzuat danışmanlığı alın.
