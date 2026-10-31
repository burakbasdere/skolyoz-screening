# Güvenlik ve gizlilik politikası

## Bildirim

**Depoda ya da geçmişinde bulunan kişisel sağlık verisini** (hasta görüntüsü, etiket, kişiye
ait herhangi bir bilgi) lütfen gizlice bildirin: herkese açık issue değil, bu depodaki GitHub
"Report a vulnerability" (özel advisory) özelliğini kullanın. Neyi ve nerede bulduğunuzu yazın,
ama verinin kendisini yapıştırmayın.

Birkaç gün içinde yanıt bekleyebilirsiniz. Bu küçük bir araştırma projesidir; ödül programı ya
da resmî bir yanıt süresi (SLA) yoktur.

## Gizlilik açısından önemli tasarım kararları

- Çıkarım tamamen cihaz üzerindedir; uygulama internet erişimi istemez.
- Yedekleme kapalıdır.
- Debug derlemeleri teşhis görüntülerini depolamaya yazar. Debug derlemesini gerçek kişilerde
  **kullanmayın**.
- Görüntü dosyaları ve model ağırlıkları bu depoya eklenmez (README'lerin başındaki ekran
  fotoğrafları ve referans fotoğraf istisnadır; bkz. [docs/DATA_AND_LICENSING.md](docs/DATA_AND_LICENSING.md)).

## Tıbbi uyarı

Bu bir araştırma yazılımı ve tarama yardımcısıdır; test aşamasındadır, son kullanıcı ürünü
değildir. Tanı aracı değildir; hastalar üzerinde doğrulanmamış ve tıbbi cihaz olarak
onaylanmamıştır.
