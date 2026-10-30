# Güvenlik ve gizlilik politikası

## Bildirim

Güvenlik açıklarını ve özellikle **depoda ya da geçmişinde bulunan kişisel sağlık verisini**
lütfen gizlice bildirin: herkese açık issue değil, bu depodaki GitHub "Report a vulnerability"
(özel advisory) özelliğini kullanın. Neyi ve nerede bulduğunuzu yazın, ama verinin kendisini
yapıştırmayın.

Birkaç gün içinde yanıt bekleyebilirsiniz. Bu küçük bir araştırma projesidir; ödül programı ya
da resmî bir yanıt süresi (SLA) yoktur.

## Kapsam

- Android uygulaması ve `android/` altındaki her şey.
- `skolyoz_metrics/` altındaki Python araçları.
- Yanlışlıkla yayımlanmış hasta görüntüleri, etiketleri ya da model ağırlıkları.

## Gizlilik açısından önemli tasarım kararları

- Çıkarım tamamen cihaz üzerindedir; uygulama `INTERNET` izni istemez.
- Yedekleme kapalıdır (`allowBackup="false"`).
- Debug derlemeleri teşhis görüntülerini depolamaya yazar. Debug derlemesini gerçek kişilerde
  **kullanmayın**.
- Görüntü dosyaları ve model ağırlıkları `.gitignore` ile dışarıda tutulur.

## Tıbbi uyarı

Bu bir araştırma yazılımı ve tarama yardımcısıdır. Tanı aracı değildir; hastalar üzerinde
doğrulanmamış ve tıbbi cihaz olarak onaylanmamıştır.
