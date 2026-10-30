# Örnek çıktılar (sentetik)

<!-- `python -m skolyoz_metrics demo` ile ÜRETİLİR (model dosyaları gerekir). Elle düzenlemeyin. -->

Bu sayfadaki hiçbir görüntü gerçek bir kişiye ait değildir: **çizilmiş bir sırt figürü** kullanıldı (omuz eğimi çizimde bilinir). İşaretler, uygulamayla aynı zincirin (`OlcumZinciri.kt`'nin Python aynası) gönderilen gerçek `.tflite` modelleri üzerindeki çıktısıdır.

> [!IMPORTANT]
> Bunlar **başarım kanıtı değil, boru hattının ne yaptığının gösterimidir.** Modeller gerçek fotoğraflarla eğitildi; çizilmiş bir figürde çalışması gerçek fotoğraflarda doğru çalışacağı anlamına gelmez (örneğin 5° çizilen eğim 3.3° bulundu: bu figürde nokta hatası etiket gürültü tabanının çok üzerinde). Gerçek telefon fotoğraflarında kararsızlık ölçüldü, bkz. [METRICS.md](METRICS.md).

> [!NOTE]
> IMU roll düzeltmesi bu örneklerde **gösterilmiyor** (roll = 0): düzeltmenin işareti cihazda doğrulanmadı, bkz. METRICS.md → Bilinen boşluklar → `imu_roll_sign`.

Modeller: `asama1.tflite` `9737b9d5792b7e6f…`, `asama2.tflite` `8f59347e45fe4354…` · 4 iş parçacığı · 2026-10-31.

## Simetrik figür

![Simetrik figür](img/ornekler/duz.png)

Omuz ve koltuk altı çizgisi yatay çizildi; beklenen eğim ≈ 0°.

Çizimde bilinen eğim: **+0.0°** · Aksiller eğim: **+0.33°** · koronal sapma: +1.2 px (+0.15°) · güven kapısı: geçti · Aşama 1 güveni 0.904

| Nokta | Kaba (640 uzayı) | Rafine (orijinal çözünürlük) | Entropi | Tepe uzaklığı (px) |
|---|---|---|---|---|
| sağ koltuk altı | (489.4, 313.3) | (681.6, 617.1) | 1.50 | 3.6 |
| sol koltuk altı | (166.4, 310.0) | (229.6, 619.6) | 1.75 | 5.3 |
| C7 | (317.7, 230.1) | (452.8, 393.1) | 0.83 | 3.4 |
| omurga 1 | (336.9, 288.6) | (451.0, 548.0) | 1.68 | 6.3 |
| omurga 2 | (330.6, 337.5) | (449.1, 633.5) | 1.78 | 8.7 |
| omurga 3 | (325.3, 386.5) | (451.3, 719.4) | 1.83 | 4.3 |
| sakrum | (326.7, 466.8) | (451.6, 859.7) | 1.69 | 4.5 |

## Sağ omuz yüksek

![Sağ omuz yüksek](img/ornekler/asimetri.png)

Omuz ve koltuk altı çizgisi 5° eğik çizildi (çizimde bilinen eğim, sağ taraf yüksek).

Çizimde bilinen eğim: **+5.0°** · Aksiller eğim: **+3.34°** · koronal sapma: -0.7 px (-0.09°) · güven kapısı: geçti · Aşama 1 güveni 0.903

| Nokta | Kaba (640 uzayı) | Rafine (orijinal çözünürlük) | Entropi | Tepe uzaklığı (px) |
|---|---|---|---|---|
| sağ koltuk altı | (482.3, 313.1) | (681.2, 604.9) | 1.57 | 3.6 |
| sol koltuk altı | (170.0, 315.0) | (235.8, 630.9) | 2.15 | 13.4 |
| C7 | (316.6, 234.7) | (451.8, 397.2) | 0.77 | 0.6 |
| omurga 1 | (335.0, 291.9) | (451.7, 550.5) | 1.65 | 7.3 |
| omurga 2 | (329.1, 339.9) | (450.4, 641.1) | 1.87 | 7.2 |
| omurga 3 | (324.4, 389.8) | (450.7, 724.5) | 1.83 | 3.9 |
| sakrum | (326.4, 469.0) | (452.5, 863.2) | 1.54 | 2.9 |

## Bulanık çekim: güven kapısı reddeder

![Bulanık çekim: güven kapısı reddeder](img/ornekler/bulanik.png)

Aynı simetrik figür, 12 piksellik Gauss bulanıklığıyla. Model yine de kişiyi bulur (Aşama 1 güveni yüksek) ama Aşama 2 ısı haritaları dağılır; entropi eşiği aşar ve uygulama yanlış bir açı göstermek yerine yeniden çekim ister. Beyaz içi boş daireler, uygulamanın göstermeyeceği güvenilmez noktalardır.

Aksiller eğim: **-1.98°** · koronal sapma: +5.7 px (+0.74°) · güven kapısı: **reddedildi** (entropi 3.74 > 2.5) · Aşama 1 güveni 0.891

| Nokta | Kaba (640 uzayı) | Rafine (orijinal çözünürlük) | Entropi | Tepe uzaklığı (px) |
|---|---|---|---|---|
| sağ koltuk altı | (504.9, 339.8) | (697.4, 647.7) | 3.74 | 6.7 |
| sol koltuk altı | (156.0, 316.3) | (219.5, 631.2) | 3.46 | 2.1 |
| C7 | (305.9, 259.2) | (455.6, 414.8) | 0.94 | 3.4 |
| omurga 1 | (328.9, 293.9) | (455.5, 552.2) | 1.92 | 8.2 |
| omurga 2 | (317.1, 345.9) | (455.6, 647.4) | 2.06 | 9.6 |
| omurga 3 | (322.1, 384.7) | (455.2, 718.3) | 1.97 | 6.3 |
| sakrum | (320.8, 469.0) | (449.9, 853.1) | 1.98 | 4.4 |

## Telefon öne yatık: çekim reddedilir

![Telefon öne yatık: çekim reddedilir](img/ornekler/pitch.png)

Telefon 11° öne yatıkken perspektif açıyı bozar; |pitch| > 8° olduğunda deklanşör kilitlenir ve ölçüm yapılmaz. Bu karar yalnızca IMU'dan gelir, model çalıştırılmaz.

Karar: ölçüm reddedildi (|pitch| 11.0° > 8°).

## Karede kimse yok

![Karede kimse yok](img/ornekler/kisi-yok.png)

Boş zemin: Aşama 1 güveni eşiğin (0.25) altında kalır, ölçüm yapılmaz.

Aşama 1 en yüksek güven: 0.001 (eşik 0.25). Ölçüm yapılmadı.

## Nasıl yeniden üretilir

```bash
pip install -r requirements/bench.txt pillow
python -m skolyoz_metrics demo --stage1 asama1.tflite --stage2 asama2.tflite
```
