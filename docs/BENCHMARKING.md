# Benchmark

Gecikme, **hiçbir hasta görüntüsü ve eğitilmiş ağırlık olmadan** ölçülür. Sabit şekilli bir ağ,
ağırlıkları ne olursa olsun aynı sürede çalışır; bu yüzden gerçek mimariler rastgele
tensörlerle zamanlanır. Sayıların herkes tarafından, her makinede yeniden üretilebilmesini
sağlayan da budur.

## Neler var

| Komut | Neyi ölçer | Gerekenler |
|---|---|---|
| `python -m skolyoz_metrics bench tflite --stage1 asama1.tflite --stage2 asama2.tflite` | **Dağıtılan** `.tflite` dosyalarını TFLite çalışma zamanıyla (XNNPACK, varsayılan 4 iş parçacığı — uygulama da 4 kullanır). Masaüstündeki en temsil edici sayı. | TFLite çalışma zamanı (`ai-edge-litert`) ve iki model dosyası |
| `python -m skolyoz_metrics bench arch` | İki PyTorch mimarisini rastgele ağırlıkla, CPU'da ve (varsa) CUDA'da; parametre ve GFLOPs da kaydedilir. Hiç model dosyası gerektirmediği için CI de çalıştırabilir. | `requirements/bench.txt` |
| `./gradlew :app:testDebugUnitTest --tests '*PerformansTest*' -i` | Kotlin ön işleme (küçültme, kırpım, tensör, soft-argmax), masaüstü JVM'de. | JDK 17 |

Her `bench` koşusu `metrics/benchmarks/<tarih>_<tür>_<cpu>.json` yazar
(`metrics/benchmark.schema.json` ile doğrulanır). İçinde CPU, işletim sistemi, kütüphane
sürümleri, iş parçacığı sayısı, ısınma/tekrar sayıları ve tüm gecikme özeti
(n, min, ortalama, std, p50, p90, p95, p99, max) bulunur. Sonra:

```bash
python -m skolyoz_metrics render    # docs/METRICS.md ve README tablolarını yeniler
python -m skolyoz_metrics check     # CI'ın çalıştırdığı denetim
```

README'deki başlıca metrik tablosu **en yeni** `tflite` koşusunu gösterir; eskiler klasörde
geçmiş olarak kalır. Eski bir dosyayı düzenlemek yerine yeni dosya commit edin.

## "pipeline" sayısı ne demek

`pipeline` = bir Aşama 1 çağrısı + **yedi** ardışık Aşama 2 çağrısı, batch boyutu 1;
`OlcumZinciri.kt` her landmark'ı ayrı rafine ettiği için onu yansıtır. (`scripts/31` hâlâ
yalnızca iki nokta rafine edilirken yazılmış "~100–200 ms/cihaz" hedefini yazdırır.)
Gruplanmış bir Aşama 2 daha hızlı olabilir; uygulama bugün bunu yapmıyor.

## Ölçülmeyenler

- **Android gecikmesi.** Buradaki hiçbir benchmark telefonda çalışmadı. Masaüstü CPU'ları ve
  Python TFLite sarmalayıcısı, düşük segment bir ARM cihazın tahmini değildir. Bu,
  [bilinen boşlukların](METRICS.md) ilkidir.
- **Modelin çevresindeki her şey:** JPEG çözme, yeniden boyutlandırma, kılavuz çizimi, IMU
  enterpolasyonu. Yalnızca JVM'deki küçültme/kırpım/tensör kısmı ayrıca kapsanır.
- **Bellek, enerji, ısınma.** Tepe RAM, pil ya da ısıl kısıtlama verisi yok.
- **Hızlandırıcılar.** GPU ya da NNAPI delegate'i denenmedi.

## Sayıları dürüst okumak

- Tipik davranış için **p50**, kuyruk için **p90/p99** kullanın; asla minimumu değil.
- Dizüstü bilgisayarda koşudan koşuya gürültü beklenir. Geliştirme makinesinde art arda iki
  PyTorch-CPU koşusu, Aşama 1 medyanında yaklaşık %13 farklıydı (96.8 ms ve 85.4 ms). Karşılaştırmayı
  aynı makinede yapın, bundan küçük farkları gürültü sayın. GitHub'ın paylaşımlı CI
  runner'ları daha da gürültülüdür; bu yüzden CI benchmark'ı duman testi olarak çalıştırır,
  sonucu commit etmek yerine artifact olarak yükler.
- Aynı CPU'da eager PyTorch, TFLite'tan yavaştır (bkz. METRICS §3.1); ürün hakkında
  konuşurken TFLite sayısını verin.

## Cihaz üstü koşu için protokol (henüz uygulanmadı)

Biri telefonda ölçerse en az şunları kaydetmeli:

1. Cihaz modeli, SoC, RAM, Android sürümü, pil seviyesi, ısıl durum.
2. **Release** derlemesi (debug derlemesi evrensel ve optimize edilmemiştir).
3. Uçak modu, ekran açık, şarj takılı değil, cihaz birkaç dakikadır boşta.
4. 10 ısınma çıkarımı, sonra ≥ 100 ölçülen; Aşama 1, Aşama 2 (çağrı başına) ve tüm zincir
   için p50 / p90 / p99.
5. Bir ölçüm sırasında tepe RSS ve ısıl kısıtlamayı görmek için 5 dakikalık tekrarlı
   koşudan sonraki p50.

Otomatik bir harness olana kadar bu sonuçları `metrics/results.json` içinde
`device_reported` altına, `status: reported` ve tam cihaz bilgisiyle yazın — ve
`on_device_inference_latency` boşluğunu ancak böyle bir kayıt oluşunca kapatın.
