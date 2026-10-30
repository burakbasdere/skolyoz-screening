package com.skolyoz.olcum.cikarim

import com.skolyoz.olcum.olcum.Nokta

/**
 * Çıkarım zincirinin saf (Android'e bağımsız) parçaları.
 *
 * NEDEN BITMAP DEĞİL INTARRAY
 * ---------------------------
 * Bu dosyadaki hiçbir fonksiyon `Bitmap` almıyor; hepsi ARGB `IntArray`
 * üzerinde çalışıyor. Sebebi test edilebilirlik: `Bitmap` android.jar'a bağlı
 * ve düz JVM birim testinde çalışmaz. Oysa buradaki adımlar
 * MOBIL_YAPILACAKLAR.md §5'te sayılan tuzakların çoğunun yaşadığı yer —
 * kanal sırası, normalizasyon, kırpım kaydırması, koordinat maplemesi.
 * Test edilemeyen bir yere bu kadar tuzak koyamayız.
 *
 * Android'e bağlı tek iş yeniden boyutlandırma; o [OlcumZinciri]'nde kalıyor.
 */
object OnIsleme {

    const val MODEL_BOYUT = 640
    const val KIRPIM = 128

    /** Aşama 2 ImageNet normalizasyonu bekliyor. Aşama 1 ise sadece /255. */
    val IMAGENET_ORT = floatArrayOf(0.485f, 0.456f, 0.406f)
    val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)

    /**
     * Görüntüyü hedef boyuta küçültür — PIL'in `BILINEAR`'ı ile aynı algoritma.
     *
     * NEDEN ANDROID'İN createScaledBitmap'İ KULLANILMIYOR
     * ---------------------------------------------------
     * Gerçek telefon fotoğrafında ölçüldü: aynı orijinalden gidildiğinde
     * Android'in ölçeklemesi ile PIL'inki **0.29°'ye kadar** farklı açı
     * veriyor. Boru hattının açı hatası medyanı 0.380°, etiket gürültüsü
     * tabanı 0.23° — yani hata bütçesinin yarısı kadar bir kalem, sırf
     * ölçekleme algoritmasından geliyordu.
     *
     * Sebep: `createScaledBitmap` düz bilineer örnekleme yapıp **aliasing**
     * bırakıyor. PIL küçültürken üçgen filtrenin DESTEĞİNİ ölçek kadar
     * genişletiyor, yani kaynaktaki tüm pikselleri hesaba katıyor.
     * 1000 → 640 küçültmede (1.56×) fark görünür hale geliyor.
     *
     * Hangisinin "doğru" olduğu keyfi değil: eğitim verisi Roboflow'da
     * 640×640'a gerildi ve o dönüşüm antialiasing'li. Model o dağılımı gördü.
     *
     * ALGORİTMA — Pillow'un `resample.c`'sinin birebir karşılığı:
     *   scale       = kaynak / hedef
     *   filtreOlcek = max(1, scale)          (büyütmede genişletme yok)
     *   destek      = 1.0 * filtreOlcek      (bilineer desteği 1.0)
     *   merkez      = (i + 0.5) * scale
     *   agirlik(j)  = ucgen((j + 0.5 - merkez) / filtreOlcek), sonra normalize
     * Önce yatay, sonra dikey geçiş (ayrılabilir filtre).
     *
     * Kotlin float ile hesaplıyor, Pillow sabit noktalı tam sayıyla; bu yüzden
     * tek tük piksel ±1 sapabilir. Birim testi bu toleransla karşılaştırıyor.
     */
    fun kucultBilinear(pikseller: IntArray, kaynakG: Int, kaynakY: Int,
                       hedefG: Int, hedefY: Int): IntArray {
        require(pikseller.size == kaynakG * kaynakY) {
            "piksel sayisi ${pikseller.size}, beklenen ${kaynakG * kaynakY}"
        }
        // Yatay geçiş: (kaynakG × kaynakY) -> (hedefG × kaynakY)
        val ara = FloatArray(hedefG * kaynakY * 3)
        val yatay = cekirdek(kaynakG, hedefG)
        for (y in 0 until kaynakY) {
            val satir = y * kaynakG
            for (x in 0 until hedefG) {
                var r = 0f; var g = 0f; var b = 0f
                val k = yatay[x]
                for (t in k.agirlik.indices) {
                    val p = pikseller[satir + k.basla + t]
                    val w = k.agirlik[t]
                    r += ((p shr 16) and 0xFF) * w
                    g += ((p shr 8) and 0xFF) * w
                    b += (p and 0xFF) * w
                }
                val i = (y * hedefG + x) * 3
                ara[i] = r; ara[i + 1] = g; ara[i + 2] = b
            }
        }
        // Dikey geçiş: (hedefG × kaynakY) -> (hedefG × hedefY)
        val cikti = IntArray(hedefG * hedefY)
        val dikey = cekirdek(kaynakY, hedefY)
        for (y in 0 until hedefY) {
            val k = dikey[y]
            for (x in 0 until hedefG) {
                var r = 0f; var g = 0f; var b = 0f
                for (t in k.agirlik.indices) {
                    val i = ((k.basla + t) * hedefG + x) * 3
                    val w = k.agirlik[t]
                    r += ara[i] * w; g += ara[i + 1] * w; b += ara[i + 2] * w
                }
                cikti[y * hedefG + x] = (0xFF shl 24) or
                    (kirp8(r) shl 16) or (kirp8(g) shl 8) or kirp8(b)
            }
        }
        return cikti
    }

    private class Cekirdek(val basla: Int, val agirlik: FloatArray)

    /** Her hedef piksel için kaynak aralığı ve normalize ağırlıklar. */
    private fun cekirdek(kaynak: Int, hedef: Int): Array<Cekirdek> {
        val olcek = kaynak.toDouble() / hedef
        val filtreOlcek = maxOf(1.0, olcek)
        val destek = 1.0 * filtreOlcek
        return Array(hedef) { i ->
            val merkez = (i + 0.5) * olcek
            val basla = maxOf(0, (merkez - destek + 0.5).toInt())
            val bitis = minOf(kaynak, (merkez + destek + 0.5).toInt())
            val n = bitis - basla
            val a = FloatArray(n)
            var toplam = 0.0
            for (t in 0 until n) {
                val d = Math.abs((basla + t + 0.5 - merkez) / filtreOlcek)
                val w = if (d < 1.0) 1.0 - d else 0.0
                a[t] = w.toFloat()
                toplam += w
            }
            if (toplam != 0.0) for (t in 0 until n) a[t] = (a[t] / toplam).toFloat()
            Cekirdek(basla, a)
        }
    }

    private fun kirp8(v: Float): Int {
        val i = Math.round(v)
        return if (i < 0) 0 else if (i > 255) 255 else i
    }

    /**
     * 640×640 ARGB piksellerden Aşama 1 tensörü: NHWC, RGB, `/255`.
     *
     * KANAL SIRASI RGB — bu ölçüldü, varsayılmadı. MOBIL_YAPILACAKLAR.md §5.4
     * BGR/RGB karışıklığının keypoint hatasını 24 px'den 40 px'e çıkardığını
     * kaydediyor. Python değerlendirmesi (`26_tflite_degerlendirme.py`) RGB
     * ile 3.44 px medyan verdi, yani doğru sıra bu.
     */
    fun asama1Tensor(pikseller: IntArray): FloatArray {
        require(pikseller.size == MODEL_BOYUT * MODEL_BOYUT) {
            "beklenen ${MODEL_BOYUT * MODEL_BOYUT} piksel, gelen ${pikseller.size}"
        }
        val t = FloatArray(pikseller.size * 3)
        var j = 0
        for (p in pikseller) {
            t[j++] = ((p shr 16) and 0xFF) / 255f   // R
            t[j++] = ((p shr 8) and 0xFF) / 255f    // G
            t[j++] = (p and 0xFF) / 255f            // B
        }
        return t
    }

    /**
     * Kırpım penceresinin sol-üst köşesi.
     *
     * Python referansı (`20_tam_degerlendirme.py::rafine_et`):
     *     sol = kaba[0] - KIRPIM / 2 ; int(round(sol))
     *
     * `Math.rint` kullanıyoruz, `Math.round` değil: Python'un `round()`'u
     * yarımları ÇİFTE yuvarlıyor (banker's rounding), Java'nın `Math.round`'u
     * yukarı. `Math.rint` Python ile aynı davranıyor. Tam .5 nadir ama
     * gerçekleştiğinde kırpım bir piksel kayar ve iki taraf sessizce ayrışır.
     */
    fun kirpimKosesi(merkez: Nokta): Pair<Int, Int> =
        Math.rint(merkez.x - KIRPIM / 2.0).toInt() to
            Math.rint(merkez.y - KIRPIM / 2.0).toInt()

    /**
     * Görüntüden [KIRPIM]×[KIRPIM] pencere keser; dışarı taşan kısmı SİYAHLA
     * doldurur.
     *
     * Siyah dolgu keyfi değil: Python tarafında PIL'in `crop`'u sınır dışını
     * 0 ile dolduruyor ve Aşama 2 bu dağılımla eğitildi/değerlendirildi.
     * Kenar yansıtma veya kırpma penceresini içeri çekme gibi "daha iyi"
     * görünen çözümler modeli görmediği bir dağılıma sokar.
     *
     * Kişi karenin kenarında durduğunda bu gerçekten devreye giriyor.
     */
    fun kirp(pikseller: IntArray, genislik: Int, yukseklik: Int,
             solX: Int, ustY: Int): IntArray {
        require(pikseller.size == genislik * yukseklik) {
            "piksel sayisi ${pikseller.size}, beklenen ${genislik * yukseklik}"
        }
        val cikti = IntArray(KIRPIM * KIRPIM)          // 0 = saydam siyah
        for (sy in 0 until KIRPIM) {
            val ky = ustY + sy
            if (ky < 0 || ky >= yukseklik) continue
            val satirKaynak = ky * genislik
            val satirHedef = sy * KIRPIM
            for (sx in 0 until KIRPIM) {
                val kx = solX + sx
                if (kx < 0 || kx >= genislik) continue
                cikti[satirHedef + sx] = pikseller[satirKaynak + kx]
            }
        }
        return cikti
    }

    /**
     * 128×128 kırpımdan Aşama 2 tensörü: NHWC, RGB, `/255`, sonra ImageNet.
     *
     * Aşama 1 sadece `/255` istiyor, Aşama 2 üstüne ImageNet normalizasyonu.
     * İkisini karıştırmak MOBIL_YAPILACAKLAR.md §5.6'da ayrıca uyarılan tuzak.
     */
    fun asama2Tensor(kirpim: IntArray): FloatArray {
        require(kirpim.size == KIRPIM * KIRPIM) {
            "beklenen ${KIRPIM * KIRPIM} piksel, gelen ${kirpim.size}"
        }
        val t = FloatArray(kirpim.size * 3)
        var j = 0
        for (p in kirpim) {
            t[j++] = (((p shr 16) and 0xFF) / 255f - IMAGENET_ORT[0]) / IMAGENET_STD[0]
            t[j++] = (((p shr 8) and 0xFF) / 255f - IMAGENET_ORT[1]) / IMAGENET_STD[1]
            t[j++] = ((p and 0xFF) / 255f - IMAGENET_ORT[2]) / IMAGENET_STD[2]
        }
        return t
    }

    /** [yumusakArgmax] çıktısı: koordinat + iki bağımsız güven ölçüsü. */
    data class HeatmapSonucu(
        val nokta: Nokta,
        /** Dağılımın yayılması. Dağınık heatmap'i yakalar. */
        val entropi: Double,
        /**
         * soft-argmax'ın verdiği nokta ile heatmap'in gerçek tepesi (argmax)
         * arasındaki mesafe, kırpım pikselinde.
         *
         * Entropinin KÖR NOKTASINI kapatıyor: iki keskin tepeli bir heatmap'te
         * entropi düşük çıkar (kütle iki dar kümede toplanmış) ama soft-argmax
         * ikisinin ortasını verir ve o nokta hiçbir tepenin üstünde değildir.
         * Bu mesafe her iki bozulmayı da yakalar.
         */
        val tepeUzakligi: Double,
    )

    /** Heatmap keskinliğini yumuşatan sıcaklık. Eğitimdeki değerle aynı. */
    const val SICAKLIK = 10.0f

    /**
     * Heatmap'ten alt-piksel koordinat + **güven ölçüsü** çıkarır.
     *
     * `17_rafinaj_model.py`'deki soft-argmax'ın birebir portu, ama ek olarak
     * dağılımın ENTROPİSİNİ de döndürüyor.
     *
     * NEDEN ENTROPİ
     * -------------
     * soft-argmax dağılımın AĞIRLIK MERKEZİNİ alıyor. Dağılım keskinse merkez
     * sağlamdır. Dağınıksa merkez, girdideki 1/255'lik bir farkla bile
     * onlarca piksel kayar — gerçek telefon fotoğrafında ölçüldü: aynı sahnenin
     * üç karesi 1.25°, 3.37°, 6.73° verdi ve Aşama 2 aynı girdide 16.8 px'e
     * kadar sapıyordu.
     *
     * Kritik nokta: model bu durumda HATA VERMİYOR, kendinden emin görünen bir
     * sayı üretiyor. Entropi, o sayının arkasında gerçek bir tepe olup
     * olmadığını söyleyen tek ölçü.
     *
     * Eşik için bkz. [ENTROPI_ESIGI].
     *
     * @param heatmap `(64, 64)` düzleştirilmiş, satır-önce
     */
    fun yumusakArgmax(heatmap: FloatArray, n: Int = 64): HeatmapSonucu {
        require(heatmap.size == n * n) {
            "beklenen ${n * n} deger, gelen ${heatmap.size}"
        }
        // Taşma koruması: en büyük değeri çıkarmak softmax'ı değiştirmez.
        var enBuyuk = Float.NEGATIVE_INFINITY
        for (v in heatmap) if (v > enBuyuk) enBuyuk = v

        val p = DoubleArray(heatmap.size)
        var toplam = 0.0
        for (i in heatmap.indices) {
            val e = Math.exp(((heatmap[i] - enBuyuk) * SICAKLIK).toDouble())
            p[i] = e
            toplam += e
        }

        var xNorm = 0.0
        var yNorm = 0.0
        var entropi = 0.0
        var enBuyukOlasilik = -1.0
        var tepeIndeks = 0
        val bolen = (n - 1).toDouble()
        for (i in p.indices) {
            val olasilik = p[i] / toplam
            val satir = i / n
            val sutun = i % n
            xNorm += olasilik * (sutun / bolen)
            yNorm += olasilik * (satir / bolen)
            entropi -= olasilik * Math.log(olasilik + 1e-12)
            if (olasilik > enBuyukOlasilik) {
                enBuyukOlasilik = olasilik
                tepeIndeks = i
            }
        }
        val olcek = (KIRPIM - 1).toDouble()
        val nokta = Nokta(xNorm * olcek, yNorm * olcek)
        val tepeX = (tepeIndeks % n) / bolen * olcek
        val tepeY = (tepeIndeks / n) / bolen * olcek
        return HeatmapSonucu(
            nokta = nokta,
            entropi = entropi,
            tepeUzakligi = Math.hypot(nokta.x - tepeX, nokta.y - tepeY),
        )
    }

    /**
     * Aşama 2'nin bu nokta için verdiği cevap kullanılabilir mi.
     *
     * İKİ ÖLÇÜT, ÇÜNKÜ İKİ FARKLI BOZULMA VAR
     * ---------------------------------------
     *   dağınık heatmap  -> entropi yüksek
     *   çift tepeli      -> entropi DÜŞÜK ama tepe uzaklığı yüksek
     *
     * İkincisi ilk taslakta gözden kaçmıştı; birim testi yakaladı.
     *
     * ÖLÇÜLEN MALİYET
     * ---------------
     * Test + doğrulama setinde 101 ölçümün 5'i reddediliyor (%5.0).
     * Kararsız çıkan gerçek telefon ölçümlerinde 3/3 reddediliyor.
     */
    fun guvenilirMi(s: HeatmapSonucu): Boolean =
        s.entropi <= ENTROPI_ESIGI && s.tepeUzakligi <= TEPE_UZAKLIK_ESIGI

    /**
     * soft-argmax ile gerçek tepe arasındaki kabul edilebilir mesafe (kırpım
     * pikselinde).
     *
     * Test + doğrulama setinde 202 noktanın dağılımı:
     *     p50 6.57   p90 9.93   p95 11.16   p99 36.96   max 55.05
     *
     * Telefon fotoğraflarında: 5.72, 16.14, 23.36, 27.58, 41.27, 51.64
     *
     * 15 px, p95 ile telefon ölçümlerinin arasında duruyor.
     */
    const val TEPE_UZAKLIK_ESIGI = 15.0

    /**
     * Bu değerin üstündeki entropi "ölçemedim" demektir.
     *
     * Keyfi değil, ölçülerek seçildi. Test + doğrulama setindeki 202 noktanın
     * entropi dağılımı:
     *
     *     p50 1.698   p90 1.997   p95 2.043   p99 2.908   max 6.165
     *
     * Eşik seçenekleri ve geçerli noktalarda yol açtığı red oranı:
     *
     *     2.0 -> %9.4     2.5 -> %2.0     3.0 -> %1.0
     *
     * 2.5 seçildi: geçerli ölçümlerin yalnızca %2'sini reddederken, kararsız
     * çıkan gerçek telefon fotoğraflarının 6 noktasından 5'ini yakalıyor
     * (oradaki entropiler 1.88, 3.09, 5.63, 6.45, 6.77, 6.92 idi).
     *
     * Bu eşik modeli İYİLEŞTİRMİYOR. Yaptığı tek şey, sessizce yanlış olanı
     * görünür biçimde reddedilen hâle çevirmek. Asıl çözüm telefon kamerasıyla
     * çekilmiş eğitim verisi toplamak — MOBIL_YAPILACAKLAR.md §7'deki birinci
     * madde.
     */
    const val ENTROPI_ESIGI = 2.5

    /**
     * Aşama 2'nin kırpım içi koordinatını tam görüntü koordinatına taşır.
     * Python: `(round(sol) + kx, round(ust) + ky)`
     */
    fun kirpimdanGoruntuye(kirpimIci: Nokta, solX: Int, ustY: Int): Nokta =
        Nokta(solX + kirpimIci.x, ustY + kirpimIci.y)

    /**
     * 640×640 model uzayından orijinal çözünürlüğe geri map'ler.
     *
     * Eğitim verisi 640×640'a STRETCH edilmişti, letterbox değil. Çıkarımda da
     * aynı stretch uygulanıyor — ama açı ASLA 640 uzayında hesaplanmıyor:
     * stretch en-boy oranını bozar. 4:3 bir fotoğrafta bu adımı atlamak ~%25
     * açı hatası demek. MOBIL_YAPILACAKLAR.md §3.
     */
    fun orijinaleMaple(nokta: Nokta, orijinalGenislik: Int, orijinalYukseklik: Int): Nokta =
        Nokta(
            x = nokta.x * orijinalGenislik / MODEL_BOYUT,
            y = nokta.y * orijinalYukseklik / MODEL_BOYUT,
        )
}
