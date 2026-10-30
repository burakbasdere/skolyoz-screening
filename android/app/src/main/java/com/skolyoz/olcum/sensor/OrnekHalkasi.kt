package com.skolyoz.olcum.sensor

/**
 * Zaman damgalı yerçekimi örneklerinin halka tamponu + deklanşör eşleştirmesi.
 *
 * Bilerek Android'e BAĞIMSIZ: `SensorEvent` de `Context` de geçmiyor. Böylece
 * eşleştirme mantığı cihaz/emülatör gerekmeden düz JVM birim testiyle
 * doğrulanabiliyor. Bu ürünün en hassas halkası — ölçtüğümüz şey bir açı ve
 * yanlış anın yerçekimi vektörü doğrudan açı hatasına yazılıyor.
 *
 * Eşzamanlılık: tek yazan (sensör thread'i), tek okuyan (kamera geri çağrısı).
 * Yazma ve okuma aynı kilidi paylaşıyor.
 */
class OrnekHalkasi(private val kapasite: Int = 256) {

    private val zaman = LongArray(kapasite)
    private val gx = FloatArray(kapasite)
    private val gy = FloatArray(kapasite)
    private val gz = FloatArray(kapasite)
    private val kilit = Any()
    private var yazilan = 0

    fun temizle() = synchronized(kilit) { yazilan = 0 }

    fun ekle(zamanNanos: Long, x: Float, y: Float, z: Float) = synchronized(kilit) {
        val i = yazilan % kapasite
        zaman[i] = zamanNanos
        gx[i] = x; gy[i] = y; gz[i] = z
        yazilan++
    }

    /**
     * Deklanşör anındaki yerçekimi vektörü — deklanşörü KUŞATAN iki örnek
     * arasında doğrusal enterpolasyonla.
     *
     * NEDEN ENTERPOLASYON, NEDEN "EN YAKIN ÖRNEK" DEĞİL
     * --------------------------------------------------
     * Sensör ~20 ms'de bir örnekliyor. En yakın örneği seçmek, deklanşör iki
     * örneğin ortasına denk geldiğinde ±10 ms'lik YAPISAL hata bırakır.
     * Redmi Note 9 Pro'da ölçüldü: sapma 2,3 ms ve 7,6 ms.
     *
     * Deklanşöre basarken el ~30°/s dönebiliyor; 10 ms'de bu 0,3° eder.
     * Etiket gürültüsü tabanımız 0,23°, boru hattının açı hatası medyanı
     * 0,380°. Yani en yakın örneği seçmek, zincirin en hassas halkasıyla aynı
     * büyüklükte hata ekliyor.
     *
     * Enterpolasyon hatayı örnekleme periyoduna değil, yerçekimi vektörünün
     * iki örnek arasındaki doğrusallıktan sapmasına indirir — el titremesi
     * ölçeğinde ihmal edilebilir.
     *
     * @param hedefNanos `ImageProxy.imageInfo.timestamp` (donanım deklanşör anı)
     * @return kuşatma yoksa ve en yakın örnek de [toleransNanos]'u aşıyorsa
     *         `null` — ölçüm reddedilmeli. Bayat örnekle sessizce ölçmektense
     *         reddetmek doğru.
     */
    fun eslestir(
        hedefNanos: Long,
        toleransNanos: Long = TOLERANS_NS,
    ): YercekimiOrnegi? = synchronized(kilit) {
        val n = minOf(yazilan, kapasite)
        if (n == 0) return null

        // önce = en büyük t <= hedef,  sonra = en küçük t >= hedef
        var onceIdx = -1
        var sonraIdx = -1
        var onceFark = Long.MAX_VALUE
        var sonraFark = Long.MAX_VALUE
        for (k in 0 until n) {
            val d = hedefNanos - zaman[k]
            if (d >= 0 && d < onceFark) { onceFark = d; onceIdx = k }
            if (d <= 0 && -d < sonraFark) { sonraFark = -d; sonraIdx = k }
        }

        if (onceIdx >= 0 && sonraIdx >= 0) {
            val aralik = zaman[sonraIdx] - zaman[onceIdx]
            // Aralığın kendisi toleransın katbekat üstündeyse örnekleme
            // durmuştur (uygulama arka plandaydı, sensör askıya alındı).
            // Böyle bir boşluğun iki ucu arasında doğrusallık varsayılamaz.
            if (aralik > BOSLUK_KATSAYISI * toleransNanos) return null
            val t = if (aralik == 0L) 0.0
                    else (hedefNanos - zaman[onceIdx]).toDouble() / aralik
            return YercekimiOrnegi(
                gx = araDeger(gx[onceIdx], gx[sonraIdx], t),
                gy = araDeger(gy[onceIdx], gy[sonraIdx], t),
                gz = araDeger(gz[onceIdx], gz[sonraIdx], t),
                sapmaMs = 0.0,
                enterpole = true,
                komsuAralikMs = aralik / 1_000_000.0,
            )
        }

        // Kuşatma yok: deklanşör tamponun bir ucunun ötesinde kalmış.
        // Yalnızca en yakın örnek toleransın içindeyse kabul et.
        val idx = if (onceIdx >= 0) onceIdx else sonraIdx
        val fark = if (onceIdx >= 0) onceFark else sonraFark
        if (idx < 0 || fark > toleransNanos) return null
        return YercekimiOrnegi(
            gx = gx[idx], gy = gy[idx], gz = gz[idx],
            sapmaMs = fark / 1_000_000.0,
            enterpole = false,
        )
    }

    private fun araDeger(a: Float, b: Float, t: Double): Float =
        (a + (b - a) * t).toFloat()

    companion object {
        /**
         * Deklanşör ile en yakın örnek arasındaki kabul edilebilir sapma.
         * Yalnızca kuşatma bulunamadığında devreye girer.
         */
        const val TOLERANS_NS = 20_000_000L  // 20 ms

        /** Bu katsayıdan geniş bir örnek boşluğu enterpolasyonu geçersiz kılar. */
        const val BOSLUK_KATSAYISI = 4
    }
}
