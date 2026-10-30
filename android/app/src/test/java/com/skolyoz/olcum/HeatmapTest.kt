package com.skolyoz.olcum

import com.skolyoz.olcum.cikarim.OnIsleme
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.exp

/**
 * Soft-argmax ve **entropi** hesabının Python referansıyla eşleştiğini
 * doğrular.
 *
 * NEDEN AYRI VE ÖNEMLİ
 * --------------------
 * Aşama 2 önce koordinat varyantıydı, soft-argmax modelin içindeydi. Entropi
 * kapısı eklenince heatmap varyantına geçildi ve soft-argmax Kotlin'e taşındı.
 * Yani artık ölçümün alt-piksel hassasiyeti **bu koda** bağlı.
 *
 * Entropi ise kapının kararını veriyor: yanlış hesaplanırsa ya geçerli
 * ölçümler reddedilir ya da güvenilmez ölçümler kullanıcıya doğruymuş gibi
 * gösterilir. İkincisi bir tarama aracında daha kötü.
 */
class HeatmapTest {

    private data class Vaka(
        val n: Int, val x: Double, val y: Double,
        val entropi: Double, val heatmap: FloatArray,
    )

    private fun altinVeri(): List<Vaka> {
        val akis = javaClass.classLoader!!
            .getResourceAsStream("cikarim_altin_veri.csv")
        assertNotNull("cikarim_altin_veri.csv yok", akis)
        return akis!!.bufferedReader().readLines()
            .filter { it.startsWith("heatmap,") }
            .map {
                val p = it.split(",")
                val h = p[5].split(";").map { v -> v.toFloat() }.toFloatArray()
                Vaka(p[1].toInt(), p[2].toDouble(), p[3].toDouble(),
                     p[4].toDouble(), h)
            }
    }

    @Test
    fun `soft-argmax Python ile ayni koordinati veriyor`() {
        val vakalar = altinVeri()
        assertTrue("heatmap vakasi yok", vakalar.isNotEmpty())
        for (v in vakalar) {
            val s = OnIsleme.yumusakArgmax(v.heatmap, v.n)
            // 1e-3 px: Kotlin double, Python float64 — fark yalnızca
            // toplama sırasından gelen kayan nokta gürültüsü.
            assertEquals("x", v.x, s.nokta.x, 1e-3)
            assertEquals("y", v.y, s.nokta.y, 1e-3)
        }
    }

    @Test
    fun `entropi Python ile ayni`() {
        val vakalar = altinVeri()
        assertTrue("heatmap vakasi yok", vakalar.isNotEmpty())
        for (v in vakalar) {
            val s = OnIsleme.yumusakArgmax(v.heatmap, v.n)
            assertEquals("entropi", v.entropi, s.entropi, 1e-6)
        }
    }

    private fun gauss(n: Int, cx: Double, cy: Double, sigma: Double): FloatArray {
        val h = FloatArray(n * n)
        for (y in 0 until n) for (x in 0 until n) {
            val d = (x - cx) * (x - cx) + (y - cy) * (y - cy)
            h[y * n + x] = (6.0 * exp(-d / (2 * sigma * sigma))).toFloat()
        }
        return h
    }

    @Test
    fun `keskin tek tepe guvenilir sayiliyor`() {
        val s = OnIsleme.yumusakArgmax(gauss(64, 31.5, 31.5, 1.2), 64)
        assertTrue(
            "keskin tepe guvenilir olmali (entropi ${s.entropi}, " +
                "tepe uzakligi ${s.tepeUzakligi})",
            OnIsleme.guvenilirMi(s),
        )
    }

    /**
     * Entropinin KÖR NOKTASI — ve ikinci ölçütün var olma sebebi.
     *
     * İki keskin tepeli bir heatmap'te soft-argmax ikisinin **ortasını** verir
     * ve o nokta hiçbir tepenin üstünde değildir; yani model "bilmiyorum"
     * derken kod kendinden emin bir koordinat üretir.
     *
     * Ama entropi bu durumu YAKALAMIYOR: kütle iki dar kümede toplandığı için
     * entropi düşük çıkıyor (~0.72). Bu test ilk taslakta kırıldı ve eksiği
     * ortaya çıkardı — tepe uzaklığı ölçütü bu yüzden eklendi.
     */
    @Test
    fun `iki tepeli heatmap entropiyle degil tepe uzakligiyla yakalaniyor`() {
        val n = 64
        val h = FloatArray(n * n)
        val a = gauss(n, 12.0, 32.0, 2.0)
        val b = gauss(n, 52.0, 32.0, 2.0)
        for (i in h.indices) h[i] = a[i] + b[i]
        val s = OnIsleme.yumusakArgmax(h, n)

        assertTrue(
            "iki keskin tepede entropi DUSUK kalir — kor nokta budur " +
                "(entropi ${s.entropi})",
            s.entropi < OnIsleme.ENTROPI_ESIGI,
        )
        assertTrue(
            "tepe uzakligi esigi asmali (elde ${s.tepeUzakligi})",
            s.tepeUzakligi > OnIsleme.TEPE_UZAKLIK_ESIGI,
        )
        assertTrue("guvenilir sayilmamali", !OnIsleme.guvenilirMi(s))

        // Cikan nokta gercekten iki tepenin ORTASINDA, yani hicbirinin
        // ustunde degil — kapinin neden gerekli oldugunun kaniti.
        val ortaX = (12.0 + 52.0) / 2.0 * (127.0 / (n - 1))
        assertTrue(
            "soft-argmax iki tepenin ortasini vermeli (elde ${s.nokta.x})",
            abs(s.nokta.x - ortaX) < 6.0,
        )
    }

    @Test
    fun `duz heatmap merkezi ve yuksek entropi veriyor`() {
        val n = 64
        val duz = FloatArray(n * n) { 0f }
        val s = OnIsleme.yumusakArgmax(duz, n)
        assertEquals("duz dagilimda x merkez olmali", 63.5, s.nokta.x, 0.01)
        assertEquals("duz dagilimda y merkez olmali", 63.5, s.nokta.y, 0.01)
        // Duzgun dagilimin entropisi ln(64*64) = 8.317
        assertEquals("duzgun dagilim entropisi", 8.3178, s.entropi, 1e-3)
        assertTrue("duz heatmap guvenilir sayilmamali", !OnIsleme.guvenilirMi(s))
    }

    /**
     * Eşikler ölçülerek seçildi; biri değiştirilirse bu test uyarır.
     *
     * entropi        test+val p95 2.043  |  telefon 1.88 … 6.92
     * tepe uzakligi  test+val p95 11.16  |  telefon 5.72 … 51.64
     *
     * Birleşik kapının ölçülen maliyeti: test+val'de 101 ölçümün 5'i (%5.0),
     * telefon ölçümlerinde 3/3.
     */
    @Test
    fun `esikler test seti ile telefon olcumlerinin arasinda duruyor`() {
        assertTrue("entropi esigi test setinin p95'inin ustunde olmali",
                   OnIsleme.ENTROPI_ESIGI > 2.043)
        assertTrue("entropi esigi kararsiz telefon olcumlerinin altinda olmali",
                   OnIsleme.ENTROPI_ESIGI < 3.09)
        assertTrue("tepe uzakligi esigi test setinin p95'inin ustunde olmali",
                   OnIsleme.TEPE_UZAKLIK_ESIGI > 11.16)
        assertTrue("tepe uzakligi esigi kararsiz olcumlerin altinda olmali",
                   OnIsleme.TEPE_UZAKLIK_ESIGI < 16.14)
    }
}
