package com.skolyoz.olcum

import com.skolyoz.olcum.cikarim.OnIsleme
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

/**
 * Küçültmenin PIL `BILINEAR` ile eşleştiğini doğrular.
 *
 * NEDEN ÖNEMLİ
 * ------------
 * Gerçek telefon fotoğrafında ölçüldü: Android'in `createScaledBitmap`'i ile
 * PIL'in ölçeklemesi arasındaki fark açıya **0.29°'ye kadar** yansıyordu.
 * Boru hattının açı hatası medyanı 0.380°, etiket gürültüsü tabanı 0.23° —
 * yani hata bütçesinin yarısı kadar bir kalem sırf ölçekleme algoritmasından
 * geliyordu.
 *
 * Beklenen değerler `scripts/29_altin_kucultme.py` tarafından PIL'in
 * kendisinden üretiliyor.
 *
 * TOLERANS ±1
 * -----------
 * Pillow konvolüsyonu 22 bit sabit noktalı tam sayıyla yapıyor, buradaki
 * uygulama float ile. Tek tük pikselin ±1 sapması bundan; algoritmanın
 * farklı olmasından değil. Kabul edilebilir, çünkü ölçtüğümüz şey pikselin
 * kendisi değil ondan çıkan açı.
 */
class KucultmeTest {

    private data class Vaka(
        val kaynakG: Int, val kaynakY: Int,
        val hedefG: Int, val hedefY: Int,
        val tohum: Int, val ornekler: String,
    )

    private fun altinVeri(): List<Vaka> {
        val akis = javaClass.classLoader!!
            .getResourceAsStream("kucultme_altin_veri.csv")
        assertNotNull(
            "kucultme_altin_veri.csv yok — once " +
                "'python scripts/29_altin_kucultme.py'",
            akis,
        )
        return akis!!.bufferedReader().readLines()
            .filter { it.isNotBlank() && !it.startsWith("#") }
            .map {
                val p = it.split(",")
                Vaka(p[1].toInt(), p[2].toInt(), p[3].toInt(), p[4].toInt(),
                     p[5].toInt(), p[6])
            }
    }

    /**
     * Altın veriyi üreten Python görüntüsünün birebir aynısı — rastgelelik yok.
     *
     * Yüksek frekanslı desen bilerek: düz gradyanda iki algoritma da
     * neredeyse aynı sonucu verir, test yeşil kalır ve gerçek farkı kaçırırdık.
     * Aliasing ancak yüksek frekansta görünür.
     */
    private fun sentetik(g: Int, y: Int): IntArray {
        val p = IntArray(g * y)
        for (sy in 0 until y) {
            for (sx in 0 until g) {
                val r = (127 + 127 * sin(sx * 0.7) * cos(sy * 0.4))
                val gg = (127 + 127 * sin((sx + sy) * 0.9))
                val b = (sx * 3 + sy * 5) % 256
                p[sy * g + sx] = (0xFF shl 24) or
                    (kes(r) shl 16) or (kes(gg) shl 8) or b
            }
        }
        return p
    }

    /** numpy'nin `clip(...).astype(uint8)` davranışı: önce kırp, sonra kes. */
    private fun kes(v: Double): Int = v.coerceIn(0.0, 255.0).toInt()

    @Test
    fun `kucultme PIL BILINEAR ile eslesiyor`() {
        val vakalar = altinVeri()
        assertTrue("kucultme vakasi yok", vakalar.isNotEmpty())

        for (v in vakalar) {
            val kaynak = sentetik(v.kaynakG, v.kaynakY)
            val sonuc = OnIsleme.kucultBilinear(
                kaynak, v.kaynakG, v.kaynakY, v.hedefG, v.hedefY)

            for (ornek in v.ornekler.split(";")) {
                val p = ornek.split(":").map { it.toInt() }
                val (x, y, r, g, b) = p
                val c = sonuc[y * v.hedefG + x]
                val yer = "${v.kaynakG}x${v.kaynakY} -> " +
                    "${v.hedefG}x${v.hedefY} piksel($x,$y)"
                // Tolerans 1: Pillow konvolusyonu 22 bit sabit noktali tam
                // sayiyla yapiyor, buradaki uygulama float ile. Algoritma
                // ayni; fark yalnizca yuvarlamadan.
                assertEquals("$yer R", r.toDouble(),
                             ((c shr 16) and 0xFF).toDouble(), 1.0)
                assertEquals("$yer G", g.toDouble(),
                             ((c shr 8) and 0xFF).toDouble(), 1.0)
                assertEquals("$yer B", b.toDouble(),
                             (c and 0xFF).toDouble(), 1.0)
            }
        }
    }

    /** Küçültme boyutu doğru mu ve alfa kanalı opak mı. */
    @Test
    fun `kucultme dogru boyut ve opak alfa uretiyor`() {
        val kaynak = sentetik(750, 1000)
        val sonuc = OnIsleme.kucultBilinear(kaynak, 750, 1000, 640, 640)
        assertEquals(640 * 640, sonuc.size)
        for (p in sonuc) {
            assertEquals("alfa opak olmali", 0xFF, (p ushr 24) and 0xFF)
        }
    }

    /**
     * Aliasing testi: küçültme, kaynaktaki TÜM pikselleri hesaba katmalı.
     *
     * Tek satır beyaz, tek satır siyah olan bir görüntüyü yarıya küçültürsek:
     *   - antialiasing'li filtre  -> gri (ortalama)
     *   - düz örnekleme (aliasing) -> ya hep beyaz ya hep siyah
     * Bu, `createScaledBitmap` yerine bu fonksiyonu yazmamızın tam sebebi.
     */
    @Test
    fun `kucultme aliasing birakmiyor`() {
        val g = 64
        val y = 64
        val kaynak = IntArray(g * y) { i ->
            val satir = i / g
            if (satir % 2 == 0) (0xFF shl 24) or 0xFFFFFF else (0xFF shl 24)
        }
        val sonuc = OnIsleme.kucultBilinear(kaynak, g, y, 32, 32)
        for (p in sonuc) {
            val r = (p shr 16) and 0xFF
            assertTrue(
                "seritli goruntu kuculturken gri olmali, elde $r " +
                    "(aliasing varsa 0 veya 255 cikar)",
                r in 100..155,
            )
        }
    }

    @Test
    fun `olcek yokken goruntu degismiyor`() {
        val kaynak = sentetik(64, 64)
        val sonuc = OnIsleme.kucultBilinear(kaynak, 64, 64, 64, 64)
        for (i in kaynak.indices) {
            for (kaydir in listOf(16, 8, 0)) {
                assertEquals(
                    "piksel $i kanal $kaydir",
                    (kaynak[i] shr kaydir) and 0xFF,
                    (sonuc[i] shr kaydir) and 0xFF,
                )
            }
        }
    }

    private operator fun <T> List<T>.component4() = this[3]
    private operator fun <T> List<T>.component5() = this[4]
}
