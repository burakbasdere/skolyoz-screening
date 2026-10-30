package com.skolyoz.olcum

import com.skolyoz.olcum.cikarim.Asama1Coz
import com.skolyoz.olcum.cikarim.OnIsleme
import com.skolyoz.olcum.olcum.Nokta
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Çıkarım zincirinin Python referansıyla eşleştiğini doğrular.
 *
 * Beklenen değerler `scripts/27_altin_cikarim.py` tarafından üretiliyor ve o
 * script ölçümü YAPAN modülü (`26_tflite_degerlendirme.py`) import ediyor.
 * Yani karşılaştırdığımız şey, test setinde gerçekten doğruladığımız
 * davranışın kendisi — ikinci bir "referans" yazılmadı.
 *
 * TFLite çağrısı burada yok: android.jar ve yerel kütüphaneler düz JVM
 * testinde çalışmıyor. Ama zincirin hata yapmaya en müsait yerleri —
 * kanal sırası, normalizasyon, kırpım kaydırması ve yuvarlaması, koordinat
 * maplemesi — hepsi saf fonksiyon olduğu için burada kapsanıyor.
 */
class CikarimTest {

    private data class Satir(val tip: String, val alan: List<String>)

    private fun altinVeri(): List<Satir> {
        val akis = javaClass.classLoader!!
            .getResourceAsStream("cikarim_altin_veri.csv")
        assertNotNull(
            "cikarim_altin_veri.csv yok — once " +
                "'.venv-tflite/Scripts/python.exe scripts/27_altin_cikarim.py'",
            akis,
        )
        return akis!!.bufferedReader().readLines()
            .filter { it.isNotBlank() && !it.startsWith("#") }
            .map { val p = it.split(","); Satir(p[0], p.drop(1)) }
    }

    private fun vakalar(tip: String) = altinVeri().filter { it.tip == tip }

    // ------------------------------------------------------------ decode

    @Test
    fun `asama 1 cozme Python ile ayni noktayi seciyor`() {
        val v = vakalar("decode")
        assertTrue("decode vakasi yok", v.isNotEmpty())
        for (s in v) {
            val aday = s.alan[0].toInt()
            val beklenenVar = s.alan[1] == "VAR"
            val tensor = FloatArray(11 * aday) { s.alan[6 + it].toFloat() }

            val sonuc = Asama1Coz.coz(tensor, aday)

            if (!beklenenVar) {
                assertNull("guven esigi altinda tespit dondu", sonuc)
                continue
            }
            assertNotNull("tespit bekleniyordu, null dondu", sonuc)
            assertEquals("sag.x", s.alan[2].toDouble(), sonuc!!.sag.x, 1e-4)
            assertEquals("sag.y", s.alan[3].toDouble(), sonuc.sag.y, 1e-4)
            assertEquals("sol.x", s.alan[4].toDouble(), sonuc.sol.x, 1e-4)
            assertEquals("sol.y", s.alan[5].toDouble(), sonuc.sol.y, 1e-4)
        }
    }

    // ------------------------------------------------------- kirpim kosesi

    /**
     * Yuvarlama testi. Python'un `round()`'u yarımları ÇİFTE yuvarlıyor,
     * Java'nın `Math.round`'u yukarı. Altın veride bilerek `.5` vakaları var;
     * `Math.rint` yerine `Math.round` kullanılsa bu test kırılır.
     */
    @Test
    fun `kirpim kosesi Python yuvarlamasiyla ayni`() {
        val v = vakalar("kose")
        assertTrue("kose vakasi yok", v.isNotEmpty())
        for (s in v) {
            val merkez = Nokta(s.alan[0].toDouble(), s.alan[1].toDouble())
            val (solX, ustY) = OnIsleme.kirpimKosesi(merkez)
            assertEquals("solX (merkez=${merkez.x})", s.alan[2].toInt(), solX)
            assertEquals("ustY (merkez=${merkez.y})", s.alan[3].toInt(), ustY)
        }
    }

    // -------------------------------------------------------------- kirpim

    /** Altın veriyi üreten Python görüntüsünün birebir aynısı. */
    private fun sentetikGoruntu(g: Int, y: Int): IntArray {
        val p = IntArray(g * y)
        for (sy in 0 until y) {
            for (sx in 0 until g) {
                val r = (sx * 7) % 256
                val gg = (sy * 11) % 256
                val b = (sx + sy) % 256
                p[sy * g + sx] = (0xFF shl 24) or (r shl 16) or (gg shl 8) or b
            }
        }
        return p
    }

    /**
     * Sınır dışına taşan kırpımda dolgu davranışı.
     *
     * PIL sınır dışını SİYAHLA dolduruyor ve Aşama 2 bu dağılımla
     * değerlendirildi. Kenar yansıtma gibi "daha iyi" görünen bir seçim
     * modeli görmediği bir dağılıma sokardı. Kişi karenin kenarında durduğunda
     * bu gerçekten devreye giriyor.
     */
    @Test
    fun `kirpim sinir disini siyahla dolduruyor`() {
        val v = vakalar("kirp")
        assertTrue("kirp vakasi yok", v.isNotEmpty())
        for (s in v) {
            val g = s.alan[0].toInt()
            val y = s.alan[1].toInt()
            val solX = s.alan[2].toInt()
            val ustY = s.alan[3].toInt()
            val kaynak = sentetikGoruntu(g, y)
            val kirpim = OnIsleme.kirp(kaynak, g, y, solX, ustY)

            for (ornek in s.alan[4].split(";")) {
                val (sx, sy, r, gg, b) = ornek.split(":").map { it.toInt() }
                val p = kirpim[sy * OnIsleme.KIRPIM + sx]
                val yer = "kirpim($solX,$ustY) piksel($sx,$sy)"
                assertEquals("$yer R", r, (p shr 16) and 0xFF)
                assertEquals("$yer G", gg, (p shr 8) and 0xFF)
                assertEquals("$yer B", b, p and 0xFF)
            }
        }
    }

    private operator fun <T> List<T>.component4() = this[3]
    private operator fun <T> List<T>.component5() = this[4]

    // ------------------------------------------------- normalizasyon / kanal

    /**
     * Kanal sırası ve normalizasyon.
     *
     * MOBIL_YAPILACAKLAR.md §5.4: RGB/BGR karışıklığı keypoint hatasını
     * 24 px'den 40 px'e çıkarmıştı. §5.6: iki aşamanın normalizasyonu farklı
     * (Aşama 1 sadece /255, Aşama 2 üstüne ImageNet). Bu iki test, ikisinden
     * biri karışırsa anında kırılır.
     */
    @Test
    fun `asama 1 tensoru RGB ve bolu 255`() {
        val v = vakalar("norm1")
        assertTrue("norm1 vakasi yok", v.isNotEmpty())
        for (s in v) {
            val argb = s.alan[0].toLong().toInt()
            val pikseller = IntArray(OnIsleme.MODEL_BOYUT * OnIsleme.MODEL_BOYUT) { argb }
            val t = OnIsleme.asama1Tensor(pikseller)
            assertEquals("R", s.alan[1].toFloat(), t[0], 1e-6f)
            assertEquals("G", s.alan[2].toFloat(), t[1], 1e-6f)
            assertEquals("B", s.alan[3].toFloat(), t[2], 1e-6f)
        }
    }

    @Test
    fun `asama 2 tensoru RGB ve ImageNet normalizasyonu`() {
        val v = vakalar("norm2")
        assertTrue("norm2 vakasi yok", v.isNotEmpty())
        for (s in v) {
            val argb = s.alan[0].toLong().toInt()
            val kirpim = IntArray(OnIsleme.KIRPIM * OnIsleme.KIRPIM) { argb }
            val t = OnIsleme.asama2Tensor(kirpim)
            assertEquals("R", s.alan[1].toFloat(), t[0], 1e-5f)
            assertEquals("G", s.alan[2].toFloat(), t[1], 1e-5f)
            assertEquals("B", s.alan[3].toFloat(), t[2], 1e-5f)
        }
    }

    /** İki aşamanın normalizasyonu birbirine karışmamalı. */
    @Test
    fun `iki asamanin normalizasyonu ayni degil`() {
        val siyah = 0xFF000000.toInt()
        val t1 = OnIsleme.asama1Tensor(
            IntArray(OnIsleme.MODEL_BOYUT * OnIsleme.MODEL_BOYUT) { siyah })
        val t2 = OnIsleme.asama2Tensor(
            IntArray(OnIsleme.KIRPIM * OnIsleme.KIRPIM) { siyah })
        assertEquals("asama 1'de siyah 0 olmali", 0f, t1[0], 1e-9f)
        assertTrue("asama 2'de siyah ImageNet ile negatife gitmeli (elde ${t2[0]})",
                   t2[0] < -2.0f)
    }

    // ------------------------------------------------------------- mapleme

    /**
     * 640 uzayından orijinal çözünürlüğe geri mapleme.
     * Bu adım atlanırsa 4:3 bir fotoğrafta ~%25 açı hatası oluşuyor.
     */
    @Test
    fun `orijinale mapleme Python ile ayni`() {
        val v = vakalar("map")
        assertTrue("map vakasi yok", v.isNotEmpty())
        for (s in v) {
            val n = Nokta(s.alan[0].toDouble(), s.alan[1].toDouble())
            val sonuc = OnIsleme.orijinaleMaple(n, s.alan[2].toInt(), s.alan[3].toInt())
            assertEquals("x", s.alan[4].toDouble(), sonuc.x, 1e-6)
            assertEquals("y", s.alan[5].toDouble(), sonuc.y, 1e-6)
        }
    }

    @Test
    fun `kirpim ici koordinat goruntuye tasiniyor`() {
        val sonuc = OnIsleme.kirpimdanGoruntuye(Nokta(63.25, 70.75), 100, -20)
        assertEquals(163.25, sonuc.x, 1e-9)
        assertEquals(50.75, sonuc.y, 1e-9)
    }
}
