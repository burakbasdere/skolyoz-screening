package com.skolyoz.olcum

import com.skolyoz.olcum.olcum.Nokta
import com.skolyoz.olcum.olcum.Olcum
import com.skolyoz.olcum.olcum.Yon
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

/**
 * Kotlin portunun Python referansıyla eşleştiğini doğrular.
 *
 * Beklenen değerler ELLE YAZILMADI — `scripts/23_altin_veri.py` bunları
 * `scripts/19_kilavuz_cizim.py`'yi çalıştırarak üretiyor. Elle yazsaydık
 * insan hesabı ile Kotlin kodu aynı yanlış işaret konvansiyonunu paylaşabilir
 * ve test yeşil kalırken ürün ters ölçerdi.
 *
 * Python tarafı değişirse altın veri yeniden üretilmeli; iki taraf ayrışırsa
 * bu test kırılır.
 */
class OlcumTest {

    /** MOBIL_YAPILACAKLAR.md Faz 3'ün koyduğu kabul eşiği. */
    private val tolerans = 0.05

    private data class Vaka(
        val tip: String,
        val a: Double, val b: Double, val c: Double,
        val d: Double, val e: Double,
        val beklenen1: Double, val beklenen2: Double?,
    )

    private fun altinVeri(): List<Vaka> {
        val akis = javaClass.classLoader!!.getResourceAsStream("olcum_altin_veri.csv")
        assertNotNull("olcum_altin_veri.csv bulunamadi — " +
            "once 'python scripts/23_altin_veri.py' calistir", akis)
        return akis!!.bufferedReader().readLines()
            .drop(1)
            .filter { it.isNotBlank() }
            .map { satir ->
                val p = satir.split(",")
                Vaka(
                    tip = p[0],
                    a = p[1].toDouble(),
                    b = p[2].toDouble(),
                    c = p[3].toDoubleOrNull() ?: 0.0,
                    d = p[4].toDoubleOrNull() ?: 0.0,
                    e = p[5].toDoubleOrNull() ?: 0.0,
                    beklenen1 = p[6].toDouble(),
                    beklenen2 = p.getOrNull(7)?.toDoubleOrNull(),
                )
            }
    }

    @Test
    fun `aksiller aci Python referansiyla eslesir`() {
        val vakalar = altinVeri().filter { it.tip == "aci" }
        assertTrue("altin veride aci vakasi yok", vakalar.isNotEmpty())
        for (v in vakalar) {
            val elde = Olcum.aksillerAci(Nokta(v.a, v.b), Nokta(v.c, v.d), v.e)
            assertEquals(
                "sag=(${v.a}, ${v.b}) sol=(${v.c}, ${v.d}) roll=${v.e}",
                v.beklenen1, elde, tolerans,
            )
        }
    }

    @Test
    fun `roll ve pitch Python referansiyla eslesir`() {
        val vakalar = altinVeri().filter { it.tip == "durus" }
        assertTrue("altin veride durus vakasi yok", vakalar.isNotEmpty())
        for (v in vakalar) {
            val roll = Olcum.rollAcisi(v.a.toFloat(), v.b.toFloat())
            val pitch = Olcum.pitchAcisi(v.a.toFloat(), v.b.toFloat(), v.c.toFloat())
            assertEquals("roll g=(${v.a}, ${v.b}, ${v.c})",
                v.beklenen1, roll, tolerans)
            assertEquals("pitch g=(${v.a}, ${v.b}, ${v.c})",
                v.beklenen2!!, pitch, tolerans)
        }
    }

    // --- konvansiyon testleri: bunlar ters donerse urun sessizce yanlis olcer

    @Test
    fun `pozitif aci sag omzun yuksek oldugu anlamina gelir`() {
        // y asagi artar: sag noktanin y'si KUCUKSE sag omuz YUKSEK demektir
        val aci = Olcum.aksillerAci(sag = Nokta(300.0, 460.0), sol = Nokta(700.0, 500.0))
        assertTrue("sag omuz yuksekken aci pozitif olmali, elde: $aci", aci > 0)
        assertEquals(Yon.SAG_YUKSEK, Olcum.yon(aci))
    }

    @Test
    fun `negatif aci sol omzun yuksek oldugu anlamina gelir`() {
        val aci = Olcum.aksillerAci(sag = Nokta(300.0, 500.0), sol = Nokta(700.0, 460.0))
        assertTrue("sol omuz yuksekken aci negatif olmali, elde: $aci", aci < 0)
        assertEquals(Yon.SOL_YUKSEK, Olcum.yon(aci))
    }

    @Test
    fun `roll duzeltmesi goruntu yatayini gercek yataya cevirir`() {
        // Kisi tamamen duz; telefon 6 derece egik. Duzeltme sonrasi olculen
        // aci telefonun egimini yansitmali, kisininkini degil.
        val duz = listOf(0.0, 3.0, -3.0, 6.0, -11.25)
        for (roll in duz) {
            val aci = Olcum.aksillerAci(Nokta(300.0, 500.0), Nokta(700.0, 500.0), roll)
            assertEquals("roll=$roll", -roll, aci, 1e-9)
        }
    }

    @Test
    fun `noktalar cakisirsa sifir doner`() {
        assertEquals(0.0, Olcum.aksillerAci(Nokta(500.0, 500.0), Nokta(500.0, 500.0)),
                     1e-12)
    }

    @Test
    fun `pitch kapisi 8 derecede kapanir`() {
        assertTrue(Olcum.olcumGecerliMi(0.0))
        assertTrue(Olcum.olcumGecerliMi(7.9))
        assertTrue(Olcum.olcumGecerliMi(-8.0))
        assertFalse(Olcum.olcumGecerliMi(8.1))
        assertFalse(Olcum.olcumGecerliMi(-12.0))
    }

    @Test
    fun `simetri bandi icindeki aci simetrik sayilir`() {
        assertEquals(Yon.SIMETRIK, Olcum.yon(0.0))
        assertEquals(Yon.SIMETRIK, Olcum.yon(0.9))
        assertEquals(Yon.SIMETRIK, Olcum.yon(-0.9))
        assertEquals(Yon.SAG_YUKSEK, Olcum.yon(1.1))
        assertEquals(Yon.SOL_YUKSEK, Olcum.yon(-1.1))
    }

    /**
     * Stretch tuzagi: 640×640 uzayinda hesaplanan aci GERCEK aci degildir.
     * MOBIL_YAPILACAKLAR.md §3: bu adim atlanirsa 4:3 fotografta ~%25 hata.
     * Test bu farkin gercekten olustugunu ve maplemenin duzelttigini gosterir.
     */
    @Test
    fun `640 uzayinda hesaplanan aci orijinalden farklidir`() {
        val g = 3000
        val y = 4000    // 3:4 portre

        val sag640 = Nokta(200.0, 300.0)
        val sol640 = Nokta(440.0, 280.0)

        val aci640 = Olcum.aksillerAci(sag640, sol640)
        val sagOrj = Olcum.orijinaleMaple(sag640, g, y)
        val solOrj = Olcum.orijinaleMaple(sol640, g, y)
        val aciOrj = Olcum.aksillerAci(sagOrj, solOrj)

        assertTrue(
            "stretch bozulmasi gorunmuyor — mapleme yanlis olabilir " +
                "(640: $aci640, orijinal: $aciOrj)",
            abs(aci640 - aciOrj) > 1.0,
        )
        // 3:4'te y ekseni x'e gore 4/3 gerildigi icin gercek aci daha buyuk
        assertTrue(abs(aciOrj) > abs(aci640))
    }

    @Test
    fun `kalibrasyon Cobb tahmini hesaplar`() {
        val res = com.skolyoz.olcum.olcum.Kalibrasyon.cobbTahmini(7.8, 3.2, 15.0, 500.0)
        assertEquals(20.5, res.tahminDerece, 0.1)
        assertEquals(3.5, res.hataPayiDerece, 0.01)
        assertEquals(17.0, res.altSinir, 0.1)
        assertEquals(24.0, res.ustSinir, 0.1)
        assertEquals("Hafif Skolyoz Şüphesi", res.kategori)
    }

    @Test
    fun `dagilim yuzdelikleri Python referansiyla ayni`() {
        // scripts/19_kilavuz_cizim.py::dagilim_yuzdelik ile ayni tablo.
        // Ikisi ayrilirsa arayuz Python ciktisindan farkli sayi gosterir.
        val D = com.skolyoz.olcum.olcum.Dagilim
        assertEquals(50.0, D.yuzdelik(2.21), 0.05)
        assertEquals(85.0, D.yuzdelik(5.46), 0.05)
        assertEquals(90.0, D.yuzdelik(7.68), 0.05)
        assertEquals(95.0, D.yuzdelik(10.27), 0.05)
        // isaret onemsiz: buyukluk kullaniliyor
        assertEquals(D.yuzdelik(7.68), D.yuzdelik(-7.68), 1e-9)
    }

    @Test
    fun `belirgin asimetri esigi p90`() {
        val D = com.skolyoz.olcum.olcum.Dagilim
        assertTrue(!D.belirginAsimetri(7.0))
        assertTrue(D.belirginAsimetri(7.68))
        assertTrue(D.belirginAsimetri(-12.0))   // sol tarafa egim de sayilir
    }

    @Test
    fun `aciklama klinik derece iddia etmez`() {
        // Cobb sinif adlari (hafif/orta/siddetli) ASLA gecmemeli: bizim
        // olctugumuz aksiller egim, Cobb degil. MOBIL_YAPILACAKLAR.md §1.
        val D = com.skolyoz.olcum.olcum.Dagilim
        for (aci in listOf(0.5, 2.21, 5.0, 7.68, 12.0, 20.0)) {
            val m = D.aciklama(aci).lowercase()
            assertTrue(!m.contains("cobb"))
            assertTrue(!m.contains("hafif"))
            assertTrue(!m.contains("orta"))
            assertTrue(!m.contains("şiddetli"))
            // nufus iddiasi da olmamali, sadece kendi veri setimiz
            assertTrue(m.contains("veri setimiz"))
        }
    }
}
