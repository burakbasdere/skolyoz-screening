package com.skolyoz.olcum

import com.skolyoz.olcum.sensor.OrnekHalkasi
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

/**
 * Deklanşör–IMU eşleştirmesinin testleri.
 *
 * Bu ürünün en hassas halkası: yanlış anın yerçekimi vektörü doğrudan açı
 * hatasına yazılıyor ve bütün ölçüm zaten bir açı. Gerçek cihazda (Redmi Note
 * 9 Pro) örnekleme periyodu ~20 ms ölçüldü; en yakın örneği seçmek ±10 ms
 * yapısal hata bırakıyordu.
 */
class OrnekHalkasiTest {

    private val ms = 1_000_000L
    private val periyot = 20 * ms      // gerçek cihazda ölçülen örnekleme aralığı

    /** t=0'dan başlayarak sabit periyotla, sabit hızda dönen bir telefon. */
    private fun donenTelefon(
        halka: OrnekHalkasi,
        adet: Int = 50,
        dereceSaniye: Double = 30.0,
        baslangicNanos: Long = 1_000_000_000L,
    ) {
        for (k in 0 until adet) {
            val t = baslangicNanos + k * periyot
            val aci = Math.toRadians(dereceSaniye * (k * periyot) / 1e9)
            // Yerçekimi vektörü telefon döndükçe cihaz düzleminde dönüyor
            halka.ekle(t, (9.81 * sin(aci)).toFloat(), (9.81 * cos(aci)).toFloat(), 0f)
        }
    }

    @Test
    fun `iki ornek arasindaki deklansor enterpole edilir`() {
        val halka = OrnekHalkasi()
        donenTelefon(halka)
        // İki örneğin tam ortasına denk gelen bir deklanşör — en kötü durum
        val hedef = 1_000_000_000L + 10 * periyot + periyot / 2
        val o = halka.eslestir(hedef)

        assertNotNull("kusatma varken eslestirme null donmemeli", o)
        assertTrue("enterpolasyon kullanilmali", o!!.enterpole)
        assertEquals("sapma enterpolasyonda 0 olmali", 0.0, o.sapmaMs, 1e-12)
        assertEquals("komsu araligi ~20 ms olmali", 20.0, o.komsuAralikMs, 0.001)
    }

    /**
     * Asıl mesele bu: enterpolasyonun en yakın örneğe göre kazancı.
     * 30°/s dönerken 10 ms'lik kayma 0,3° hata demek — boru hattının açı
     * hatası medyanı 0,380°, etiket gürültüsü tabanı 0,23°.
     */
    @Test
    fun `enterpolasyon en yakin ornekten belirgin daha dogru`() {
        val halka = OrnekHalkasi()
        val hiz = 30.0
        val t0 = 1_000_000_000L
        donenTelefon(halka, dereceSaniye = hiz, baslangicNanos = t0)

        val k = 10
        val hedef = t0 + k * periyot + periyot / 2
        val gercekAci = hiz * (hedef - t0) / 1e9

        val enterpole = halka.eslestir(hedef)!!
        val enterpoleAci = enterpole.roll
        val enterpoleHata = abs(enterpoleAci - gercekAci)

        // Aynı anda "en yakın örnek" yaklaşımının yapacağı hata:
        // deklanşör tam ortada olduğu için en yakın örnek 10 ms uzakta.
        val enYakinAci = hiz * (k * periyot) / 1e9
        val enYakinHata = abs(enYakinAci - gercekAci)

        assertTrue(
            "en yakin ornek yaklasimi bu senaryoda >=0.25 derece hata yapmali " +
                "(elde: $enYakinHata) — test senaryosu anlamli degil",
            enYakinHata >= 0.25,
        )
        assertTrue(
            "enterpolasyon hatasi 0.01 derecenin altinda olmali (elde: $enterpoleHata)",
            enterpoleHata < 0.01,
        )
        assertTrue(
            "enterpolasyon en az 10x kazandirmali " +
                "(enterpole: $enterpoleHata, en yakin: $enYakinHata)",
            enYakinHata / enterpoleHata > 10,
        )
    }

    @Test
    fun `tam ornek zamanina denk gelen deklansor o orneği verir`() {
        val halka = OrnekHalkasi()
        val t0 = 1_000_000_000L
        donenTelefon(halka, baslangicNanos = t0)
        val hedef = t0 + 7 * periyot
        val o = halka.eslestir(hedef)!!
        val beklenen = 30.0 * (7 * periyot) / 1e9
        assertEquals(beklenen, o.roll, 1e-6)
    }

    @Test
    fun `bos halka null doner`() {
        assertNull(OrnekHalkasi().eslestir(1_000_000_000L))
    }

    @Test
    fun `cok eski deklansor reddedilir`() {
        val halka = OrnekHalkasi()
        val t0 = 1_000_000_000L
        donenTelefon(halka, baslangicNanos = t0)
        // Tamponun en eski örneğinden 1 saniye önce: kuşatma yok, tolerans dışı
        assertNull(halka.eslestir(t0 - 1_000 * ms))
    }

    @Test
    fun `cok yeni deklansor reddedilir`() {
        val halka = OrnekHalkasi()
        val t0 = 1_000_000_000L
        donenTelefon(halka, adet = 20, baslangicNanos = t0)
        val son = t0 + 19 * periyot
        assertNull(halka.eslestir(son + 1_000 * ms))
    }

    @Test
    fun `tolerans icindeki tek tarafli deklansor kabul edilir`() {
        val halka = OrnekHalkasi()
        val t0 = 1_000_000_000L
        donenTelefon(halka, adet = 20, baslangicNanos = t0)
        val son = t0 + 19 * periyot
        // Son örnekten 5 ms sonra: kuşatma yok ama tolerans (20 ms) içinde
        val o = halka.eslestir(son + 5 * ms)
        assertNotNull(o)
        assertTrue("kusatma yokken enterpolasyon yapilmamali", !o!!.enterpole)
        assertEquals(5.0, o.sapmaMs, 1e-9)
    }

    /**
     * Uygulama arka plandayken sensör askıya alınır. Boşluğun iki ucu arasında
     * doğrusallık varsayılamaz — telefon o sürede bambaşka bir konuma gitmiş
     * olabilir. Enterpolasyon burada sessizce yanlış cevap üretirdi.
     */
    @Test
    fun `ornekleme bosluğunda enterpolasyon yapilmaz`() {
        val halka = OrnekHalkasi()
        val t0 = 1_000_000_000L
        halka.ekle(t0, 0f, 9.81f, 0f)
        // 500 ms sonra telefon tamamen başka konumda
        halka.ekle(t0 + 500 * ms, 9.81f, 0f, 0f)
        assertNull(halka.eslestir(t0 + 250 * ms))
    }

    @Test
    fun `halka dolunca eski ornekler dusurulur`() {
        val halka = OrnekHalkasi(kapasite = 8)
        val t0 = 1_000_000_000L
        for (k in 0 until 20) halka.ekle(t0 + k * periyot, 0f, 9.81f, 0f)
        // İlk 12 örnek düşmüş olmalı: en eskisi artık k=12
        assertNull("dusen ornek hala eslesiyor", halka.eslestir(t0 + 2 * periyot))
        assertNotNull(halka.eslestir(t0 + 15 * periyot))
    }

    @Test
    fun `temizle sonrasi eslestirme yapilmaz`() {
        val halka = OrnekHalkasi()
        donenTelefon(halka)
        halka.temizle()
        assertNull(halka.eslestir(1_000_000_000L + 10 * periyot))
    }
}
