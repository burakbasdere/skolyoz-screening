package com.skolyoz.olcum.cikarim

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.skolyoz.olcum.olcum.Nokta
import com.skolyoz.olcum.olcum.Olcum
import com.skolyoz.olcum.olcum.Yon
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.Locale

private const val ETIKET = "OlcumZinciri"

/**
 * Ölçümün çıktısı.
 *
 * Üç durum ayrı ayrı taşınıyor çünkü kullanıcıya söylenecek şey farklı:
 * kişi bulunamadıysa kadraj düzeltilmeli, ölçüm güvenilmezse koşullar
 * (ışık, mesafe, poz) düzeltilip tekrar çekilmeli. İkisini tek "başarısız"
 * durumuna indirgemek kullanıcıyı yanlış yöne iter.
 */
sealed interface OlcumCiktisi {
    data class Basarili(val sonuc: OlcumSonucu) : OlcumCiktisi

    /** Aşama 1 güven eşiğini geçen aday bulamadı. */
    data object KisiBulunamadi : OlcumCiktisi

    /**
     * Heatmap dağınık — çıkan sayı kendinden emin görünür ama girdideki
     * 1/255'lik fark bile onu derecelerce oynatır. Bkz. [OnIsleme.ENTROPI_ESIGI].
     */
    data class Guvenilmez(val entropi: Double, val tepeUzakligi: Double) : OlcumCiktisi
}

data class OlcumSonucu(
    /** Orijinal fotoğraf koordinatlarında. */
    val sag: Nokta,
    val sol: Nokta,
    val c7: Nokta? = null,
    val o1: Nokta? = null,
    val o2: Nokta? = null,
    val o3: Nokta? = null,
    val sakrum: Nokta? = null,
    /** Aksiller eğim açısı, roll düzeltilmiş. Pozitif = sağ omuz yüksek. */
    val aci: Double,
    val yon: Yon,
    val rollDerece: Double,
    val guven: Float,
    /** İki noktanın heatmap entropisinden büyüğü — küçük olan iyi. */
    val entropi: Double,
    val sureMs: Long,
) {
    /** C7 ve Sakrum mevcutsa Koronal Denge (dikey şakül sapması) hesabı. */
    val koronalDenge: com.skolyoz.olcum.olcum.KoronalDengeResult?
        get() = if (c7 != null && sakrum != null) {
            com.skolyoz.olcum.olcum.Olcum.koronalDenge(c7, sakrum, rollDerece)
        } else null

    /** Omurga noktalarının (O1, O2, O3) yanal sapma hesabı. */
    val omurgaSapmalari: com.skolyoz.olcum.olcum.OmurgaSapmalariResult?
        get() = if (c7 != null && o1 != null && o2 != null && o3 != null && sakrum != null) {
            com.skolyoz.olcum.olcum.Olcum.omurgaSapmalari(c7, o1, o2, o3, sakrum, rollDerece)
        } else null

    /** Kalibre edilmiş tahmini Cobb açısı sonucu. */
    val cobbTahmini: com.skolyoz.olcum.olcum.CobbTahminiResult
        get() {
            val kd = koronalDenge
            val os = omurgaSapmalari
            return com.skolyoz.olcum.olcum.Kalibrasyon.cobbTahmini(
                aksillerAciDeg = aci,
                koronalAciDeg = kd?.sapmaDerece ?: 0.0,
                apeksSapmaPx = os?.maksSapmaPx ?: 0.0,
                govdeBoyuPx = kd?.govdeBoyuPx ?: 1.0
            )
        }
}

/**
 * İki aşamalı TFLite çıkarım zinciri.
 *
 * ```
 * Bitmap → 640×640 STRETCH → Aşama 1 → 2 kaba nokta (640 uzayında)
 *        → her nokta için 128×128 kırpım → Aşama 2 → alt-piksel nokta
 *        → orijinal çözünürlüğe map → açı (roll düzeltmeli)
 * ```
 *
 * KIRPIM 640 UZAYINDAN ALINIYOR, ORİJİNALDEN DEĞİL
 * ------------------------------------------------
 * Aşama 2, eğitim setindeki 640×640 görüntülerden kesilmiş 128×128
 * kırpımlarla eğitildi. Orijinal 3 MP fotoğraftan 128×128 kesmek, modele çok
 * daha dar bir görüş açısı verirdi — gördüğü dağılım değişirdi. Bu yüzden
 * kırpım da 640 uzayından alınıyor.
 *
 * AÇI 640 UZAYINDA HESAPLANMIYOR
 * ------------------------------
 * Stretch en-boy oranını bozuyor; oradaki açı gerçek açı değil. Noktalar önce
 * orijinal çözünürlüğe map'leniyor, açı orada hesaplanıyor.
 * MOBIL_YAPILACAKLAR.md §3.
 */
class OlcumZinciri(baglam: Context) : Closeable {

    /**
     * Ayarlanırsa her ölçümün ara çıktıları buraya yazılır (PNG + metin).
     *
     * NEDEN VAR
     * ---------
     * MOBIL_YAPILACAKLAR.md Faz 3 şartı: "Python çıktısıyla aynı görüntüde
     * ≤0.05° farkla eşleşmeli." Bunu doğrulamak için iki tarafın AYNI
     * görüntüyü işlemesi gerekiyor; uygulama fotoğrafı saklamadığı sürece
     * karşılaştırma yapılamaz.
     *
     * Hem orijinal hem 640×640 girdi kaydediliyor, çünkü iki farklı soru var:
     *   - 640 girdiyle: zincirin kendisi (decode, kırpım, mapleme) aynı mı?
     *   - orijinalle  : Android'in yeniden boyutlandırması PIL'den ne kadar
     *                   farklı? Bu ikisi ayrılmazsa fark nereden geliyor
     *                   bilinemez.
     *
     * PNG çünkü KAYIPSIZ. JPEG kaydetmek karşılaştırmaya kendi gürültüsünü
     * katardı.
     *
     * TEŞHİS AMAÇLIDIR: hasta fotoğrafı diske yazar. Yalnızca hata ayıklama
     * derlemesinde ve elle açılmalı; üründe kapalı kalmalı.
     */
    var teshisDizini: File? = null

    private val asama1 = yukle(baglam, "asama1.tflite")
    private val asama2 = yukle(baglam, "asama2.tflite")

    private val asama1Girdi = tampon(OnIsleme.MODEL_BOYUT * OnIsleme.MODEL_BOYUT * 3)
    private val asama1Cikti = tampon(KANAL * ADAY)
    private val asama2Girdi = tampon(OnIsleme.KIRPIM * OnIsleme.KIRPIM * 3)
    private val asama2Cikti = tampon(HEATMAP * HEATMAP * TIP_SAYISI)

    private fun yukle(baglam: Context, ad: String): Interpreter {
        val tanim = baglam.assets.openFd(ad)
        val model = tanim.createInputStream().channel.use { kanal ->
            kanal.map(FileChannel.MapMode.READ_ONLY, tanim.startOffset,
                      tanim.declaredLength)
        }
        return Interpreter(model, Interpreter.Options().apply { numThreads = 4 })
    }

    private fun tampon(elemanSayisi: Int): ByteBuffer =
        ByteBuffer.allocateDirect(elemanSayisi * 4).order(ByteOrder.nativeOrder())

    @Synchronized
    fun olc(foto: Bitmap, rollDerece: Double): OlcumCiktisi {
        val basla = SystemClock.elapsedRealtime()

        val ham = IntArray(foto.width * foto.height)
        foto.getPixels(ham, 0, foto.width, 0, 0, foto.width, foto.height)
        val pikseller = OnIsleme.kucultBilinear(
            ham, foto.width, foto.height,
            OnIsleme.MODEL_BOYUT, OnIsleme.MODEL_BOYUT)

        val kaba = calistir(asama1, asama1Girdi, asama1Cikti,
                            OnIsleme.asama1Tensor(pikseller), KANAL * ADAY)
            .let { Asama1Coz.coz(it, ADAY) }
        if (kaba == null) {
            Log.w(ETIKET, "kisi bulunamadi (guven esigi ${Asama1Coz.GUVEN_ESIGI})")
            return OlcumCiktisi.KisiBulunamadi
        }

        // --- Aşama 2: 7 nokta için 4 tipli heatmap rafinajı ---
        val kabaNoktalar = listOf(kaba.sag, kaba.sol, kaba.c7, kaba.omurga1, kaba.omurga2, kaba.omurga3, kaba.sakrum)
        val sonuclar = kabaNoktalar.mapIndexed { idx, nokta ->
            val tip = NOKTA_TIPI[idx]
            val (solX, ustY) = OnIsleme.kirpimKosesi(nokta)
            val kirpim = OnIsleme.kirp(pikseller, OnIsleme.MODEL_BOYUT,
                                       OnIsleme.MODEL_BOYUT, solX, ustY)
            val heatmapHepsi = calistir(asama2, asama2Girdi, asama2Cikti,
                                          OnIsleme.asama2Tensor(kirpim), HEATMAP * HEATMAP * TIP_SAYISI)
            
            // İlgili tipin heatmap kanalını NHWC (64x64x4) düzeninde oku
            val tekHeatmap = FloatArray(HEATMAP * HEATMAP)
            val adim = TIP_SAYISI
            for (i in 0 until HEATMAP * HEATMAP) {
                tekHeatmap[i] = heatmapHepsi[i * adim + tip]
            }
            
            val h = OnIsleme.yumusakArgmax(tekHeatmap, HEATMAP)
            h to OnIsleme.kirpimdanGoruntuye(h.nokta, solX, ustY)
        }

        val enBuyukEntropi = sonuclar.take(2).maxOf { it.first.entropi }
        val enBuyukUzaklik = sonuclar.take(2).maxOf { it.first.tepeUzakligi }
        val ince = sonuclar.map { it.second }
        val guvenilir = sonuclar.take(2).all { OnIsleme.guvenilirMi(it.first) }

        // --- Orijinal çözünürlüğe geri map
        val sag = OnIsleme.orijinaleMaple(ince[0], foto.width, foto.height)
        val sol = OnIsleme.orijinaleMaple(ince[1], foto.width, foto.height)
        val c7 = OnIsleme.orijinaleMaple(ince[2], foto.width, foto.height)
        val o1 = OnIsleme.orijinaleMaple(ince[3], foto.width, foto.height)
        val o2 = OnIsleme.orijinaleMaple(ince[4], foto.width, foto.height)
        val o3 = OnIsleme.orijinaleMaple(ince[5], foto.width, foto.height)
        val sakrum = OnIsleme.orijinaleMaple(ince[6], foto.width, foto.height)

        val aci = Olcum.aksillerAci(sag, sol, rollDerece)
        val sure = SystemClock.elapsedRealtime() - basla

        teshisDizini?.let { dizin ->
            teshisYaz(dizin, foto, pikseller, kaba, ince, sag, sol, aci,
                      rollDerece, enBuyukEntropi, enBuyukUzaklik, guvenilir)
        }

        if (!guvenilir) {
            Log.w(ETIKET, "olcum reddedildi: entropi %.2f (esik %.2f), tepe uzakligi %.1f px (esik %.1f), atilan aci %.3f"
                .format(enBuyukEntropi, OnIsleme.ENTROPI_ESIGI,
                        enBuyukUzaklik, OnIsleme.TEPE_UZAKLIK_ESIGI, aci))
            return OlcumCiktisi.Guvenilmez(enBuyukEntropi, enBuyukUzaklik)
        }

        Log.i(ETIKET, "olcum %.3f° (roll %+.2f°) c7=(%.1f, %.1f) sakrum=(%.1f, %.1f) guven %.3f  %dms"
            .format(aci, rollDerece, c7.x, c7.y, sakrum.x, sakrum.y, kaba.guven, sure))

        return OlcumCiktisi.Basarili(
            OlcumSonucu(
                sag = sag, sol = sol, c7 = c7, o1 = o1, o2 = o2, o3 = o3, sakrum = sakrum,
                aci = aci, yon = Olcum.yon(aci),
                rollDerece = rollDerece, guven = kaba.guven,
                entropi = enBuyukEntropi, sureMs = sure,
            )
        )
    }

    private fun calistir(yorumlayici: Interpreter, girdi: ByteBuffer,
                         cikti: ByteBuffer, veri: FloatArray,
                         ciktiBoyut: Int): FloatArray {
        girdi.rewind()
        for (v in veri) girdi.putFloat(v)
        girdi.rewind()
        cikti.rewind()
        yorumlayici.run(girdi, cikti)
        cikti.rewind()
        return FloatArray(ciktiBoyut) { cikti.float }
    }

    /**
     * Karşılaştırma için ara çıktıları diske yazar.
     *
     * PNG kullanılıyor çünkü KAYIPSIZ — JPEG kaydetmek Python tarafına kendi
     * sıkıştırma gürültüsünü taşır ve ölçtüğümüz farkın ne kadarının zincirden
     * geldiği belirsizleşirdi.
     */
    private fun teshisYaz(
        dizin: File, foto: Bitmap, pikseller: IntArray,
        kaba: Asama1Coz.KabaNoktalar, ince: List<Nokta>,
        sag: Nokta, sol: Nokta, aci: Double, rollDerece: Double,
        entropi: Double, tepeUzakligi: Double, guvenilir: Boolean,
    ) = runCatching {
        dizin.mkdirs()
        val damga = System.currentTimeMillis()

        File(dizin, "${damga}_orijinal.png").outputStream().use {
            foto.compress(Bitmap.CompressFormat.PNG, 100, it)
        }
        // Modele giden 640x640'i AYRICA yaziyoruz: boylece Python ayni tensoru
        // isleyebiliyor ve Android'in yeniden boyutlandirmasi ile PIL'inki
        // arasindaki fark olcumden ayrilabiliyor.
        val girdi = Bitmap.createBitmap(pikseller, OnIsleme.MODEL_BOYUT,
                                        OnIsleme.MODEL_BOYUT, Bitmap.Config.ARGB_8888)
        File(dizin, "${damga}_girdi640.png").outputStream().use {
            girdi.compress(Bitmap.CompressFormat.PNG, 100, it)
        }
        girdi.recycle()

        // Locale.ROOT SART: cihaz Turkce yerel ayarindayken "%.10f" ondalik
        // ayirici olarak VIRGUL yaziyor ("0,3410142147"). Ekranda dogru ama
        // bu dosya makine tarafindan okunuyor; virgul hem ayirici hem ondalik
        // olunca ayristirma belirsizlesiyor.
        fun f(v: Double) = String.format(Locale.ROOT, "%.10f", v)

        File(dizin, "${damga}_sonuc.txt").writeText(
            buildString {
                appendLine("orijinal=${foto.width}x${foto.height}")
                appendLine("roll=${f(rollDerece)}")
                appendLine("guven=${f(kaba.guven.toDouble())}")
                appendLine("kaba_sag=${f(kaba.sag.x)},${f(kaba.sag.y)}")
                appendLine("kaba_sol=${f(kaba.sol.x)},${f(kaba.sol.y)}")
                appendLine("ince640_sag=${f(ince[0].x)},${f(ince[0].y)}")
                appendLine("ince640_sol=${f(ince[1].x)},${f(ince[1].y)}")
                appendLine("orijinal_sag=${f(sag.x)},${f(sag.y)}")
                appendLine("orijinal_sol=${f(sol.x)},${f(sol.y)}")
                appendLine("aci=${f(aci)}")
                appendLine("entropi=${f(entropi)}")
                appendLine("tepe_uzakligi=${f(tepeUzakligi)}")
                appendLine("guvenilir=$guvenilir")
            },
            Charsets.UTF_8,
        )
        Log.i(ETIKET, "teshis yazildi: $dizin/${damga}_*")
    }.onFailure { Log.e(ETIKET, "teshis yazilamadi", it) }

    override fun close() {
        asama1.close()
        asama2.close()
    }

    private companion object {
        const val KANAL = 26
        const val ADAY = 8400
        const val HEATMAP = 64
        const val TIP_SAYISI = 4
        val NOKTA_TIPI = intArrayOf(0, 0, 1, 2, 2, 2, 3)
    }
}
