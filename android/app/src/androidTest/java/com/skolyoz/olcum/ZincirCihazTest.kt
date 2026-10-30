package com.skolyoz.olcum

import android.graphics.Bitmap
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.skolyoz.olcum.cikarim.OlcumCiktisi
import com.skolyoz.olcum.cikarim.OlcumZinciri
import com.skolyoz.olcum.cikarim.OnIsleme
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.cos
import kotlin.math.sin

/**
 * Zincirin GERÇEK CİHAZDA koştuğunu doğrular.
 *
 * NEDEN AYRI VE NEDEN GEREKLİ
 * ---------------------------
 * Düz JVM birim testleri zincirin saf parçalarını kapsıyor (decode, kırpım,
 * normalizasyon, küçültme, soft-argmax) ama TFLite'ın kendisini KAPSAMIYOR:
 * yerel kütüphane ve `assets/` düz JVM'de yok.
 *
 * Dolayısıyla şu sorular şimdiye kadar her seferinde ELLE doğrulanıyordu:
 *   - modeller assets'ten yükleniyor mu
 *   - tensör şekilleri kodun beklediğiyle aynı mı
 *   - çıkarım çökmeden tamamlanıyor mu
 *
 * Bunlar sessizce bozulabilecek şeyler: modeli yeniden üretip assets'e
 * kopyalarken heatmap yerine koordinat varyantını koymak yeterdi — birim
 * testler yeşil kalır, uygulama ilk çekimde çökerdi.
 *
 * Çalıştırma (telefon bağlıyken):
 *     ./gradlew :app:connectedDebugAndroidTest
 */
@RunWith(AndroidJUnit4::class)
class ZincirCihazTest {

    private val baglam get() = InstrumentationRegistry.getInstrumentation().targetContext

    /** Model girdisiyle aynı boyutta, deterministik sentetik görüntü. */
    private fun sentetikBitmap(g: Int, y: Int): Bitmap {
        val p = IntArray(g * y)
        for (sy in 0 until y) for (sx in 0 until g) {
            val r = (127 + 127 * sin(sx * 0.05)).toInt().coerceIn(0, 255)
            val gg = (127 + 127 * cos(sy * 0.05)).toInt().coerceIn(0, 255)
            p[sy * g + sx] = (0xFF shl 24) or (r shl 16) or (gg shl 8) or 90
        }
        return Bitmap.createBitmap(p, g, y, Bitmap.Config.ARGB_8888)
    }

    @Test
    fun modeller_assetsten_yukleniyor_ve_cikarim_cokmuyor() {
        OlcumZinciri(baglam).use { zincir ->
            val foto = sentetikBitmap(750, 1000)
            val cikti = zincir.olc(foto, rollDerece = 0.0)

            // Sentetik goruntude kisi YOK; beklenen sonuc KisiBulunamadi.
            // Onemli olan zincirin cokmeden bir karar uretmesi: modeller
            // yuklendi, tensor sekilleri tuttu, cikti cozuldu.
            assertNotNull(cikti)
            assertTrue(
                "sentetik goruntude tespit veya red bekleniyordu, gelen: $cikti",
                cikti is OlcumCiktisi.KisiBulunamadi ||
                    cikti is OlcumCiktisi.Guvenilmez ||
                    cikti is OlcumCiktisi.Basarili,
            )
            foto.recycle()
        }
    }

    /**
     * Aynı girdi hep aynı çıktıyı vermeli.
     *
     * Belirlenimlilik bu üründe önemli: kullanıcı aynı fotoğrafta farklı sayı
     * görürse ölçüme güvenemez. TFLite çok iş parçacığıyla koşuyor (numThreads
     * = 4); indirgeme sırası değişirse sonuç oynardı.
     */
    @Test
    fun ayni_girdi_ayni_cikti() {
        OlcumZinciri(baglam).use { zincir ->
            val foto = sentetikBitmap(750, 1000)
            val a = zincir.olc(foto, 0.0)
            val b = zincir.olc(foto, 0.0)
            if (a is OlcumCiktisi.Basarili && b is OlcumCiktisi.Basarili) {
                assertEquals("ayni girdide aci degisti", a.sonuc.aci, b.sonuc.aci, 1e-9)
            } else {
                assertEquals("ayni girdide karar degisti",
                             a.javaClass.name, b.javaClass.name)
            }
            foto.recycle()
        }
    }

    /**
     * Aşama 2'nin assets'teki modeli HEATMAP varyantı olmalı.
     *
     * Koordinat varyantı `(1, 2)` döndürüyor, heatmap varyantı `(1, 64, 64, 1)`.
     * Yanlışlıkla koordinat varyantı kopyalanırsa güven kapısı çalışamaz —
     * entropi hesaplanacak heatmap olmaz. Bu test o karışıklığı yakalar.
     */
    @Test
    fun asama2_heatmap_varyanti_kullaniliyor() {
        val tanim = baglam.assets.openFd("asama2.tflite")
        val boyut = tanim.declaredLength
        tanim.close()
        // Heatmap varyanti ~0.86 MB, koordinat varyanti ~0.88 MB — boyut
        // ayirt edici degil. Gercek kontrol cikti sekli uzerinden yapiliyor:
        // zincir HEATMAP*HEATMAP eleman okuyor; koordinat varyanti olsaydi
        // tampon boyutu tutmaz ve cikarim hata verirdi.
        assertTrue("asama2.tflite bos", boyut > 100_000)

        OlcumZinciri(baglam).use { zincir ->
            val foto = sentetikBitmap(750, 1000)
            // Cokmeden tamamlanmasi, 64x64 heatmap okundugunun kanitidir.
            zincir.olc(foto, 0.0)
            foto.recycle()
        }
    }

    /** Kırpım ve model boyutu sabitleri, assets'teki modelle uyumlu olmalı. */
    @Test
    fun sabitler_beklenen_degerlerde() {
        assertEquals(640, OnIsleme.MODEL_BOYUT)
        assertEquals(128, OnIsleme.KIRPIM)
    }
}
