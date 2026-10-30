package com.skolyoz.olcum

import com.skolyoz.olcum.cikarim.OnIsleme
import com.skolyoz.olcum.olcum.Nokta
import com.skolyoz.olcum.olcum.Olcum
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.cos
import kotlin.math.sin
import kotlin.system.measureNanoTime

/**
 * Saf JVM on-isleme hiz olcumu — TFLite olmadan.
 *
 * Mobil cihazda cikarimin buyuk kismi TFLite'ta; bu test yalnizca
 * Kotlin on-islemenin (kucultme, kirpim, tensor, soft-argmax) maliyetini olcer.
 */
class PerformansTest {

    private fun sentetik(g: Int, y: Int): IntArray {
        val p = IntArray(g * y)
        for (sy in 0 until y) for (sx in 0 until g) {
            val r = (127 + 127 * sin(sx * 0.05)).toInt().coerceIn(0, 255)
            val gg = (127 + 127 * cos(sy * 0.05)).toInt().coerceIn(0, 255)
            p[sy * g + sx] = (0xFF shl 24) or (r shl 16) or (gg shl 8) or 90
        }
        return p
    }

    @Test
    fun `on isleme zinciri makul surede tamamlanir`() {
        // Tipik telefon foto: ~3000x4000 -> 640 + 2x kirpim + tensor + argmax
        val ham = sentetik(3000, 4000)
        val tekrar = 5
        var kucultmeNs = 0L
        var kirpimNs = 0L
        var tensorNs = 0L
        var argmaxNs = 0L

        repeat(tekrar) {
            kucultmeNs += measureNanoTime {
                OnIsleme.kucultBilinear(ham, 3000, 4000, 640, 640)
            }
            val p640 = OnIsleme.kucultBilinear(ham, 3000, 4000, 640, 640)
            val merkez = Nokta(320.0, 280.0)
            kirpimNs += measureNanoTime {
                val (sx, sy) = OnIsleme.kirpimKosesi(merkez)
                OnIsleme.kirp(p640, 640, 640, sx, sy)
            }
            val (sx, sy) = OnIsleme.kirpimKosesi(merkez)
            val kirpim = OnIsleme.kirp(p640, 640, 640, sx, sy)
            tensorNs += measureNanoTime {
                OnIsleme.asama1Tensor(p640)
                OnIsleme.asama2Tensor(kirpim)
            }
            val hm = FloatArray(64 * 64) { i ->
                val x = i % 64; val y = i / 64
                kotlin.math.exp(-((x - 32.0) * (x - 32.0) + (y - 28.0) * (y - 28.0)) / 20.0).toFloat()
            }
            argmaxNs += measureNanoTime { OnIsleme.yumusakArgmax(hm, 64) }
            Olcum.aksillerAci(Nokta(400.0, 300.0), Nokta(600.0, 320.0), 2.5)
        }

        val kucultmeMs = kucultmeNs / tekrar / 1_000_000.0
        val kirpimMs = kirpimNs / tekrar / 1_000_000.0
        val tensorMs = tensorNs / tekrar / 1_000_000.0
        val argmaxMs = argmaxNs / tekrar / 1_000_000.0
        val toplamMs = kucultmeMs + kirpimMs * 2 + tensorMs + argmaxMs * 2

        println("JVM on-isleme (3000x4000, ortalama $tekrar tekrar):")
        println("  kucultme 640   : ${"%.1f".format(kucultmeMs)} ms")
        println("  kirpim x2       : ${"%.1f".format(kirpimMs * 2)} ms")
        println("  tensor x2       : ${"%.1f".format(tensorMs)} ms")
        println("  soft-argmax x2  : ${"%.1f".format(argmaxMs * 2)} ms")
        println("  toplam (TFLite haric): ${"%.1f".format(toplamMs)} ms")

        // Kucultme en agir adim; 3000x4000 icin 2 sn uzeri anormal
        assertTrue("kucultme cok yavas: $kucultmeMs ms", kucultmeMs < 2000.0)
    }
}
