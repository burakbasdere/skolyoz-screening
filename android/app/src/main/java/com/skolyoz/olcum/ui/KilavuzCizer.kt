package com.skolyoz.olcum.ui

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.graphics.Typeface
import com.skolyoz.olcum.cikarim.OlcumSonucu
import com.skolyoz.olcum.olcum.Nokta
import com.skolyoz.olcum.olcum.Yon
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * Ölçüm kılavuzunu fotoğraf üzerine çizer — `scripts/19_kilavuz_cizim.py`
 * `kilavuz_ciz` fonksiyonunun Android portu.
 *
 * NEDEN BITMAP DÖNDÜRÜYOR (Compose Canvas değil)
 * ----------------------------------------------
 * Referans PIL üzerinde çalışıyor ve annote edilmiş bir görüntü üretiyor.
 * Bitmap döndürmek üç işe birden yarıyor:
 *   1) sonuç ekranında gösterilir,
 *   2) Faz 5'te kayıt/paylaşım için saklanabilir,
 *   3) TEŞHİS: R/L noktalarının fotoğrafın neresine düştüğünü gözle görürüz.
 *      Benzer çekimlerde açı ~2° oynuyor; noktaların koltuk altında mı
 *      yoksa alakasız bir yerde mi olduğunu ancak çizince anlarız.
 *
 * Açı matematiği burada TEKRAR HESAPLANMIYOR — `OlcumSonucu`'ndan geliyor,
 * o da test edilmiş `Olcum.aksillerAci`'den. Bu dosya yalnızca sunum.
 */
object KilavuzCizer {

    private const val SARI = 0xFFFFCD00.toInt()
    private const val KIRMIZI = 0xFFFF4646.toInt()
    private const val MAVI = 0xFF46A0FF.toInt()
    private const val GRI = 0xFFAAAAAA.toInt()
    private const val BEYAZ = 0xFFFFFFFF.toInt()

    /**
     * @param foto orijinal fotoğraf (değiştirilmez, kopyalanır)
     * @param sonuc ölçüm — noktalar foto koordinat uzayında
     */
    fun ciz(foto: Bitmap, sonuc: OlcumSonucu): Bitmap {
        val im = foto.copy(Bitmap.Config.ARGB_8888, true)
        val c = Canvas(im)
        val w = im.width
        val h = im.height
        val olcek = max(w, h) / 640.0f

        val sag = sonuc.sag
        val sol = sonuc.sol
        val mx = ((sag.x + sol.x) / 2.0).toFloat()
        val my = ((sag.y + sol.y) / 2.0).toFloat()

        val cizgi = Paint(Paint.ANTI_ALIAS_FLAG)

        // --- yerçekimi yatayı ve dikey şakül çizgileri (kesikli). Roll kadar döndürülür. ---
        val r = Math.toRadians(sonuc.rollDerece)
        val uzanYatay = w * 0.75f
        val uzanDikey = h * 0.75f
        
        val hx = (cos(r) * uzanYatay).toFloat()
        val hy = (sin(r) * uzanYatay).toFloat()
        val vx = (-sin(r) * uzanDikey).toFloat()
        val vy = (cos(r) * uzanDikey).toFloat()

        // Yatay referans çizgisi
        kesikli(c, mx - hx, my - hy, mx + hx, my + hy, GRI,
                max(1f, 2 * olcek), 10 * olcek, 14 * olcek)

        // Dikey şakül referans çizgisi
        kesikli(c, mx - vx, my - vy, mx + vx, my + vy, GRI,
                max(1f, 2 * olcek), 10 * olcek, 14 * olcek)

        // --- açı yayı: yerçekimi yatayından koltuk altı çizgisine ---
        // Çizgi yönü SOLDAN SAĞA (sol -> sag) ölçülüyor; sag karede sağda
        // olduğu için bu yön ~0°'dir. Tersi alınırsa yay 180° taranır.
        val yaricap = max(26.0, hypot(sag.x - sol.x, sag.y - sol.y) * 0.20).toFloat()
        val aCizgi = Math.toDegrees(Math.atan2(sag.y - sol.y, sag.x - sol.x)).toFloat()
        val aYatay = sonuc.rollDerece.toFloat()
        cizgi.color = SARI
        cizgi.style = Paint.Style.STROKE
        cizgi.strokeWidth = max(2f, 3 * olcek)
        c.drawArc(
            RectF(mx - yaricap, my - yaricap, mx + yaricap, my + yaricap),
            min(aYatay, aCizgi), abs(aCizgi - aYatay), false, cizgi,
        )

        // --- koltuk altı çizgisi ---
        cizgi.strokeWidth = max(3f, 4 * olcek)
        c.drawLine(sag.x.toFloat(), sag.y.toFloat(),
                   sol.x.toFloat(), sol.y.toFloat(), cizgi)

        // --- R / L nokta işaretleri ve etiketler ---
        val nr = max(6f, 8 * olcek)
        val fEtiket = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = BEYAZ
            textSize = 15 * olcek
            typeface = Typeface.create(Typeface.SANS_SERIF, Typeface.BOLD)
        }

        for ((nokta, renk, etiket) in listOf(
            Triple(sag, KIRMIZI, "R"),
            Triple(sol, MAVI, "L")
        )) {
            // Dış glow halkası
            cizgi.style = Paint.Style.STROKE
            cizgi.color = renk and 0x50FFFFFF
            cizgi.strokeWidth = max(4f, 6 * olcek)
            c.drawCircle(nokta.x.toFloat(), nokta.y.toFloat(), nr * 1.5f, cizgi)

            cizgi.style = Paint.Style.FILL
            cizgi.color = renk
            c.drawCircle(nokta.x.toFloat(), nokta.y.toFloat(), nr, cizgi)
            cizgi.style = Paint.Style.STROKE
            cizgi.color = BEYAZ
            cizgi.strokeWidth = max(2f, 2 * olcek)
            c.drawCircle(nokta.x.toFloat(), nokta.y.toFloat(), nr, cizgi)

            val ofs = if (etiket == "R") nr * 1.6f else -nr * 2.2f
            c.drawText(etiket, nokta.x.toFloat() + ofs, nokta.y.toFloat() - nr * 0.5f, fEtiket)
        }

        // --- C7 - Sakrum Dikey Şakül ve 5 Parçalı Omurga Eğrisi ---
        val c7 = sonuc.c7
        val o1 = sonuc.o1
        val o2 = sonuc.o2
        val o3 = sonuc.o3
        val sakrum = sonuc.sakrum
        val mor = 0xFFBE78FF.toInt()
        val turkuaz = 0xFF00E5FF.toInt()

        if (c7 != null && sakrum != null) {
            val uz = h * 0.95f
            val dikX = (-sin(r) * uz).toFloat()
            val dikY = (cos(r) * uz).toFloat()

            // Sakrumdan geçen dikey yerçekimi şakülü
            kesikli(c, sakrum.x.toFloat() - dikX, sakrum.y.toFloat() - dikY,
                    sakrum.x.toFloat() + dikX * 0.15f, sakrum.y.toFloat() + dikY * 0.15f,
                    mor, max(1.5f, 2.5f * olcek), 10 * olcek, 14 * olcek)

            // C7 -> Sakrum düz referans hattı
            cizgi.color = mor
            cizgi.style = Paint.Style.STROKE
            cizgi.strokeWidth = max(1.5f, 2f * olcek)
            c.drawLine(c7.x.toFloat(), c7.y.toFloat(), sakrum.x.toFloat(), sakrum.y.toFloat(), cizgi)

            // 5 parçalı omurga eğri hattını çiz (C7 -> O1 -> O2 -> O3 -> Sakrum)
            val omurgaNoktalari = listOfNotNull(
                c7 to "C7",
                o1?.let { it to "O1" },
                o2?.let { it to "O2" },
                o3?.let { it to "O3" },
                sakrum to "S1"
            )
            if (omurgaNoktalari.size > 2) {
                val omurgaYolu = Path()
                omurgaYolu.moveTo(omurgaNoktalari[0].first.x.toFloat(), omurgaNoktalari[0].first.y.toFloat())
                for (i in 1 until omurgaNoktalari.size) {
                    val p = omurgaNoktalari[i].first
                    omurgaYolu.lineTo(p.x.toFloat(), p.y.toFloat())
                }
                cizgi.color = turkuaz
                cizgi.style = Paint.Style.STROKE
                cizgi.strokeWidth = max(2.5f, 3.5f * olcek)
                c.drawPath(omurgaYolu, cizgi)
            }

            for ((nokta, etiket) in omurgaNoktalari) {
                val pRenk = if (etiket == "C7" || etiket == "S1") mor else turkuaz

                cizgi.style = Paint.Style.STROKE
                cizgi.color = pRenk and 0x50FFFFFF
                cizgi.strokeWidth = max(4f, 5 * olcek)
                c.drawCircle(nokta.x.toFloat(), nokta.y.toFloat(), nr * 1.3f, cizgi)

                cizgi.style = Paint.Style.FILL
                cizgi.color = pRenk
                c.drawCircle(nokta.x.toFloat(), nokta.y.toFloat(), nr, cizgi)
                cizgi.style = Paint.Style.STROKE
                cizgi.color = BEYAZ
                cizgi.strokeWidth = max(2f, 2 * olcek)
                c.drawCircle(nokta.x.toFloat(), nokta.y.toFloat(), nr, cizgi)

                c.drawText(etiket, nokta.x.toFloat() + nr * 1.5f, nokta.y.toFloat() + nr * 0.5f, fEtiket)
            }
        }

        okumaPaneli(c, olcek, sonuc)
        tibbiUyari(c, w, h, olcek)
        return im
    }

    /** Kesikli çizgi — PIL'deki `_kesikli`'nin karşılığı. */
    private fun kesikli(c: Canvas, x1: Float, y1: Float, x2: Float, y2: Float,
                        renk: Int, kalinlik: Float, bosluk: Float, dolu: Float) {
        val uzunluk = hypot((x2 - x1).toDouble(), (y2 - y1).toDouble()).toFloat()
        if (uzunluk < 1f) return
        val ux = (x2 - x1) / uzunluk
        val uy = (y2 - y1) / uzunluk
        val p = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = renk; strokeWidth = kalinlik; style = Paint.Style.STROKE
        }
        var t = 0f
        while (t < uzunluk) {
            val s = min(t + dolu, uzunluk)
            c.drawLine(x1 + ux * t, y1 + uy * t, x1 + ux * s, y1 + uy * s, p)
            t = s + bosluk
        }
    }

    private fun okumaPaneli(c: Canvas, olcek: Float, sonuc: OlcumSonucu) {
        val fBuyuk = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = BEYAZ; textSize = 34 * olcek
            typeface = Typeface.create(Typeface.SANS_SERIF, Typeface.BOLD)
        }
        val fKucuk = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = 0xFFCCCCCC.toInt(); textSize = 17 * olcek
        }
        val fMor = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = 0xFFCE93D8.toInt(); textSize = 17 * olcek
            typeface = Typeface.create(Typeface.SANS_SERIF, Typeface.BOLD)
        }
        val fTurkuaz = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = 0xFF80DEEA.toInt(); textSize = 16 * olcek
        }
        val fCobb = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = 0xFFFFD700.toInt(); textSize = 18 * olcek
            typeface = Typeface.create(Typeface.SANS_SERIF, Typeface.BOLD)
        }
        val yon = when (sonuc.yon) {
            Yon.SAG_YUKSEK -> "sağ omuz yüksek"
            Yon.SOL_YUKSEK -> "sol omuz yüksek"
            Yon.SIMETRIK -> "simetrik"
        }
        val baslik = "%.1f°  %s".format(abs(sonuc.aci), yon)
        val alt = "aksiller eğim · IMU roll %+.1f° düzeltildi".format(sonuc.rollDerece)
        
        val kd = sonuc.koronalDenge
        val koronalMetin = kd?.let {
            val taraf = if (abs(it.sapmaPx) >= 1.0) (if (it.sapmaPx > 0) "sağa" else "sola") else "—"
            "koronal denge: %.1f° %s (%.0f px)".format(abs(it.sapmaDerece), taraf, abs(it.sapmaPx))
        }

        val os = sonuc.omurgaSapmalari
        val omurgaMetin = os?.let {
            "omurga sapması: O1:%+.0f  O2:%+.0f  O3:%+.0f px".format(it.o1SapmaPx, it.o2SapmaPx, it.o3SapmaPx)
        }

        val cobbRes = sonuc.cobbTahmini
        val cobbMetin = "Cobb tahmini: ~%.1f° (±%.1f°) — %s".format(
            cobbRes.tahminDerece, cobbRes.hataPayiDerece, cobbRes.kategori
        )

        val satirlar = mutableListOf(alt)
        if (koronalMetin != null) satirlar.add(koronalMetin)
        if (omurgaMetin != null) satirlar.add(omurgaMetin)
        satirlar.add(cobbMetin)

        val pad = 14 * olcek
        val ic = 6 * olcek
        val h1 = fBuyuk.fontMetrics.let { it.descent - it.ascent }
        val h2 = fKucuk.fontMetrics.let { it.descent - it.ascent }
        
        val maxGenislik = max(
            fBuyuk.measureText(baslik),
            max(fKucuk.measureText(alt), max(fMor.measureText(koronalMetin ?: ""), max(fTurkuaz.measureText(omurgaMetin ?: ""), fCobb.measureText(cobbMetin))))
        )
        val pw = maxGenislik + pad * 2.2f
        val ph = h1 + h2 * satirlar.size + ic * (satirlar.size + 2)

        // Cam efektli koyu panel
        val panel = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xCE0E0E12.toInt() }
        val cerceve = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = 0x40FFFFFF.toInt()
            style = Paint.Style.STROKE
            strokeWidth = max(1f, 1.5f * olcek)
        }
        val solCizgi = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = 0xFFFFD700.toInt()
            strokeWidth = 4 * olcek
        }

        val rBox = RectF(pad, pad, pad + pw, pad + ph)
        c.drawRoundRect(rBox, 12 * olcek, 12 * olcek, panel)
        c.drawRoundRect(rBox, 12 * olcek, 12 * olcek, cerceve)

        // Sol altın accent bar
        c.drawLine(pad + 2 * olcek, pad + 10 * olcek, pad + 2 * olcek, pad + ph - 10 * olcek, solCizgi)

        c.drawText(baslik, pad + pad + 4 * olcek, pad + ic - fBuyuk.fontMetrics.ascent, fBuyuk)
        
        var yPos = pad + ic * 2 + h1 - fKucuk.fontMetrics.ascent
        c.drawText(alt, pad + pad + 4 * olcek, yPos, fKucuk)
        
        if (koronalMetin != null) {
            yPos += h2 + ic
            c.drawText(koronalMetin, pad + pad + 4 * olcek, yPos, fMor)
        }
        
        if (omurgaMetin != null) {
            yPos += h2 + ic
            c.drawText(omurgaMetin, pad + pad + 4 * olcek, yPos, fTurkuaz)
        }
        
        yPos += h2 + ic
        c.drawText(cobbMetin, pad + pad + 4 * olcek, yPos, fCobb)
    }

    /** Tıbbi uyarı şeridi — her sonuçta zorunlu. */
    private fun tibbiUyari(c: Canvas, w: Int, h: Int, olcek: Float) {
        val f = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = 0xFFE0E0E0.toInt(); textSize = 15 * olcek
            typeface = Typeface.create(Typeface.SANS_SERIF, Typeface.BOLD)
        }
        val uyari = "Tanı aracı değildir · Klinik değerlendirme yerine geçmez"
        val serit = Paint().apply { color = 0xB0000000.toInt() }
        c.drawRect(0f, h - 28 * olcek, w.toFloat(), h.toFloat(), serit)
        val uw = f.measureText(uyari)
        c.drawText(uyari, (w - uw) / 2f, h - 22 * olcek - f.fontMetrics.ascent, f)
    }
}
