package com.skolyoz.olcum.cikarim

import com.skolyoz.olcum.olcum.Nokta

/**
 * YOLO11n-pose'un ham çıktısını iki noktaya çevirir.
 *
 * ÇIKTI BİÇİMİ
 * ------------
 * Tensör `(1, 11, 8400)`: 8400 aday, her aday için 11 kanal.
 *
 *     0..3  : cx, cy, w, h   (kutu, 640×640 piksel uzayında)
 *     4     : güven
 *     5..7  : kx0, ky0, kv0  → index 0 = SAĞ
 *     8..10 : kx1, ky1, kv1  → index 1 = SOL
 *
 * Koordinatlar zaten piksel cinsinden; ultralytics'in başı onları modelin
 * içinde çözüyor, ayrıca anchor/stride hesabı yapmamız gerekmiyor.
 *
 * NMS YOK
 * -------
 * Karede tek kişi bekliyoruz; en yüksek güvenli adayı almak yeterli. NMS
 * yazmak burada gereksiz karmaşıklık olurdu. Python değerlendirmesi
 * (`26_tflite_degerlendirme.py`) aynı sadeleştirmeyi kullanıyor, dolayısıyla
 * iki taraf birebir aynı kararı veriyor.
 *
 * SOL/SAĞ POSE MODELİNDEN ALINMIYOR
 * ---------------------------------
 * Index 0 = sağ, index 1 = sol; bu bizim eğittiğimiz modelin konvansiyonu.
 * MOBIL_YAPILACAKLAR.md §5.3: hazır COCO pose modellerinin arkadan çekimde
 * anatomik sol/sağ ataması yalnızca %11.7 doğru. Bizimki konum kuralıyla
 * eğitildi, o yüzden indekslere güvenilir.
 */
object Asama1Coz {

    const val GUVEN_ESIGI = 0.25f
    private const val KANAL = 26
    private const val ADAY = 8400

    /** Aşama 1'in bulduğu 7 kaba nokta, 640×640 uzayında. */
    data class KabaNoktalar(
        val sag: Nokta,
        val sol: Nokta,
        val c7: Nokta,
        val omurga1: Nokta,
        val omurga2: Nokta,
        val omurga3: Nokta,
        val sakrum: Nokta,
        val guven: Float
    )

    /**
     * @param cikti `(1, 26, 8400)` veya `(1, 11, 8400)` tensörünün düzleştirilmiş hâli
     * @return güven eşiğini geçen aday yoksa `null` — kişi bulunamadı
     */
    fun coz(cikti: FloatArray, adaySayisi: Int = ADAY): KabaNoktalar? {
        val kanal = cikti.size / adaySayisi
        require(kanal == 11 || kanal == 26) {
            "beklenen 11 veya 26 kanal * $adaySayisi, gelen ${cikti.size}"
        }

        var enIyi = -1
        var enIyiGuven = Float.NEGATIVE_INFINITY
        val guvenTabani = 4 * adaySayisi
        for (i in 0 until adaySayisi) {
            val g = cikti[guvenTabani + i]
            if (g > enIyiGuven) {
                enIyiGuven = g
                enIyi = i
            }
        }
        if (enIyi < 0 || enIyiGuven < GUVEN_ESIGI) return null

        fun noktaAl(kptIdx: Int): Nokta {
            if (5 + kptIdx * 3 >= kanal) return Nokta(0.0, 0.0)
            return Nokta(
                x = cikti[(5 + kptIdx * 3) * adaySayisi + enIyi].toDouble(),
                y = cikti[(6 + kptIdx * 3) * adaySayisi + enIyi].toDouble()
            )
        }

        return KabaNoktalar(
            sag = noktaAl(0),
            sol = noktaAl(1),
            c7 = noktaAl(2),
            omurga1 = noktaAl(3),
            omurga2 = noktaAl(4),
            omurga3 = noktaAl(5),
            sakrum = noktaAl(6),
            guven = enIyiGuven
        )
    }
}
