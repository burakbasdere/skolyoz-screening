package com.skolyoz.olcum.olcum

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot

/**
 * Ölçüm matematiği — `scripts/19_kilavuz_cizim.py` dosyasının birebir portu.
 *
 * Bu dosya bilerek Android'e bağımlı değildir (ne `PointF` ne `SensorEvent`).
 * Böylece emülatör/cihaz gerekmeden düz JVM birim testiyle Python çıktısına
 * karşı doğrulanabiliyor. Bkz. `src/test/.../OlcumTest.kt`.
 *
 * KONVANSİYONLAR — bunlar değişmez, MOBIL_YAPILACAKLAR.md §4:
 *   - Çekim arkadan yapılır (hastanın sırtı kameraya dönük)
 *   - Model çıktısında index 0 = SAĞ, index 1 = SOL
 *   - Arkadan çekimde kişinin sağı karede sağdadır
 *   - Açı işareti: pozitif = SAĞ omuz yüksek
 *   - y aşağı doğru artar (görüntü konvansiyonu)
 */
object Olcum {

    /** Telefon öne/arkaya bu açıdan fazla yatıksa perspektif ölçümü bozar. */
    const val PITCH_ESIGI = 8.0

    /** |açı| bu değerin altındaysa "simetrik" sayılır. */
    const val SIMETRI_BANDI = 1.0

    /**
     * Telefonun roll açısı (derece), IMU yerçekimi vektöründen.
     *
     * Android sensör eksenleri: x sağa, y yukarı (ekran düzleminde),
     * z ekrandan dışarı. Telefon dik tutulduğunda g = (0, +9.81, 0) → roll 0.
     * Telefon saat yönünde döndükçe roll pozitif büyür.
     */
    fun rollAcisi(gx: Float, gy: Float): Double =
        Math.toDegrees(atan2(gx.toDouble(), gy.toDouble()))

    /**
     * Telefonun pitch açısı (derece) — öne/arkaya yatıklık.
     *
     * Yerçekimi vektörünün ekran DÜZLEMİYLE yaptığı açı. Bu tanım roll'dan
     * bağımsızdır: telefonu kendi ekseninde döndürmek pitch'i değiştirmez,
     * sadece öne/arkaya yatırmak değiştirir.
     *
     * Ekran düzlemi dikey → g'nin z bileşeni 0 → pitch 0.
     * Pozitif = telefonun üstü geriye yatık (ekran yukarı bakıyor).
     */
    fun pitchAcisi(gx: Float, gy: Float, gz: Float): Double =
        Math.toDegrees(atan2(gz.toDouble(), hypot(gx.toDouble(), gy.toDouble())))

    /**
     * Aksiller eğim açısı — iki koltuk altı noktasının yerçekimi yatayına göre
     * eğimi.
     *
     * DİKKAT: bu Cobb açısı DEĞİLDİR, yüzey asimetrisidir. Arayüzde asla
     * "Cobb" yazılmaz.
     *
     * @param sag kişinin sağ koltuk altı, ORİJİNAL görüntü pikselinde
     * @param sol kişinin sol koltuk altı, ORİJİNAL görüntü pikselinde
     * @param rollDerece telefonun roll açısı; görüntü yatayını gerçek yataya
     *        çevirmek için çıkarılır
     * @return pozitif = kişinin SAĞ omzu daha yüksek
     */
    fun aksillerAci(sag: Nokta, sol: Nokta, rollDerece: Double = 0.0): Double {
        val dx = sol.x - sag.x
        val dy = sol.y - sag.y
        if (abs(dx) < 1e-9 && abs(dy) < 1e-9) return 0.0
        val ham = Math.toDegrees(atan2(dy, abs(dx)))
        return ham - rollDerece
    }

    /**
     * Telefon öne/arkaya yatıkken perspektif açıyı bozar → çekimi reddet.
     * Bu bir uyarı değil, sert kapıdır: deklanşör kilitlenir.
     */
    fun olcumGecerliMi(pitchDerece: Double, esik: Double = PITCH_ESIGI): Boolean =
        abs(pitchDerece) <= esik

    /** Açıdan karar sınıfı — ürünün gerçekten sınıflandırma yaptığı yer. */
    fun yon(aciDerece: Double): Yon = when {
        abs(aciDerece) < SIMETRI_BANDI -> Yon.SIMETRIK
        aciDerece > 0 -> Yon.SAG_YUKSEK
        else -> Yon.SOL_YUKSEK
    }

    /**
     * 640×640 model uzayındaki noktayı orijinal çözünürlüğe geri map'ler.
     *
     * Eğitim verisi 640×640'a STRETCH edilmişti (letterbox değil), o yüzden
     * çıkarımda da stretch uygulanır — ama açı ASLA 640×640 uzayında
     * hesaplanmaz: stretch en-boy oranını bozar. 4:3 bir fotoğrafta bu adımı
     * atlamak ~%25 açı hatası demektir. MOBIL_YAPILACAKLAR.md §3.
     */
    fun orijinaleMaple(nokta: Nokta, orijinalGenislik: Int, orijinalYukseklik: Int,
                       modelBoyut: Int = 640): Nokta = Nokta(
        x = nokta.x * orijinalGenislik / modelBoyut,
        y = nokta.y * orijinalYukseklik / modelBoyut,
    )

    /**
     * C7'nin sakrumdan çekilen YERÇEKİMİ ŞAKÜLÜNDEN (dikey eksen) yatay sapması.
     *
     * Skolyoz muayenesinde C7 vertebra prominensinden sarkıtılan şakül ipliği ölçümünün karşılığıdır.
     *
     * @param c7 C7 noktası (boyun tabanı), orijinal görüntü pikselinde
     * @param sakrum Sakrum noktası (kuyruk sokumu üstü), orijinal görüntü pikselinde
     * @param rollDerece telefonun roll açısı (derece)
     * @return KoronalDengeResult (sapmaPx, sapmaDerece, govdeBoyuPx)
     */
    fun koronalDenge(c7: Nokta, sakrum: Nokta, rollDerece: Double = 0.0): KoronalDengeResult {
        val r = Math.toRadians(rollDerece)
        val dikX = -kotlin.math.sin(r)
        val dikY = kotlin.math.cos(r)
        val yatX = kotlin.math.cos(r)
        val yatY = kotlin.math.sin(r)

        val vx = c7.x - sakrum.x
        val vy = c7.y - sakrum.y

        val sapmaPx = vx * yatX + vy * yatY
        val govdeBoyuPx = abs(vx * dikX + vy * dikY)

        val sapmaDerece = if (govdeBoyuPx < 1e-6) 0.0 else Math.toDegrees(atan2(sapmaPx, govdeBoyuPx))

        return KoronalDengeResult(
            sapmaPx = sapmaPx,
            sapmaDerece = sapmaDerece,
            govdeBoyuPx = govdeBoyuPx
        )
    }

    /**
     * O1, O2, O3 omurga noktalarının Sakrum şakül hattından yanal sapması.
     */
    fun omurgaSapmalari(
        c7: Nokta, o1: Nokta, o2: Nokta, o3: Nokta, sakrum: Nokta, rollDerece: Double = 0.0
    ): OmurgaSapmalariResult {
        val r = Math.toRadians(rollDerece)
        val yatX = kotlin.math.cos(r)
        val yatY = kotlin.math.sin(r)

        fun yanalSapma(p: Nokta): Double {
            val vx = p.x - sakrum.x
            val vy = p.y - sakrum.y
            return vx * yatX + vy * yatY
        }

        val d1 = yanalSapma(o1)
        val d2 = yanalSapma(o2)
        val d3 = yanalSapma(o3)
        val maks = maxOf(abs(d1), maxOf(abs(d2), abs(d3)))

        return OmurgaSapmalariResult(
            o1SapmaPx = d1,
            o2SapmaPx = d2,
            o3SapmaPx = d3,
            maksSapmaPx = maks
        )
    }
}

/** C7-Sakrum şakül ölçüm sonucu. */
data class KoronalDengeResult(
    val sapmaPx: Double,
    val sapmaDerece: Double,
    val govdeBoyuPx: Double
)

/** Omurga noktalarının (O1, O2, O3) yanal sapma sonucu. */
data class OmurgaSapmalariResult(
    val o1SapmaPx: Double,
    val o2SapmaPx: Double,
    val o3SapmaPx: Double,
    val maksSapmaPx: Double
)

/** Görüntü pikselinde nokta. y aşağı doğru artar. */
data class Nokta(val x: Double, val y: Double)

enum class Yon { SAG_YUKSEK, SOL_YUKSEK, SIMETRIK }

data class CobbTahminiResult(
    val tahminDerece: Double,
    val hataPayiDerece: Double = 3.5,
    val altSinir: Double,
    val ustSinir: Double,
    val kategori: String,
    val aciklama: String
)

/**
 * Yüzey topoğrafyasından Cobb Açısı tahmini kalibrasyon katmanı.
 *
 * UYARI: Bu tıbbi röntgen ölçümü değildir; AI yüzey asimetrisi türevi
 * tarama tahminidir.
 */
object Kalibrasyon {
    const val KATSAYI_AKSILLER = 1.25
    const val KATSAYI_KORONAL = 1.10
    const val KATSAYI_APEKS = 2.40
    const val STANDART_HATA_DERECE = 3.5

    fun cobbTahmini(
        aksillerAciDeg: Double,
        koronalAciDeg: Double = 0.0,
        apeksSapmaPx: Double = 0.0,
        govdeBoyuPx: Double = 1.0
    ): CobbTahminiResult {
        val absAksiller = abs(aksillerAciDeg)
        val absKoronal = abs(koronalAciDeg)
        val apeksOran = (abs(apeksSapmaPx) / maxOf(govdeBoyuPx, 1e-6)) * 100.0

        val hamCobb = KATSAYI_AKSILLER * absAksiller +
                      KATSAYI_KORONAL * absKoronal +
                      KATSAYI_APEKS * apeksOran

        val cobbVal = Math.round(maxOf(0.0, hamCobb) * 10.0) / 10.0
        val alt = Math.round(maxOf(0.0, cobbVal - STANDART_HATA_DERECE) * 10.0) / 10.0
        val ust = Math.round((cobbVal + STANDART_HATA_DERECE) * 10.0) / 10.0

        val (kategori, aciklama) = when {
            cobbVal < 10.0 -> "Normal / Minimal Asimetri" to "Cobb tahmini 10° altında — fizyolojik sınırlar dahilinde"
            cobbVal < 25.0 -> "Hafif Skolyoz Şüphesi" to "Cobb tahmini 10°-25° aralığında — periyodik takip önerilir"
            cobbVal < 45.0 -> "Orta Skolyoz Şüphesi" to "Cobb tahmini 25°-45° aralığında — ortopedi / fizyoterapi değerlendirmesi önerilir"
            else -> "Şiddetli Skolyoz Şüphesi" to "Cobb tahmini 45° üzerinde — uzman hekim değerlendirmesi gerekir"
        }

        return CobbTahminiResult(
            tahminDerece = cobbVal,
            hataPayiDerece = STANDART_HATA_DERECE,
            altSinir = alt,
            ustSinir = ust,
            kategori = kategori,
            aciklama = aciklama
        )
    }
}

/**
 * Ölçülen açının, ETİKETLİ VERİ SETİMİZDEKİ dağılıma göre konumu.
 *
 * NEDEN COBB SINIFI DEĞİL
 * -----------------------
 * Klinik eşikler (10-25 hafif, 25-45 orta, >45 şiddetli) COBB açısı içindir.
 * Bizim ölçtüğümüz aksiller eğim farklı bir büyüklük: veri setimizde medyan
 * 2.21°, p90 7.68°. Aynı birimde ama aynı ölçek değil. Aksiller açısı 8° olan
 * birinin Cobb'u 5° de olabilir 35° de — hangi klinik etiketi yapıştırırsak
 * bir kısmına yanlış söylemiş oluruz.
 *
 * KRİTİK KISIT — arayüz metnini buna göre yazın
 * ---------------------------------------------
 * Bu yüzdelikler 337 görüntülük ETİKETLİ SETİMİZDEN geliyor. O set okul
 * taramasından gelen rastgele bir örneklem DEĞİL; klinik/skolyoz kaynaklı
 * görüntülerle zenginleşmiş. Dolayısıyla "üst %10" ifadesi GENEL NÜFUSTA
 * ne demek, bilmiyoruz.
 *
 * Arayüzde "insanların %90'ından yüksek" YAZILMAZ.
 * "Ölçtüklerimizin üst %10'unda" yazılır — bu doğrulanabilir bir ifade.
 *
 * Röntgen eşleşmeli veri gelirse bu sınıf yerini gerçek Cobb sınıfına bırakır.
 */
object Dagilim {
    /** (|açı| derece, yüzdelik) — data/final 337 görüntüden ölçüldü. */
    private val TABLO = listOf(
        2.21 to 50.0, 2.73 to 60.0, 3.27 to 70.0, 3.75 to 75.0,
        4.45 to 80.0, 5.46 to 85.0, 7.68 to 90.0, 10.27 to 95.0,
        12.46 to 97.5, 14.15 to 99.0,
    )

    /** Üstünde "belirgin asimetri" denen eşik. Veri setinin p90'ı. */
    const val YONLENDIRME_ESIGI = 7.68

    /** |açı| -> yüzdelik (0-100), tablo arası doğrusal aradeğerleme. */
    fun yuzdelik(aciDerece: Double): Double {
        val a = kotlin.math.abs(aciDerece)
        if (a <= TABLO.first().first) {
            return a / TABLO.first().first * TABLO.first().second
        }
        for (i in 0 until TABLO.size - 1) {
            val (x0, y0) = TABLO[i]
            val (x1, y1) = TABLO[i + 1]
            if (a <= x1) return y0 + (a - x0) / (x1 - x0) * (y1 - y0)
        }
        return 99.5
    }

    /** Ölçüm yönlendirme eşiğinin üstünde mi. */
    fun belirginAsimetri(aciDerece: Double): Boolean =
        kotlin.math.abs(aciDerece) >= YONLENDIRME_ESIGI

    /**
     * Arayüzde gösterilecek satır. Klinik derece İDDİA ETMEZ; yalnızca
     * ölçümün kendi veri setimizdeki konumunu bildirir.
     */
    fun aciklama(aciDerece: Double): String {
        val ust = 100.0 - yuzdelik(aciDerece)
        // Türkçe iyelik eki her sayıda değişiyor ("%10'unda" ama "%2'sinde"),
        // o yüzden ekten kaçınıp sabit "dilimde" kullanıyoruz.
        val konum = when {
            ust >= 50.0 -> "veri setimizin alt yarısında"
            ust < 1.0 -> "veri setimizde üst %1 dilimde"
            else -> "veri setimizde üst %${"%.0f".format(ust)} dilimde"
        }
        return if (belirginAsimetri(aciDerece)) {
            "$konum — belirgin asimetri, hekime başvurmanız önerilir"
        } else {
            konum
        }
    }
}
