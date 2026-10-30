package com.skolyoz.olcum.sensor

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import com.skolyoz.olcum.olcum.Olcum
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Yerçekimi sensörünü dinler; örnekleri [OrnekHalkasi]'na yazar ve canlı duruşu
 * arayüze yayınlar.
 *
 * NEDEN GEÇMİŞ SAKLIYORUZ
 * -----------------------
 * "Şu anki" yerçekimini okuyup fotoğrafla eşleştirmek yanlıştır: fotoğrafın
 * gerçekte yakalandığı an ile geri çağrının çalıştığı an arasında onlarca ms
 * gecikme var. Bu sürede telefon birkaç derece dönebilir ve bu doğrudan ölçüm
 * hatasına yazılır — ölçtüğümüz şeyin kendisi bir açı.
 *
 * Eşleştirme mantığı bilerek [OrnekHalkasi]'nda: orası Android'e bağımsız ve
 * birim testi var. MOBIL_YAPILACAKLAR.md Faz 2: "Bu eşleşme ölçüm doğruluğunun
 * temeli — gevşek yapma."
 */
class YercekimiTakipci(context: Context) : SensorEventListener {

    private val sensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager

    /** TYPE_GRAVITY yoksa ivmeölçere düşüp alçak geçiren filtre uygularız. */
    private val yercekimiSensoru: Sensor? =
        sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY)
    private val ivmeSensoru: Sensor? =
        sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

    private val kullanilanSensor: Sensor? = yercekimiSensoru ?: ivmeSensoru

    /** Donanımda TYPE_GRAVITY var mı — yoksa ölçüm bir tık daha gürültülü. */
    val donanimYercekimi: Boolean get() = yercekimiSensoru != null

    private val halka = OrnekHalkasi()

    private val _durus = MutableStateFlow(Durus())
    val durus: StateFlow<Durus> = _durus.asStateFlow()

    // alçak geçiren filtre durumu (yalnızca ivmeölçer yedeğinde kullanılır)
    private var fx = 0f
    private var fy = 0f
    private var fz = 0f
    private var filtreHazir = false

    fun basla() {
        val s = kullanilanSensor ?: return
        sensorManager.registerListener(this, s, SensorManager.SENSOR_DELAY_GAME)
    }

    fun dur() {
        sensorManager.unregisterListener(this)
        filtreHazir = false
        // Duraklamadan önceki örnekler artık geçerli değil: uygulama arka
        // plandayken telefon bambaşka bir konuma gitmiş olabilir. Bayat
        // örneklerle enterpolasyon yapılmasın.
        halka.temizle()
    }

    override fun onSensorChanged(event: SensorEvent) {
        val (x, y, z) = if (event.sensor.type == Sensor.TYPE_GRAVITY) {
            Triple(event.values[0], event.values[1], event.values[2])
        } else {
            // Alçak geçiren: ivmeölçerden yerçekimini ayıklar. İlk örnekte
            // filtreyi doğrudan ölçümle başlat, yoksa sıfırdan yavaşça
            // yaklaşırken ilk saniyeler hatalı açı gösterir.
            if (!filtreHazir) {
                fx = event.values[0]; fy = event.values[1]; fz = event.values[2]
                filtreHazir = true
            } else {
                fx += ALFA * (event.values[0] - fx)
                fy += ALFA * (event.values[1] - fy)
                fz += ALFA * (event.values[2] - fz)
            }
            Triple(fx, fy, fz)
        }

        halka.ekle(event.timestamp, x, y, z)

        val pitch = Olcum.pitchAcisi(x, y, z)
        _durus.value = Durus(
            roll = Olcum.rollAcisi(x, y),
            pitch = pitch,
            gecerli = Olcum.olcumGecerliMi(pitch),
            hazir = true,
        )
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    /**
     * Deklanşör anındaki yerçekimi vektörü.
     *
     * @param hedefNanos `ImageProxy.imageInfo.timestamp`
     * @return eşleştirilemezse `null` — ölçüm reddedilmeli
     */
    fun ornekEslestir(hedefNanos: Long): YercekimiOrnegi? = halka.eslestir(hedefNanos)

    private companion object {
        /** Alçak geçiren filtre katsayısı (yalnızca ivmeölçer yedeğinde). */
        const val ALFA = 0.15f
    }
}

/** Canlı seviye göstergesinin okuduğu anlık duruş. */
data class Durus(
    val roll: Double = 0.0,
    val pitch: Double = 0.0,
    val gecerli: Boolean = false,
    val hazir: Boolean = false,
)

/** Deklanşör anıyla eşleştirilmiş yerçekimi örneği. */
data class YercekimiOrnegi(
    val gx: Float,
    val gy: Float,
    val gz: Float,
    /**
     * Deklanşör ile kullanılan örnek arasındaki zaman farkı (ms).
     * Enterpolasyon yapıldıysa 0 — değer tam deklanşör anına aittir.
     */
    val sapmaMs: Double,
    /** Değer iki örnek arasından enterpole edildi mi? */
    val enterpole: Boolean = false,
    /** Enterpolasyonda kullanılan iki örnek arasındaki mesafe (ms). */
    val komsuAralikMs: Double = 0.0,
) {
    val roll: Double get() = Olcum.rollAcisi(gx, gy)
    val pitch: Double get() = Olcum.pitchAcisi(gx, gy, gz)
}
