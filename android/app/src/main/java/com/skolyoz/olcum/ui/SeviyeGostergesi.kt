package com.skolyoz.olcum.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.material3.Text
import com.skolyoz.olcum.olcum.Olcum
import com.skolyoz.olcum.sensor.Durus
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin

private val SARI = Color(0xFFFFCD00)
private val KIRMIZI = Color(0xFFFF4646)
private val YESIL = Color(0xFF4CD07A)
private val GRI = Color(0xFFAAAAAA)

/**
 * Canlı seviye göstergesi — kullanıcı telefonu dikleştirsin diye.
 *
 * İki ayrı bilgi gösteriyor, çünkü ikisinin sonucu farklı:
 *
 *   ROLL  : ölçümü BOZMAZ, yazılımda düzeltilir (açıdan çıkarılır). Yine de
 *           gösteriyoruz; büyük roll'da kadraj kayar ve kullanıcı hastayı
 *           çerçeveden kaçırır.
 *   PITCH : yazılımda düzeltilemez. Öne/arkaya yatıklık perspektifi bozar,
 *           iki koltuk altı kameraya farklı uzaklıkta kalır ve aradaki açı
 *           gerçek açı olmaktan çıkar. Bu yüzden 8°'yi aşınca deklanşör
 *           kilitlenir — uyarı değil, sert kapı.
 */
@Composable
fun SeviyeGostergesi(durus: Durus, modifier: Modifier = Modifier) {
    val pitchAsim = abs(durus.pitch) > Olcum.PITCH_ESIGI
    Column(modifier = modifier.fillMaxWidth().padding(horizontal = 24.dp)) {
        Canvas(modifier = Modifier.fillMaxWidth().height(120.dp)) {
            ufukCiz(durus.roll, pitchAsim)
            pitchCetveliCiz(durus.pitch, pitchAsim)
        }
        Row(
            modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = "yan eğim %+.1f°".format(durus.roll),
                color = GRI, fontSize = 13.sp, fontFamily = FontFamily.Monospace,
            )
            Text(
                text = "öne/arka %+.1f°".format(durus.pitch),
                color = if (pitchAsim) KIRMIZI else YESIL,
                fontSize = 13.sp,
                fontWeight = if (pitchAsim) FontWeight.Bold else FontWeight.Normal,
                fontFamily = FontFamily.Monospace,
            )
        }
    }
}

/** Roll'u gösteren yapay ufuk: çizgi telefon döndükçe ters yönde döner. */
private fun DrawScope.ufukCiz(rollDerece: Double, pitchAsim: Boolean) {
    val mx = size.width / 2f
    val my = size.height / 2f
    val yari = min(size.width, size.height * 3f) * 0.32f

    // Sabit referans: gerçek yatay (kesikli)
    val kesik = 14f
    var t = -yari
    while (t < yari) {
        val s = min(t + kesik, yari)
        drawLine(GRI.copy(alpha = 0.45f), Offset(mx + t, my), Offset(mx + s, my),
                 strokeWidth = 2f)
        t = s + kesik
    }

    // Telefonun ufku: roll kadar döner. Roll pozitifken telefon saat yönünde
    // dönmüş demektir, dolayısıyla ufuk ekranda ters yöne eğik görünür.
    val r = Math.toRadians(-rollDerece)
    val dx = (cos(r) * yari).toFloat()
    val dy = (sin(r) * yari).toFloat()
    val renk = if (pitchAsim) KIRMIZI else SARI
    drawLine(renk, Offset(mx - dx, my - dy), Offset(mx + dx, my + dy),
             strokeWidth = 5f, cap = StrokeCap.Round)
    drawCircle(renk, radius = 5f, center = Offset(mx, my))
}

/** Pitch'i gösteren dikey cetvel: kabul bandı yeşil, dışı kırmızı. */
private fun DrawScope.pitchCetveliCiz(pitchDerece: Double, pitchAsim: Boolean) {
    val x = size.width - 14f
    val ust = 8f
    val alt = size.height - 8f
    val orta = (ust + alt) / 2f
    val yariYukseklik = (alt - ust) / 2f

    // Cetvel 20° aralığı kapsıyor; 8°'lik kabul bandı görsel olarak işaretli.
    val olcek = yariYukseklik / 20f
    val bant = (Olcum.PITCH_ESIGI * olcek).toFloat()

    drawLine(GRI.copy(alpha = 0.35f), Offset(x, ust), Offset(x, alt),
             strokeWidth = 3f)
    drawLine(YESIL.copy(alpha = 0.7f), Offset(x, orta - bant), Offset(x, orta + bant),
             strokeWidth = 3f)

    val y = (orta - (pitchDerece * olcek).toFloat())
        .coerceIn(ust, alt)
    drawCircle(if (pitchAsim) KIRMIZI else YESIL, radius = 8f, center = Offset(x, y))
    drawCircle(Color.White, radius = 8f, center = Offset(x, y),
               style = Stroke(width = 2f))
}
