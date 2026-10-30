package com.skolyoz.olcum

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.skolyoz.olcum.cikarim.OlcumCiktisi
import com.skolyoz.olcum.cikarim.OlcumZinciri
import com.skolyoz.olcum.cikarim.OnIsleme
import com.skolyoz.olcum.kamera.CekimEkrani
import com.skolyoz.olcum.kamera.CekimHatasi
import com.skolyoz.olcum.kamera.CekimSerisi
import com.skolyoz.olcum.kamera.ozet
import com.skolyoz.olcum.olcum.Yon
import com.skolyoz.olcum.sensor.YercekimiTakipci
import com.skolyoz.olcum.ui.KilavuzCizer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

private val SARI = Color(0xFFFFCD00)
private val KIRMIZI = Color(0xFFFF4646)
private val GRI = Color(0xFFAAAAAA)

class MainActivity : ComponentActivity() {

    private lateinit var takipci: YercekimiTakipci
    private var zincir: OlcumZinciri? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        takipci = YercekimiTakipci(this)
        zincir = OlcumZinciri(this).apply {
            // Yalnizca hata ayiklama derlemesinde: Kotlin-Python
            // karsilastirmasi icin ara ciktilari diske yazar.
            // HASTA FOTOGRAFI DISKE YAZAR — urun derlemesinde kapali.
            if (BuildConfig.DEBUG) {
                teshisDizini = java.io.File(getExternalFilesDir(null), "teshis")
            }
        }

        setContent {
            MaterialTheme(colorScheme = darkColorScheme()) {
                Surface(modifier = Modifier.fillMaxSize(), color = Color.Black) {
                    Uygulama(takipci, zincir!!)
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        takipci.basla()
    }

    override fun onPause() {
        takipci.dur()
        super.onPause()
    }

    override fun onDestroy() {
        zincir?.close()
        zincir = null
        super.onDestroy()
    }
}

/** Çekim sonrası ölçümün durumu. */
private sealed interface Durum {
    data object Cekim : Durum
    data class Hesaplaniyor(val seri: CekimSerisi) : Durum
    data class Bitti(
        val seri: CekimSerisi,
        val cikti: OlcumCiktisi?,
        /** Başarılı ölçümde fotoğraf üzerine çizilmiş kılavuz; yoksa null. */
        val kilavuz: android.graphics.Bitmap?,
    ) : Durum
}

@Composable
private fun Uygulama(takipci: YercekimiTakipci, zincir: OlcumZinciri) {
    val baglam = LocalContext.current
    var izinVar by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(baglam, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED
        )
    }
    var durum by remember { mutableStateOf<Durum>(Durum.Cekim) }
    var hata by remember { mutableStateOf<CekimHatasi?>(null) }

    val izinIsteyici = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { verildi -> izinVar = verildi }

    // Çıkarım ana thread'de koşarsa arayüz donar; iki model ~yüz ms'ler
    // mertebesinde iş yapıyor.
    LaunchedEffect(durum) {
        val d = durum
        if (d is Durum.Hesaplaniyor) {
            val kare = d.seri.gecerliKareler.firstOrNull()
                ?: d.seri.kareler.firstOrNull()
            // Çıkarım ve çizim aynı arka plan bloğunda: çizim de büyük bir
            // bitmap kopyalıyor, ana thread'de yapılırsa donma yaratır.
            val (cikti, kilavuz) = if (kare == null) null to null
            else withContext(Dispatchers.Default) {
                val c = zincir.olc(kare.foto, kare.yercekimi.roll)
                val k = (c as? OlcumCiktisi.Basarili)
                    ?.let { KilavuzCizer.ciz(kare.foto, it.sonuc) }
                c to k
            }
            durum = Durum.Bitti(d.seri, cikti, kilavuz)
        }
    }

    when {
        !izinVar -> IzinEkrani { izinIsteyici.launch(Manifest.permission.CAMERA) }

        durum is Durum.Hesaplaniyor -> BilgiEkrani("Ölçülüyor…")

        durum is Durum.Bitti -> {
            val d = durum as Durum.Bitti
            SonucEkrani(d.seri, d.cikti, d.kilavuz) {
                // Bitmap'ler burada serbest bırakılıyor: seri artık hiçbir
                // yerde kullanılmıyor. Kılavuz da (fotoğrafın kopyası).
                d.seri.serbestBirak()
                d.kilavuz?.let { if (!it.isRecycled) it.recycle() }
                durum = Durum.Cekim
            }
        }

        else -> {
            CekimEkrani(
                takipci = takipci,
                onCekim = { durum = Durum.Hesaplaniyor(it); hata = null },
                onHata = { hata = it },
            )
            hata?.let { HataSeridi(it) }
        }
    }
}

@Composable
private fun BilgiEkrani(mesaj: String) {
    Column(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) { Text(mesaj, color = SARI, fontSize = 18.sp) }
}

@Composable
private fun IzinEkrani(onIste: () -> Unit) {
    Column(
        modifier = Modifier.fillMaxSize().padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("Ölçüm için kamera izni gerekiyor.", color = Color.White)
        Text(
            "Fotoğraflar cihazdan çıkmaz.",
            color = GRI, fontSize = 13.sp,
            modifier = Modifier.padding(top = 8.dp, bottom = 24.dp),
        )
        Button(onClick = onIste) { Text("İzin ver") }
    }
}

/**
 * Ölçüm sonucu.
 *
 * Arayüzde "Cobb açısı" YAZMIYOR ve yazmayacak. Ölçtüğümüz şey aksiller eğim
 * açısı — yüzey asimetrisi. Röntgenle eşleşmiş Cobb ground truth olmadığı için
 * Cobb iddiası doğrulanamaz. MOBIL_YAPILACAKLAR.md §1.
 */
@Composable
private fun SonucEkrani(
    seri: CekimSerisi,
    cikti: OlcumCiktisi?,
    kilavuz: android.graphics.Bitmap?,
    onGeri: () -> Unit,
) {
    val olcum = (cikti as? OlcumCiktisi.Basarili)?.sonuc

    // Başarılı ölçümde asıl çıktı çizilmiş fotoğraf: R/L noktaları, sarı
    // koltuk altı çizgisi, yerçekimi yatayı, açı yayı ve okuma paneli hepsi
    // görüntünün üzerinde. Metin özeti altta teşhis için kalıyor.
    if (olcum != null && kilavuz != null) {
        Column(
            modifier = Modifier.fillMaxSize().background(Color(0xFF0C0C0F)).padding(12.dp),
            verticalArrangement = Arrangement.SpaceBetween,
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            androidx.compose.foundation.Image(
                bitmap = kilavuz.asImageBitmap(),
                contentDescription = "ölçüm kılavuzu",
                modifier = Modifier.weight(1f).fillMaxWidth(),
            )
            Column(
                modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "Sağ (%.0f, %.0f) · Sol (%.0f, %.0f) · Entropi %.2f · %d ms"
                        .format(olcum.sag.x, olcum.sag.y, olcum.sol.x, olcum.sol.y,
                                olcum.entropi, olcum.sureMs),
                    color = Color(0xFFAAAAAA),
                    fontFamily = FontFamily.Monospace, fontSize = 11.sp,
                    modifier = Modifier.padding(bottom = 8.dp),
                )
                Button(
                    onClick = onGeri,
                    modifier = Modifier.fillMaxWidth(0.85f),
                    colors = androidx.compose.material3.ButtonDefaults.buttonColors(
                        containerColor = Color(0xFFFFD700),
                        contentColor = Color.Black
                    )
                ) {
                    Text("Yeni Çekim Yap", fontWeight = FontWeight.Bold, fontSize = 16.sp)
                }
            }
        }
        return
    }

    Column(
        modifier = Modifier.fillMaxSize().padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        if (olcum == null) {
            // İki başarısızlık ayrı ayrı anlatılıyor: kişi bulunamadıysa
            // kadraj sorunu var, ölçüm güvenilmezse koşullar düzeltilmeli.
            // Tek "hata" mesajı kullanıcıyı yanlış yöne iter.
            val guvenilmez = cikti is OlcumCiktisi.Guvenilmez
            Text(
                text = if (guvenilmez) "Ölçüm güvenilir değil" else "Kişi bulunamadı",
                color = KIRMIZI, fontSize = 22.sp, fontWeight = FontWeight.Bold,
            )
            Text(
                text = if (guvenilmez)
                    "Model koltuk altlarını net seçemedi. Işığı artırın, " +
                        "hastanın sırtı tamamen görünsün, biraz yaklaşın ve " +
                        "yeniden çekin."
                else
                    "Hastanın tamamı karede olsun, arkadan ve ayakta çekin.",
                color = GRI, fontSize = 14.sp,
                modifier = Modifier.padding(top = 8.dp),
            )
            Text(
                text = "Yanlış bir açı göstermektense ölçmemeyi tercih ediyoruz.",
                color = Color(0xFF777777), fontSize = 12.sp,
                modifier = Modifier.padding(top = 12.dp),
            )
        } else {
            val yon = when (olcum.yon) {
                Yon.SAG_YUKSEK -> "sağ omuz yüksek"
                Yon.SOL_YUKSEK -> "sol omuz yüksek"
                Yon.SIMETRIK -> "simetrik"
            }
            Text(
                text = "%.1f°  %s".format(kotlin.math.abs(olcum.aci), yon),
                color = SARI, fontSize = 34.sp, fontWeight = FontWeight.Bold,
            )
            Text(
                text = "aksiller eğim · IMU roll %+.1f° düzeltildi"
                    .format(olcum.rollDerece),
                color = GRI, fontSize = 13.sp,
                modifier = Modifier.padding(top = 6.dp),
            )
            // Ölçümün kendi veri setimizdeki konumu. Klinik derece (hafif/orta/
            // şiddetli) İDDİA ETMEZ — o eşikler Cobb içindir, bizim ölçtüğümüz
            // aksiller eğim farklı bir büyüklük. Bkz. Olcum.kt::Dagilim.
            val belirgin = com.skolyoz.olcum.olcum.Dagilim.belirginAsimetri(olcum.aci)
            Text(
                text = com.skolyoz.olcum.olcum.Dagilim.aciklama(olcum.aci),
                color = if (belirgin) Color(0xFFFFA05A) else GRI,
                fontSize = 13.sp,
                modifier = Modifier.padding(top = 4.dp),
            )
        }

        Text(
            text = buildString {
                append(seri.ozet())
                if (olcum != null) {
                    append("\nsağ (%.1f, %.1f)  sol (%.1f, %.1f)"
                        .format(olcum.sag.x, olcum.sag.y, olcum.sol.x, olcum.sol.y))
                    append("\nguven %.3f  entropi %.2f  cikarim %d ms"
                        .format(olcum.guven, olcum.entropi, olcum.sureMs))
                }
                (cikti as? OlcumCiktisi.Guvenilmez)?.let {
                    append("\nentropi %.2f (esik %.2f)  tepe uzakligi %.1f px (esik %.1f)"
                        .format(it.entropi, OnIsleme.ENTROPI_ESIGI,
                                it.tepeUzakligi, OnIsleme.TEPE_UZAKLIK_ESIGI))
                }
            },
            color = Color(0xFF888888),
            fontFamily = FontFamily.Monospace,
            fontSize = 11.sp,
            modifier = Modifier.padding(top = 24.dp),
        )

        Button(onClick = onGeri, modifier = Modifier.padding(top = 24.dp)) {
            Text("Yeni çekim")
        }

        Text(
            text = "Tanı aracı değildir · tarama amaçlıdır",
            color = GRI, fontSize = 12.sp,
            modifier = Modifier.padding(top = 32.dp),
        )
    }
}

@Composable
private fun HataSeridi(hata: CekimHatasi) {
    val mesaj = when (hata) {
        CekimHatasi.IMU_ESLESMEDI ->
            "Sensör okuması deklanşörle eşleşmedi — telefonu sabit tutup tekrar deneyin"
        CekimHatasi.KAMERA -> "Kamera hatası"
    }
    Column(modifier = Modifier.fillMaxSize(), verticalArrangement = Arrangement.Top) {
        Text(
            text = mesaj,
            color = Color.White,
            fontSize = 13.sp,
            modifier = Modifier.fillMaxWidth()
                .background(Color(0xCC963030)).padding(12.dp),
        )
    }
}
