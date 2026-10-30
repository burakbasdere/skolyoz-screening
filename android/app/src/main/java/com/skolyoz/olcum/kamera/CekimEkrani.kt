package com.skolyoz.olcum.kamera

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import android.view.Surface
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.skolyoz.olcum.olcum.Olcum
import com.skolyoz.olcum.sensor.YercekimiOrnegi
import com.skolyoz.olcum.sensor.YercekimiTakipci
import com.skolyoz.olcum.ui.SeviyeGostergesi
import java.util.concurrent.Executor
import java.util.concurrent.Executors

private const val ETIKET = "CekimEkrani"

/**
 * Kaç kare çekilecek.
 *
 * ŞU AN 1 — yani seri değil, tek kare. Gerekçe:
 *
 * Çok kareli çekim + medyan, hata kuyruğunu (bkz. [CekimSerisi]) bastırmak
 * için cazip. Ama kare başına deklanşör gecikmesi gerçek cihazda ~950 ms
 * ölçüldü; 3 kare ~3 saniye eder. Bu bedeli ödemeden önce kazancın GERÇEK
 * olduğunu görmek gerekiyor — medyan yalnızca bağımsız gürültüyü bastırır ve
 * arka arkaya çekilen kareler neredeyse aynı görüntüdür.
 *
 * Kazanç telefonda ölçülemez (ground truth yok). Ölçüm Python tarafında,
 * 50 görüntülük test setinde yapılacak. Sonuç olumluysa burada tek sabiti
 * değiştirmek yeterli — çekim ve birleştirme yapısı hazır duruyor.
 */
const val SERI_KARE_SAYISI = 1

/** Seri içindeki tek bir kare. */
data class Kare(
    val foto: Bitmap,
    val yercekimi: YercekimiOrnegi,
) {
    /** Bu karenin pitch'i kabul bandında mı. */
    val gecerli: Boolean get() = Olcum.olcumGecerliMi(yercekimi.pitch)
}

/**
 * Tek deklanşör basışında çekilen kare serisi.
 *
 * NEDEN SERİ
 * ----------
 * MOBIL_YAPILACAKLAR.md §7: "Noktaların %10'u >10 px sapıyor, çoğunlukla
 * eğilmiş/atipik pozlarda." Tek kare çekmek, bu hata kuyruğuna denk gelme
 * riskini tamamen kullanıcıya yıkar.
 *
 * NEDEN MEDYAN, NEDEN ORTALAMA DEĞİL
 * ----------------------------------
 * Aynı dokümandaki test sonuçları: keypoint hatası medyan 3,55 px, ortalama
 * 7,83 px. Aradaki iki kat fark hata kuyruğunun ta kendisi. Serinin amacı o
 * kuyruktaki kareyi ELEMEK; ortalama alırsak elemeyiz, etkisini 1/3'e indirip
 * sonuca taşırız. Üç değerde medyan ortadaki değerdir — bozuk kareyi tamamen
 * dışarıda bırakır.
 *
 * DÜRÜST UYARI
 * ------------
 * Medyan yalnızca BAĞIMSIZ gürültüyü bastırır. Arka arkaya çekilen kareler
 * neredeyse aynı görüntüdür; model üçünde de aynı hatayı yapabilir. Kazancın
 * gerçek olup olmadığı ölçülmeli — telefonda ground truth yok, bu deney
 * Python tarafında 50 görüntülük test setinde yapılmalı.
 *
 * Kazanç ölçülemese bile serinin bedavaya gelen bir faydası var: pitch'i
 * bozulmuş veya kişinin bulunamadığı kareler elenebiliyor.
 */
data class CekimSerisi(val kareler: List<Kare>) {
    /** Pitch kapısını geçen kareler — ölçüm yalnızca bunlarla yapılır. */
    val gecerliKareler: List<Kare> get() = kareler.filter { it.gecerli }

    /**
     * Serideki bitmap'leri serbest bırakır.
     *
     * Çöp toplayıcıya bırakmak yetmiyor: her kare tam çözünürlükte bir bitmap
     * ve art arda çekimde bunlar birikiyor. `Bitmap.recycle()` yerel belleği
     * hemen geri verir, GC'nin sırasını beklemez.
     *
     * ÇAĞIRDIKTAN SONRA bu serideki `foto` alanlarına DOKUNULMAZ — geri
     * verilmiş bir bitmap'i okumak çökme sebebidir. Bu yüzden yalnızca sonuç
     * ekranından çıkarken, seri artık hiçbir yerde kullanılmıyorken çağrılır.
     */
    fun serbestBirak() {
        for (k in kareler) if (!k.foto.isRecycled) k.foto.recycle()
    }

    /**
     * Faz 3 kancası: her karenin açısı hesaplandıktan sonra medyan burada
     * alınacak. Ortalama DEĞİL — gerekçesi sınıf belgesinde.
     */
    fun medyanAci(aciHesapla: (Kare) -> Double): Double? {
        val acilar = gecerliKareler.map(aciHesapla).sorted()
        if (acilar.isEmpty()) return null
        return acilar[acilar.size / 2]
    }
}

/** Çekim başarısız olduğunda nedeni. */
enum class CekimHatasi {
    /** Serideki hiçbir kare deklanşör anıyla eşleşmedi. */
    IMU_ESLESMEDI,

    /** Kamera katmanı hata verdi. */
    KAMERA,
}

/**
 * Çekim ekranı: önizleme + canlı seviye göstergesi + deklanşör.
 *
 * ARKA KAMERA ZORUNLU
 * -------------------
 * Ön kamera görüntüyü aynalar. Sağ/sol koltuk altı ayrımı bu üründe ANLAM
 * taşıyor (açının işareti hangi omzun yüksek olduğunu söylüyor), dolayısıyla
 * aynalanmış bir görüntü ölçümü sessizce ters çevirir.
 * MOBIL_YAPILACAKLAR.md §5.1.
 */
@Composable
fun CekimEkrani(
    takipci: YercekimiTakipci,
    onCekim: (CekimSerisi) -> Unit,
    onHata: (CekimHatasi) -> Unit,
) {
    val baglam = LocalContext.current
    val yasamSahibi = LocalLifecycleOwner.current
    val durus by takipci.durus.collectAsStateWithLifecycle()
    var cekiliyor by remember { mutableStateOf(false) }
    var ilerleme by remember { mutableIntStateOf(0) }

    // JPEG cozme ve 12 MP bitmap dondurme ana thread'de yapilirsa arayuz
    // donuyor (gercek cihazda 'Slow main thread' uyarisi aliniyordu).
    // Cekim geri cagrisi ayri bir thread'de kosuyor; sonuc ana thread'e
    // geri gonderiliyor cunku Compose durumu oradan yazilmali.
    val anaYurutucu = remember(baglam) { ContextCompat.getMainExecutor(baglam) }
    val yurutucu = remember { Executors.newSingleThreadExecutor() }
    DisposableEffect(Unit) { onDispose { yurutucu.shutdown() } }

    val imageCapture = remember {
        ImageCapture.Builder()
            // ZERO_SHUTTER_LAG. Kamera surekli bir kare tamponu tutar ve
            // deklansora basildiginda YENI kare cekmek yerine tampondan
            // basis anina ait kareyi verir.
            //
            // NEDEN: gercek cihazda olculdu, deklansor gecikmesi ~950 ms.
            // Once bunu sensor okuma / JPEG kodlama sandik ve cozunurlugu
            // 12 MP -> 2 MP dusurduk: gecikme 956 -> 902 ms, yani DEGISMEDI.
            // Demek ki sureyi yiyen sey her cekimde calisan otomatik odak +
            // pozlama yakinsamasi (3A on-cekim dizisi). ZSL tam olarak bunu
            // atlar.
            //
            // MAXIMIZE_QUALITY neden degil: cok kareli fuzyonda uretilen
            // goruntu birden fazla poza ait; imageInfo.timestamp'in hangi
            // kareye karsilik geldigi belirsizlesir. Butun IMU eslestirmemiz
            // o damganin TEK BIR ana ait olmasina dayaniyor. ZSL bu acidan
            // da dogru secim: donen kare zaten tek bir yakalanmis kare.
            //
            // Cihaz ZSL desteklemiyorsa CameraX sessizce MINIMIZE_LATENCY'ye
            // duser.
            .setCaptureMode(ImageCapture.CAPTURE_MODE_ZERO_SHUTTER_LAG)
            // Aktivite portreye kilitli; hedef rotasyonu da sabitliyoruz ki
            // ImageProxy.rotationDegrees'in anlamı cihazdan cihaza kaymasın.
            .setTargetRotation(Surface.ROTATION_0)
            .setResolutionSelector(
                ResolutionSelector.Builder()
                    // 4:3 = sensorun tam alani. 16:9 sensoru kirpar, ayakta
                    // duran bir insani cerceveye almak zorlasir.
                    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                    // TAM COZUNURLUK.
                    //
                    // Cozunurlugu dusurmeyi IKI KEZ denedik, ikisinde de
                    // gecikmeye faydasi olmadi (gercek cihazda olculdu):
                    //   12 MP, ZSL yok  : 956 ms      2 MP, ZSL yok : 902 ms
                    //   12 MP, ZSL var  : 588 ms      4 MP, ZSL var : 602 ms
                    //
                    // Yani "yakalama -> teslim" kalemi (~700 ms) piksel
                    // sayisiyla olceklenmiyor; sabit bir boru hatti maliyeti.
                    // Gecikmeyi asil dusuren sey ZSL oldu.
                    //
                    // Karsiliksiz bedel odememek icin tam cozunurluge
                    // donuldu: olcume etkisi zaten yok (Asama 1 goruntuyu
                    // 640x640'a geriyor) ama Faz 5'te arsivlenecek fotografin
                    // kalitesi icin onemli.
                    .setResolutionStrategy(ResolutionStrategy.HIGHEST_AVAILABLE_STRATEGY)
                    .build()
            )
            .build()
    }

    Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { ctx ->
                val onizleme = PreviewView(ctx).apply {
                    scaleType = PreviewView.ScaleType.FIT_CENTER
                }
                val gelecek = ProcessCameraProvider.getInstance(ctx)
                gelecek.addListener({
                    val saglayici = gelecek.get()
                    val preview = Preview.Builder().build().also {
                        it.surfaceProvider = onizleme.surfaceProvider
                    }
                    try {
                        saglayici.unbindAll()
                        val kamera = saglayici.bindToLifecycle(
                            yasamSahibi,
                            CameraSelector.DEFAULT_BACK_CAMERA,   // ön kamera YASAK
                            preview,
                            imageCapture,
                        )
                        // ZSL desteklenmiyorsa CameraX sessizce geri duser;
                        // sessizce olmasin, bilelim.
                        Log.i(ETIKET, "kamera bagli | ZSL destegi: " +
                            "${kamera.cameraInfo.isZslSupported}")
                    } catch (e: Exception) {
                        Log.e(ETIKET, "kamera baglanamadi", e)
                        onHata(CekimHatasi.KAMERA)
                    }
                }, ContextCompat.getMainExecutor(ctx))
                onizleme
            },
        )

        Column(
            modifier = Modifier.align(Alignment.BottomCenter).fillMaxWidth()
                .background(Color(0xCC000000)).padding(vertical = 16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            SeviyeGostergesi(durus)

            if (!durus.gecerli && durus.hazir) {
                Text(
                    text = "Telefon %.0f° öne/arkaya yatık — dikleştirin"
                        .format(durus.pitch),
                    color = Color(0xFFFF4646),
                    fontWeight = FontWeight.Bold,
                    fontSize = 14.sp,
                )
            } else if (cekiliyor) {
                Text(
                    text = if (SERI_KARE_SAYISI > 1)
                        "çekiliyor… $ilerleme/$SERI_KARE_SAYISI" else "çekiliyor…",
                    color = Color(0xFFFFCD00),
                    fontSize = 14.sp,
                )
            }

            Button(
                onClick = {
                    cekiliyor = true
                    ilerleme = 0
                    cekSeri(
                        imageCapture, yurutucu, anaYurutucu, takipci,
                        onIlerleme = { ilerleme = it },
                        onSeri = { cekiliyor = false; onCekim(it) },
                        onHata = { cekiliyor = false; onHata(it) },
                    )
                },
                // Çekim sürerken kilitli: ikinci basış ilkini iptal etmez,
                // sadece kuyruğa bir çekim daha ekler ve gecikmeyi büyütür.
                enabled = durus.gecerli && durus.hazir && !cekiliyor,
                modifier = Modifier.size(76.dp),
                shape = CircleShape,
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFFFFCD00),
                    disabledContainerColor = Color(0xFF555555),
                ),
            ) { Text("") }

            Text(
                text = "Tanı aracı değildir · tarama amaçlıdır",
                color = Color(0xFFBBBBBB),
                fontSize = 12.sp,
            )
        }
    }
}

/**
 * [SERI_KARE_SAYISI] kareyi arka arkaya çeker, sonra seriyi teslim eder.
 *
 * Kareler SIRAYLA çekiliyor (her biri bir öncekinin geri çağrısında
 * başlıyor), aynı anda değil. Aynı anda göndermek CameraX'te kuyruğa girip
 * kareler arası zamanlamayı öngörülemez yapıyor; ayrıca her karenin kendi IMU
 * eşleştirmesini yapabilmek için geri çağrıların ayrışması gerekiyor.
 *
 * TEK KARENİN BAŞARISIZLIĞI SERİYİ DÜŞÜRMEZ
 * -----------------------------------------
 * Bir karede IMU eşleşmezse o kare atlanır, seri devam eder. Üç karenin
 * hiçbiri eşleşmezse hata döner. Sağlam kareyi elde varken ölçümü iptal etmek
 * anlamsız olurdu.
 */
private fun cekSeri(
    imageCapture: ImageCapture,
    yurutucu: Executor,
    anaYurutucu: Executor,
    takipci: YercekimiTakipci,
    onIlerleme: (Int) -> Unit,
    onSeri: (CekimSerisi) -> Unit,
    onHata: (CekimHatasi) -> Unit,
) {
    val kareler = mutableListOf<Kare>()
    val seriBasi = SystemClock.elapsedRealtime()

    fun sonraki() {
        if (kareler.size >= SERI_KARE_SAYISI) {
            val sure = SystemClock.elapsedRealtime() - seriBasi
            Log.i(ETIKET, "seri tamam: ${kareler.size}/$SERI_KARE_SAYISI kare, " +
                "${kareler.count { it.gecerli }} gecerli, ${sure}ms")
            anaYurutucu.execute { onSeri(CekimSerisi(kareler.toList())) }
            return
        }
        cekTek(
            imageCapture, yurutucu, takipci,
            onKare = { kare ->
                kareler += kare
                anaYurutucu.execute { onIlerleme(kareler.size) }
                sonraki()
            },
            onKareBasarisiz = { hata ->
                // Bu kare gitti; kalanları denemeye devam et.
                val kalanDeneme = SERI_KARE_SAYISI - kareler.size
                if (kalanDeneme > 1) {
                    Log.w(ETIKET, "kare atlandi ($hata), seri suruyor")
                    sonraki()
                } else if (kareler.isNotEmpty()) {
                    Log.w(ETIKET, "kare atlandi ($hata), eldeki ${kareler.size} " +
                        "kareyle devam")
                    anaYurutucu.execute { onSeri(CekimSerisi(kareler.toList())) }
                } else {
                    anaYurutucu.execute { onHata(hata) }
                }
            },
        )
    }
    sonraki()
}

private fun cekTek(
    imageCapture: ImageCapture,
    yurutucu: Executor,
    takipci: YercekimiTakipci,
    onKare: (Kare) -> Unit,
    onKareBasarisiz: (CekimHatasi) -> Unit,
) {
    val basisAnda = SystemClock.elapsedRealtime()
    imageCapture.takePicture(
        yurutucu,           // arka plan: JPEG cozme ve dondurme burada
        object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureStarted() {
                Log.d(ETIKET, "3A bitti, cekim basladi: " +
                    "${SystemClock.elapsedRealtime() - basisAnda}ms")
            }

            override fun onCaptureSuccess(image: ImageProxy) {
                try {
                    // Donanım deklanşör zamanı. "Şu anki" yerçekimini okumak
                    // yerine bu damgayı kuşatan iki örnek arasında enterpole
                    // ediyoruz; aradaki gecikmede telefon birkaç derece
                    // dönebiliyor ve bu doğrudan ölçüm hatasına yazılır.
                    //
                    // Bu yüzden deklanşör gecikmesi DOĞRULUĞU bozmuyor —
                    // yalnızca kullanıcı deneyimi meselesi.
                    val damga = image.imageInfo.timestamp
                    val ornek = takipci.ornekEslestir(damga)
                    if (ornek == null) {
                        Log.w(ETIKET, "IMU eslesmedi, damga=$damga")
                        onKareBasarisiz(CekimHatasi.IMU_ESLESMEDI)
                        return
                    }
                    val geriCagriAnda = SystemClock.elapsedRealtime()
                    val bitmap = image.dogrultulmusBitmap()
                    val cozmeAnda = SystemClock.elapsedRealtime()

                    // Karenin KENDI donanim damgasi, elapsedRealtime tabaninda.
                    // Bununla toplam gecikmeyi ikiye boluyoruz:
                    //   basis -> yakalama : kameranin cekmeye baslamasi
                    //   yakalama -> geri cagri : JPEG kodlama + aktarma
                    // Ikisi bambaska sorunlar; ayirmadan cozum aranamaz.
                    val kareAni = damga / 1_000_000L
                    Log.i(ETIKET, "cekim ${bitmap.width}x${bitmap.height} " +
                        "roll=%.3f pitch=%.3f ".format(ornek.roll, ornek.pitch) +
                        (if (ornek.enterpole)
                            "enterpole (komsu aralik %.1fms)".format(ornek.komsuAralikMs)
                        else
                            "en yakin ornek (sapma %.1fms)".format(ornek.sapmaMs)) +
                        " | basis→yakalama ${kareAni - basisAnda}ms" +
                        " yakalama→teslim ${geriCagriAnda - kareAni}ms" +
                        " cozme ${cozmeAnda - geriCagriAnda}ms" +
                        " (toplam ${cozmeAnda - basisAnda}ms)")

                    onKare(Kare(bitmap, ornek))
                } finally {
                    image.close()
                }
            }

            override fun onError(exception: ImageCaptureException) {
                Log.e(ETIKET, "cekim hatasi", exception)
                onKareBasarisiz(CekimHatasi.KAMERA)
            }
        },
    )
}

/**
 * ImageProxy (JPEG) → dik duran Bitmap, çözerken küçültülmüş.
 *
 * NEDEN TAM ÇÖZÜNÜRLÜKTE ÇÖZMÜYORUZ
 * ---------------------------------
 * 3000×4000 JPEG'i tam çözmek + döndürmek gerçek cihazda 496 ms sürdü ve
 * 48 MB'lik iki ayrı tampon ayırdı. Bunun karşılığında ölçüme kazandırdığı
 * hiçbir şey yok: Aşama 1 görüntüyü 640×640'a geriyor, Aşama 2 de o uzayda
 * 128×128 kırpım alıyor. Yani 640'ın ötesindeki piksellerin tamamı atılıyor.
 *
 * `inSampleSize` JPEG'i çözerken küçültür — tam boyutlu tampon hiç
 * ayrılmaz.
 *
 * AÇI HESABI AÇISINDAN GÜVENLİ
 * ----------------------------
 * `inSampleSize` iki ekseni de aynı katsayıyla böler, yani EN-BOY ORANI
 * korunur. Açıyı bozan şey çözünürlük değil, en-boy oranının bozulmasıdır
 * (640×640 stretch'in yaptığı şey). Ölçüm bu bitmap'in kendi koordinat
 * uzayında yapılacağı için oranı ne ise açı da ona göre doğru çıkar.
 *
 * Döndürme yapılır ama AYNALAMA ASLA yapılmaz. Döndürme sonrası görüntünün
 * x ekseni cihazın x ekseniyle hizalı olur; roll düzeltmesi ancak bu hizalama
 * doğruysa anlamlıdır.
 *
 * @param enKisaKenarHedefi çözülen görüntünün kısa kenarı bunun altına
 *        düşmez. Model girdisi 640 olduğu için 700 yeterli pay bırakıyor.
 */
private fun ImageProxy.dogrultulmusBitmap(enKisaKenarHedefi: Int = 700): Bitmap {
    val t0 = SystemClock.elapsedRealtime()
    val tampon = planes[0].buffer
    val baytlar = ByteArray(tampon.remaining()).also { tampon.get(it) }
    val t1 = SystemClock.elapsedRealtime()

    // Once sadece boyutlari oku — bu adim bellek ayirmaz.
    val olcum = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    BitmapFactory.decodeByteArray(baytlar, 0, baytlar.size, olcum)

    var ornekleme = 1
    val kisaKenar = minOf(olcum.outWidth, olcum.outHeight)
    while (kisaKenar / (ornekleme * 2) >= enKisaKenarHedefi) ornekleme *= 2

    val secenek = BitmapFactory.Options().apply { inSampleSize = ornekleme }
    val ham = BitmapFactory.decodeByteArray(baytlar, 0, baytlar.size, secenek)
    val t2 = SystemClock.elapsedRealtime()

    val derece = imageInfo.rotationDegrees
    val sonuc = if (derece == 0) ham else {
        val m = Matrix().apply { postRotate(derece.toFloat()) }
        // filtreleme KAPALI: 90°'nin katlarında döndürme piksellerin birebir
        // yer değiştirmesidir, interpolasyona gerek yok. Açık bırakmak hem
        // yavaşlatır hem gereksiz yumuşatma yapar.
        Bitmap.createBitmap(ham, 0, 0, ham.width, ham.height, m, false)
            .also { if (it !== ham) ham.recycle() }
    }
    val t3 = SystemClock.elapsedRealtime()

    Log.d(ETIKET, "bitmap ${olcum.outWidth}x${olcum.outHeight}" +
        " -> ${sonuc.width}x${sonuc.height} (inSampleSize=$ornekleme)" +
        " | kopya ${t1 - t0}ms cozme ${t2 - t1}ms dondurme ${t3 - t2}ms")
    return sonuc
}

/** Seriyi teşhis amaçlı özetler — Faz 3'te yerini ölçüm sonucu alacak. */
fun CekimSerisi.ozet(): String = buildString {
    append("${kareler.size} kare, ${gecerliKareler.size} geçerli\n")
    kareler.forEachIndexed { i, k ->
        append("${i + 1}) ${k.foto.width}×${k.foto.height}  ")
        append("roll %+.2f°  pitch %+.2f°  ".format(k.yercekimi.roll, k.yercekimi.pitch))
        append(
            if (k.yercekimi.enterpole)
                "enterpole %.1fms".format(k.yercekimi.komsuAralikMs)
            else
                "sapma %.1fms".format(k.yercekimi.sapmaMs)
        )
        append(if (k.gecerli) "" else "  GEÇERSİZ")
        if (i < kareler.lastIndex) append("\n")
    }
}
