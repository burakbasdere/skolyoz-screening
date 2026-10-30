plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.skolyoz.olcum"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.skolyoz.olcum"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"),
                          "proguard-rules.pro")
            // TFLite dort ABI icin yerel kutuphane getiriyor ve APK'yi
            // 32 MB'a cikariyor. x86/x86_64 yalnizca emulator icin gerekli,
            // gercek telefonlar arm. Filtre SADECE release'te: debug'i
            // universal birakiyoruz ki emulator calismaya devam etsin.
            ndk { abiFilters += listOf("arm64-v8a", "armeabi-v7a") }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        compose = true
        // BuildConfig.DEBUG icin: teshis ciktisi yalnizca hata ayiklama
        // derlemesinde yaziliyor (hasta fotografi diske yaziyor).
        buildConfig = true
    }

    // TFLite modelleri sikistirilmamali; aksi halde bellek esleme (mmap) yerine
    // aciliyorlar, hem yavas hem RAM yiyor.
    androidResources {
        noCompress += listOf("tflite")
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.activity.compose)

    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)
    debugImplementation(libs.androidx.ui.tooling)

    implementation(libs.camera.core)
    implementation(libs.camera.camera2)
    implementation(libs.camera.lifecycle)
    implementation(libs.camera.view)

    implementation(libs.tensorflow.lite)

    testImplementation(libs.junit)

    // Cihaz ustu test: TFLite yerel kutuphanesi ve assets duz JVM'de yok.
    androidTestImplementation(libs.androidx.test.junit)
    androidTestImplementation(libs.androidx.test.runner)
}
