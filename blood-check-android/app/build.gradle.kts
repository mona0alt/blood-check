plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.chaquo.python")
}

android {
    namespace = "com.bloodcheck.app"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.bloodcheck.app"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0.0"

        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

chaquopy {
    defaultConfig {
        if (System.getProperty("os.name").startsWith("Windows", ignoreCase = true)) {
            buildPython("py", "-3.10")
        } else {
            buildPython("python3.10")
        }
        version = "3.10"
        pip {
            // Keep versions explicit for reproducible Chaquopy builds.
            install("numpy==1.23.3")
            install("pandas==2.1.3")
            install("scipy==1.8.1")
            install("scikit-learn==1.3.2")
            install("PyWavelets==1.1.1")
            install("joblib==1.3.2")
            // Phase 0 validation on this machine confirms Chaquopy can't resolve xgboost.
            // The current model still needs a compatibility fallback (for example JSON/UBJ export)
            // before on-device inference can load successfully.
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.activity:activity-ktx:1.8.2")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.lifecycle:lifecycle-livedata-ktx:2.7.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.7.0")
    testImplementation("junit:junit:4.12")
}
