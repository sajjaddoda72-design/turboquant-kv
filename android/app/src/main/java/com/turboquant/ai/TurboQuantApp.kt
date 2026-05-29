package com.turboquant.ai

import android.app.Application
import android.util.Log

/**
 * Application class for TurboQuant AI.
 *
 * Currently a thin shell; can be extended with DI frameworks (Hilt/Koin)
 * or shared application-level resources as the project grows.
 */
class TurboQuantApp : Application() {

    override fun onCreate() {
        super.onCreate()
        Log.i("TurboQuantApp", "Application started")
    }
}
