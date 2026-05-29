package com.turboquant.ai

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import android.util.Log
import com.turboquant.ai.service.DownloadForegroundService
import com.turboquant.ai.service.DownloadServiceState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Application entry point.
 *
 * Hosts two application-level singletons:
 *
 * 1. The download notification channel — must be created before any
 *    notification is posted (Android 8+).
 *
 * 2. [downloadFlow] — a [StateFlow] that carries [DownloadServiceState]
 *    emitted by [DownloadForegroundService].  The ViewModel (AndroidViewModel)
 *    collects from this flow so state is preserved across Activity recreations
 *    without requiring a service binding.
 */
class TurboQuantApp : Application() {

    // ── Shared download state (Service → ViewModel) ───────────────────────
    private val _downloadFlow = MutableStateFlow<DownloadServiceState>(DownloadServiceState.Idle)
    val downloadFlow: StateFlow<DownloadServiceState> = _downloadFlow.asStateFlow()

    /** Called by [DownloadForegroundService] to broadcast state changes. */
    fun emitDownloadState(state: DownloadServiceState) {
        _downloadFlow.value = state
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────

    override fun onCreate() {
        super.onCreate()
        createNotificationChannels()
        Log.i("TurboQuantApp", "Application started")
    }

    private fun createNotificationChannels() {
        val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        NotificationChannel(
            DownloadForegroundService.CHANNEL_ID,
            "Model Download",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Shows Gemma 2B model download progress"
            setShowBadge(false)
            enableVibration(false)
            setSound(null, null)
            nm.createNotificationChannel(this)
        }
        Log.i("TurboQuantApp", "Notification channels created")
    }
}
