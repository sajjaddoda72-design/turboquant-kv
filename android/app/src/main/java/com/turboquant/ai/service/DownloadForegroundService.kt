package com.turboquant.ai.service

import android.app.Notification
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Binder
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import com.turboquant.ai.MainActivity
import com.turboquant.ai.engine.DownloadState
import com.turboquant.ai.engine.ModelDownloader
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.File

private const val TAG = "DownloadForegroundService"

/** States exposed to the bound ViewModel via [DownloadForegroundService.state]. */
sealed class DownloadServiceState {
    object Idle : DownloadServiceState()
    data class Downloading(
        val fraction: Float,
        val downloadedMb: Float,
        val totalMb: Float
    ) : DownloadServiceState()
    data class Complete(val file: File) : DownloadServiceState()
    data class Failed(val message: String) : DownloadServiceState()
}

/**
 * Foreground service that runs the model download independently of the
 * Activity lifecycle.  The user can minimise or close the app and the
 * download continues; a persistent notification shows live progress.
 *
 * Lifecycle:
 *  1. Started via [startForegroundService] from [ChatViewModel.downloadModel].
 *  2. Bound via [bindService] so the ViewModel can collect [state] directly.
 *  3. Calls [stopSelf] when the download finishes or fails.
 *
 * Notification behaviour:
 *  - Ongoing progress notification while downloading.
 *  - "Download complete" tap-to-open notification on success.
 *  - "Download failed" notification on error.
 */
class DownloadForegroundService : Service() {

    companion object {
        const val CHANNEL_ID          = "tq_model_download"
        const val NOTIF_ID_PROGRESS   = 1001
        const val NOTIF_ID_COMPLETE   = 1002

        /** Convenience factory for the start intent. */
        fun intent(context: Context): Intent =
            Intent(context, DownloadForegroundService::class.java)
    }

    // ── Binder — gives bound clients direct access to this service ────────
    inner class LocalBinder : Binder() {
        fun getService(): DownloadForegroundService = this@DownloadForegroundService
    }
    private val binder = LocalBinder()

    // ── Live download state for bound clients (ViewModel) ─────────────────
    private val _state = MutableStateFlow<DownloadServiceState>(DownloadServiceState.Idle)
    val state: StateFlow<DownloadServiceState> = _state.asStateFlow()

    // ── Internal ──────────────────────────────────────────────────────────
    private val scope          = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private lateinit var downloader: ModelDownloader
    private lateinit var notifManager: NotificationManager

    // ── Service lifecycle ─────────────────────────────────────────────────

    override fun onCreate() {
        super.onCreate()
        downloader   = ModelDownloader(applicationContext)
        notifManager = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        Log.i(TAG, "Created")
    }

    /**
     * Called when the ViewModel does startForegroundService().
     * We MUST call startForeground() here within the 5-second ANR window.
     */
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.i(TAG, "onStartCommand — launching foreground download")
        // Show notification immediately (Android 12+ enforces < 5s window)
        startForeground(NOTIF_ID_PROGRESS, buildProgressNotif(0f, 0f, 0f))
        launchDownload()
        return START_NOT_STICKY   // don't restart after process kill
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onDestroy() {
        scope.cancel()
        Log.i(TAG, "Destroyed")
        super.onDestroy()
    }

    // ── Download coroutine ─────────────────────────────────────────────────

    private fun launchDownload() {
        scope.launch {
            downloader.download().collect { result ->
                when (result) {
                    is DownloadState.Progress -> {
                        val dlMb    = result.downloadedBytes.toFloat() / (1024f * 1024f)
                        val totalMb = if (result.totalBytes > 0L)
                            result.totalBytes.toFloat() / (1024f * 1024f) else 0f

                        val serviceState = DownloadServiceState.Downloading(
                            fraction     = result.fraction,
                            downloadedMb = dlMb,
                            totalMb      = totalMb
                        )
                        // Broadcast to ViewModel via Application-level StateFlow
                        _state.value = serviceState
                        (application as? com.turboquant.ai.TurboQuantApp)
                            ?.emitDownloadState(serviceState)

                        notifManager.notify(NOTIF_ID_PROGRESS,
                            buildProgressNotif(result.fraction, dlMb, totalMb))
                    }

                    is DownloadState.Complete -> {
                        Log.i(TAG, "Download complete: ${result.file.path}")
                        val serviceState = DownloadServiceState.Complete(result.file)
                        _state.value = serviceState
                        (application as? com.turboquant.ai.TurboQuantApp)
                            ?.emitDownloadState(serviceState)

                        stopForeground(STOP_FOREGROUND_REMOVE)
                        notifManager.notify(NOTIF_ID_COMPLETE, buildCompleteNotif())
                        stopSelf()
                    }

                    is DownloadState.Error -> {
                        Log.e(TAG, "Download failed: ${result.message}")
                        val serviceState = DownloadServiceState.Failed(result.message)
                        _state.value = serviceState
                        (application as? com.turboquant.ai.TurboQuantApp)
                            ?.emitDownloadState(serviceState)

                        stopForeground(STOP_FOREGROUND_REMOVE)
                        notifManager.notify(NOTIF_ID_COMPLETE,
                            buildErrorNotif(result.message))
                        stopSelf()
                    }
                }
            }
        }
    }

    // ── Notification builders ──────────────────────────────────────────────

    private fun tapPendingIntent(): PendingIntent {
        val intent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        return PendingIntent.getActivity(
            this, 0, intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
    }

    private fun buildProgressNotif(fraction: Float, dlMb: Float, totalMb: Float): Notification {
        val pct       = (fraction * 100).toInt()
        val sizeLabel = when {
            totalMb > 0f -> "%.0f MB / %.0f MB".format(dlMb, totalMb)
            dlMb  > 0f   -> "%.0f MB downloaded".format(dlMb)
            else          -> "Connecting…"
        }
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle("Downloading Gemma 2B")
            .setContentText("$pct%  ·  $sizeLabel")
            .setProgress(100, pct, pct == 0)
            .setOngoing(true)           // can't be swiped away while downloading
            .setOnlyAlertOnce(true)     // don't re-ping on every progress update
            .setContentIntent(tapPendingIntent())
            .build()
    }

    private fun buildCompleteNotif(): Notification =
        NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_download_done)
            .setContentTitle("Gemma 2B ready")
            .setContentText("Tap to open TurboQuant AI and run the model")
            .setAutoCancel(true)
            .setContentIntent(tapPendingIntent())
            .build()

    private fun buildErrorNotif(message: String): Notification =
        NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_notify_error)
            .setContentTitle("Download failed")
            .setContentText(message.take(100))
            .setAutoCancel(true)
            .setContentIntent(tapPendingIntent())
            .build()
}
