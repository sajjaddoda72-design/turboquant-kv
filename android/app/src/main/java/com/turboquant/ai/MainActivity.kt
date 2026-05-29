package com.turboquant.ai

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import com.turboquant.ai.ui.components.LoadingOverlay
import com.turboquant.ai.ui.screens.ChatScreen
import com.turboquant.ai.ui.screens.ModelHubScreen
import com.turboquant.ai.ui.theme.TurboQuantTheme
import com.turboquant.ai.viewmodel.AppScreen
import com.turboquant.ai.viewmodel.ChatViewModel

private const val TAG = "MainActivity"

/**
 * Single-activity host for TurboQuant AI.
 *
 * Responsibilities:
 *  1. Request POST_NOTIFICATIONS permission on first launch (Android 13+)
 *     so the [DownloadForegroundService] can show download-progress notifications.
 *  2. Render the three-screen Compose UI driven by [ChatViewModel].
 *
 * Download state flows back from [DownloadForegroundService] via
 * [TurboQuantApp.downloadFlow], so no service binding is required here.
 */
class MainActivity : ComponentActivity() {

    private val viewModel: ChatViewModel by viewModels {
        ChatViewModel.factory(applicationContext)
    }

    // ── Notification permission (Android 13+ / API 33+) ───────────────────

    private val requestNotifPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            Log.i(TAG, "POST_NOTIFICATIONS granted")
        } else {
            Log.w(TAG, "POST_NOTIFICATIONS denied — download will continue in background " +
                    "but progress notifications will not appear")
        }
    }

    private fun requestNotificationPermissionIfNeeded() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            val alreadyGranted = ContextCompat.checkSelfPermission(
                this, Manifest.permission.POST_NOTIFICATIONS
            ) == PackageManager.PERMISSION_GRANTED

            if (!alreadyGranted) {
                Log.i(TAG, "Requesting POST_NOTIFICATIONS")
                requestNotifPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
            }
        }
        // Below API 33: notifications are granted by default, nothing to request.
    }

    // ── Activity lifecycle ─────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // Show the notification permission dialog immediately on first launch.
        // The OS remembers the user's decision and won't show it again.
        requestNotificationPermissionIfNeeded()

        setContent {
            TurboQuantTheme {
                val currentScreen by viewModel.currentScreen.collectAsState()

                Box(modifier = Modifier.fillMaxSize()) {

                    AnimatedContent(
                        targetState  = currentScreen,
                        transitionSpec = { fadeIn() togetherWith fadeOut() },
                        label        = "screen_transition"
                    ) { screen ->
                        when (screen) {
                            AppScreen.MODEL_HUB, AppScreen.LOADING ->
                                ModelHubScreen(viewModel = viewModel)
                            AppScreen.CHAT ->
                                ChatScreen(viewModel = viewModel)
                        }
                    }

                    // Loading overlay sits on top of the Hub while the model initialises
                    if (currentScreen == AppScreen.LOADING) {
                        LoadingOverlay()
                    }
                }
            }
        }
    }
}
