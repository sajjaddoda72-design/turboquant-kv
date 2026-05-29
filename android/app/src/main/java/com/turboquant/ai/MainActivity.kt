package com.turboquant.ai

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
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
import com.turboquant.ai.ui.components.LoadingOverlay
import com.turboquant.ai.ui.screens.ChatScreen
import com.turboquant.ai.ui.screens.ModelHubScreen
import com.turboquant.ai.ui.theme.TurboQuantTheme
import com.turboquant.ai.viewmodel.AppScreen
import com.turboquant.ai.viewmodel.ChatViewModel

/**
 * Single-activity host.
 *
 * Screen routing is handled by [AppScreen] from the [ChatViewModel] —
 * no NavController needed for this three-screen flow.
 */
class MainActivity : ComponentActivity() {

    private val viewModel: ChatViewModel by viewModels {
        ChatViewModel.factory(applicationContext)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            TurboQuantTheme {
                val currentScreen by viewModel.currentScreen.collectAsState()

                // ── Hub + Loading overlay share the same z-stack ───────────
                Box(modifier = Modifier.fillMaxSize()) {

                    AnimatedContent(
                        targetState = currentScreen,
                        transitionSpec = { fadeIn() togetherWith fadeOut() },
                        label = "screen_transition"
                    ) { screen ->
                        when (screen) {
                            AppScreen.MODEL_HUB, AppScreen.LOADING ->
                                ModelHubScreen(viewModel = viewModel)

                            AppScreen.CHAT ->
                                ChatScreen(viewModel = viewModel)
                        }
                    }

                    // Loading overlay sits on top of the Hub during init
                    if (currentScreen == AppScreen.LOADING) {
                        LoadingOverlay()
                    }
                }
            }
        }
    }
}
