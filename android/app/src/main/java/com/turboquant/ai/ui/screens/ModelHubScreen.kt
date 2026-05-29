package com.turboquant.ai.ui.screens

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.ArrowDownward
import androidx.compose.material.icons.rounded.Bolt
import androidx.compose.material.icons.rounded.Compress
import androidx.compose.material.icons.rounded.DataObject
import androidx.compose.material.icons.rounded.Memory
import androidx.compose.material.icons.rounded.PlayArrow
import androidx.compose.material.icons.rounded.Refresh
import androidx.compose.material.icons.rounded.SdStorage
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.turboquant.ai.ui.components.CrimsonGlowButton
import com.turboquant.ai.ui.components.GlassCard
import com.turboquant.ai.ui.components.crimsonGlow
import com.turboquant.ai.ui.theme.ColorBackground
import com.turboquant.ai.ui.theme.ColorCrimson
import com.turboquant.ai.ui.theme.ColorCrimson20
import com.turboquant.ai.ui.theme.ColorOnSurface
import com.turboquant.ai.ui.theme.ColorOnSurfaceVariant
import com.turboquant.ai.ui.theme.ColorPrimary
import com.turboquant.ai.ui.theme.ColorSurfaceContainer
import com.turboquant.ai.ui.theme.ColorSurfaceContainerHigh
import com.turboquant.ai.ui.theme.ColorSurfaceContainerLow
import com.turboquant.ai.ui.theme.ColorWhite10
import com.turboquant.ai.ui.theme.ColorWhite5
import com.turboquant.ai.viewmodel.ChatViewModel
import com.turboquant.ai.viewmodel.ModelState

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelHubScreen(viewModel: ChatViewModel) {

    val modelState by viewModel.modelState.collectAsState()

    Scaffold(
        containerColor = ColorBackground,
        topBar = {
            TopAppBar(
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Rounded.Memory,
                            contentDescription = "App Icon",
                            tint     = ColorCrimson,
                            modifier = Modifier.size(26.dp)
                        )
                        Spacer(Modifier.width(10.dp))
                        Text(
                            text       = "TurboQuant AI",
                            color      = ColorCrimson,
                            fontWeight = FontWeight.Bold,
                            fontSize   = 22.sp
                        )
                    }
                },
                actions = {
                    // Avatar placeholder
                    Box(
                        modifier = Modifier
                            .padding(end = 16.dp)
                            .size(32.dp)
                            .clip(CircleShape)
                            .background(ColorSurfaceContainerHigh)
                            .border(1.dp, ColorWhite10, CircleShape),
                        contentAlignment = Alignment.Center
                    ) {
                        Text("U", color = Color.White, fontSize = 13.sp, fontWeight = FontWeight.Bold)
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = ColorBackground.copy(alpha = 0.90f)
                )
            )
        }
    ) { paddingValues ->

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(horizontal = 24.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Spacer(Modifier.height(8.dp))

            // ── Gemma 2B model card ───────────────────────────────────────
            GlassCard(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(24.dp)) {

                    // "Optimized" badge
                    Row(
                        modifier = Modifier
                            .clip(RoundedCornerShape(50.dp))
                            .background(ColorSurfaceContainerHigh)
                            .padding(horizontal = 12.dp, vertical = 5.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            imageVector = Icons.Rounded.Bolt,
                            contentDescription = null,
                            tint     = ColorCrimson,
                            modifier = Modifier.size(16.dp)
                        )
                        Spacer(Modifier.width(5.dp))
                        Text(
                            text       = "Optimized",
                            color      = ColorCrimson,
                            fontSize   = 11.sp,
                            fontWeight = FontWeight.SemiBold,
                            letterSpacing = 0.6.sp
                        )
                    }

                    Spacer(Modifier.height(16.dp))

                    // Model name
                    Text(
                        text       = "Gemma 2B",
                        style      = MaterialTheme.typography.displayLarge,
                        color      = ColorOnSurface
                    )
                    Spacer(Modifier.height(6.dp))
                    Text(
                        text  = "Lightweight, state-of-the-art open model optimized for " +
                                "on-device inference with TurboQuant 3-bit KV cache compression.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = ColorOnSurfaceVariant
                    )

                    Spacer(Modifier.height(20.dp))

                    // ── Model details grid ────────────────────────────────
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        ModelDetailCard(
                            icon  = Icons.Rounded.SdStorage,
                            label = "Model Size",
                            value = "1.5 GB",
                            modifier = Modifier.weight(1f)
                        )
                        ModelDetailCard(
                            icon  = Icons.Rounded.DataObject,
                            label = "Context",
                            value = "2048",
                            modifier = Modifier.weight(1f)
                        )
                        ModelDetailCard(
                            icon  = Icons.Rounded.Compress,
                            label = "Quantization",
                            value = "Q4_K_M",
                            isAccent = true,
                            modifier = Modifier.weight(1f)
                        )
                    }

                    Spacer(Modifier.height(20.dp))

                    // ── Download progress (visible while downloading) ─────
                    AnimatedVisibility(
                        visible = modelState is ModelState.Downloading,
                        enter   = fadeIn(),
                        exit    = fadeOut()
                    ) {
                        val state = modelState
                        if (state is ModelState.Downloading) {
                            Column(modifier = Modifier.fillMaxWidth()) {
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween
                                ) {
                                    Text(
                                        text  = "Downloading Gemma 2B...",
                                        color = ColorOnSurfaceVariant,
                                        fontSize = 12.sp,
                                        fontWeight = FontWeight.Medium
                                    )
                                    Text(
                                        text  = if (state.totalMb > 0f)
                                            "${formatMb(state.downloadedMb)} / ${formatMb(state.totalMb)}"
                                        else "${(state.progress * 100).toInt()}%",
                                        color = ColorCrimson,
                                        fontSize = 12.sp,
                                        fontWeight = FontWeight.SemiBold
                                    )
                                }
                                Spacer(Modifier.height(8.dp))
                                LinearProgressIndicator(
                                    progress       = { state.progress },
                                    modifier       = Modifier
                                        .fillMaxWidth()
                                        .height(4.dp)
                                        .clip(RoundedCornerShape(2.dp)),
                                    color          = ColorCrimson,
                                    trackColor     = ColorSurfaceContainerHigh
                                )
                                Spacer(Modifier.height(16.dp))
                            }
                        }
                    }

                    // ── Error message ─────────────────────────────────────
                    AnimatedVisibility(visible = modelState is ModelState.Error) {
                        val state = modelState
                        if (state is ModelState.Error) {
                            Text(
                                text     = "⚠ ${state.message}",
                                color    = MaterialTheme.colorScheme.error,
                                fontSize = 13.sp,
                                modifier = Modifier.padding(bottom = 12.dp)
                            )
                        }
                    }

                    // ── Action button ─────────────────────────────────────
                    ActionButton(modelState = modelState, viewModel = viewModel)
                }
            }

            // ── Footer metadata ───────────────────────────────────────────
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 4.dp, vertical = 4.dp),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text  = "Requires 2 GB Unified Memory",
                    color = ColorOnSurface.copy(alpha = 0.4f),
                    fontSize = 11.sp
                )
                Text(
                    text  = "v1.2.0",
                    color = ColorOnSurface.copy(alpha = 0.4f),
                    fontSize = 11.sp
                )
            }

            Spacer(Modifier.height(24.dp))
        }
    }
}

// ── Action button (adapts to current model state) ────────────────────────────

@Composable
private fun ActionButton(modelState: ModelState, viewModel: ChatViewModel) {
    when (modelState) {
        is ModelState.NotDownloaded ->
            CrimsonGlowButton(
                text    = "Download Model",
                icon    = Icons.Rounded.ArrowDownward,
                onClick = { viewModel.downloadModel() },
                modifier = Modifier.fillMaxWidth()
            )

        is ModelState.Downloading ->
            CrimsonGlowButton(
                text    = "${(modelState.progress * 100).toInt()}%  ·  Downloading",
                icon    = Icons.Rounded.ArrowDownward,
                onClick = {},
                enabled  = false,
                modifier = Modifier.fillMaxWidth()
            )

        is ModelState.Downloaded ->
            CrimsonGlowButton(
                text    = "Run Model",
                icon    = Icons.Rounded.PlayArrow,
                onClick = { viewModel.runModel() },
                modifier = Modifier
                    .fillMaxWidth()
                    .crimsonGlow(glowRadius = 24.dp, alpha = 0.50f)
            )

        is ModelState.Initializing ->
            CrimsonGlowButton(
                text    = "Loading Model...",
                icon    = Icons.Rounded.Memory,
                onClick = {},
                enabled  = false,
                modifier = Modifier.fillMaxWidth()
            )

        is ModelState.Ready ->
            CrimsonGlowButton(
                text    = "Run Model",
                icon    = Icons.Rounded.PlayArrow,
                onClick = { viewModel.runModel() },
                modifier = Modifier
                    .fillMaxWidth()
                    .crimsonGlow(glowRadius = 24.dp, alpha = 0.50f)
            )

        is ModelState.Error ->
            CrimsonGlowButton(
                text    = "Retry Download",
                icon    = Icons.Rounded.Refresh,
                onClick = { viewModel.downloadModel() },
                modifier = Modifier.fillMaxWidth()
            )
    }
}

// ── Model detail stat card ─────────────────────────────────────────────────────

@Composable
private fun ModelDetailCard(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    label: String,
    value: String,
    modifier: Modifier = Modifier,
    isAccent: Boolean = false
) {
    Column(
        modifier = modifier
            .clip(RoundedCornerShape(10.dp))
            .background(ColorSurfaceContainerLow)
            .border(1.dp, ColorWhite5, RoundedCornerShape(10.dp))
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint     = ColorOnSurfaceVariant,
                modifier = Modifier.size(18.dp)
            )
            Spacer(Modifier.width(5.dp))
            Text(
                text       = label.uppercase(),
                color      = ColorOnSurfaceVariant,
                fontSize   = 9.sp,
                fontWeight = FontWeight.SemiBold,
                letterSpacing = 0.5.sp,
                maxLines   = 1
            )
        }
        Text(
            text       = value,
            color      = if (isAccent) ColorCrimson else ColorOnSurface,
            fontSize   = 16.sp,
            fontWeight = FontWeight.SemiBold
        )
    }
}

// ── Format megabytes (e.g. "1 234.5 MB" or "1.2 GB") ──────────────────────────

private fun formatMb(mb: Float): String {
    return if (mb >= 1024f) "%.1f GB".format(mb / 1024f)
    else "%.0f MB".format(mb)
}
