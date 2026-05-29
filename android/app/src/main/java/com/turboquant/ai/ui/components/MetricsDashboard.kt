package com.turboquant.ai.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Analytics
import androidx.compose.material.icons.rounded.Bolt
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.Memory
import androidx.compose.material.icons.rounded.Save
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.turboquant.ai.engine.InferenceMetrics
import com.turboquant.ai.ui.theme.ColorCrimson
import com.turboquant.ai.ui.theme.ColorCrimson20
import com.turboquant.ai.ui.theme.ColorCrimson80
import com.turboquant.ai.ui.theme.ColorOnSurface
import com.turboquant.ai.ui.theme.ColorOnSurfaceVariant
import com.turboquant.ai.ui.theme.ColorSurfaceContainerHigh
import com.turboquant.ai.ui.theme.ColorSurfaceContainerLow
import com.turboquant.ai.ui.theme.ColorSurfaceDim
import com.turboquant.ai.ui.theme.ColorWhite10
import com.turboquant.ai.ui.theme.ColorWhite5

/**
 * Full-screen metrics overlay: a semi-transparent backdrop + slide-up panel.
 *
 * Clicking the backdrop dismisses the overlay (same as pressing X).
 * The panel itself does NOT propagate click events to the backdrop.
 *
 * All metric values are rendered live from [InferenceMetrics]; "--" is shown
 * when tokenCount == 0 (generation has not started yet).
 */
@Composable
fun MetricsDashboard(
    metrics: InferenceMetrics,
    onDismiss: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
    ) {
        // ── Semi-transparent backdrop (click to dismiss) ──────────────────
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Black.copy(alpha = 0.68f))
                .clickable(onClick = onDismiss)
        )

        // ── Sliding panel ─────────────────────────────────────────────────
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .clip(RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp))
                .background(ColorSurfaceDim.copy(alpha = 0.97f))
                .border(
                    1.dp,
                    ColorWhite10,
                    RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp)
                )
                .crimsonGlow(glowRadius = 10.dp, alpha = 0.08f)
                .navigationBarsPadding()
                // Prevent backdrop click from firing when touching the panel
                .clickable(enabled = false, onClick = {})
        ) {
            // Drag handle
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 12.dp, bottom = 4.dp),
                contentAlignment = Alignment.Center
            ) {
                Box(
                    modifier = Modifier
                        .width(48.dp)
                        .height(4.dp)
                        .clip(RoundedCornerShape(2.dp))
                        .background(ColorWhite10)
                )
            }

            // Header
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 20.dp, vertical = 12.dp),
                verticalAlignment     = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text       = "Performance Metrics",
                    color      = ColorCrimson,
                    fontWeight = FontWeight.Bold,
                    fontSize   = 22.sp
                )

                // Crimson X close button
                Box(
                    modifier = Modifier
                        .size(32.dp)
                        .clip(CircleShape)
                        .background(ColorCrimson20)
                        .border(1.dp, ColorCrimson80, CircleShape)
                        .crimsonGlow(glowRadius = 8.dp, alpha = 0.4f)
                        .clickable(onClick = onDismiss),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector     = Icons.Rounded.Close,
                        contentDescription = "Close Metrics",
                        tint            = ColorCrimson,
                        modifier        = Modifier.size(18.dp)
                    )
                }
            }

            HorizontalDivider(color = ColorWhite5, thickness = 1.dp)

            // Scrollable metrics content
            Column(
                modifier = Modifier
                    .verticalScroll(rememberScrollState())
                    .padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {

                // ── Top row: Speed (wide) + RAM Saved + MSE ───────────────
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    // Generation Speed — big card
                    StatCard(
                        icon       = Icons.Rounded.Bolt,
                        title      = "Generation Speed",
                        largeValue = if (metrics.tokenCount > 0)
                            "%.1f".format(metrics.tokensPerSecond)
                        else "--",
                        unit       = "Token/sec",
                        modifier   = Modifier.weight(2f)
                    )
                    // RAM Saved
                    StatCard(
                        icon       = Icons.Rounded.Save,
                        title      = "RAM Saved",
                        largeValue = if (metrics.tokenCount > 0)
                            "%.1f".format(metrics.ramSavedPct)
                        else "--",
                        unit       = "%",
                        accentUnit = true,
                        modifier   = Modifier.weight(1f)
                    )
                    // MSE
                    StatCard(
                        icon       = Icons.Rounded.Analytics,
                        title      = "MSE",
                        largeValue = if (metrics.tokenCount > 0)
                            "%.4f".format(metrics.mse)
                        else "--",
                        unit       = "",
                        modifier   = Modifier.weight(1f)
                    )
                }

                // ── KV Cache Compression (full width) ─────────────────────
                KvCacheCard(metrics = metrics)

                Spacer(Modifier.height(4.dp))
            }
        }
    }
}

// ── Stat card component ───────────────────────────────────────────────────────

@Composable
private fun StatCard(
    icon: ImageVector,
    title: String,
    largeValue: String,
    unit: String,
    modifier: Modifier = Modifier,
    accentUnit: Boolean = false
) {
    Column(
        modifier = modifier
            .clip(RoundedCornerShape(16.dp))
            .background(ColorSurfaceContainerLow)
            .border(1.dp, ColorWhite5, RoundedCornerShape(16.dp))
            .padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(icon, null, tint = ColorOnSurfaceVariant, modifier = Modifier.size(16.dp))
            Spacer(Modifier.width(4.dp))
            Text(
                text          = title.uppercase(),
                color         = ColorOnSurfaceVariant,
                fontSize      = 9.sp,
                fontWeight    = FontWeight.SemiBold,
                letterSpacing = 0.4.sp,
                maxLines      = 1
            )
        }
        Row(
            verticalAlignment = Alignment.Baseline,
            horizontalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            Text(
                text       = largeValue,
                color      = ColorCrimson,
                fontSize   = 22.sp,
                fontWeight = FontWeight.Bold
            )
            if (unit.isNotEmpty()) {
                Text(
                    text     = unit,
                    color    = if (accentUnit) ColorCrimson else ColorOnSurfaceVariant,
                    fontSize = 12.sp
                )
            }
        }
    }
}

// ── KV Cache compression card ─────────────────────────────────────────────────

@Composable
private fun KvCacheCard(metrics: InferenceMetrics) {
    val isLive = metrics.tokenCount > 0

    // Ratio of TurboQuant to Original (for the visual bar fill)
    val compressionRatio = if (isLive && metrics.originalKvCacheMb > 0f)
        (metrics.turboQuantKvCacheMb / metrics.originalKvCacheMb).coerceIn(0f, 1f)
    else 0f

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(ColorSurfaceContainerLow)
            .border(1.dp, ColorWhite5, RoundedCornerShape(16.dp))
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text(
            text          = "KV Cache Compression (3.5-bit TurboQuant)",
            color         = ColorOnSurfaceVariant,
            fontSize      = 10.sp,
            fontWeight    = FontWeight.SemiBold,
            letterSpacing = 0.5.sp
        )

        // ── Visual bars ───────────────────────────────────────────────
        // Original bar (full width = reference)
        CacheBar(
            label     = "Original KV Cache (FP16)",
            valueText = if (isLive) "%.2f MB".format(metrics.originalKvCacheMb) else "-- MB",
            fraction  = 1f,
            color     = ColorSurfaceContainerHigh,
            labelColor = ColorOnSurfaceVariant
        )
        // TurboQuant bar (fraction of original)
        CacheBar(
            label     = "TurboQuant (3.5-bit)",
            valueText = if (isLive) "%.2f MB".format(metrics.turboQuantKvCacheMb) else "-- MB",
            fraction  = compressionRatio,
            color     = ColorCrimson,
            labelColor = ColorCrimson
        )

        HorizontalDivider(color = ColorWhite5, thickness = 0.5.dp)

        // ── Stat rows ─────────────────────────────────────────────────
        val fmtMb  : (Float) -> String = { if (isLive) "%.2f MB".format(it) else "-- MB"  }
        val fmtPct : (Float) -> String = { if (isLive) "%.1f %%".format(it) else "--%"   }
        val fmtTps : (Float) -> String = { if (isLive) "%.1f Token/s".format(it) else "-- Token/s" }
        val fmtMse : (Float) -> String = { if (isLive) "%.4f".format(it) else "--" }

        MetricRow("Mean Squared Error (MSE)",   fmtMse(metrics.mse))
        MetricRow("Original KV Cache Size",      fmtMb(metrics.originalKvCacheMb))
        MetricRow("TurboQuant Cache Size",        fmtMb(metrics.turboQuantKvCacheMb))
        MetricRow("RAM Saved",                    fmtPct(metrics.ramSavedPct))
        MetricRow("Generation Speed",             fmtTps(metrics.tokensPerSecond))
    }
}

@Composable
private fun CacheBar(
    label: String,
    valueText: String,
    fraction: Float,
    color: Color,
    labelColor: Color
) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(label,     color = labelColor,  fontSize = 11.sp, fontWeight = FontWeight.Medium)
            Text(valueText, color = ColorOnSurface, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
        }
        LinearProgressIndicator(
            progress   = { fraction },
            modifier   = Modifier
                .fillMaxWidth()
                .height(6.dp)
                .clip(RoundedCornerShape(3.dp)),
            color      = color,
            trackColor = ColorSurfaceContainerHigh
        )
    }
}
