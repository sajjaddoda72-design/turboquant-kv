package com.turboquant.ai.ui.components

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.turboquant.ai.ui.theme.ColorCrimson
import com.turboquant.ai.ui.theme.ColorCrimson20
import com.turboquant.ai.ui.theme.ColorCrimson80
import com.turboquant.ai.ui.theme.ColorOnSurface
import com.turboquant.ai.ui.theme.ColorSurfaceDim
import com.turboquant.ai.ui.theme.ColorWhite8

// ── Glass card container ──────────────────────────────────────────────────────

/**
 * A glassmorphic panel: semi-transparent charcoal background with
 * a subtle inner-edge white highlight and an optional crimson tint.
 *
 * On Android 12+ the background can be blurred with [Modifier.blur()].
 * On older APIs we approximate the glass look via transparency layering.
 */
@Composable
fun GlassCard(
    modifier: Modifier = Modifier,
    cornerRadius: Dp = 16.dp,
    content: @Composable BoxScope.() -> Unit
) {
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(cornerRadius))
            .background(ColorSurfaceDim.copy(alpha = 0.90f))
            .border(
                width = 1.dp,
                color = ColorWhite8,
                shape = RoundedCornerShape(cornerRadius)
            )
    ) {
        content()
    }
}

// ── Crimson glow extension ────────────────────────────────────────────────────

/**
 * Draws a soft crimson halo around a composable using three concentric
 * semi-transparent [drawRoundRect] layers.
 *
 * Works on all API levels (no [android.graphics.BlurMaskFilter] required).
 * The outermost layer is most transparent; the innermost is most opaque,
 * approximating a Gaussian glow falloff.
 *
 * @param glowRadius  maximum spread of the outermost glow layer
 * @param alpha       peak alpha of the innermost glow layer (0f–1f)
 */
fun Modifier.crimsonGlow(
    glowRadius: Dp = 20.dp,
    alpha: Float = 0.35f
): Modifier = this.drawBehind {
    val cornerPx  = 16.dp.toPx()
    val glow      = Color(0xFFD32F2F)

    // Three concentric rings with decreasing spread and increasing alpha
    listOf(
        glowRadius.toPx()        to alpha * 0.25f,
        glowRadius.toPx() * 0.6f to alpha * 0.55f,
        glowRadius.toPx() * 0.3f to alpha * 0.85f
    ).forEach { (spread, layerAlpha) ->
        drawRoundRect(
            color       = glow.copy(alpha = layerAlpha),
            topLeft     = Offset(-spread, -spread),
            size        = Size(size.width + spread * 2f, size.height + spread * 2f),
            cornerRadius = CornerRadius(cornerPx + spread)
        )
    }
}

// ── Crimson primary button ────────────────────────────────────────────────────

/**
 * Glassmorphic Crimson Red action button.
 *
 * Matches the `.crimson-glow-btn` style from the HTML prototype:
 *   • Semi-transparent crimson fill
 *   • 80 % opaque crimson border
 *   • Soft crimson shadow/glow behind the button
 */
@Composable
fun CrimsonGlowButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    icon: ImageVector? = null,
    enabled: Boolean = true
) {
    Button(
        onClick  = onClick,
        enabled  = enabled,
        modifier = modifier.crimsonGlow(glowRadius = 18.dp, alpha = 0.35f),
        shape    = RoundedCornerShape(12.dp),
        colors   = ButtonDefaults.buttonColors(
            containerColor    = ColorCrimson20,
            contentColor      = Color.White,
            disabledContainerColor = ColorCrimson20.copy(alpha = 0.4f),
            disabledContentColor   = Color.White.copy(alpha = 0.5f)
        ),
        border        = BorderStroke(1.dp, if (enabled) ColorCrimson80 else ColorCrimson80.copy(alpha = 0.4f)),
        contentPadding = PaddingValues(horizontal = 24.dp, vertical = 14.dp)
    ) {
        if (icon != null) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                modifier    = Modifier
                    .size(22.dp)
                    .padding(end = 8.dp),
                tint        = Color.White
            )
        }
        Text(
            text       = text,
            fontWeight = FontWeight.SemiBold,
            fontSize   = 15.sp,
            letterSpacing = 0.3.sp
        )
    }
}

// ── KV Cache status badge ─────────────────────────────────────────────────────

/**
 * The glowing "● 3-bit KV Cache Compression Active" badge shown at the
 * top of the chat screen.
 */
@Composable
fun KvCacheBadge(modifier: Modifier = Modifier) {
    Row(
        modifier = modifier
            .clip(RoundedCornerShape(50.dp))
            .background(ColorSurfaceDim.copy(alpha = 0.8f))
            .border(1.dp, ColorCrimson.copy(alpha = 0.25f), RoundedCornerShape(50.dp))
            .crimsonGlow(glowRadius = 12.dp, alpha = 0.25f)
            .padding(horizontal = 16.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Animated pulsing dot
        Box(
            modifier = Modifier
                .size(8.dp)
                .clip(RoundedCornerShape(50.dp))
                .background(ColorCrimson)
        )
        Text(
            text          = "  3-bit KV Cache Compression Active",
            color         = ColorCrimson,
            fontSize      = 12.sp,
            fontWeight    = FontWeight.SemiBold,
            letterSpacing = 0.6.sp
        )
    }
}

// ── Metric stat row ───────────────────────────────────────────────────────────

/**
 * A single row in the metrics dashboard: "Label .............. value"
 */
@Composable
fun MetricRow(
    label: String,
    value: String,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text     = label,
            color    = ColorOnSurface.copy(alpha = 0.6f),
            fontSize = 13.sp,
            modifier = Modifier.weight(1f)
        )
        Text(
            text       = value,
            color      = ColorOnSurface,
            fontSize   = 13.sp,
            fontWeight = FontWeight.Medium
        )
    }
}
