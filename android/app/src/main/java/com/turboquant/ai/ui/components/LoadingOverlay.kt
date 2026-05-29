package com.turboquant.ai.ui.components

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
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
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Memory
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.turboquant.ai.ui.theme.ColorBackground
import com.turboquant.ai.ui.theme.ColorCrimson
import com.turboquant.ai.ui.theme.ColorOnSurface
import com.turboquant.ai.ui.theme.ColorOnSurfaceVariant
import com.turboquant.ai.ui.theme.ColorSurfaceContainerHigh
import com.turboquant.ai.ui.theme.ColorSurfaceDim
import com.turboquant.ai.ui.theme.ColorWhite5

/**
 * Full-screen loading overlay displayed while the model is being initialised.
 *
 * Visually matches the HTML prototype's "loading_model" screen:
 *  • Blurred-dark backdrop (approximated via 85 % opaque background)
 *  • Three-layer animated spinner (slow track, fast arc, pulse ring, centre icon)
 *  • "Initializing…" heading with bouncing dots
 *  • TurboQuant SIMD engine message
 *  • Indeterminate progress bar
 */
@Composable
fun LoadingOverlay() {

    val transition = rememberInfiniteTransition(label = "loading")

    // Fast arc rotation
    val arcRotation by transition.animateFloat(
        initialValue  = 0f,
        targetValue   = 360f,
        animationSpec = infiniteRepeatable(tween(1500, easing = LinearEasing)),
        label         = "arc"
    )

    // Slow track rotation (reverse)
    val trackRotation by transition.animateFloat(
        initialValue  = 360f,
        targetValue   = 0f,
        animationSpec = infiniteRepeatable(tween(4000, easing = LinearEasing)),
        label         = "track"
    )

    // Outer ring pulse scale
    val pulseScale by transition.animateFloat(
        initialValue  = 0.80f,
        targetValue   = 1.05f,
        animationSpec = infiniteRepeatable(
            tween(1000),
            repeatMode = RepeatMode.Reverse
        ),
        label = "pulse"
    )

    // Bouncing dot offsets
    val dot0 by transition.animateFloat(
        initialValue  = 0f, targetValue = -7f,
        animationSpec = infiniteRepeatable(tween(400, delayMillis = 0), RepeatMode.Reverse),
        label = "d0"
    )
    val dot1 by transition.animateFloat(
        initialValue  = 0f, targetValue = -7f,
        animationSpec = infiniteRepeatable(tween(400, delayMillis = 140), RepeatMode.Reverse),
        label = "d1"
    )
    val dot2 by transition.animateFloat(
        initialValue  = 0f, targetValue = -7f,
        animationSpec = infiniteRepeatable(tween(400, delayMillis = 280), RepeatMode.Reverse),
        label = "d2"
    )
    val dotOffsets = listOf(dot0, dot1, dot2)

    // Full-screen dark overlay (approximates backdrop-blur)
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(ColorBackground.copy(alpha = 0.90f)),
        contentAlignment = Alignment.Center
    ) {
        Column(
            modifier           = Modifier
                .fillMaxWidth(0.88f)
                .padding(horizontal = 8.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(28.dp)
        ) {

            // ── Layered spinner ──────────────────────────────────────────
            Box(
                modifier        = Modifier.size(136.dp),
                contentAlignment = Alignment.Center
            ) {
                // Outer pulsing ring
                androidx.compose.foundation.Canvas(
                    modifier = Modifier
                        .size(136.dp * pulseScale)
                        .align(Alignment.Center)
                ) {
                    drawCircle(
                        color = ColorCrimson.copy(alpha = 0.20f),
                        style = Stroke(width = 2.dp.toPx())
                    )
                }

                // Slow dashed track
                androidx.compose.foundation.Canvas(
                    modifier = Modifier
                        .size(128.dp)
                        .rotate(trackRotation)
                ) {
                    val intervals = floatArrayOf(8.dp.toPx(), 16.dp.toPx())
                    drawCircle(
                        color  = ColorSurfaceContainerHigh,
                        radius = size.minDimension / 2f - 2.dp.toPx(),
                        style  = Stroke(
                            width        = 2.dp.toPx(),
                            pathEffect   = androidx.compose.ui.graphics.PathEffect
                                .dashPathEffect(intervals, 0f)
                        )
                    )
                }

                // Fast gradient arc
                androidx.compose.foundation.Canvas(
                    modifier = Modifier
                        .size(120.dp)
                        .rotate(arcRotation)
                ) {
                    drawArc(
                        brush      = Brush.sweepGradient(
                            0.0f to Color.Transparent,
                            0.6f to ColorCrimson.copy(alpha = 0.6f),
                            1.0f to ColorCrimson
                        ),
                        startAngle = 0f,
                        sweepAngle = 270f,
                        useCenter  = false,
                        style      = Stroke(width = 4.dp.toPx(), cap = StrokeCap.Round)
                    )
                }

                // Centre chip icon
                Box(
                    modifier = Modifier
                        .size(64.dp)
                        .clip(CircleShape)
                        .background(ColorSurfaceDim.copy(alpha = 0.6f))
                        .border(1.dp, ColorWhite5, CircleShape),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector     = Icons.Rounded.Memory,
                        contentDescription = null,
                        tint            = ColorCrimson,
                        modifier        = Modifier.size(32.dp)
                    )
                }
            }

            // ── Text card ────────────────────────────────────────────────
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(16.dp))
                    .background(ColorSurfaceDim.copy(alpha = 0.88f))
                    .border(1.dp, ColorWhite5, RoundedCornerShape(16.dp))
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                // "Initializing" + bouncing dots
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text       = "Initializing",
                        style      = androidx.compose.material3.MaterialTheme.typography.headlineMedium,
                        color      = ColorOnSurface,
                        fontWeight = FontWeight.SemiBold
                    )
                    Spacer(Modifier.width(8.dp))
                    Row(
                        horizontalArrangement = Arrangement.spacedBy(4.dp),
                        verticalAlignment     = Alignment.CenterVertically
                    ) {
                        dotOffsets.forEach { offset ->
                            Box(
                                modifier = Modifier
                                    .size(6.dp)
                                    .offset(y = offset.dp)
                                    .clip(CircleShape)
                                    .background(ColorCrimson)
                            )
                        }
                    }
                }

                Text(
                    text  = "Optimizing & Loading Model with TurboQuant SIMD Engine… Please wait.",
                    style = androidx.compose.material3.MaterialTheme.typography.bodyMedium,
                    color = ColorOnSurfaceVariant,
                    textAlign = androidx.compose.ui.text.style.TextAlign.Center
                )

                // Indeterminate progress bar
                LinearProgressIndicator(
                    modifier   = Modifier
                        .fillMaxWidth()
                        .height(3.dp)
                        .clip(RoundedCornerShape(2.dp)),
                    color      = ColorCrimson,
                    trackColor = ColorSurfaceContainerHigh
                )
            }
        }
    }
}
