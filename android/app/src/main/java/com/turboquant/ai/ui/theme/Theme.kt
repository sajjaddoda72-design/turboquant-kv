package com.turboquant.ai.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

// TurboQuant AI always uses a dark Material3 scheme.
// Light mode is intentionally not supported (OLED dark only).
private val DarkColorScheme = darkColorScheme(
    primary               = ColorCrimson,
    onPrimary             = Color.White,
    primaryContainer      = ColorCrimson20,
    onPrimaryContainer    = Color.White,

    secondary             = ColorPrimary,          // salmon — for tint accents
    onSecondary           = Color.Black,
    secondaryContainer    = ColorSurfaceContainerHigh,
    onSecondaryContainer  = ColorSecondary,

    tertiary              = ColorTertiary,
    onTertiary            = Color.Black,

    background            = ColorBackground,
    onBackground          = ColorOnSurface,
    surface               = ColorSurfaceDim,
    onSurface             = ColorOnSurface,
    surfaceVariant        = ColorSurfaceContainerHigh,
    onSurfaceVariant      = ColorOnSurfaceVariant,
    surfaceContainer      = ColorSurfaceContainer,
    surfaceContainerLow   = ColorSurfaceContainerLow,
    surfaceContainerHigh  = ColorSurfaceContainerHigh,

    outline               = ColorCrimson80,
    outlineVariant        = ColorWhite10,

    error                 = Color(0xFFFFB4AB),
    onError               = Color(0xFF690005),
    errorContainer        = Color(0xFF93000A),
    onErrorContainer      = Color(0xFFFFDAD6)
)

@Composable
fun TurboQuantTheme(
    // Kept for API symmetry; always renders in dark mode.
    @Suppress("UNUSED_PARAMETER") darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = DarkColorScheme,
        typography  = TurboQuantTypography,
        content     = content
    )
}
