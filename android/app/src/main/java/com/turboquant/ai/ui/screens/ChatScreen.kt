package com.turboquant.ai.ui.screens

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
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
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.GridView
import androidx.compose.material.icons.rounded.Insights
import androidx.compose.material.icons.rounded.Memory
import androidx.compose.material.icons.rounded.Send
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.FloatingActionButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.turboquant.ai.engine.MessageRole
import com.turboquant.ai.ui.components.KvCacheBadge
import com.turboquant.ai.ui.components.MetricsDashboard
import com.turboquant.ai.ui.components.crimsonGlow
import com.turboquant.ai.ui.theme.ColorBackground
import com.turboquant.ai.ui.theme.ColorCrimson
import com.turboquant.ai.ui.theme.ColorCrimson20
import com.turboquant.ai.ui.theme.ColorOnSurface
import com.turboquant.ai.ui.theme.ColorOnSurfaceVariant
import com.turboquant.ai.ui.theme.ColorSurfaceContainer
import com.turboquant.ai.ui.theme.ColorSurfaceContainerHigh
import com.turboquant.ai.ui.theme.ColorSurfaceContainerLow
import com.turboquant.ai.ui.theme.ColorSurfaceDim
import com.turboquant.ai.ui.theme.ColorWhite10
import com.turboquant.ai.ui.theme.ColorWhite5
import com.turboquant.ai.viewmodel.ChatViewModel
import com.turboquant.ai.viewmodel.UiMessage

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(viewModel: ChatViewModel) {

    val messages         by viewModel.messages.collectAsState()
    val streamingText    by viewModel.streamingResponse.collectAsState()
    val isGenerating     by viewModel.isGenerating.collectAsState()
    val metricsVisible   by viewModel.metricsVisible.collectAsState()
    val liveMetrics      by viewModel.liveMetrics.collectAsState()

    var inputText        by remember { mutableStateOf("") }
    val listState        = rememberLazyListState()

    // ── Auto-scroll whenever a new token or message arrives ──────────────
    val totalItems = messages.size + (if (streamingText.isNotEmpty()) 1 else 0)
    LaunchedEffect(totalItems) {
        if (totalItems > 0) listState.animateScrollToItem(totalItems - 1)
    }

    // ── Send helper ───────────────────────────────────────────────────────
    fun sendMessage() {
        val text = inputText.trim()
        if (text.isNotEmpty() && !isGenerating) {
            viewModel.sendMessage(text)
            inputText = ""
        }
    }

    Box(modifier = Modifier.fillMaxSize().background(ColorBackground)) {

        // ── Main layout ──────────────────────────────────────────────────
        Column(modifier = Modifier.fillMaxSize()) {

            // Top App Bar
            TopAppBar(
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Rounded.Memory,
                            contentDescription = null,
                            tint     = ColorCrimson,
                            modifier = Modifier.size(24.dp)
                        )
                        Spacer(Modifier.width(8.dp))
                        Text(
                            text       = "TurboQuant AI",
                            color      = ColorCrimson,
                            fontWeight = FontWeight.Bold,
                            fontSize   = 20.sp
                        )
                    }
                },
                actions = {
                    Box(
                        modifier = Modifier
                            .padding(end = 16.dp)
                            .size(30.dp)
                            .clip(CircleShape)
                            .background(ColorSurfaceContainerHigh)
                            .border(1.dp, ColorWhite10, CircleShape),
                        contentAlignment = Alignment.Center
                    ) {
                        Text("U", color = Color.White, fontSize = 12.sp, fontWeight = FontWeight.Bold)
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = ColorBackground.copy(alpha = 0.92f)
                )
            )

            // KV Cache badge
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 12.dp, bottom = 4.dp),
                contentAlignment = Alignment.Center
            ) {
                KvCacheBadge()
            }

            // ── Chat message list ────────────────────────────────────────
            LazyColumn(
                state        = listState,
                modifier     = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
                contentPadding = androidx.compose.foundation.layout.PaddingValues(
                    top = 8.dp, bottom = 120.dp
                )
            ) {
                // Real messages (no demo data — only messages the user sent)
                items(
                    items = messages,
                    key   = { it.id }
                ) { message ->
                    when (message.role) {
                        MessageRole.USER      -> UserBubble(message)
                        MessageRole.ASSISTANT -> AssistantBubble(message.content, isStreaming = false)
                    }
                }

                // Live streaming bubble (appears while tokens are arriving)
                if (streamingText.isNotEmpty()) {
                    item(key = "streaming") {
                        AssistantBubble(text = streamingText, isStreaming = true)
                    }
                }
            }
        }

        // ── Input bar (fixed above system navigation) ────────────────────
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.BottomCenter)
                .background(
                    androidx.compose.ui.graphics.Brush.verticalGradient(
                        colors = listOf(Color.Transparent, ColorBackground, ColorBackground)
                    )
                )
                .navigationBarsPadding()
                .imePadding()
                .padding(horizontal = 16.dp, vertical = 10.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Text field
                TextField(
                    value         = inputText,
                    onValueChange = { inputText = it },
                    modifier      = Modifier
                        .weight(1f)
                        .clip(RoundedCornerShape(28.dp))
                        .border(
                            width = 1.dp,
                            color = if (inputText.isNotEmpty()) ColorCrimson.copy(alpha = 0.7f)
                                    else ColorWhite10,
                            shape = RoundedCornerShape(28.dp)
                        ),
                    placeholder = {
                        Text(
                            "Ask TurboQuant…",
                            color    = ColorOnSurfaceVariant.copy(alpha = 0.5f),
                            fontSize = 15.sp
                        )
                    },
                    textStyle = androidx.compose.ui.text.TextStyle(
                        color    = ColorOnSurface,
                        fontSize = 15.sp
                    ),
                    singleLine  = false,
                    maxLines    = 5,
                    keyboardOptions = KeyboardOptions(
                        capitalization = KeyboardCapitalization.Sentences,
                        imeAction      = ImeAction.Send
                    ),
                    keyboardActions = KeyboardActions(
                        onSend = { sendMessage() }
                    ),
                    colors = TextFieldDefaults.colors(
                        focusedContainerColor   = ColorSurfaceContainerLow,
                        unfocusedContainerColor = ColorSurfaceContainerLow,
                        cursorColor             = ColorCrimson,
                        focusedIndicatorColor   = Color.Transparent,
                        unfocusedIndicatorColor = Color.Transparent
                    )
                )

                Spacer(Modifier.width(8.dp))

                // Send button
                Box(
                    modifier = Modifier
                        .size(48.dp)
                        .clip(CircleShape)
                        .background(
                            if (!isGenerating && inputText.isNotBlank()) ColorCrimson20
                            else ColorSurfaceContainerHigh
                        )
                        .border(
                            1.dp,
                            if (!isGenerating && inputText.isNotBlank()) ColorCrimson.copy(0.7f)
                            else ColorWhite10,
                            CircleShape
                        )
                        .then(
                            if (!isGenerating && inputText.isNotBlank())
                                Modifier.crimsonGlow(glowRadius = 10.dp, alpha = 0.3f)
                            else Modifier
                        )
                        .clickable(enabled = !isGenerating && inputText.isNotBlank()) {
                            sendMessage()
                        },
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector     = Icons.Rounded.Send,
                        contentDescription = "Send",
                        tint = if (!isGenerating && inputText.isNotBlank()) ColorCrimson
                               else ColorOnSurfaceVariant.copy(alpha = 0.4f),
                        modifier = Modifier.size(22.dp)
                    )
                }
            }
        }

        // ── FAB — open metrics dashboard ─────────────────────────────────
        FloatingActionButton(
            onClick           = { viewModel.toggleMetrics() },
            modifier          = Modifier
                .align(Alignment.BottomEnd)
                .padding(end = 20.dp, bottom = 90.dp)
                .crimsonGlow(glowRadius = 16.dp, alpha = 0.4f),
            containerColor    = ColorCrimson20,
            contentColor      = ColorCrimson,
            elevation         = FloatingActionButtonDefaults.elevation(0.dp, 0.dp),
            shape             = CircleShape
        ) {
            Box(
                modifier = Modifier
                    .size(56.dp)
                    .border(1.dp, ColorCrimson.copy(alpha = 0.75f), CircleShape),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Rounded.Insights,
                    contentDescription = "Performance Metrics",
                    modifier = Modifier.size(26.dp)
                )
            }
        }

        // ── Bottom navigation ─────────────────────────────────────────────
        Row(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .background(ColorSurfaceContainerLow.copy(alpha = 0.75f))
                .navigationBarsPadding()
                .padding(horizontal = 16.dp, vertical = 6.dp),
            horizontalArrangement = Arrangement.SpaceAround
        ) {
            // Hub tab
            NavTab(
                icon     = Icons.Rounded.GridView,
                label    = "Hub",
                isActive = false,
                onClick  = { viewModel.navigateToHub() }
            )
            // Chat tab (active)
            NavTab(
                icon       = Icons.Rounded.Memory,
                label      = "Chat",
                isActive   = true,
                onClick    = {}
            )
            // Metrics tab
            NavTab(
                icon     = Icons.Rounded.Insights,
                label    = "Metrics",
                isActive = false,
                onClick  = { viewModel.toggleMetrics() }
            )
        }

        // ── Metrics dashboard overlay (slides up from bottom) ────────────
        AnimatedVisibility(
            visible = metricsVisible,
            enter   = fadeIn() + slideInVertically(initialOffsetY = { it }),
            exit    = fadeOut() + slideOutVertically(targetOffsetY = { it }),
            modifier = Modifier.fillMaxSize()
        ) {
            MetricsDashboard(
                metrics   = liveMetrics,
                onDismiss = { viewModel.hideMetrics() }
            )
        }
    }
}

// ── Chat bubble composables ───────────────────────────────────────────────────

@Composable
private fun UserBubble(message: UiMessage) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.End
    ) {
        Box(
            modifier = Modifier
                .widthIn(max = 300.dp)
                .clip(RoundedCornerShape(18.dp, 4.dp, 18.dp, 18.dp))
                .background(ColorSurfaceContainer)
                .border(
                    1.dp,
                    ColorWhite5,
                    RoundedCornerShape(18.dp, 4.dp, 18.dp, 18.dp)
                )
                .padding(horizontal = 16.dp, vertical = 12.dp)
        ) {
            Text(
                text  = message.content,
                color = ColorOnSurface,
                fontSize = 15.sp,
                lineHeight = 22.sp
            )
        }
    }
}

@Composable
private fun AssistantBubble(text: String, isStreaming: Boolean) {

    // Blinking cursor animation for streaming state
    val infiniteTransition = rememberInfiniteTransition(label = "cursor")
    val cursorAlpha by infiniteTransition.animateFloat(
        initialValue  = 1f,
        targetValue   = 0f,
        animationSpec = infiniteRepeatable(
            tween(500, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "cursor_alpha"
    )

    Row(
        modifier            = Modifier.fillMaxWidth(),
        verticalAlignment   = Alignment.Top,
        horizontalArrangement = Arrangement.Start
    ) {
        // Model avatar
        Box(
            modifier = Modifier
                .size(32.dp)
                .clip(CircleShape)
                .background(ColorCrimson20)
                .border(1.dp, ColorCrimson.copy(alpha = 0.30f), CircleShape),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = Icons.Rounded.Memory,
                contentDescription = null,
                tint     = ColorCrimson,
                modifier = Modifier.size(18.dp)
            )
        }

        Spacer(Modifier.width(10.dp))

        // Bubble
        Box(
            modifier = Modifier
                .widthIn(max = 300.dp)
                .clip(RoundedCornerShape(4.dp, 18.dp, 18.dp, 18.dp))
                .background(ColorSurfaceContainerLow)
                .border(
                    1.dp,
                    ColorWhite5,
                    RoundedCornerShape(4.dp, 18.dp, 18.dp, 18.dp)
                )
                .padding(horizontal = 16.dp, vertical = 12.dp)
        ) {
            // Check if text contains code blocks and render appropriately
            val displayText = if (isStreaming) text else text.trim()

            if (displayText.contains("```")) {
                MixedTextContent(displayText, isStreaming, cursorAlpha)
            } else {
                Text(
                    text     = if (isStreaming) "$displayText${buildCursor(cursorAlpha)}" else displayText,
                    color    = ColorOnSurface,
                    fontSize = 15.sp,
                    lineHeight = 22.sp
                )
            }
        }
    }
}

/** Renders a message that may contain ``` code blocks. */
@Composable
private fun MixedTextContent(text: String, isStreaming: Boolean, cursorAlpha: Float) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        val segments = text.split("```")
        segments.forEachIndexed { index, segment ->
            val isCode = index % 2 == 1
            if (segment.isBlank()) return@forEachIndexed

            if (isCode) {
                // Code block
                val codeText = segment.trimStart('\n').trimEnd('\n')
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(8.dp))
                        .background(ColorSurfaceDim)
                        .border(1.dp, ColorWhite10, RoundedCornerShape(8.dp))
                        .padding(12.dp)
                ) {
                    Text(
                        text       = codeText,
                        color      = ColorOnSurfaceVariant,
                        fontFamily = FontFamily.Monospace,
                        fontSize   = 13.sp,
                        lineHeight = 18.sp
                    )
                }
            } else {
                val suffix = if (isStreaming && index == segments.lastIndex)
                    buildCursor(cursorAlpha) else ""
                Text(
                    text     = segment.trim() + suffix,
                    color    = ColorOnSurface,
                    fontSize = 15.sp,
                    lineHeight = 22.sp
                )
            }
        }
    }
}

/** Returns a cursor character with the given alpha (approximated via space/pipe). */
private fun buildCursor(alpha: Float): String = if (alpha > 0.5f) "▌" else ""

// ── Bottom nav tab ────────────────────────────────────────────────────────────

@Composable
private fun NavTab(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    label: String,
    isActive: Boolean,
    onClick: () -> Unit
) {
    Column(
        modifier = Modifier
            .clip(RoundedCornerShape(12.dp))
            .background(
                if (isActive) ColorCrimson20 else Color.Transparent
            )
            .border(
                1.dp,
                if (isActive) ColorCrimson.copy(alpha = 0.3f) else Color.Transparent,
                RoundedCornerShape(12.dp)
            )
            .clickable(onClick = onClick)
            .padding(horizontal = 16.dp, vertical = 6.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(2.dp)
    ) {
        Icon(
            imageVector = icon,
            contentDescription = label,
            tint     = if (isActive) ColorCrimson else ColorOnSurfaceVariant.copy(alpha = 0.5f),
            modifier = Modifier.size(22.dp)
        )
        Text(
            text     = label,
            color    = if (isActive) ColorCrimson else ColorOnSurfaceVariant.copy(alpha = 0.5f),
            fontSize = 10.sp,
            fontWeight = if (isActive) FontWeight.SemiBold else FontWeight.Normal
        )
    }
}
