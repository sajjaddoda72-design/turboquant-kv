package com.turboquant.ai.viewmodel

import android.content.Context
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.turboquant.ai.engine.DownloadState
import com.turboquant.ai.engine.InferenceManager
import com.turboquant.ai.engine.InferenceMetrics
import com.turboquant.ai.engine.InferenceToken
import com.turboquant.ai.engine.MessageRole
import com.turboquant.ai.engine.MetricsCalculator
import com.turboquant.ai.engine.ModelDownloader
import com.turboquant.ai.engine.TurboQuantEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.UUID

private const val TAG = "ChatViewModel"

// ── UI data models ────────────────────────────────────────────────────────────

/** A single chat message shown in the conversation list. */
data class UiMessage(
    val id: String = UUID.randomUUID().toString(),
    val role: MessageRole,
    val content: String
)

/** Describes the current lifecycle phase of the model. */
sealed class ModelState {
    /** No model file on disk. */
    object NotDownloaded : ModelState()

    /** Model is being downloaded (progress in [0f, 1f]). */
    data class Downloading(
        val progress: Float,
        val downloadedMb: Float,
        val totalMb: Float
    ) : ModelState()

    /** Model file is on disk and ready to load. */
    object Downloaded : ModelState()

    /** Model is being loaded and context created. */
    object Initializing : ModelState()

    /** Inference engine is ready for queries. */
    object Ready : ModelState()

    /** A recoverable or fatal error occurred. */
    data class Error(val message: String) : ModelState()
}

/** Which top-level screen the app is showing. */
enum class AppScreen { MODEL_HUB, LOADING, CHAT }

// ── ViewModel ─────────────────────────────────────────────────────────────────

/**
 * Single ViewModel that drives the full TurboQuant AI app lifecycle:
 *
 *  1. Model Hub     — download / run-model button
 *  2. Loading       — initialisation overlay
 *  3. Chat          — message list + streaming response + metrics dashboard
 *
 * All heavy work is dispatched to [Dispatchers.IO]; UI state is updated
 * on the main thread via [MutableStateFlow].
 */
class ChatViewModel(
    private val downloader: ModelDownloader,
    private val inference: InferenceManager,
    private val metrics: MetricsCalculator
) : ViewModel() {

    // ── Exposed state ────────────────────────────────────────────────────────

    private val _currentScreen   = MutableStateFlow(AppScreen.MODEL_HUB)
    val currentScreen: StateFlow<AppScreen> = _currentScreen.asStateFlow()

    private val _modelState      = MutableStateFlow<ModelState>(ModelState.NotDownloaded)
    val modelState: StateFlow<ModelState> = _modelState.asStateFlow()

    private val _messages        = MutableStateFlow<List<UiMessage>>(emptyList())
    val messages: StateFlow<List<UiMessage>> = _messages.asStateFlow()

    /**
     * Partial text of the assistant turn that is currently being streamed.
     * The UI shows this as an in-progress "typing" bubble.
     * Empty string when no generation is in progress.
     */
    private val _streamingResponse = MutableStateFlow("")
    val streamingResponse: StateFlow<String> = _streamingResponse.asStateFlow()

    private val _isGenerating    = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating.asStateFlow()

    private val _liveMetrics     = MutableStateFlow(metrics.emptyMetrics())
    val liveMetrics: StateFlow<InferenceMetrics> = _liveMetrics.asStateFlow()

    private val _metricsVisible  = MutableStateFlow(false)
    val metricsVisible: StateFlow<Boolean> = _metricsVisible.asStateFlow()

    // ── Initialization ───────────────────────────────────────────────────────

    init {
        checkModelStatus()
    }

    /** Synchronously check the local model file and set initial state. */
    private fun checkModelStatus() {
        viewModelScope.launch(Dispatchers.IO) {
            val status = downloader.modelStatus()
            val newState = when (status) {
                com.turboquant.ai.engine.ModelFileStatus.READY ->
                    ModelState.Downloaded

                com.turboquant.ai.engine.ModelFileStatus.INCOMPLETE ->
                    ModelState.Downloading(0f, 0f, 0f)   // will show resume option

                com.turboquant.ai.engine.ModelFileStatus.NOT_FOUND ->
                    ModelState.NotDownloaded
            }
            _modelState.value = newState
            Log.i(TAG, "Model status checked: $newState")
        }
    }

    // ── Model Download ───────────────────────────────────────────────────────

    /**
     * Start (or resume) downloading the model file from Hugging Face.
     * Progress is mapped directly to [modelState].
     */
    fun downloadModel() {
        if (_modelState.value is ModelState.Downloading) return

        viewModelScope.launch {
            downloader.download().collect { state ->
                when (state) {
                    is DownloadState.Progress -> {
                        val dlMb    = state.downloadedBytes.toFloat() / (1024f * 1024f)
                        val totalMb = if (state.totalBytes > 0L)
                            state.totalBytes.toFloat() / (1024f * 1024f) else 0f
                        _modelState.value = ModelState.Downloading(state.fraction, dlMb, totalMb)
                    }

                    is DownloadState.Complete -> {
                        Log.i(TAG, "Download complete: ${state.file.path}")
                        _modelState.value = ModelState.Downloaded
                    }

                    is DownloadState.Error -> {
                        Log.e(TAG, "Download error: ${state.message}")
                        _modelState.value = ModelState.Error(state.message)
                    }
                }
            }
        }
    }

    // ── Model Initialization ─────────────────────────────────────────────────

    /**
     * Load the model from disk and create the TurboQuant inference context.
     * Transitions: MODEL_HUB → LOADING → CHAT on success.
     */
    fun runModel() {
        if (inference.isInitialized) {
            // Already loaded — jump straight to chat
            _currentScreen.value = AppScreen.CHAT
            return
        }

        _currentScreen.value = AppScreen.LOADING
        _modelState.value    = ModelState.Initializing

        viewModelScope.launch(Dispatchers.IO) {
            try {
                inference.initialize(downloader.modelFile)
                withContext(Dispatchers.Main) {
                    _modelState.value  = ModelState.Ready
                    _currentScreen.value = AppScreen.CHAT
                }
                Log.i(TAG, "Model initialized, transitioning to chat")
            } catch (e: Exception) {
                Log.e(TAG, "Model initialization failed: ${e.message}")
                withContext(Dispatchers.Main) {
                    _modelState.value  = ModelState.Error(e.message ?: "Unknown error")
                    _currentScreen.value = AppScreen.MODEL_HUB
                }
            }
        }
    }

    // ── Chat ─────────────────────────────────────────────────────────────────

    /**
     * Send [text] to the model and stream the response token-by-token.
     *
     * - Adds the user message to [messages] immediately.
     * - Streams assistant tokens into [streamingResponse].
     * - On completion, commits the full response to [messages].
     * - Updates [liveMetrics] after every token.
     */
    fun sendMessage(text: String) {
        val trimmed = text.trim()
        if (trimmed.isEmpty() || _isGenerating.value) return
        if (!inference.isInitialized) {
            Log.w(TAG, "sendMessage called but engine not ready")
            return
        }

        // ── Add user message to UI ───────────────────────────────────────
        _messages.update { it + UiMessage(role = MessageRole.USER, content = trimmed) }
        _isGenerating.value    = true
        _streamingResponse.value = ""

        metrics.startGeneration()

        viewModelScope.launch {
            inference.generateStreaming(trimmed).collect { token ->
                when (token) {
                    is InferenceToken.Piece -> {
                        // Append piece to the live streaming bubble
                        _streamingResponse.update { it + token.text }
                        // Update all metrics (TPS, KV sizes, RAM%, MSE) per token
                        _liveMetrics.value = metrics.onTokenGenerated()
                    }

                    is InferenceToken.Done -> {
                        // Commit streamed text as a permanent message
                        val finalText = _streamingResponse.value
                        if (finalText.isNotEmpty()) {
                            _messages.update {
                                it + UiMessage(role = MessageRole.ASSISTANT, content = finalText)
                            }
                        }
                        _streamingResponse.value = ""
                        _isGenerating.value      = false
                        Log.i(TAG, "Generation done. Tokens: ${_liveMetrics.value.tokenCount}")
                    }

                    is InferenceToken.Error -> {
                        Log.e(TAG, "Inference error: ${token.message}")
                        _messages.update {
                            it + UiMessage(
                                role    = MessageRole.ASSISTANT,
                                content = "⚠ Error: ${token.message}"
                            )
                        }
                        _streamingResponse.value = ""
                        _isGenerating.value      = false
                    }
                }
            }
        }
    }

    /** Stop current generation mid-stream. */
    fun stopGeneration() {
        inference.stopGeneration()
        _isGenerating.value = false
    }

    /** Clear all chat messages and reset the model KV cache. */
    fun resetConversation() {
        viewModelScope.launch {
            inference.resetConversation()
            _messages.value          = emptyList()
            _streamingResponse.value = ""
            _isGenerating.value      = false
            _liveMetrics.value       = metrics.emptyMetrics()
        }
    }

    // ── Metrics Dashboard ────────────────────────────────────────────────────

    fun toggleMetrics()  { _metricsVisible.update { !it } }
    fun hideMetrics()    { _metricsVisible.value = false }

    // ── Navigation helpers ───────────────────────────────────────────────────

    fun navigateToHub() {
        _currentScreen.value = AppScreen.MODEL_HUB
    }

    // ── Cleanup ──────────────────────────────────────────────────────────────

    override fun onCleared() {
        super.onCleared()
        inference.release()
        Log.i(TAG, "ViewModel cleared, engine released")
    }

    // ── Factory ──────────────────────────────────────────────────────────────

    companion object {
        fun factory(context: Context): ViewModelProvider.Factory =
            object : ViewModelProvider.Factory {
                @Suppress("UNCHECKED_CAST")
                override fun <T : ViewModel> create(modelClass: Class<T>): T {
                    val appCtx     = context.applicationContext
                    val engine     = TurboQuantEngine()
                    val downloader = ModelDownloader(appCtx)
                    val inference  = InferenceManager(engine)
                    val metrics    = MetricsCalculator(inference)
                    return ChatViewModel(downloader, inference, metrics) as T
                }
            }
    }
}
