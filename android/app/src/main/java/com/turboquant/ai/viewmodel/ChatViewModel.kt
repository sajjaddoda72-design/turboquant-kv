package com.turboquant.ai.viewmodel

import android.app.Application
import android.content.Context
import android.util.Log
import androidx.core.content.ContextCompat
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.turboquant.ai.TurboQuantApp
import com.turboquant.ai.engine.InferenceManager
import com.turboquant.ai.engine.InferenceMetrics
import com.turboquant.ai.engine.InferenceToken
import com.turboquant.ai.engine.MessageRole
import com.turboquant.ai.engine.MetricsCalculator
import com.turboquant.ai.engine.ModelDownloader
import com.turboquant.ai.engine.TurboQuantEngine
import com.turboquant.ai.service.DownloadForegroundService
import com.turboquant.ai.service.DownloadServiceState
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

data class UiMessage(
    val id: String = UUID.randomUUID().toString(),
    val role: MessageRole,
    val content: String
)

sealed class ModelState {
    object NotDownloaded : ModelState()
    data class Downloading(val progress: Float, val downloadedMb: Float, val totalMb: Float) : ModelState()
    object Downloaded : ModelState()
    object Initializing : ModelState()
    object Ready : ModelState()
    data class Error(val message: String) : ModelState()
}

enum class AppScreen { MODEL_HUB, LOADING, CHAT }

// ── ViewModel ─────────────────────────────────────────────────────────────────

/**
 * Single ViewModel driving the entire TurboQuant AI app lifecycle.
 *
 * Extends [AndroidViewModel] so it can access [Application] context to:
 *  - Start [DownloadForegroundService] via `startForegroundService`
 *  - Read [TurboQuantApp.downloadFlow] — the application-level relay that
 *    the service writes download progress to, allowing state to survive
 *    Activity recreation and background/foreground transitions.
 */
class ChatViewModel(
    application: Application,
    private val downloader: ModelDownloader,
    private val inference: InferenceManager,
    private val metrics: MetricsCalculator
) : AndroidViewModel(application) {

    // ── Exposed state ────────────────────────────────────────────────────────

    private val _currentScreen    = MutableStateFlow(AppScreen.MODEL_HUB)
    val currentScreen: StateFlow<AppScreen> = _currentScreen.asStateFlow()

    private val _modelState       = MutableStateFlow<ModelState>(ModelState.NotDownloaded)
    val modelState: StateFlow<ModelState> = _modelState.asStateFlow()

    private val _messages         = MutableStateFlow<List<UiMessage>>(emptyList())
    val messages: StateFlow<List<UiMessage>> = _messages.asStateFlow()

    private val _streamingResponse = MutableStateFlow("")
    val streamingResponse: StateFlow<String> = _streamingResponse.asStateFlow()

    private val _isGenerating     = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating.asStateFlow()

    private val _liveMetrics      = MutableStateFlow(metrics.emptyMetrics())
    val liveMetrics: StateFlow<InferenceMetrics> = _liveMetrics.asStateFlow()

    private val _metricsVisible   = MutableStateFlow(false)
    val metricsVisible: StateFlow<Boolean> = _metricsVisible.asStateFlow()

    // ── Initialization ───────────────────────────────────────────────────────

    init {
        checkModelStatus()
        observeDownloadService()
    }

    /** Check whether the model file is already on disk and set the initial button state. */
    private fun checkModelStatus() {
        viewModelScope.launch(Dispatchers.IO) {
            val status = downloader.modelStatus()
            _modelState.value = when (status) {
                com.turboquant.ai.engine.ModelFileStatus.READY       -> ModelState.Downloaded
                com.turboquant.ai.engine.ModelFileStatus.INCOMPLETE  -> ModelState.Downloading(0f, 0f, 0f)
                com.turboquant.ai.engine.ModelFileStatus.NOT_FOUND   -> ModelState.NotDownloaded
            }
            Log.i(TAG, "Initial model status: ${_modelState.value}")
        }
    }

    /**
     * Observe the application-level [TurboQuantApp.downloadFlow] to pick up
     * download progress emitted by [DownloadForegroundService].
     *
     * This works regardless of whether the Activity is bound to the service,
     * so state is correctly restored when the user returns to the app after
     * leaving it during a background download.
     */
    private fun observeDownloadService() {
        viewModelScope.launch {
            getApplication<TurboQuantApp>().downloadFlow.collect { serviceState ->
                when (serviceState) {
                    is DownloadServiceState.Idle -> { /* no-op */ }

                    is DownloadServiceState.Downloading ->
                        _modelState.value = ModelState.Downloading(
                            progress     = serviceState.fraction,
                            downloadedMb = serviceState.downloadedMb,
                            totalMb      = serviceState.totalMb
                        )

                    is DownloadServiceState.Complete -> {
                        _modelState.value = ModelState.Downloaded
                        Log.i(TAG, "Model download complete: ${serviceState.file.path}")
                        // Reset the app-level flow so a re-open doesn't re-trigger this
                        getApplication<TurboQuantApp>().emitDownloadState(DownloadServiceState.Idle)
                    }

                    is DownloadServiceState.Failed -> {
                        _modelState.value = ModelState.Error(serviceState.message)
                        Log.e(TAG, "Download failed: ${serviceState.message}")
                        getApplication<TurboQuantApp>().emitDownloadState(DownloadServiceState.Idle)
                    }
                }
            }
        }
    }

    // ── Model Download ───────────────────────────────────────────────────────

    /**
     * Start the model download via [DownloadForegroundService].
     *
     * The service runs independently of the Activity lifecycle: the user can
     * press Home and the download continues with a progress notification.
     * State flows back through [TurboQuantApp.downloadFlow].
     */
    fun downloadModel() {
        if (_modelState.value is ModelState.Downloading) return

        val ctx = getApplication<Application>()
        Log.i(TAG, "Starting DownloadForegroundService")
        ContextCompat.startForegroundService(ctx, DownloadForegroundService.intent(ctx))

        // Optimistically set state so the UI responds immediately (before first
        // progress callback from the service).
        _modelState.value = ModelState.Downloading(0f, 0f, 0f)
    }

    // ── Model Initialization ─────────────────────────────────────────────────

    /**
     * Load the model from disk and create the TurboQuant inference context.
     * Transitions: MODEL_HUB → LOADING overlay → CHAT on success.
     */
    fun runModel() {
        if (inference.isInitialized) {
            _currentScreen.value = AppScreen.CHAT
            return
        }

        _currentScreen.value = AppScreen.LOADING
        _modelState.value    = ModelState.Initializing

        viewModelScope.launch(Dispatchers.IO) {
            try {
                inference.initialize(downloader.modelFile)
                withContext(Dispatchers.Main) {
                    _modelState.value    = ModelState.Ready
                    _currentScreen.value = AppScreen.CHAT
                }
                Log.i(TAG, "Model initialized → Chat screen")
            } catch (e: Exception) {
                Log.e(TAG, "Initialization failed: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    _modelState.value    = ModelState.Error(e.message ?: "Init failed")
                    _currentScreen.value = AppScreen.MODEL_HUB
                }
            }
        }
    }

    // ── Chat ─────────────────────────────────────────────────────────────────

    /**
     * Send [text] to the model and stream the response token-by-token.
     *
     * - Immediately appends the user message to [messages].
     * - Streams assistant tokens live into [streamingResponse].
     * - On completion, commits the full response to [messages].
     * - Updates [liveMetrics] after every single token.
     */
    fun sendMessage(text: String) {
        val trimmed = text.trim()
        if (trimmed.isEmpty() || _isGenerating.value) return
        if (!inference.isInitialized) {
            Log.w(TAG, "Engine not ready")
            return
        }

        _messages.update { it + UiMessage(role = MessageRole.USER, content = trimmed) }
        _isGenerating.value      = true
        _streamingResponse.value = ""
        metrics.startGeneration()

        viewModelScope.launch {
            inference.generateStreaming(trimmed).collect { token ->
                when (token) {
                    is InferenceToken.Piece -> {
                        _streamingResponse.update { it + token.text }
                        _liveMetrics.value = metrics.onTokenGenerated()
                    }
                    is InferenceToken.Done -> {
                        val finalText = _streamingResponse.value
                        if (finalText.isNotEmpty()) {
                            _messages.update {
                                it + UiMessage(role = MessageRole.ASSISTANT, content = finalText)
                            }
                        }
                        _streamingResponse.value = ""
                        _isGenerating.value      = false
                        Log.i(TAG, "Done — ${_liveMetrics.value.tokenCount} tokens")
                    }
                    is InferenceToken.Error -> {
                        Log.e(TAG, "Inference error: ${token.message}")
                        _messages.update {
                            it + UiMessage(role = MessageRole.ASSISTANT,
                                content = "⚠ Error: ${token.message}")
                        }
                        _streamingResponse.value = ""
                        _isGenerating.value      = false
                    }
                }
            }
        }
    }

    fun stopGeneration() {
        inference.stopGeneration()
        _isGenerating.value = false
    }

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

    fun toggleMetrics() { _metricsVisible.update { !it } }
    fun hideMetrics()   { _metricsVisible.value = false }

    // ── Navigation ───────────────────────────────────────────────────────────

    fun navigateToHub() { _currentScreen.value = AppScreen.MODEL_HUB }

    // ── Cleanup ──────────────────────────────────────────────────────────────

    override fun onCleared() {
        super.onCleared()
        inference.release()
        Log.i(TAG, "ViewModel cleared")
    }

    // ── Factory ──────────────────────────────────────────────────────────────

    companion object {
        fun factory(context: Context): ViewModelProvider.Factory =
            object : ViewModelProvider.Factory {
                @Suppress("UNCHECKED_CAST")
                override fun <T : androidx.lifecycle.ViewModel> create(modelClass: Class<T>): T {
                    val app        = context.applicationContext as Application
                    val engine     = TurboQuantEngine()
                    val downloader = ModelDownloader(app)
                    val inference  = InferenceManager(engine)
                    val metrics    = MetricsCalculator(inference)
                    return ChatViewModel(app, downloader, inference, metrics) as T
                }
            }
    }
}
