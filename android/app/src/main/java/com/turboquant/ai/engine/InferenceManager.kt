package com.turboquant.ai.engine

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File
private const val TAG = "InferenceManager"

// Developer-defined context window for Gemma 2B
private const val GEMMA_N_CTX = 2048

/** Represents a single turn in the conversation. */
data class ChatMessage(val role: MessageRole, val content: String)

enum class MessageRole { USER, ASSISTANT }

/** Emitted by [InferenceManager.generateStreaming] for each streamed piece. */
sealed class InferenceToken {
    data class Piece(val text: String) : InferenceToken()
    object Done : InferenceToken()
    data class Error(val message: String) : InferenceToken()
}

/**
 * Manages the full inference lifecycle:
 *  - Model loading / context creation (one-time, cached in memory)
 *  - Multi-turn conversation via incremental KV-cache extension
 *  - Real-time token streaming via Kotlin [channelFlow]
 *
 * Thread-safety guarantees:
 *  - [initialize] and [resetConversation] are serialised by [initMutex].
 *  - [generateStreaming] is serialised by [generationMutex] — concurrent calls
 *    will queue rather than corrupt the KV cache or crash the native layer.
 *  - [ctxPtr] / [modelPtr] are @Volatile so reads on IO threads always see the
 *    latest value written by the init/release paths on the main thread.
 *  - [sessionStarted] has been REMOVED.  Whether to prepend BOS is now
 *    determined directly from [kvUsedCells] which queries the native KV cache
 *    occupancy.  This is more reliable and automatically correct after a reset.
 *
 * Gemma 2B chat template:
 * ```
 * Turn 1 (kv_used == 0):
 *   [BOS] <start_of_turn>user\n{msg}<end_of_turn>\n<start_of_turn>model\n
 *
 * Turn N (kv_used > 0):
 *   <end_of_turn>\n<start_of_turn>user\n{msg}<end_of_turn>\n<start_of_turn>model\n
 * ```
 * `add_special` in llama_tokenize is set in the C++ bridge based on whether
 * `llama_get_kv_cache_used_cells(ctx) == 0`, ensuring BOS is added exactly once.
 */
class InferenceManager(private val engine: TurboQuantEngine) {

    /** Guards initialize / resetConversation / release */
    private val initMutex = Mutex()

    /**
     * Guards generateStreaming — prevents two messages from being processed
     * concurrently, which would corrupt the KV cache and crash the model.
     */
    private val generationMutex = Mutex()

    // Opaque native pointers — @Volatile so IO-thread reads see the latest value.
    @Volatile private var modelPtr: Long = 0L
    @Volatile private var ctxPtr:   Long = 0L

    /** History kept for potential debugging / context-rebuild use-cases. */
    private val history = mutableListOf<ChatMessage>()

    // ── Model initialization ───────────────────────────────────────────────

    /**
     * Load the model from [modelFile] and create a native inference context.
     * Idempotent — safe to call if already initialized.
     */
    suspend fun initialize(modelFile: File) {
        initMutex.withLock {
            if (ctxPtr != 0L) {
                Log.i(TAG, "Model already loaded — skipping init")
                return
            }
            require(modelFile.exists()) {
                "Model file not found: ${modelFile.absolutePath}"
            }

            Log.i(TAG, "Initializing TurboQuant engine, model=${modelFile.name}")
            engine.nativeBackendInit()

            modelPtr = engine.nativeLoadModel(modelFile.absolutePath)
            check(modelPtr != 0L) { "nativeLoadModel returned NULL — check logcat" }

            ctxPtr = engine.nativeCreateContext(modelPtr, GEMMA_N_CTX)
            check(ctxPtr != 0L) { "nativeCreateContext returned NULL — OOM or bad KV type?" }

            history.clear()
            Log.i(TAG, "Engine ready — n_ctx=$GEMMA_N_CTX, KV type=TQ2_0 (w/ fallback)")
        }
    }

    val isInitialized: Boolean get() = ctxPtr != 0L

    // ── Conversation management ────────────────────────────────────────────

    /**
     * Clear KV cache and history, keeping the model loaded.
     * Next [generateStreaming] call will start a fresh conversation.
     */
    suspend fun resetConversation() {
        initMutex.withLock {
            if (ctxPtr != 0L) engine.nativeResetContext(ctxPtr)
            history.clear()
            Log.i(TAG, "Conversation reset — KV cache cleared")
        }
    }

    // ── Streaming token generation ─────────────────────────────────────────

    /**
     * Generate a response to [userMessage] using the current conversation context.
     *
     * Tokens are emitted immediately as the native decode loop produces them —
     * no buffering.  The stream ends with exactly one [InferenceToken.Done] or
     * [InferenceToken.Error].
     *
     * Serialised by [generationMutex]: if the caller sends a second message
     * before the first has finished, it will suspend here until the current
     * generation completes.
     */
    fun generateStreaming(userMessage: String): Flow<InferenceToken> = channelFlow {
        generationMutex.withLock {
            val ctx   = ctxPtr
            val model = modelPtr
            if (ctx == 0L || model == 0L) {
                send(InferenceToken.Error("Engine not initialized"))
                return@withLock
            }

            // ── Decide on the correct Gemma template ──────────────────────
            // We query the native KV cache instead of tracking a Kotlin flag.
            // This is authoritative: kvUsedCells() == 0 ↔ fresh context.
            val isFirstTurn = (engine.nativeGetKvUsedCells(ctx) == 0)
            val formattedPrompt = if (isFirstTurn)
                buildFirstTurnPrompt(userMessage)
            else
                buildContinuationPrompt(userMessage)

            Log.d(TAG, "Streaming (firstTurn=$isFirstTurn, prompt=${formattedPrompt.length} chars)")

            val responseBuffer = StringBuilder()
            var errorMsg: String? = null

            // nativeGenerate blocks on the current IO thread.
            // Each onToken callback fires for every new token piece;
            // trySend delivers it immediately to the Flow collector.
            val success = engine.nativeGenerate(
                ctx, model, formattedPrompt,
                object : TurboQuantEngine.TokenCallback {
                    override fun onToken(token: String) {
                        responseBuffer.append(token)
                        trySend(InferenceToken.Piece(token))
                    }
                    override fun onComplete() { /* handled below */ }
                    override fun onError(message: String) { errorMsg = message }
                }
            )

            if (!success || errorMsg != null) {
                send(InferenceToken.Error(errorMsg ?: "Native generate returned false"))
                return@withLock
            }

            history += ChatMessage(MessageRole.USER, userMessage)
            history += ChatMessage(MessageRole.ASSISTANT, responseBuffer.toString())

            send(InferenceToken.Done)
        }
    }.flowOn(Dispatchers.IO)

    // ── KV cache queries (used by MetricsCalculator) ───────────────────────

    /** Tokens currently occupying the KV cache (live, from native). */
    fun kvUsedCells(): Int = ctxPtr.let { if (it != 0L) engine.nativeGetKvUsedCells(it) else 0 }

    /** Instantaneous MSE from KV-cache quantisation error (live, from native). */
    fun computeMse(): Float = ctxPtr.let { if (it != 0L) engine.nativeComputeMse(it) else 0f }

    /** Request the running generation to stop at the next decode step. */
    fun stopGeneration() { ctxPtr.let { if (it != 0L) engine.nativeStopGeneration(it) } }

    // ── Cleanup ────────────────────────────────────────────────────────────

    fun release() {
        val ctx   = ctxPtr
        val model = modelPtr
        ctxPtr    = 0L
        modelPtr  = 0L
        if (ctx   != 0L) engine.nativeFreeContext(ctx)
        if (model != 0L) engine.nativeFreeModel(model)
        engine.nativeBackendFree()
        history.clear()
        Log.i(TAG, "InferenceManager released")
    }

    // ── Gemma 2B chat-template helpers ─────────────────────────────────────

    /**
     * Turn 1 — BOS will be inserted by `llama_tokenize(add_special=true)` in
     * the C++ bridge (because `kv_used == 0`).
     */
    private fun buildFirstTurnPrompt(msg: String) =
        "<start_of_turn>user\n${msg}<end_of_turn>\n<start_of_turn>model\n"

    /**
     * Turn N — closes the previous model turn with `<end_of_turn>`, then opens
     * the new user turn.  BOS is NOT added (C++ bridge sees `kv_used > 0`).
     */
    private fun buildContinuationPrompt(msg: String) =
        "<end_of_turn>\n<start_of_turn>user\n${msg}<end_of_turn>\n<start_of_turn>model\n"
}
