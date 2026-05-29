package com.turboquant.ai.engine

import android.util.Log

/**
 * JNI wrapper for the TurboQuant native inference engine.
 *
 * All function names must exactly match the JNI exports in turboquant_bridge.cpp.
 * Handles model lifetime and exposes a clean Kotlin API over raw pointer handles.
 */
class TurboQuantEngine {

    companion object {
        private const val TAG = "TurboQuantEngine"

        init {
            try {
                System.loadLibrary("turboquant")
                Log.i(TAG, "libturboquant.so loaded")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load native library: ${e.message}")
                throw e
            }
        }
    }

    // ── Callback interface (called from the native inference loop) ───────────
    interface TokenCallback {
        /** Called for each generated token piece (UTF-8 string). */
        fun onToken(token: String)

        /** Called when generation completes normally or hits EOS. */
        fun onComplete()

        /** Called when a fatal error occurs during inference. */
        fun onError(message: String)
    }

    // ── Native declarations (implemented in turboquant_bridge.cpp) ───────────

    /** Initialise the llama.cpp backend (call once per process). */
    external fun nativeBackendInit()

    /**
     * Load a GGUF model file.
     * @return opaque model pointer, or 0 on failure.
     */
    external fun nativeLoadModel(modelPath: String): Long

    /**
     * Create an inference context.
     * Forces KV cache to GGML_TYPE_TQ2_0 (id=35) for 3-bit TurboQuant layout.
     * @param modelPtr   pointer returned by [nativeLoadModel]
     * @param nCtx       context window size (2048)
     * @return opaque context pointer, or 0 on failure.
     */
    external fun nativeCreateContext(modelPtr: Long, nCtx: Int): Long

    /**
     * Run the auto-regressive generation loop on a background thread.
     * Calls [TokenCallback.onToken] for every generated piece,
     * [TokenCallback.onComplete] when done, [TokenCallback.onError] on failure.
     *
     * @param ctxPtr     context pointer
     * @param modelPtr   model pointer
     * @param prompt     fully-formatted conversation text (Gemma template applied by Kotlin)
     * @param callback   token-by-token callback interface
     */
    external fun nativeGenerate(
        ctxPtr: Long,
        modelPtr: Long,
        prompt: String,
        callback: TokenCallback
    ): Boolean

    /** Signal the running generation to stop after the current token. */
    external fun nativeStopGeneration(ctxPtr: Long)

    /** Clear KV cache (start a new conversation on the same context). */
    external fun nativeResetContext(ctxPtr: Long)

    /**
     * Return the number of tokens currently occupying the KV cache.
     * Used by [MetricsCalculator] to derive live cache-size stats.
     */
    external fun nativeGetKvUsedCells(ctxPtr: Long): Int

    /**
     * Sample a representative block of the KV cache and compute
     * the Mean Squared Error between TQ2_0 reconstruction and FP16 reference.
     * Value fluctuates naturally with token entropy.
     */
    external fun nativeComputeMse(ctxPtr: Long): Float

    /** Free an inference context (also frees the sampler chain). */
    external fun nativeFreeContext(ctxPtr: Long)

    /** Free a loaded model. */
    external fun nativeFreeModel(modelPtr: Long)

    /** Shut down the llama.cpp backend (call at process exit). */
    external fun nativeBackendFree()
}
