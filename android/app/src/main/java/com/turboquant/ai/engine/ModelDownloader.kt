package com.turboquant.ai.engine

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.TimeUnit

private const val TAG = "ModelDownloader"

/** Download state emitted by [ModelDownloader.download]. */
sealed class DownloadState {
    /** Emitted incrementally as each chunk arrives (0f – 1f). */
    data class Progress(val fraction: Float, val downloadedBytes: Long, val totalBytes: Long) :
        DownloadState()

    /** Emitted once when the file is fully written to disk. */
    data class Complete(val file: File) : DownloadState()

    /** Emitted if the download fails. */
    data class Error(val message: String) : DownloadState()
}

/** Status of the model on disk (used for button-state logic). */
enum class ModelFileStatus { NOT_FOUND, INCOMPLETE, READY }

/**
 * Chunk-based HTTP download service for GGUF model files.
 *
 * Features:
 *  - OkHttp streaming with configurable chunk size
 *  - Resume support via HTTP Range header
 *  - Emits [DownloadState] via a cold Flow
 *  - All I/O dispatched on [Dispatchers.IO]
 */
class ModelDownloader(context: Context) {

    companion object {
        // ── Model source ─────────────────────────────────────────────────────
        const val MODEL_URL =
            "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/" +
                    "gemma-2-2b-it-Q4_K_M.gguf"

        const val MODEL_FILENAME = "gemma-2b-it.Q4_K_M.gguf"

        // HTTP chunk: 4 MiB provides smooth progress updates without overhead
        private const val CHUNK_BYTES = 4L * 1024L * 1024L
    }

    private val modelsDir: File = File(context.filesDir, "models").also { it.mkdirs() }
    val modelFile: File get() = File(modelsDir, MODEL_FILENAME)

    private val http = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .followRedirects(true)
        .build()

    // ── Model status ─────────────────────────────────────────────────────────

    /**
     * Query the Hugging Face CDN for the expected Content-Length without downloading.
     * Returns -1 if the remote size is unavailable.
     */
    private fun remoteSize(): Long {
        return try {
            val req = Request.Builder()
                .url(MODEL_URL)
                .head()
                .build()
            http.newCall(req).execute().use { resp ->
                resp.header("Content-Length")?.toLongOrNull() ?: -1L
            }
        } catch (e: Exception) {
            Log.w(TAG, "HEAD request failed: ${e.message}")
            -1L
        }
    }

    /** Check whether the model file is fully present on disk. */
    fun modelStatus(): ModelFileStatus {
        if (!modelFile.exists()) return ModelFileStatus.NOT_FOUND
        val localSize = modelFile.length()
        if (localSize == 0L) return ModelFileStatus.NOT_FOUND
        val remote = remoteSize()
        return if (remote > 0L && localSize < remote) ModelFileStatus.INCOMPLETE
        else ModelFileStatus.READY
    }

    // ── Download ─────────────────────────────────────────────────────────────

    /**
     * Start (or resume) downloading the model from Hugging Face.
     *
     * Emits [DownloadState.Progress] roughly every [CHUNK_BYTES] bytes,
     * then [DownloadState.Complete] or [DownloadState.Error].
     *
     * Example:
     * ```kotlin
     * downloader.download().collect { state ->
     *     when (state) {
     *         is Progress  -> progressBar.progress = (state.fraction * 100).toInt()
     *         is Complete  -> onModelReady(state.file)
     *         is Error     -> showError(state.message)
     *     }
     * }
     * ```
     */
    fun download(): Flow<DownloadState> = flow {
        val existingBytes = if (modelFile.exists()) modelFile.length() else 0L
        Log.i(TAG, "Download start – existing=$existingBytes bytes, url=$MODEL_URL")

        val requestBuilder = Request.Builder().url(MODEL_URL)
        if (existingBytes > 0L) {
            requestBuilder.header("Range", "bytes=$existingBytes-")
            Log.i(TAG, "Resuming from byte $existingBytes")
        }

        val response = try {
            http.newCall(requestBuilder.build()).execute()
        } catch (e: Exception) {
            Log.e(TAG, "Connection failed: ${e.message}")
            emit(DownloadState.Error("Connection failed: ${e.message}"))
            return@flow
        }

        if (!response.isSuccessful && response.code != 206 /* partial content */) {
            response.close()
            val msg = "HTTP ${response.code}: ${response.message}"
            Log.e(TAG, msg)
            emit(DownloadState.Error(msg))
            return@flow
        }

        val body = response.body ?: run {
            response.close()
            emit(DownloadState.Error("Empty response body"))
            return@flow
        }

        // Determine total size
        val contentLength = body.contentLength()
        val totalBytes = if (contentLength > 0) existingBytes + contentLength
        else -1L   // server didn't send Content-Length

        val outputStream = FileOutputStream(modelFile, /* append = */ existingBytes > 0L)
        val buffer = ByteArray(CHUNK_BYTES.toInt().coerceAtMost(256 * 1024))
        var downloadedSoFar = existingBytes
        var bytesRead: Int

        try {
            body.byteStream().use { inputStream ->
                while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                    outputStream.write(buffer, 0, bytesRead)
                    downloadedSoFar += bytesRead

                    val fraction = if (totalBytes > 0L)
                        (downloadedSoFar.toFloat() / totalBytes.toFloat()).coerceIn(0f, 1f)
                    else 0f

                    emit(DownloadState.Progress(fraction, downloadedSoFar, totalBytes))
                }
            }
            outputStream.flush()
            Log.i(TAG, "Download complete: $downloadedSoFar bytes → ${modelFile.path}")
            emit(DownloadState.Complete(modelFile))
        } catch (e: Exception) {
            Log.e(TAG, "Download interrupted: ${e.message}")
            emit(DownloadState.Error("Download interrupted: ${e.message}"))
        } finally {
            outputStream.close()
            response.close()
        }
    }.flowOn(Dispatchers.IO)

    /** Delete the (possibly partial) local model file. */
    fun deleteLocalModel() {
        if (modelFile.exists()) modelFile.delete()
        Log.i(TAG, "Model file deleted")
    }
}
