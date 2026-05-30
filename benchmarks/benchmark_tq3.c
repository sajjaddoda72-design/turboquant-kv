/*
 * benchmark_tq3.c — TurboQuant TQ3_0 Standalone Benchmark
 *
 * Measures compression ratio, MSE, quantize speed, and dequantize speed
 * for the TQ3_0 KV cache quantization scheme.
 *
 * Architecture constants use Gemma 2B defaults (18 layers, 16 KV heads, 256 dim).
 * Adjust the #defines below for other models.
 *
 * Build:
 *   gcc -O3 -march=native benchmark_tq3.c -lm -o bench_tq3
 *
 * Run:
 *   ./bench_tq3
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <assert.h>

/* ─── Model architecture (edit for other models) ───────────────────────── */
#define LAYERS    18    /* transformer layers               */
#define KV_HEADS  16    /* key/value attention heads        */
#define HEAD_DIM  256   /* head dimension (d_kv per head)   */

/* ─── TQ3_0 block layout ─────────────────────────────────────────────────
 *   d     : ggml_fp16_t  (2 bytes) — per-block scale
 *   qs[12]: uint8_t[12]  (12 bytes) — 32 × 3-bit values packed LSB-first
 *   Total : 14 bytes / 32 elements = 3.5 bits per element
 * ──────────────────────────────────────────────────────────────────────── */
#define QK_TQ3_0   32   /* elements per block */
#define QKB_TQ3_0  12   /* packed-data bytes per block */

typedef uint16_t fp16_t;

typedef struct {
    fp16_t  d;
    uint8_t qs[QKB_TQ3_0];
} block_tq3_0;

/* ─── FP32 ↔ FP16 helpers ────────────────────────────────────────────── */
static fp16_t f32_to_f16(float v) {
    uint32_t b; memcpy(&b, &v, 4);
    return (fp16_t)(((b >> 16) & 0x8000u)
                  | (((b & 0x7f800000u) - 0x38000000u) >> 13)
                  |  ((b >> 13) & 0x03ffu));
}
static float f16_to_f32(fp16_t h) {
    uint32_t x = ((uint32_t)(h & 0x8000u) << 16)
               | (((uint32_t)(h & 0x7c00u) + 0x1C000u) << 13)
               |  ((uint32_t)(h & 0x03ffu) << 13);
    float f; memcpy(&f, &x, 4); return f;
}

/* ─── Bit-packing helpers ────────────────────────────────────────────── */
static void tq3_pack8(const uint8_t *t, uint8_t *o) {
    o[0] =  (t[0] & 7)        | ((t[1] & 7) << 3) | ((t[2] & 7) << 6);
    o[1] = ((t[2] & 7) >> 2)  | ((t[3] & 7) << 1) | ((t[4] & 7) << 4) | ((t[5] & 7) << 7);
    o[2] = ((t[5] & 7) >> 1)  | ((t[6] & 7) << 2) | ((t[7] & 7) << 5);
}

/* ─── Quantise: float32 → TQ3_0 ─────────────────────────────────────── */
static void quantize_tq3_0(const float *src, block_tq3_0 *dst, int64_t n) {
    assert(n % QK_TQ3_0 == 0);
    const int64_t nb = n / QK_TQ3_0;
    for (int64_t i = 0; i < nb; ++i) {
        float amax = 0.f;
        for (int j = 0; j < QK_TQ3_0; ++j) {
            float v = fabsf(src[i * QK_TQ3_0 + j]);
            if (v > amax) amax = v;
        }
        const float d  = amax / 3.0f;
        const float id = d > 0.f ? 1.f / d : 0.f;
        dst[i].d = f32_to_f16(d);
        for (int g = 0; g < 4; ++g) {
            uint8_t t[8];
            for (int j = 0; j < 8; ++j) {
                int q = (int)roundf(src[i * QK_TQ3_0 + g * 8 + j] * id) + 4;
                if (q < 0) q = 0; if (q > 7) q = 7;
                t[j] = (uint8_t)q;
            }
            tq3_pack8(t, dst[i].qs + g * 3);
        }
    }
}

/* ─── Dequantise: TQ3_0 → float32 ───────────────────────────────────── */
static void dequantize_tq3_0(const block_tq3_0 *src, float *dst, int64_t n) {
    assert(n % QK_TQ3_0 == 0);
    const int64_t nb = n / QK_TQ3_0;
    for (int64_t i = 0; i < nb; ++i) {
        const float d = f16_to_f32(src[i].d);
        for (int g = 0; g < 4; ++g) {
            const uint8_t *b = src[i].qs + g * 3;
            float *o = dst + i * QK_TQ3_0 + g * 8;
            o[0] = ((float)((int)( b[0]                     & 7) - 4)) * d;
            o[1] = ((float)((int)((b[0] >> 3)               & 7) - 4)) * d;
            o[2] = ((float)((int)(((b[0] >> 6) | (b[1] << 2)) & 7) - 4)) * d;
            o[3] = ((float)((int)((b[1] >> 1)               & 7) - 4)) * d;
            o[4] = ((float)((int)((b[1] >> 4)               & 7) - 4)) * d;
            o[5] = ((float)((int)(((b[1] >> 7) | (b[2] << 1)) & 7) - 4)) * d;
            o[6] = ((float)((int)((b[2] >> 2)               & 7) - 4)) * d;
            o[7] = ((float)((int)((b[2] >> 5)               & 7) - 4)) * d;
        }
    }
}

/* ─── LCG random (reproducible) ─────────────────────────────────────── */
static uint32_t rng_state = 0xDEADBEEFu;
static float rand_normal(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    float u1 = (float)((rng_state >> 8)) / (float)(1u << 24);
    rng_state = rng_state * 1664525u + 1013904223u;
    float u2 = (float)((rng_state >> 8)) / (float)(1u << 24);
    return sqrtf(-2.f * logf(u1 + 1e-7f)) * cosf(6.283185f * u2) * 0.5f;
}

static double now_sec(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* ─── Main benchmark ─────────────────────────────────────────────────── */
int main(void) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║            TurboQuant TQ3_0 — Benchmark Report              ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  Architecture: %d layers × %d KV heads × %d head dim         ║\n",
           LAYERS, KV_HEADS, HEAD_DIM);
    printf("║  Block format: %d elements / %zu bytes = %.2f bits/element     ║\n",
           QK_TQ3_0, sizeof(block_tq3_0),
           (double)(sizeof(block_tq3_0) * 8) / QK_TQ3_0);
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    /* ── Header ── */
    printf("%-8s  %-13s  %-13s  %-10s  %-10s  %-10s  %-10s\n",
           "Tokens", "FP16 KV (MB)", "TQ3_0 (MB)", "Saved (%)",
           "MSE", "Q speed", "DQ speed");
    printf("%-8s  %-13s  %-13s  %-10s  %-10s  %-10s  %-10s\n",
           "──────", "────────────", "──────────", "─────────",
           "───", "────────", "─────────");

    /* Token counts to test */
    const int tokens[] = { 32, 64, 128, 256, 512, 1024, 2048 };
    const int ntests   = (int)(sizeof(tokens) / sizeof(tokens[0]));

    for (int ti = 0; ti < ntests; ++ti) {
        const int tok = tokens[ti];
        const int64_t ne = (int64_t)tok * LAYERS * KV_HEADS * HEAD_DIM;

        float       *orig  = (float *)       malloc(ne * sizeof(float));
        block_tq3_0 *comp  = (block_tq3_0 *) malloc((ne / QK_TQ3_0) * sizeof(block_tq3_0));
        float       *recon = (float *)        malloc(ne * sizeof(float));
        if (!orig || !comp || !recon) { printf("OOM at %d tokens\n", tok); break; }

        /* Fill with Gaussian data (σ = 0.5) representing attention activations */
        rng_state = 0xDEADBEEFu;
        for (int64_t i = 0; i < ne; ++i) orig[i] = rand_normal();

        /* ── Quantize ── */
        double t0 = now_sec();
        quantize_tq3_0(orig, comp, ne);
        double t_quant = now_sec() - t0;

        /* ── Dequantize ── */
        double t1 = now_sec();
        dequantize_tq3_0(comp, recon, ne);
        double t_dequant = now_sec() - t1;

        /* ── Metrics ── */
        float fp16_mb = (float)(ne * 2)                      / (1024.f * 1024.f);
        float tq3_mb  = (float)(ne / QK_TQ3_0 * (int64_t)sizeof(block_tq3_0)) / (1024.f * 1024.f);
        float saved   = (1.f - tq3_mb / fp16_mb) * 100.f;

        double mse = 0.0;
        for (int64_t i = 0; i < ne; ++i) {
            double e = orig[i] - recon[i]; mse += e * e;
        }
        mse /= (double)ne;

        /* Throughput in MB/s */
        float q_speed  = (float)(ne * sizeof(float)) / (1024.f * 1024.f) / (float)t_quant;
        float dq_speed = tq3_mb / (float)t_dequant;

        printf("%-8d  %-13.2f  %-13.2f  %-9.2f%%  %-10.6f  %5.0f MB/s  %5.0f MB/s\n",
               tok, fp16_mb, tq3_mb, saved, (float)mse, q_speed, dq_speed);

        free(orig); free(comp); free(recon);
    }

    /* ── Summary ── */
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  KV Cache Compression Formulae                              ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  FP16   = tokens × layers × heads × dim × 2 bytes          ║\n");
    printf("║  TQ3_0  = tokens × layers × heads × dim × (14/32) bytes    ║\n");
    printf("║  RAM %% saved = (1 − TQ3_0_size / FP16_size) × 100          ║\n");
    printf("║  → Fixed ratio: 1 − 0.4375/2 = 78.125 %%                   ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  Comparison with other ggml KV types                        ║\n");
    printf("║  FP16  16.00 bpw  100.0 %% memory                          ║\n");
    printf("║  Q8_0   8.25 bpw   53.1 %% memory                          ║\n");
    printf("║  Q4_0   4.50 bpw   28.1 %% memory                          ║\n");
    printf("║  TQ3_0  3.50 bpw   21.9 %% memory  ← this work             ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    return 0;
}
