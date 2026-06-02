# TQ3_0 — Files for llama.cpp Pull Request

هذا المجلد يحتوي على كل الملفات الجاهزة للإرسال لمستودع llama.cpp.

---

## الملفات

```
ggml/include/ggml.h                      ← GGML_TYPE_TQ3_0 = 36
ggml/src/ggml-common.h                   ← block_tq3_0 struct
ggml/src/ggml-quants.h                   ← declarations
ggml/src/ggml-quants.c                   ← quantize + dequantize (NEON/AVX2)
ggml/src/ggml.c                          ← type_traits table
ggml/src/ggml-cpu/quants.h               ← vec_dot declaration
ggml/src/ggml-cpu/quants.c               ← vec_dot generic
ggml/src/ggml-cpu/ggml-cpu.c             ← type_traits_cpu
ggml/src/ggml-cpu/ops.cpp                ← dispatch (7 switch blocks)
ggml/src/ggml-cpu/arch/arm/quants.c      ← ARM NEON implementation
ggml/src/ggml-cpu/arch/x86/quants.c      ← x86 fallback
ggml/src/ggml-cpu/arch/riscv/quants.c    ← RISC-V fallback
llama-context.cpp.patch                  ← Flash Attention fix (apply manually)
```

---

## كيف تستخدمها

1. انزّل هذا المجلد كـ ZIP من GitHub
2. افتح مستودع llama.cpp على جهازك
3. انسخ كل ملف من ggml/ إلى نفس المسار في llama.cpp
4. طبّق الـ patch على `src/llama-context.cpp`:
   - افتح الملف
   - ابحث عن: `cparams.auto_fa = false;`
   - أضف بعده مباشرة:
   ```cpp
   if ((int)cparams.type_k == 36 || (int)cparams.type_v == 36) {
       cparams.flash_attn = false;
   }
   ```

---

## نص طلب السحب (PR)

**العنوان:**
```
feat: add GGML_TYPE_TQ3_0 — 3.5bpw symmetric KV cache quantization
```

**الوصف:**
```
Adds TQ3_0, a new 3-bit symmetric quantization type for KV cache compression.

Results:
- 78.1% RAM savings vs FP16
- MSE = 0.002109
- Roundtrip: 64/64
- ARM NEON (vdotq_s32) + AVX2 + scalar fallback
- F16 perplexity baseline: PPL = 19.2358 (Gemma 2B, WikiText-2, 512 ctx)

Reference implementation: https://github.com/sajjaddoda72-design/turboquant-kv
```
