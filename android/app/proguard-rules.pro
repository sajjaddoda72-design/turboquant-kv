# Keep all JNI-exposed classes intact – the native bridge calls these by name.
-keep class com.turboquant.ai.engine.TurboQuantEngine { *; }
-keep interface com.turboquant.ai.engine.TurboQuantEngine$TokenCallback { *; }

# Keep llama.cpp-related log classes
-keep class com.turboquant.ai.engine.** { *; }

# OkHttp
-dontwarn okhttp3.**
-dontwarn okio.**
