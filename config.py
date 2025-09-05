"""
Configuration file for the real-time translation system.
Adjust these parameters to optimize for your specific use case.
"""

# Model Configuration
WHISPER_MODEL_SIZE = "medium"  # Options: "tiny", "base", "small", "medium", "large", "large-v2", "large-v3"
WHISPER_COMPUTE_TYPE = "int8"  # Options: "int8", "float16", "float32" (int8 is fastest)
WHISPER_DEVICE = "auto"  # Options: "cpu", "auto" (auto tries GPU first, falls back to CPU)

# OpenAI Translation Model  
TRANSLATION_MODEL = "gpt-4o-mini"  # Options: "gpt-4o-mini", "gpt-4o", "gpt-4"

# Audio Processing
MIN_CHUNK_DURATION = 0.2  # Minimum audio chunk duration in seconds (reduced for speed)
OVERLAP_DURATION = 1.0     # Audio overlap for context in seconds
SAMPLE_RATE = 16000        # Audio sample rate in Hz

# Voice Activity Detection
VAD_AGGRESSIVENESS = 2     # 0-3, higher = more aggressive filtering
VAD_SPEECH_THRESHOLD = 0.3 # Minimum ratio of speech frames required

# Translation Caching
CACHE_MAX_SIZE = 1000      # Maximum number of cached translations
ENABLE_CACHING = True      # Set to False to disable caching

# Processing Pipeline
MAX_QUEUE_SIZE = 10        # Maximum audio chunks in processing queue
BEAM_SIZE = 1             # Whisper beam size (1-5, higher = more accurate but slower)

# Context Management
MAX_SRC_CTX = 2           # Maximum source segments for context
MAX_RECENT = 10           # Maximum recent translations for context

# Performance Tuning
ENABLE_VAD_FILTER = True   # Use Whisper's built-in VAD
MIN_SILENCE_DURATION_MS = 500  # Minimum silence duration for VAD

# Model Quality vs Speed Presets
PRESETS = {
    "speed": {
        "model": "base",
        "compute_type": "int8", 
        "beam_size": 1,
        "vad_aggressiveness": 3,
        "min_chunk_duration": 0.1
    },
    "balanced": {
        "model": "medium",
        "compute_type": "int8",
        "beam_size": 1,  # Reduced for speed
        "vad_aggressiveness": 2,
        "min_chunk_duration": 0.2
    },
    "quality": {
        "model": "large-v3",
        "compute_type": "float16",
        "beam_size": 3,  # Reduced from 5 for speed
        "vad_aggressiveness": 1,
        "min_chunk_duration": 0.5
    },
    "original": {
        "model": "large-v3",
        "compute_type": "float16",
        "beam_size": 5,
        "vad_aggressiveness": 0,  # Disable VAD
        "min_chunk_duration": 1.0
    }
}

def load_preset(preset_name: str):
    """Load a performance preset."""
    if preset_name in PRESETS:
        preset = PRESETS[preset_name]
        globals().update({
            "WHISPER_MODEL_SIZE": preset["model"],
            "WHISPER_COMPUTE_TYPE": preset["compute_type"],
            "BEAM_SIZE": preset["beam_size"],
            "VAD_AGGRESSIVENESS": preset["vad_aggressiveness"],
            "MIN_CHUNK_DURATION": preset["min_chunk_duration"]
        })
        print(f"✅ Loaded {preset_name} preset: {preset['model']} model, beam_size={preset['beam_size']}")
    else:
        print(f"❌ Unknown preset: {preset_name}")
        print(f"Available presets: {list(PRESETS.keys())}")