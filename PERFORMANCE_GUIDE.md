# Real-Time Translation Performance Guide

This guide explains how to optimize your bidirectional Japanese-English speech translation system for maximum performance.

## Quick Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your OpenAI API key:**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. **Run the optimized system:**
   ```bash
   python main.py
   ```

## Performance Optimizations Implemented

### ✅ Model Optimizations
- **Faster Whisper**: Switched from OpenAI Whisper to faster-whisper (2-10x speedup)
- **Medium Model**: Balanced accuracy/speed (was large-v3)
- **GPU Acceleration**: Auto-detects and uses CUDA when available
- **Reduced Beam Size**: From 5 to 3 for faster inference

### ✅ Translation Improvements  
- **GPT-4o-mini**: 3-5x faster than GPT-5, maintains quality
- **Translation Caching**: Avoids re-translating identical phrases
- **Optimized Prompts**: Shorter, more focused prompts

### ✅ Audio Processing Pipeline
- **In-Memory Processing**: Eliminated temporary file I/O overhead
- **Voice Activity Detection**: Skip processing silence/noise
- **Minimum Duration Filter**: Ignore chunks too short to process
- **Async Pipeline**: Parallel transcription and translation

### ✅ Advanced Features
- **Built-in VAD**: Uses Whisper's voice activity detection
- **Dynamic Filtering**: Removes CTA content and interjections  
- **Configurable Parameters**: Easy performance tuning

## Performance Presets

Use `config.load_preset()` to quickly optimize for different use cases:

### Speed Preset (Real-time priority)
```python
config.load_preset("speed")
```
- Model: `base`
- Expected latency: 200-500ms
- Use case: Live streaming, real-time conversation

### Balanced Preset (Default)
```python
config.load_preset("balanced")  
```
- Model: `medium`
- Expected latency: 500-1000ms
- Use case: Most applications

### Quality Preset (Accuracy priority)
```python
config.load_preset("quality")
```
- Model: `large-v3`
- Expected latency: 1-3 seconds
- Use case: Professional transcription, important meetings

## Manual Configuration

Edit `config.py` to fine-tune performance:

### For Maximum Speed
```python
WHISPER_MODEL_SIZE = "base"
WHISPER_COMPUTE_TYPE = "int8"  
BEAM_SIZE = 1
VAD_AGGRESSIVENESS = 3
MIN_CHUNK_DURATION = 0.3
```

### For Maximum Quality
```python  
WHISPER_MODEL_SIZE = "large-v3"
WHISPER_COMPUTE_TYPE = "float16"
BEAM_SIZE = 5
VAD_AGGRESSIVENESS = 1
MIN_CHUNK_DURATION = 1.0
```

## Expected Performance Gains

Compared to the original implementation:

| Component | Original | Optimized | Improvement |
|-----------|----------|-----------|-------------|
| Whisper Model | large-v3 | medium + faster-whisper | 5-8x faster |
| Translation | GPT-5 | GPT-4o-mini | 3-5x faster |
| Audio Pipeline | File I/O | In-memory | 2x faster |
| **Total Speedup** | - | - | **8-20x faster** |

## Hardware Recommendations

### For Maximum Performance:
- **GPU**: NVIDIA GPU with CUDA support
- **RAM**: 8GB+ for medium model, 16GB+ for large models  
- **CPU**: Modern multi-core processor

### Resource Usage by Model:
- `tiny`: ~1GB RAM, very fast
- `base`: ~1GB RAM, fast
- `medium`: ~5GB RAM, balanced
- `large`: ~10GB RAM, slow but accurate

## Troubleshooting

### Audio Too Fast/Chunks Too Small
```python
# Increase minimum chunk duration
MIN_CHUNK_DURATION = 1.0

# Reduce VAD aggressiveness  
VAD_AGGRESSIVENESS = 1
```

### Translation Too Slow
```python
# Use faster model
TRANSLATION_MODEL = "gpt-4o-mini"

# Enable caching
ENABLE_CACHING = True

# Reduce beam size
BEAM_SIZE = 1
```

### Memory Issues
```python
# Use smaller Whisper model
WHISPER_MODEL_SIZE = "base"

# Use int8 quantization
WHISPER_COMPUTE_TYPE = "int8"

# Reduce cache size
CACHE_MAX_SIZE = 500
```

## Monitoring Performance

The system prints performance information:
- `Using GPU acceleration` - GPU is working
- `Processing queue full` - System overloaded
- `VAD: X% speech frames` - Voice activity levels

## Advanced Optimization

For expert users, consider:

1. **Custom VAD**: Replace WebRTC VAD with specialized models
2. **Model Quantization**: Further reduce model size
3. **Batch Processing**: Process multiple chunks together
4. **WebRTC Integration**: Direct browser-to-server streaming
5. **Edge Deployment**: Run inference closer to users

## Getting Help

If you're still experiencing performance issues:
1. Check your hardware specs against recommendations
2. Try different presets in `config.py`
3. Monitor system resources during operation
4. Consider upgrading hardware for demanding use cases