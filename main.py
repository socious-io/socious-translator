import tempfile, subprocess, uvicorn, openai, os, json, uuid, re, asyncio, hashlib, time, io
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from faster_whisper import WhisperModel
# webrtcvad import is optional
try:
    import webrtcvad
    HAS_WEBRTCVAD = True
except ImportError:
    HAS_WEBRTCVAD = False
    print("webrtcvad not available - using built-in VAD only")
import numpy as np
from collections import deque
import config

# Load performance preset - uncomment one of these:
config.load_preset("speed")      # Fastest, lower quality - ENABLED FOR TESTING
# config.load_preset("balanced")   # Default - good balance  
# config.load_preset("quality")    # Slower, highest quality
# config.load_preset("original")   # Match your original setup

# basic setup and client
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = openai.AsyncOpenAI(api_key=api_key)

# pre-warm ffmpeg so the first request isn't slow
try:
    subprocess.run(
        ["ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "0.5", "-ar", "16000", "-ac", "1", "-y", "/tmp/warm.wav"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
except:
    pass

# Load whisper model with robust GPU detection
model = None
device_used = "cpu"
compute_type_used = config.WHISPER_COMPUTE_TYPE

# Try GPU first if set to "auto" or "cuda"
if config.WHISPER_DEVICE in ["auto", "cuda"]:
    try:
        print("Attempting to load GPU model...")
        gpu_model = WhisperModel(config.WHISPER_MODEL_SIZE, device="cuda", compute_type="float16")
        # Quick test to ensure GPU actually works
        test_segments, _ = gpu_model.transcribe("/tmp/warm.wav", language="en")
        model = gpu_model
        device_used = "cuda"
        compute_type_used = "float16"
        print("✅ Using GPU acceleration (CUDA)")
    except Exception as e:
        print(f"❌ GPU failed: {e}")
        print("Falling back to CPU...")
        model = None

# Fallback to CPU if GPU failed or not requested
if model is None:
    try:
        model = WhisperModel(config.WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        device_used = "cpu"
        compute_type_used = "int8"
        print("✅ Using CPU with int8 quantization")
    except Exception as e:
        print(f"❌ Critical error loading model: {e}")
        raise

# Warm up the model
try:
    segments, _ = model.transcribe("/tmp/warm.wav", language="en")
    print("Model warmed up successfully")
except Exception as e:
    print(f"Model warmup failed: {e}")
    pass

# Print configuration summary
print(f"=== Translation System Configuration ===")
print(f"Whisper Model: {config.WHISPER_MODEL_SIZE}")
print(f"Device: {device_used}")
print(f"Compute Type: {compute_type_used}")
print(f"Translation Model: {config.TRANSLATION_MODEL}")
print(f"VAD Available: {HAS_WEBRTCVAD}")
print(f"Caching Enabled: {config.ENABLE_CACHING}")
print(f"Min Chunk Duration: {config.MIN_CHUNK_DURATION}s")
print(f"Max Queue Size: {config.MAX_QUEUE_SIZE}")
print("=" * 40)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


# tiny context buffers for current-line translation only
recent_src_segments: list[str] = []
recent_targets: list[str] = []

# All configuration now handled by config.py module

# Translation cache
translation_cache = {}

# VAD configuration (optional)
vad = None
if HAS_WEBRTCVAD:
    vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)

# Audio buffer for overlapping chunks
audio_buffer = deque(maxlen=int(config.SAMPLE_RATE * 5))  # 5 second buffer

# exact short interjections only (and standalone "you")
THANKS_RE = re.compile(
    r"""
    ^\s*
    (?:
        (?:thank\s*(?:you|u)|thanks|thanx|thx|tks|ty|tysm|tyvm|
         many\s*thanks|much\s*thanks|much\s*appreciated|appreciate\s*it|
         cheers|ta|nice\s*one)
        (?:\s*(?:so\s*much|very\s*much|a\s*lot|
            everyone|everybody|all|y['’]?all|guys|folks|team))?
        |you
    )
    \s*[!.…😊🙏💖✨👏👍]*\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

def is_interjection_thanks(text: str) -> bool:
    if not text:
        return False
    return bool(THANKS_RE.match(text.strip()))


# Broad CTA / meta filler (intros, outros, like/subscribe, links, comments, follows, sponsors, merch, goals)
_CTA_PATTERNS = [
    # --- Intros / greetings (channel meta) ---
    r"(?i)^\s*(?:hey|hi|hello|what'?s\s*up|yo|good\s+(?:morning|afternoon|evening))\s+(?:guys|everyone|everybody|folks|y['’]?all|friends)\b",
    r"(?i)\bwelcome\s+(?:back\s+)?to\s+my\s+channel\b",
    r"(?i)\bwelcome\s+(?:back\s+)?(?:everyone|guys|y['’]?all|friends)\b",
    r"(?i)\bthanks?\s+for\s+joining\s+(?:me|us)\b",

    # --- Like / comment / subscribe combos ---
    r"(?i)\blike\s*(?:,?\s*comment\s*)?(?:,?\s*and\s*)?subscribe\b",
    r"(?i)\bsubscribe\s*(?:,?\s*and\s*)?(?:like|comment)\b",
    r"(?i)\b(?:like|comment|subscribe)\b(?:\s*[,&/]\s*\b(?:like|comment|subscribe)\b){1,2}",
    r"(?i)\b(?:drop|leave|smash|hit|press|tap)\s+(?:the\s+)?like\b",
    r"(?i)\bgive\s+(?:it\s+)?a\s+thumbs?\s*up\b",
    r"(?i)\bdrop\s+a\s+like\b",
    r"(?i)\bsubscribe\s+(?:for|to)\s+more\b",
    r"(?i)\bsubscribe\s+now\b",
    r"(?i)\bhelp\s+(?:the\s+)?channel\s+by\s+(?:liking|subscribing)\b",

    # --- Notifications ---
    r"(?i)\b(?:ring|hit|tap|click)\s+(?:the\s+)?(?:notification\s+)?bell\b",
    r"(?i)\bturn\s+on\s+(?:the\s+)?notifications?\b",
    r"(?i)\benable\s+(?:post\s+)?notifications?\b",
    r"(?i)\bdon'?t\s+forget\s+(?:to\s+)?(?:like|comment|share|subscribe|turn\s+on\s+notifications?)\b",

    # --- Share / links / description ---
    r"(?i)\bshare\s+(?:this|the)\s+(?:video|stream|clip|content|tutorial)\b",
    r"(?i)\bplease\s+share\b",
    r"(?i)\b(?:link|links?)\s+in\s+(?:the\s+)?(?:bio|description|comments?)\b",
    r"(?i)\bcheck\s+(?:the\s+)?link\s+(?:below|above)\b",
    r"(?i)\bmore\s+info\s+in\s+(?:the\s+)?description\b",

    # --- Comment prompts ---
    r"(?i)\b(?:leave|drop|post)\s+(?:a|your)?\s*comment[s]?\b",
    r"(?i)\bcomment\s+(?:below|down\s+below)\b",
    r"(?i)\btell\s+me\s+in\s+the\s+comments?\b",
    r"(?i)\bwhat\s+do\s+you\s+think\s+in\s+the\s+comments?\b",

    # --- Outros / closings ---
    r"(?i)\bthanks?\s+for\s+(?:watching|tuning\s+in|coming\s+by|listening)\b",
    r"(?i)\bthank\s+you\s+for\s+(?:watching|tuning\s+in|coming\s+by|listening)\b",
    r"(?i)\bsee\s+you\s+(?:next\s*time|tomorrow|soon|in\s+the\s+next(?:\s+one|video)?)\b",
    r"(?i)\bthat'?s\s+(?:it|all)(?:\s+for\s+(?:today|now))?\b",
    r"(?i)\bcatch\s+you\s+later\b",
    r"(?i)\bpeace\s+out\b",
    r"(?i)\btake\s+care\b",

    # --- Sponsorship / affiliate / support ---
    r"(?i)\bthis\s+video\s+is\s+sponsored\s+by\b",
    r"(?i)\b(?:sponsor(?:ed)?|partner(?:ed)?)\s+(?:with|by)\b",
    r"(?i)\buse\s+code\s+[A-Z0-9]{3,}\b",
    r"(?i)\baffiliate\s+links?\b",
    r"(?i)\b(?:support|supporting)\s+(?:the\s+)?channel\b",
    r"(?i)\b(?:join|check\s+out)\s+my\s+patreon\b",
    r"(?i)\bpatreon\.com\b",
    r"(?i)\bko-?fi\.com\b",
    r"(?i)\bbuy\s+me\s+a\s+coffee\b",
    r"(?i)\bmy\s+merch\b",
    r"(?i)\bmerch(?:andise)?\s+link\b",

    # --- Social follows ---
    r"(?i)\bfollow\s+me\s+on\s+(?:instagram|tiktok|twitter|x|twitch|youtube|facebook|threads)\b",
    r"(?i)\b(?:instagram|tiktok|twitter|x|twitch|youtube|facebook|threads)\.com\/\w+",
    r"(?i)\bmy\s+(?:instagram|tiktok|twitter|x|twitch|facebook|threads)\s+is\b",

    # --- Community / membership ---
    r"(?i)\bjoin\s+(?:my|our)\s+discord\b",
    r"(?i)\bdiscord\.gg\/?[A-Za-z0-9]+",
    r"(?i)\bjoin\s+(?:the\s+)?channel\s+as\s+a\s+member\b",
    r"(?i)\bbecome\s+a\s+member\b",

    # --- Giveaways / goals / algorithm meta ---
    r"(?i)\bgiveaway\b",
    r"(?i)\blike\s+goal\b",
    r"(?i)\blet'?s\s+get\s+to\s+\d+\s+likes\b",
    r"(?i)\bhelps?\s+(?:with\s+)?the\s+algorithm\b",

    # --- Captions/credits (existing) ---
    r"(?i)\bsubtitles?\s+by\b",
    r"(?i)\bcaptions?\s+by\b",
    r"(?i)\bsubtitled\s+by\b",
    r"(?i)\bclosed\s+captions?\s+by\b",
    r"(?i)\bamara\.org\b",
    r"(?i)\btranscription\s+by\b",

 
]

_CTA_REGEXES = [re.compile(p) for p in _CTA_PATTERNS]

def is_cta_like(text: str) -> bool:
    if not text or len(text.strip()) < 2: return False
    return any(rx.search(text) for rx in _CTA_REGEXES)


def get_cache_key(text: str, source_lang: str, target_lang: str) -> str:
    """Generate cache key for translation."""
    content = f"{text}|{source_lang}|{target_lang}"
    return hashlib.md5(content.encode()).hexdigest()


def get_cached_translation(text: str, source_lang: str, target_lang: str) -> str:
    """Get cached translation if available."""
    key = get_cache_key(text, source_lang, target_lang)
    return translation_cache.get(key)


def cache_translation(text: str, source_lang: str, target_lang: str, translation: str):
    """Cache translation with LRU eviction."""
    key = get_cache_key(text, source_lang, target_lang)
    
    if len(translation_cache) >= config.CACHE_MAX_SIZE:
        # Remove oldest entry
        oldest_key = next(iter(translation_cache))
        del translation_cache[oldest_key]
    
    translation_cache[key] = translation


def has_speech(audio_data: bytes, sample_rate: int = 16000) -> bool:
    """Check if audio contains speech using WebRTC VAD (if available)."""
    if not HAS_WEBRTCVAD or vad is None:
        # If WebRTC VAD is not available, use simple energy-based detection
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            # Simple energy-based voice activity detection
            energy = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            # Threshold for speech (adjust as needed)
            has_activity = energy > 1000
            # Uncomment for debugging: print(f"Energy VAD: {energy:.1f}, has_speech: {has_activity}")
            return has_activity
        except Exception as e:
            print(f"VAD fallback error: {e}")
            return True
    
    try:
        # Convert to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # VAD requires specific frame sizes (10ms, 20ms, 30ms)
        frame_duration = 30  # ms
        frame_size = int(sample_rate * frame_duration / 1000)
        
        # Check if we have enough data
        if len(audio_array) < frame_size:
            return False
        
        # Process in 30ms frames
        speech_frames = 0
        total_frames = 0
        
        for i in range(0, len(audio_array) - frame_size, frame_size):
            frame = audio_array[i:i + frame_size]
            if vad.is_speech(frame.tobytes(), sample_rate):
                speech_frames += 1
            total_frames += 1
        
        # Return True if speech ratio exceeds threshold
        return speech_frames / total_frames > config.VAD_SPEECH_THRESHOLD if total_frames > 0 else False
    except Exception:
        # If VAD fails, assume there's speech
        return True


def convert_audio_in_memory(input_data: bytes) -> bytes:
    """Convert audio to WAV format in memory using FFmpeg."""
    try:
        process = subprocess.Popen(
            ["ffmpeg", "-i", "pipe:0", "-ar", str(config.SAMPLE_RATE), "-ac", "1", "-f", "wav", "pipe:1"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        
        wav_data, _ = process.communicate(input=input_data)
        return wav_data if process.returncode == 0 else b""
    except Exception:
        return b""


def get_audio_duration(wav_data: bytes) -> float:
    """Get duration of WAV audio data in seconds."""
    try:
        # Skip WAV header (44 bytes) and calculate duration
        audio_data = wav_data[44:]  # Skip WAV header
        num_samples = len(audio_data) // 2  # 16-bit samples
        return num_samples / config.SAMPLE_RATE
    except Exception:
        return 0.0


async def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    # Check cache first if enabled
    if config.ENABLE_CACHING:
        cached = get_cached_translation(text, source_lang, target_lang)
        if cached:
            return cached
    
    # Simplified, much shorter prompt for speed
    system = "Translate to natural Japanese. Return ONLY the translation."
    user = f"Translate: {text}"

    try:
        resp = await client.chat.completions.create(
            model=config.TRANSLATION_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_completion_tokens=64,  # Reduced for speed
            temperature=0.0  # Deterministic for consistency
        )
        translation = (resp.choices[0].message.content or "").strip()
        
        # Cache the result if caching is enabled
        if config.ENABLE_CACHING:
            cache_translation(text, source_lang, target_lang, translation)
        
        return translation
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original on error

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join("frontend", "index.html"))

async def process_audio_chunk_fast(audio_data: bytes, source_lang: str, target_lang: str, whisper_task: str, direction: str) -> tuple[str, str]:
    """Process audio chunk with minimal overhead for speed."""
    try:
        # Direct file write without extra checks for speed
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as raw_file:
            raw_file.write(audio_data)
            raw_file.flush()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_file:
                # Fast conversion
                try:
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", raw_file.name, "-ar", str(config.SAMPLE_RATE), "-ac", "1", wav_file.name],
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL, 
                        check=True,
                        timeout=5  # 5 second timeout
                    )
                except:
                    return "", ""
                
                # Fast transcription with minimal parameters
                try:
                    segments, _ = model.transcribe(
                        wav_file.name,
                        language="en" if source_lang == "English" else "ja",
                        task=whisper_task,
                        beam_size=1,  # Fastest setting
                        temperature=0.0,
                        condition_on_previous_text=False,
                    )
                    
                    # Extract text quickly
                    src_text = " ".join([segment.text for segment in segments]).strip()
                    
                    if not src_text:
                        return "", ""
                    
                    print(f"ASR ({direction}): {src_text}")
                    
                    # Skip basic filtering
                    if len(src_text) < 3:  # Too short
                        return "", ""
                    
                    # Fast translation
                    if direction == "en-ja":
                        translated = await translate_text(src_text, source_lang, target_lang)
                    else:
                        translated = src_text
                    
                    return src_text, translated
                    
                finally:
                    # Cleanup
                    try:
                        os.unlink(raw_file.name)
                        os.unlink(wav_file.name)
                    except:
                        pass
                        
    except Exception as e:
        print(f"Fast processing error: {e}")
        return "", ""


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected")

    try:
        settings = await websocket.receive_text()
        client_config = json.loads(settings)
        direction = client_config.get("direction")

        if direction == "en-ja":
            source_lang, target_lang = "English", "Japanese"
            whisper_task = "transcribe"
        elif direction == "ja-en":
            source_lang, target_lang = "Japanese", "English"
            whisper_task = "translate"
        else:
            await websocket.close()
            return

        print(f"Processing {direction} translation")

        # Simple direct processing for speed
        while True:
            try:
                msg = await websocket.receive()
                if "bytes" not in msg:
                    continue

                audio = msg["bytes"]
                if not audio or len(audio) < 1000:  # Skip very small chunks
                    continue

                # Process directly without queue overhead
                src_text, translated = await process_audio_chunk_fast(
                    audio, source_lang, target_lang, whisper_task, direction
                )
                
                if src_text and translated:
                    # Update context quickly
                    recent_src_segments.append(src_text)
                    if len(recent_src_segments) > config.MAX_SRC_CTX * 3:
                        recent_src_segments.pop(0)

                    recent_targets.append(translated)
                    if len(recent_targets) > config.MAX_RECENT:
                        recent_targets.pop(0)

                    # Send immediately
                    segment_id = str(uuid.uuid4())
                    await websocket.send_text(f"[DONE]{json.dumps({'id': segment_id, 'text': translated})}")
                        
            except Exception as e:
                print(f"Processing error: {e}")
                continue  # Keep going on individual errors
            
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)