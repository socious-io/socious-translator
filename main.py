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

# Load whisper model with configuration
model = WhisperModel(config.WHISPER_MODEL_SIZE, device=config.WHISPER_DEVICE, compute_type=config.WHISPER_COMPUTE_TYPE)

# Try to use GPU if available and configured
if config.WHISPER_DEVICE == "cpu":
    try:
        gpu_model = WhisperModel(config.WHISPER_MODEL_SIZE, device="cuda", compute_type="float16")
        # Test GPU model
        test_segments, _ = gpu_model.transcribe("/tmp/warm.wav", language="en")
        model = gpu_model
        print("Using GPU acceleration")
    except:
        print("Using CPU - GPU not available")
else:
    print(f"Using {config.WHISPER_DEVICE} with {config.WHISPER_COMPUTE_TYPE}")

# Warm up the model
try:
    segments, _ = model.transcribe("/tmp/warm.wav", language="en")
except:
    pass

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


# tiny context buffers for current-line translation only
recent_src_segments: list[str] = []
recent_targets: list[str] = []
MAX_SRC_CTX = 2
MAX_RECENT = 10

# Audio processing configuration
MIN_CHUNK_DURATION = 0.5  # seconds
OVERLAP_DURATION = 1.0     # seconds
SAMPLE_RATE = 16000

# Translation cache
translation_cache = {}
CACHE_MAX_SIZE = 1000

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
            return energy > 1000
        except Exception:
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
    
    source_context = " ".join(recent_src_segments[-config.MAX_SRC_CTX:])
    recent_target_str = "\n".join(recent_targets[-config.MAX_RECENT:])

    system = "Translate live ASR segments into natural, idiomatic target-language captions. Return ONLY the translation text."
    user = f"""
You are translating a single ASR segment from English → Japanese.

<goal>
Produce fluent, idiomatic {target_lang} for THIS single ASR segment only. Preserve original word strength, tone, and register. Use context only to resolve ambiguous pronouns or cut-offs.
</goal>

<context_use>
- Use <source_context> only for ambiguity in pronouns/ellipses/cut-offs.
- Do not merge or restate earlier material beyond what’s needed for this segment.
- If input repeats a clause from <source_context>, keep it once in the cleanest form.
- Do not re-translate content already fully covered in <recent_target> unless new info is added.
</context_use>

<priorities>
1) Fidelity first — preserve meaning and *emotional strength*.
2) No intensity shifts (e.g., “annoyed” ≠ 「激怒」; “kind of good” ≠ 「最高」).
3) Don’t invent filler, asides, or emphasis not in the source.
4) Match register/politeness exactly (casual/neutral/formal).
5) Overlap handling — remove duplicated words; don’t paraphrase earlier material unless literally repeated here.
6) Numbers/units/symbols/proper names — keep digits/symbols; use established JP renderings or one consistent form.
7) Describe nonverbal events only if explicitly audible/mentioned.
8) If input is already {target_lang}, return unchanged.
9) Labels/titles/meta — translate as labels/titles; don’t expand to full sentences.
10) Preserve grammatical person/mood; don’t add/soften/strengthen imperatives or politeness.
11) Drop standalone low-content interjections (“uh/erm”); keep one token max if needed (e.g., 「えっと」).
12) Consistency — keep proper-noun spellings consistent across the session.
13) Redundancy — if fully covered in <recent_target> with no new detail, output nothing.
</priorities>

<style_targets>
- 明確・簡潔。
- 自然だが、意味・トーン・強度は一切変更しない。
- 脚色・強調・コメントは加えない。
</style_targets>

<source_context>
{source_context}
</source_context>

<recent_target>
{recent_target_str}
</recent_target>

<examples_positive>
<input>Please double-check the numbers.</input>
<output>数値を再確認してください。</output>

<input>It’s kind of slow—nothing on screen yet.</input>
<output>少し遅いですね。まだ画面に何も表示されていません。</output>

<input>We may roll it back if needed.</input>
<output>必要であればロールバックする可能性があります。</output>
</examples_positive>

<examples_boundary>
<input>Could you fix it?</input>
<bad_output>すぐに直してください！</bad_output>
<why>Intensity was upgraded.</why>
<good_output>修正してもらえますか。</good_output>

<input>Do it.</input>
<bad_output>やっていただけますか。</bad_output>
<why>Imperative was softened.</why>
<good_output>やれ。</good_output>
</examples_boundary>

<target_lang>日本語</target_lang>

<input>
{text}
</input>

""".strip()

    try:
        resp = await client.chat.completions.create(
            model=config.TRANSLATION_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_completion_tokens=128
        )
        translation = (resp.choices[0].message.content or "").strip()
        # Cache the result if caching is enabled
        if config.ENABLE_CACHING:
            cache_translation(text, source_lang, target_lang, translation)
        return translation
    except Exception as e:
        print("Translation error:", e)
        return ""

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join("frontend", "index.html"))

async def process_audio_chunk(audio_data: bytes, source_lang: str, target_lang: str, whisper_task: str, direction: str) -> tuple[str, str]:
    """Process a single audio chunk asynchronously."""
    try:
        # Convert audio to WAV in memory
        wav_data = convert_audio_in_memory(audio_data)
        if not wav_data:
            return "", ""

        # Check minimum duration
        duration = get_audio_duration(wav_data)
        if duration < config.MIN_CHUNK_DURATION:
            return "", ""

        # Check for speech activity
        if not has_speech(wav_data):
            return "", ""

        # Save to temporary file for Whisper (faster-whisper needs file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_file:
            wav_file.write(wav_data)
            wav_file.flush()
            
            try:
                # Transcribe with faster-whisper
                segments, info = model.transcribe(
                    wav_file.name,
                    temperature=0.0,
                    beam_size=config.BEAM_SIZE,
                    condition_on_previous_text=False,
                    hallucination_silence_threshold=0.30,
                    no_speech_threshold=0.6,
                    language="en" if source_lang == "English" else "ja",
                    compression_ratio_threshold=1.7,
                    logprob_threshold=-0.3,
                    task=whisper_task,
                    vad_filter=config.ENABLE_VAD_FILTER,
                    vad_parameters=dict(min_silence_duration_ms=config.MIN_SILENCE_DURATION_MS)
                )
                
                # Extract text from segments
                src_text = " ".join([segment.text for segment in segments]).strip()
                
                if not src_text:
                    return "", ""
                
                print("ASR:", src_text)
                
                # Skip unwanted content
                if is_interjection_thanks(src_text) or is_cta_like(src_text):
                    return "", ""
                
                # Translate if needed
                if direction == "en-ja":
                    translated = await translate_text(src_text, source_lang, target_lang)
                else:
                    translated = src_text  # Whisper already gave English
                
                # Skip unwanted translations
                if is_interjection_thanks(translated) or is_cta_like(translated):
                    return "", ""
                
                translated = translated.strip()
                if not translated or not re.search(r'[A-Za-z0-9ぁ-んァ-ン一-龯々ー]', translated):
                    return "", ""
                
                return src_text, translated
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(wav_file.name)
                except:
                    pass
                    
    except Exception as e:
        print(f"Audio processing error: {e}")
        return "", ""


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected")

    # Processing queue for async pipeline
    processing_queue = asyncio.Queue(maxsize=config.MAX_QUEUE_SIZE)
    results_queue = asyncio.Queue()
    
    async def audio_processor():
        """Background task to process audio chunks."""
        while True:
            try:
                audio_data, source_lang, target_lang, whisper_task, direction = await processing_queue.get()
                src_text, translated = await process_audio_chunk(
                    audio_data, source_lang, target_lang, whisper_task, direction
                )
                
                if src_text and translated:
                    await results_queue.put((src_text, translated))
                
                processing_queue.task_done()
            except Exception as e:
                print(f"Audio processor error: {e}")

    async def result_sender():
        """Background task to send results."""
        while True:
            try:
                src_text, translated = await results_queue.get()
                
                # Update context buffers
                recent_src_segments.append(src_text)
                if len(recent_src_segments) > config.MAX_SRC_CTX * 3:
                    recent_src_segments.pop(0)

                recent_targets.append(translated)
                if len(recent_targets) > config.MAX_RECENT:
                    recent_targets.pop(0)

                segment_id = str(uuid.uuid4())
                await websocket.send_text(f"[DONE]{json.dumps({'id': segment_id, 'text': translated})}")
                
                results_queue.task_done()
            except Exception as e:
                print(f"Result sender error: {e}")
                break

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

        # Start background tasks
        processor_task = asyncio.create_task(audio_processor())
        sender_task = asyncio.create_task(result_sender())

        try:
            while True:
                msg = await websocket.receive()
                if "bytes" not in msg:
                    continue

                audio = msg["bytes"]
                if not audio:
                    continue

                # Add to processing queue (non-blocking)
                try:
                    processing_queue.put_nowait((audio, source_lang, target_lang, whisper_task, direction))
                except asyncio.QueueFull:
                    print("Processing queue full, dropping audio chunk")
                    
        finally:
            processor_task.cancel()
            sender_task.cancel()
            
    except Exception as e:
        print("WebSocket error:", e)
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)