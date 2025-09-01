# main.py
# - Browser sends raw 16 kHz Int16 PCM frames over WS
# - Rolling window decode; finalize on brief pause
# - Whisper settings + filters + prompt remain the same

import os, json, re, uuid, time, contextlib, asyncio, subprocess
from collections import deque

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles  # serve /static for the worklet

import webrtcvad
import whisper
import torch
import openai
import uvicorn

# ---------- setup ----------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = openai.AsyncOpenAI(api_key=api_key)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Serve the UI at /
@app.get("/")
async def serve_index():
    return FileResponse(os.path.join("frontend", "index.html"))

# Serve the worklet at /static/pcm-capture.worklet.js
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Pre-warm ffmpeg only to generate a tiny wav for Whisper warmup 
try:
    subprocess.run(
        ["ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "0.5", "-ar", "16000", "-ac", "1", "-y", "/tmp/warm.wav"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
except:
    pass

# Load Whisper once (large-v3), tiny warmup 
torch.set_num_threads(1)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("large-v3", device=DEVICE)
try:
    model.transcribe("/tmp/warm.wav", language="en", fp16=True)
except:
    pass

# ---------- tiny live context  ----------
recent_src_segments: list[str] = []
recent_targets: list[str] = []
MAX_SRC_CTX = 2
MAX_RECENT = 10

# ---------- filters  ----------
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
    r"(?i)\bmy\s+(?:instagram|tiktok|twitter|x|twitch|youtube|facebook|threads)\s+is\b",
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
    # --- Captions/credits  ---
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

# ---------- translation prompt ----------
async def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    source_context = " ".join(recent_src_segments[-MAX_SRC_CTX:])
    recent_target_str = "\n".join(recent_targets[-MAX_RECENT:])

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
            model="gpt-5",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            reasoning_effort="minimal",
            max_completion_tokens=128
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print("Translation error:", e)
        return ""

# ---------- streaming constants ----------
SAMPLE_RATE = 16000
FRAME_BYTES = 320 * 2              # 20ms @16k mono, int16 → 640 bytes
TICK_MS = 200
WIN_MS  = 1000
HANG_MS = 300

# ---------- websocket ----------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected")

    # first message is the direction
    cfg = json.loads(await websocket.receive_text())
    direction = cfg.get("direction")

    if direction == "en-ja":
        source_lang, target_lang = "English", "Japanese"
        whisper_task = "transcribe"
        lang_code = "en"
    elif direction == "ja-en":
        source_lang, target_lang = "Japanese", "English"
        whisper_task = "translate"
        lang_code = "ja"
    else:
        await websocket.close()
        return

    vad = webrtcvad.Vad(2)

    ring = bytearray()
    last_partial = ""
    hang_ms = 0

    win_bytes  = int(WIN_MS / 1000.0 * SAMPLE_RATE) * 2
    ring_max   = win_bytes * 3              # cap buffer to ~3s so slices stay cheap
    tail_bytes = FRAME_BYTES * 5            # 100ms for quick VAD

    # --- instrumentation ---
    last_bytes_log = 0
    bytes_since_log = 0
    last_audio_ts = time.time()

    async def recv_loop():
        nonlocal ring, bytes_since_log, last_audio_ts
        while True:
            msg = await websocket.receive()
            if "bytes" in msg and msg["bytes"]:
                b = msg["bytes"]
                ring.extend(b)
                bytes_since_log += len(b)
                last_audio_ts = time.time()
                # keep buffer bounded
                if len(ring) > ring_max:
                    del ring[:len(ring) - ring_max]
            # ignore control pings etc.

    recv_task = asyncio.create_task(recv_loop())

    try:
        while True:
            await asyncio.sleep(TICK_MS / 1000.0)

            # periodic byte counter so you can see audio flow
            now = time.time()
            if now - last_bytes_log >= 0.5:
                if bytes_since_log:
                    print(f"WS bytes: +{bytes_since_log} (ring={len(ring)} bytes, ~{len(ring)/2/SAMPLE_RATE*1000:.0f} ms)")
                    bytes_since_log = 0
                if now - last_audio_ts > 2.0:
                    print("⚠️  No audio arriving for >2s (check mic/worklet/HTTPS).")
                    last_audio_ts = now  # avoid spamming
                last_bytes_log = now

            if len(ring) < win_bytes:
                continue

            # zero-copy slice of the last window
            window_mv = memoryview(ring)[-win_bytes:]
            pcm = np.frombuffer(window_mv, dtype=np.int16).astype(np.float32) / 32768.0

            # run Whisper in a worker thread so the event loop stays responsive
            loop = asyncio.get_running_loop()
            def _run():
                return model.transcribe(
                    pcm,
                    fp16=True,
                    temperature=0.0,
                    beam_size=5,
                    condition_on_previous_text=False,
                    hallucination_silence_threshold=0.30,
                    no_speech_threshold=0.6,
                    language=lang_code,
                    compression_ratio_threshold=1.7,
                    logprob_threshold=-0.3,
                    task=whisper_task,
                )
            result = await loop.run_in_executor(None, _run)
            partial = (result.get("text") or "").strip()

            if partial and partial != last_partial:
                print(f"ASR(partial): {partial}")
                await websocket.send_text(f"[UPDATE]{json.dumps({'id':'live','text': partial})}")
                last_partial = partial

            # quick VAD over the most recent ~100ms (5x20ms)
            tail_mv = memoryview(ring)[-tail_bytes:]
            speech = False
            for i in range(0, len(tail_mv), FRAME_BYTES):
                frame_mv = tail_mv[i:i+FRAME_BYTES]
                if len(frame_mv) == FRAME_BYTES and vad.is_speech(frame_mv.tobytes(), SAMPLE_RATE):
                    speech = True
                    break

            if speech:
                hang_ms = 0
            else:
                hang_ms += TICK_MS
                if hang_ms >= HANG_MS and last_partial:
                    seg_id = str(uuid.uuid4())
                    src_text = last_partial.strip()
                    last_partial = ""
                    hang_ms = 0

                    # toss low-value stuff early
                    if is_interjection_thanks(src_text) or is_cta_like(src_text):
                        continue

                    print(f"ASR(final): {src_text}")

                    if direction == "en-ja":
                        async def _translate_and_send():
                            translated = await translate_text(src_text, source_lang, target_lang)
                            if translated and not is_interjection_thanks(translated) and not is_cta_like(translated):
                                print(f"Translation: {translated}")
                                recent_src_segments.append(src_text)
                                if len(recent_src_segments) > MAX_SRC_CTX * 3:
                                    recent_src_segments.pop(0)
                                recent_targets.append(translated)
                                if len(recent_targets) > MAX_RECENT:
                                    recent_targets.pop(0)
                                await websocket.send_text(f"[DONE]{json.dumps({'id': seg_id, 'text': translated})}")
                        asyncio.create_task(_translate_and_send())
                    else:
                        # ja→en uses Whisper translate output as final
                        translated = src_text
                        if translated and not is_interjection_thanks(translated) and not is_cta_like(translated):
                            print(f"Translation: {translated}")
                            recent_src_segments.append(src_text)
                            if len(recent_src_segments) > MAX_SRC_CTX * 3:
                                recent_src_segments.pop(0)
                            recent_targets.append(translated)
                            if len(recent_targets) > MAX_RECENT:
                                recent_targets.pop(0)
                            await websocket.send_text(f"[DONE]{json.dumps({'id': seg_id, 'text': translated})}")

    except Exception as e:
        print("WebSocket error:", e)
    finally:
        with contextlib.suppress(Exception):
            recv_task.cancel()
        with contextlib.suppress(Exception):
            await websocket.close()
        print("WebSocket closed")

# ---------- dev entry ----------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)