import tempfile, subprocess, uvicorn, openai, whisper, os, json, uuid, re
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

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

# load whisper once
model = whisper.load_model("large-v3")
try:
    model.transcribe("/tmp/warm.wav", language="en", fp16=True)
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

# exact short interjections only (and standalone "you")
THANKS_RE = re.compile(
    r'^\s*(?:thank\s*you|thank\s*you\s*so\s*much|thanks|thx|you)\s*[!.…]*\s*$',
    re.IGNORECASE
)

def is_interjection_thanks(text: str) -> bool:
    if not text:
        return False
    return bool(THANKS_RE.match(text.strip()))

_CTA_PATTERNS = [
    r'(?i)\blike\s*(?:and\s*)?subscribe\b',
    r'(?i)\bsubscribe\s*(?:and\s*)?like\b',
    r'(?i)\bshare\s+(?:this|the)\s+(?:video|stream|clip|content)\b',
    r'(?i)\bplease\s+share\b',
    r'(?i)\bhit\s+(?:the\s+)?bell\b',
    r'(?i)\bturn\s+on\s+notifications?\b',
    r'(?i)\bdon\'?t\s+forget\s+to\s+turn\s+on\s+notifications?\b',
    r'(?i)\blink\s+in\s+(?:the\s+)?(?:bio|description)\b',
    r'(?i)\bcheck\s+(?:the\s+)?link\s+(?:below|above)\b',
    r'(?i)\bsee\s+you\s+(?:next\s*time|tomorrow|soon|in\s+the\s+next)\b',
    r'(?i)\bthanks?\s+for\s+watching\b',
    r'(?i)\bthank\s+you\s+for\s+watching\b',
    r'(?i)\bthank\s+you\s+so\s+much\s+for\s+watching\b',
    r"(?i)\bthank\s+you\s+y['’]?all\b",
    r"(?i)\bthanks?\s+y['’]?all\b",
    r'(?i)\bthat\'?s\s+(?:it|all)\s+for\s+(?:today|now)\b',
    r'(?i)\bthat\'?s\s+(?:it|all)\b',
    r'(?i)\bsmash\s+(?:that\s+)?like\b',
    r'(?i)\bsubtitles?\s+by\b',
    r'(?i)\bcaptions?\s+by\b',
    r'(?i)\bsubtitled\s+by\b',
    r'(?i)\bclosed\s+captions?\s+by\b',
    r'(?i)\bamara\.org\b',
    r'(?i)\btranscription\s+by\b'
]

def is_cta_like(text: str) -> bool:
    if not text or len(text.strip()) < 2:
        return False
    for pat in _CTA_PATTERNS:
        if re.search(pat, text):
            return True
    return False

async def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    source_context = " ".join(recent_src_segments[-MAX_SRC_CTX:])
    recent_target_str = "\n".join(recent_targets[-MAX_RECENT:])

    system = "Translate live ASR segments into natural, idiomatic target-language captions. Return ONLY the translation text."
    user = f"""
You are translating a single ASR segment from English → Japanese.

<goal>
Produce fluent, idiomatic {target_lang} for THIS single ASR segment exactly as spoken. Preserve original word strength, tone, and register without upgrading or downgrading. Use context only to resolve ambiguous pronouns or incomplete words.
</goal>

<context_use>
- Refer to <source_context> only to resolve ambiguity in pronouns, ellipses, or cut-off words. Do not merge, rewrite, or restate the current input beyond what is necessary to produce a faithful translation of *this segment alone*.
- Do not re-translate content already fully covered in <recent_target> unless the new input adds substantive information.
- If the current input repeats a clause from <source_context>, keep it once in the cleanest form.
</context_use>

<priorities>
1) Fidelity above all — Translate exactly what is said, preserving both meaning and *emotional strength* of each word/phrase.
2) No upgrades/downgrades in intensity — keep subjective strength identical. (e.g., “annoyed” ≠ 「激怒」, “kind of good” ≠ 「最高」)
3) Do NOT invent filler, rhetorical asides, or emphasis markers unless explicitly present in the source.
4) Maintain register/tone — mirror the source (casual ↔ casual; neutral/polite ↔ です/ます; formal/deferential ↔ 丁寧/敬語) without escalation.
5) Overlap handling — if the input overlaps with <source_context> or <recent_target>, remove duplicated words; do not paraphrase earlier material unless literally repeated here.
6) Numbers/units/symbols/proper names — keep digits and symbols as-is; preserve proper names exactly as heard. Use standard Japanese renderings when widely established; otherwise keep the original form (Latin) or katakana once, then stay consistent.
7) Only describe nonverbal events if explicitly audible/mentioned; do not invent them.
8) If the input is already {target_lang}, return it unchanged.
9) Labels/titles/meta — translate as labels/titles; do not expand into full sentences.
10) Preserve grammatical person and mood — don’t turn declaratives into imperatives or add politeness that isn’t in the source.
11) Drop standalone low-content interjections (“uh,” “erm”) unless they convey hesitation/stance needed for meaning; keep one token max if needed (e.g., 「えっと」).
12) Consistency — maintain the same spelling/form for each proper noun across the session.
13) Redundancy filter — if the current line restates content already fully covered in <recent_target> with no new detail, output nothing.
</priorities>

<en_to_ja_specifics>
- Subject omission: omit subjects when natural in Japanese unless clarity requires them. Prefer phrasing that avoids inventing gendered pronouns.
- Hedges/modality: map softly and proportionally ( “maybe/kinda/sort of” → 「たぶん／やや／少し」, “I think” → 「と思います／と思う」 ) without strengthening.
- Requests/imperatives: preserve mood precisely.
  * “Please … / Could you … ?” → 「…してください／…してもらえますか」 (choose the minimal form matching source politeness).
  * Bare imperatives (“Do it.”) → imperative/plain without softening.
- Sentence fragments & repairs: keep fragments if the source is fragmentary; don’t force full sentences. Use 「—」 or 「…」 sparingly to reflect real hesitation or cut-offs.
- Connectives: split long English chains into short, natural Japanese clauses without adding new content.
- Terminology: use established Japanese technical terms; avoid awkward katakana calques where a standard term exists (e.g., “latency” → 「レイテンシ」 if domain-standard; otherwise 「遅延」). Stay consistent.
- Punctuation/width: prefer Japanese punctuation （、。）; keep ASCII for code, URLs, file paths, CLI flags, and units. Do not add decorative punctuation or extra exclamation marks.
- Counters: if unavoidable for grammaticality, choose the most neutral counter; otherwise rephrase to avoid guessing.
</en_to_ja_specifics>

<style_targets>
- 明確で簡潔。
- 自然だが、意味・トーン・強度を一切変更しない。
- 追加の脚色・強調・コメントはしない。
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

<input>Agenda — Thursday</input>
<output>アジェンダ — 木曜日</output>

<input>They arrived late because because the train was delayed.</input>
<output>電車の遅延で到着が遅れました。</output>
</examples_positive>

<examples_boundary>
<input>Could you fix it?</input>
<bad_output>すぐに直してください！</bad_output>
<why>Intensity was upgraded (“please” → strong imperative with emphasis).</why>
<good_output>修正してもらえますか。</good_output>

<input>Do it.</input>
<bad_output>やっていただけますか。</bad_output>
<why>Imperative was softened.</why>
<good_output>やれ。</good_output>

<input>It’s okay, kind of.</input>
<bad_output>とても大丈夫です。</bad_output>
<why>Strength was upgraded.</why>
<good_output>まあ大丈夫です。</good_output>
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
            max_completion_tokens=140
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print("Translation error:", e)
        return ""

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join("frontend", "index.html"))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected")

    try:
        settings = await websocket.receive_text()
        config = json.loads(settings)
        direction = config.get("direction")

        if direction == "en-ja":
            source_lang, target_lang = "English", "Japanese"
            whisper_task = "transcribe"
        elif direction == "ja-en":
            source_lang, target_lang = "Japanese", "English"
            whisper_task = "translate"
        else:
            await websocket.close()
            return

        while True:
            msg = await websocket.receive()
            if "bytes" not in msg:
                continue

            audio = msg["bytes"]
            if not audio:
                continue

            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as raw:
                raw.write(audio)
                raw.flush()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav:
                    try:
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", raw.name, "-ar", "16000", "-ac", "1", wav.name],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True
                        )
                    except:
                        continue

                    result = model.transcribe(
                        wav.name,
                        fp16=True,
                        temperature=0.0,
                        beam_size=6,
                        condition_on_previous_text=False,
                        hallucination_silence_threshold=0.30,
                        no_speech_threshold=0.6,
                        language="en" if source_lang == "English" else "ja",
                        compression_ratio_threshold=2.0,
                        logprob_threshold=-1.2,
                        task=whisper_task,
                    )

                    src_text = (result.get("text") or "").strip()
                    if not src_text:
                        continue
                    print("ASR:", src_text)

                    if is_interjection_thanks(src_text):
                        print("Skipped short thank-you interjection (source).")
                        continue
                    if is_cta_like(src_text):
                        print("Dropped CTA/meta filler (source).")
                        continue

                    segment_id = str(uuid.uuid4())

                    if direction == "en-ja":
                        translated = await translate_text(src_text, source_lang, target_lang)
                    else:
                        translated = src_text  # Whisper already gave English

                    if is_interjection_thanks(translated):
                        print("Skipped short thank-you interjection (target).")
                        continue
                    if is_cta_like(translated):
                        print("Dropped CTA/meta filler (target).")
                        continue

                    translated = translated.strip()
                    if not translated or not re.search(r'[A-Za-z0-9ぁ-んァ-ン一-龯々ー]', translated):
                        print("Suppressed empty/no-op output.")
                        continue

                    # update tiny context buffers only when emitting
                    recent_src_segments.append(src_text)
                    if len(recent_src_segments) > MAX_SRC_CTX * 3:
                        recent_src_segments.pop(0)

                    recent_targets.append(translated)
                    if len(recent_targets) > MAX_RECENT:
                        recent_targets.pop(0)

                    await websocket.send_text(f"[DONE]{json.dumps({'id': segment_id, 'text': translated})}")

    except Exception as e:
        print("WebSocket error:", e)
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
