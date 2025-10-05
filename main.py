# --------- VERY TOP: keep memory low on small instances ----------
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# Optional persistent caches (safe to keep even if no disk mounted)
os.environ.setdefault("INSIGHTFACE_HOME", "/data/.insightface")
os.environ.setdefault("HF_HOME", "/data/.cache/huggingface")
# ---------------------------------------------------------------

import re, io, json, time, base64, hashlib, asyncio, logging, shutil
from pathlib import Path
from typing import Any, Iterable, Tuple, Dict, Optional, List
from contextlib import contextmanager

import requests
import numpy as np
import cv2
import replicate
from PIL import Image

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes,
    CallbackQueryHandler, filters
)

# ======================
# ENV / FLAGS
# ======================
TOKEN = os.getenv("BOT_TOKEN")
os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN") or ""
if not TOKEN or not re.match(r"^\d+:[A-Za-z0-9_-]{20,}$", TOKEN):
    raise RuntimeError("Некорректный BOT_TOKEN.")
if not os.getenv("REPLICATE_API_TOKEN"):
    raise RuntimeError("Нет REPLICATE_API_TOKEN.")

# Режимы
FAST_DEBUG = os.getenv("FAST_DEBUG", "1") == "1"           # быстрый режим по умолчанию
USE_INSIGHTFACE = os.getenv("USE_INSIGHTFACE", "0") == "1" # по умолчанию выкл., чтобы билдился мгновенно

# Параметры под режимы
INFERENCE_STEPS_FAST = int(os.getenv("INFERENCE_STEPS_FAST", "12"))
INFERENCE_STEPS_FULL = int(os.getenv("INFERENCE_STEPS_FULL", "18"))
TOPK_REFS_FAST = int(os.getenv("TOPK_REFS_FAST", "1"))
TOPK_REFS_FULL = int(os.getenv("TOPK_REFS_FULL", "3"))

# Replicate модели
STYLE_BACKEND = "instantid"
INSTANTID_MODEL = os.getenv("INSTANTID_MODEL", "grandlineai/instant-id-photorealistic")
QWEN_EDIT_MODEL = os.getenv("QWEN_EDIT_MODEL", "qwen/qwen-image-edit-plus")

# «ультра-похожесть»
ULTRA_LOCK_STRENGTH = float(os.getenv("ULTRA_LOCK_STRENGTH") or 0.14)   # 0.12–0.16
ULTRA_LOCK_GUIDANCE = float(os.getenv("ULTRA_LOCK_GUIDANCE") or 2.3)

NEGATIVE_PROMPT = (
    "cartoon, anime, cgi, 3d, plastic skin, waxy skin, porcelain, airbrushed, beauty filter, smoothing, "
    "overprocessed, oversharpen, hdr effect, halo, neon skin, garish, fake skin, cosplay wig, doll, "
    "ai-artifacts, deformed, bad anatomy, watermark, text, logo, "
    "warped face, distorted face, changed facial proportions, geometry change, face reshape, exaggerated makeup, "
    "enlarged nose, wider nose, altered nose shape, flat hair, missing hair volume, bluish nose"
)
AESTHETIC_SUFFIX = (
    ", natural healthy skin, preserved pores, balanced contrast, soft realistic light, "
    "no beauty filter, no plastic look"
)

STYLE_PRESETS: Dict[str, str] = {
    "natural":     "ultra realistic portrait, real skin texture, subtle makeup, neutral grading",
    "boho":        "boho portrait, earthy palette, soft daylight",
    "vogue":       "beauty cover shot, soft studio light, calibrated colors, photographic grain",
    "beauty_soft": "beauty portrait, clean studio light, controlled highlights, visible pores",
    "windowlight": "soft window light portrait, natural diffusion, balanced exposure",
    "editorial":   "editorial fashion portrait, preserved natural imperfections, pro color grading",
    "moody":       "moody cinematic portrait, controlled shadows, subtle rim light",
}

# ======================
# LOGGING + TIMERS
# ======================
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("bot")

@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = (time.perf_counter() - t0) * 1000
        logger.info("[TIMER] %s: %.1f ms", label, dt)

# ======================
# PROFILES STORAGE
# ======================
DATA_DIR = Path("profiles"); DATA_DIR.mkdir(exist_ok=True)

def user_dir(uid:int) -> Path:
    p = DATA_DIR / str(uid); p.mkdir(parents=True, exist_ok=True); return p

def list_ref_images(uid:int) -> List[Path]:
    return sorted((user_dir(uid)).glob("ref_*.jpg"))

def profile_path(uid:int) -> Path:
    return user_dir(uid) / "profile.json"

def save_profile(uid:int, prof:Dict[str,Any]):
    profile_path(uid).write_text(json.dumps(prof))

def load_profile(uid:int) -> Dict[str, Any]:
    p = profile_path(uid)
    if p.exists():
        return json.loads(p.read_text())
    return {"embeds": [], "images": []}

# Даунскейл перед сохранением (экономия RAM/скорости)
def save_ref_downscaled(path: Path, b: bytes, max_side=1024, quality=92):
    im = Image.open(io.BytesIO(b)).convert("RGB")
    im.thumbnail((max_side, max_side))
    im.save(path, "JPEG", quality=quality)

# ======================
# (OPTIONAL) InsightFace – грузим только если включен флаг
# ======================
_FACE_APP = None
_ARCFACE = None

def _init_face_models():
    if not USE_INSIGHTFACE:
        return
    global _FACE_APP, _ARCFACE
    if _FACE_APP is None:
        from insightface.app import FaceAnalysis
        _FACE_APP = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
        _FACE_APP.prepare(ctx_id=0, det_size=(320, 320))
    if _ARCFACE is None:
        from insightface.model_zoo import get_model
        _ARCFACE = get_model("w600k_r50.onnx", download=True, providers=["CPUExecutionProvider"])

def face_embed_from_bytes(b: bytes) -> Optional[np.ndarray]:
    if not USE_INSIGHTFACE:
        return None
    _init_face_models()
    img = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    if img is None: return None
    faces = _FACE_APP.get(img)
    if not faces: return None
    face = max(faces, key=lambda f: f.det_score)
    emb = _ARCFACE.get(img, face.bbox.astype(int))
    if emb is None: return None
    emb = emb.astype(np.float32)
    emb /= (np.linalg.norm(emb) + 1e-8)
    return emb

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8))

def build_or_update_profile(uid:int):
    """Пересчёт эмбеддингов, если включён InsightFace; иначе – только список файлов."""
    refs = list_ref_images(uid)
    prof = {"embeds": [], "images": []}
    if USE_INSIGHTFACE:
        for fp in refs:
            try:
                emb = face_embed_from_bytes(fp.read_bytes())
                if emb is not None:
                    prof["embeds"].append(emb.tolist())
                    prof["images"].append(fp.name)
            except Exception as e:
                logger.warning("Embed failed for %s: %s", fp, e)
    else:
        prof["images"] = [fp.name for fp in refs]
    save_profile(uid, prof)
    return prof

def best_refs_for(uid:int, topk:int) -> List[Path]:
    refs = list_ref_images(uid)
    if not USE_INSIGHTFACE:
        return refs[:topk]
    prof = load_profile(uid)
    if not prof["embeds"] or not refs:
        return refs[:topk]
    centroid = np.mean(np.array(prof["embeds"], dtype=np.float32), axis=0)
    scores = []
    for name, emb_list in zip(prof["images"], prof["embeds"]):
        emb = np.array(emb_list, dtype=np.float32)
        scores.append((cosine(centroid, emb), name))
    scores.sort(reverse=True)
    return [user_dir(uid)/name for _, name in scores[:topk]]

# ======================
# REPLICATE UTILS
# ======================
def resolve_model_version(slug: str) -> str:
    if ":" in slug: return slug
    model = replicate.models.get(slug)
    versions = list(model.versions.list())
    if not versions: raise RuntimeError(f"Нет версий модели {slug}")
    return f"{slug}:{versions[0].id}"

def extract_any_url(output: Any) -> Optional[str]:
    if isinstance(output, str) and output.startswith(("http", "https")):
        return output
    if isinstance(output, list):
        for v in output:
            u = extract_any_url(v)
            if u: return u
    if isinstance(output, dict):
        for v in output.values():
            u = extract_any_url(v)
            if u: return u
    return None

def replicate_run_flexible(model: str, inputs_list: Iterable[dict]) -> str:
    last = None
    for payload in inputs_list:
        try:
            out = replicate.run(model, input=payload)
            url = extract_any_url(out)
            if not url: raise RuntimeError("Empty output")
            return url
        except Exception as e:
            last = e
            logger.warning("Replicate payload rejected: %s", e)
    raise last or RuntimeError("Все варианты отклонены")

def _seed_from(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16)

def _data_url_from_path(p: Path) -> str:
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return "data:image/jpeg;base64," + b64

def instantid_from_profile(base_ref_data_url:str, face_ref_data_url:str,
                           prompt:str, denoise:float, guidance:float, seed:int) -> Tuple[str,str]:
    resolved = resolve_model_version(INSTANTID_MODEL)
    steps = INFERENCE_STEPS_FAST if FAST_DEBUG else INFERENCE_STEPS_FULL
    payload = {
        "face_image": face_ref_data_url,
        "input_image": face_ref_data_url,
        "input_image_ref": face_ref_data_url,
        "image": base_ref_data_url,
        "prompt": prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "ip_adapter_scale": 1.0,
        "controlnet_conditioning_scale": 0.18,
        "strength": denoise,
        "guidance_scale": guidance,
        "num_inference_steps": steps,
        "seed": seed,
    }
    with timed("replicate run"):
        url = replicate_run_flexible(resolved, [payload])
    return url, resolved

def pick_most_similar(candidate_urls: List[str], centroid: Optional[np.ndarray]) -> str:
    if not candidate_urls or centroid is None:
        return candidate_urls[0]
    best_u, best_s = candidate_urls[0], -1.0
    for u in candidate_urls:
        try:
            b = requests.get(u, timeout=90).content
            emb = face_embed_from_bytes(b)
            if emb is None: continue
            s = cosine(centroid, emb)
            if s > best_s: best_s, best_u = s, u
        except Exception:
            continue
    logger.info("Picked best candidate with score=%.4f", best_s)
    return best_u

def render_with_profile(uid:int, preset_prompt:str) -> Tuple[str,str]:
    topk = TOPK_REFS_FAST if FAST_DEBUG else TOPK_REFS_FULL
    refs = best_refs_for(uid, topk)
    if not refs:
        raise RuntimeError("Профиль пуст. Сначала /idenroll и /iddone.")

    with timed("encode base ref"):
        base_data_url = _data_url_from_path(refs[0])

    denoise  = ULTRA_LOCK_STRENGTH
    guidance = ULTRA_LOCK_GUIDANCE
    seed = _seed_from(str(uid) + preset_prompt)
    prompt = (
        "highly realistic portrait, exact facial identity, preserve original facial proportions and features, "
        "keep natural hair volume and hairline, matched skin tone, balanced natural lighting, "
        "no stylization of anatomy, no geometry change, "
        + preset_prompt + AESTHETIC_SUFFIX
    )

    if FAST_DEBUG:
        with timed("instantid fast pass"):
            face_data_url = _data_url_from_path(refs[0])
            u, _ = instantid_from_profile(base_data_url, face_data_url, prompt, denoise, guidance, seed)
            return u, "instantid"

    # full mode: несколько кандидатов
    candidates: List[str] = []
    for face_ref in refs:
        face_data_url = _data_url_from_path(face_ref)
        with timed(f"instantid candidate {face_ref.name}"):
            u, _ = instantid_from_profile(base_data_url, face_data_url, prompt, denoise, guidance, seed)
            candidates.append(u)

    if USE_INSIGHTFACE:
        prof = load_profile(uid)
        centroid = None
        if prof["embeds"]:
            centroid = np.mean(np.array(prof["embeds"], dtype=np.float32), axis=0)
        with timed("pick most similar"):
            best = pick_most_similar(candidates, centroid)
        return best, "instantid"

    return candidates[0], "instantid"

# ======================
# TELEGRAM HANDLERS
# ======================
ENROLL_FLAG: Dict[int,bool] = {}  # user_id -> True если идёт набор

def styles_keyboard() -> InlineKeyboardMarkup:
    rows, row = [], []
    names = list(STYLE_PRESETS.keys())
    for i, name in enumerate(names, 1):
        row.append(InlineKeyboardButton(name.title(), callback_data=f"style:{name}"))
        if i % 3 == 0: rows.append(row); row=[]
    if row: rows.append(row)
    return InlineKeyboardMarkup(rows)

async def tg_download_bytes(message) -> bytes:
    f = await message.photo[-1].get_file()
    buf = await f.download_as_bytearray()
    return bytes(buf)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mode = "FAST" if FAST_DEBUG else "FULL"
    await update.message.reply_text(
        f"Привет! Режим: {mode} (InsightFace={'ON' if USE_INSIGHTFACE else 'OFF'}).\n\n"
        "1) /idenroll — включить набор (пришли подряд до 10 фото)\n"
        "2) /iddone — сохранить профиль (переживает перезапуски)\n"
        "3) /styles — список стилей\n"
        "4) /style <preset> — сгенерировать из сохранённого профиля (новые фото НЕ нужны)\n"
        "Сервисные: /idstatus, /idreset, /fast_on, /fast_off"
    )

async def id_enroll(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    ENROLL_FLAG[uid] = True
    await update.message.reply_text("Набор включён. Пришли до 10 фото. Когда закончишь — /iddone.")

async def id_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    ENROLL_FLAG[uid] = False
    await update.message.reply_text("Сохраняю профиль…")
    prof = await asyncio.to_thread(build_or_update_profile, uid)
    if USE_INSIGHTFACE and not FAST_DEBUG:
        await asyncio.to_thread(_init_face_models)   # прогрев весов
    await update.message.reply_text(f"Готово. В профиле {len(prof['images'])} фото. Теперь /styles → /style <preset>.")

async def id_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    cnt = len(list_ref_images(uid))
    await update.message.reply_text(f"В профиле сохранено {cnt} фото.")

async def id_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    p = user_dir(uid)
    if p.exists(): shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    ENROLL_FLAG[uid] = False
    await update.message.reply_text("Профиль удалён. Запусти /idenroll, чтобы собрать заново.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    m = update.message
    if ENROLL_FLAG.get(uid):
        refs = list_ref_images(uid)
        if len(refs) >= 10:
            await m.reply_text("Уже 10/10. Жми /iddone."); return
        b = await tg_download_bytes(m)
        save_ref_downscaled(user_dir(uid) / f"ref_{int(time.time()*1000)}.jpg", b)
        await m.reply_text(f"Сохранила ({len(refs)+1}/10). Ещё?")
        return
    await m.reply_text("Сначала включи набор: /idenroll. После /iddone используй /style <preset> без новых фото.")

async def styles_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Выбери стиль:", reply_markup=styles_keyboard())

async def callback_style(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    preset = q.data.split(":",1)[1]
    await q.message.reply_text(f"Окей, стиль: {preset}. Теперь просто напиши `/style {preset}`.")

async def style_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    args = context.args
    if not args:
        await update.message.reply_text("Укажи стиль, например: `/style natural`", reply_markup=styles_keyboard()); return
    preset = args[0].lower()
    if preset not in STYLE_PRESETS:
        await update.message.reply_text(f"Не знаю стиль '{preset}'. Смотри /styles"); return
    if not list_ref_images(uid):
        await update.message.reply_text("Профиль пуст. Сначала /idenroll и /iddone."); return

    await update.message.chat.send_action(ChatAction.UPLOAD_PHOTO)
    await update.message.reply_text(f"Стилизация ({'быстрый' if FAST_DEBUG else 'полный'} режим): {preset}… 🎨")
    try:
        url, model = await asyncio.to_thread(render_with_profile, uid, STYLE_PRESETS[preset])
        await safe_send_image(update, url, f"Готово ✨\nСтиль: {preset}\nBackend: {STYLE_BACKEND}\nМодель: {model}")
    except Exception as e:
        logger.exception("style_command failed")
        await update.message.reply_text(f"Упс, не вышло. Причина: {e}")

# быстрые тумблеры режима без ребилда
async def fast_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global FAST_DEBUG
    FAST_DEBUG = True
    await update.message.reply_text("FAST mode ON: 1 реф, меньше шагов, InsightFace OFF.")

async def fast_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global FAST_DEBUG
    FAST_DEBUG = False
    await update.message.reply_text("FAST mode OFF: полный режим. Для отбора по лицу включи USE_INSIGHTFACE=1 в env.")

# ======================
# SEND IMAGE
# ======================
async def safe_send_image(update: Update, url: str, caption: str = ""):
    msg = update.message
    try:
        await msg.reply_photo(photo=url, caption=caption); return
    except Exception:
        try:
            r = requests.get(url, timeout=90); r.raise_for_status()
            bio = io.BytesIO(r.content); bio.name = "result.jpg"
            await msg.reply_photo(photo=bio, caption=caption); return
        except Exception as e:
            await msg.reply_text(f"Готово, но вложить не удалось. Ссылка:\n{url}\n({e})")

# ======================
# SYSTEM
# ======================
async def _post_init(app):
    await app.bot.delete_webhook(drop_pending_updates=True)
    if USE_INSIGHTFACE and not FAST_DEBUG:
        await asyncio.to_thread(_init_face_models)

async def _error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Unhandled error", exc_info=context.error)

def main():
    app = (ApplicationBuilder().token(TOKEN).post_init(_post_init).build())

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("idenroll", id_enroll))
    app.add_handler(CommandHandler("iddone", id_done))
    app.add_handler(CommandHandler("idstatus", id_status))
    app.add_handler(CommandHandler("idreset", id_reset))

    app.add_handler(CommandHandler("styles", styles_cmd))
    app.add_handler(CallbackQueryHandler(callback_style, pattern=r"^style:"))
    app.add_handler(CommandHandler("style", style_command))

    app.add_handler(CommandHandler("fast_on", fast_on))
    app.add_handler(CommandHandler("fast_off", fast_off))

    # Фото нужны только для набора профиля
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    app.add_error_handler(_error_handler)
    logger.info("Bot ready (polling) | FAST_DEBUG=%s | USE_INSIGHTFACE=%s", FAST_DEBUG, USE_INSIGHTFACE)
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()


