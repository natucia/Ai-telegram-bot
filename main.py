import os, re, io, json, time, glob, shutil, asyncio, logging, hashlib
from pathlib import Path
from typing import Any, Iterable, Tuple, Dict, Optional, List

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
# ОКРУЖЕНИЕ
# ======================
TOKEN = os.getenv("BOT_TOKEN")
os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")
if not TOKEN or not re.match(r"^\d+:[A-Za-z0-9_-]{20,}$", TOKEN):
    raise RuntimeError("Некорректный BOT_TOKEN.")
if not os.getenv("REPLICATE_API_TOKEN"):
    raise RuntimeError("Нет REPLICATE_API_TOKEN.")

# Replicate модели
STYLE_BACKEND = "instantid"
INSTANTID_MODEL = os.getenv("INSTANTID_MODEL", "grandlineai/instant-id-photorealistic")
QWEN_EDIT_MODEL = os.getenv("QWEN_EDIT_MODEL", "qwen/qwen-image-edit-plus")

# Ультра-похожесть (почти без перерисовки)
ULTRA_LOCK_STRENGTH = float(os.getenv("ULTRA_LOCK_STRENGTH") or 0.14)  # 0.12–0.16
ULTRA_LOCK_GUIDANCE = float(os.getenv("ULTRA_LOCK_GUIDANCE") or 2.3)
TOPK_REFS = int(os.getenv("TOPK_REFS") or 3)  # сколько лучших референсов брать из профиля

# Реставрация (необязательная)
PRIMARY_MODEL = os.getenv("REPLICATE_MODEL") or "tencentarc/gfpgan"
FALLBACK_MODELS = ["tencentarc/gfpgan", "xinntao/realesrgan"]

NEGATIVE_PROMPT = (
    "cartoon, anime, cgi, 3d, plastic skin, waxy skin, porcelain, airbrushed, beauty filter, smoothing, "
    "overprocessed, oversharpen, hdr effect, halo, neon skin, garish, fake skin, cosplay wig, doll, "
    "ai-artifacts, deformed, bad anatomy, extra fingers, duplicated features, watermark, text, logo, "
    "overly saturated, extreme skin retouch, low detail, lowres, jpeg artifacts, warped face, distorted face, "
    "changed facial proportions, geometry change, face reshape, exaggerated makeup"
)
AESTHETIC_SUFFIX = (
    ", natural healthy skin, preserved pores, subtle makeup, balanced contrast, soft realistic light, "
    "no beauty filter, no plastic look"
)

STYLE_PRESETS: Dict[str, str] = {
    "natural": "ultra realistic portrait, real skin texture, subtle makeup, soft natural light, neutral grading",
    "boho": "boho portrait, natural fabrics, earthy palette, soft daylight",
    "vogue": "beauty cover shot, calibrated colors, soft studio light, photographic grain",
    "beauty_soft": "beauty portrait, clean studio light, controlled highlights, visible pores",
    "windowlight": "soft window light portrait, balanced exposure, natural diffusion",
    "editorial": "editorial fashion portrait, preserved natural imperfections, pro color grading",
    "moody": "moody cinematic portrait, controlled shadows, subtle rim, realistic texture",
}

# ======================
# ЛОГИ
# ======================
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("bot")

# ======================
# ЛОКАЛЬНОЕ ХРАНИЛИЩЕ ПРОФИЛЕЙ
# ======================
DATA_DIR = Path("profiles"); DATA_DIR.mkdir(exist_ok=True)

def user_dir(uid:int) -> Path:
    p = DATA_DIR / str(uid); p.mkdir(parents=True, exist_ok=True); return p

def save_image_bytes(path: Path, content: bytes):
    path.write_bytes(content)

def list_ref_images(uid:int) -> List[Path]:
    return sorted((user_dir(uid)).glob("ref_*.jpg"))

# ======================
# FACE EMBEDDINGS (InsightFace)
# ======================
_FACE_APP = None
_ARCFACE = None

def _init_face_models():
    global _FACE_APP, _ARCFACE
    if _FACE_APP is None:
        from insightface.app import FaceAnalysis
        _FACE_APP = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _FACE_APP.prepare(ctx_id=0, det_size=(640, 640))
    if _ARCFACE is None:
        from insightface.model_zoo import get_model
        _ARCFACE = get_model("glintr100.onnx", download=True, providers=["CPUExecutionProvider"])

def face_embed_from_bytes(b: bytes) -> Optional[np.ndarray]:
    _init_face_models()
    img = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    if img is None: return None
    faces = _FACE_APP.get(img)
    if not faces: return None
    face = max(faces, key=lambda f: f.det_score)
    emb = _ARCFACE.get(img, face.bbox.astype(int))
    if emb is None: return None
    return emb.astype(np.float32) / (np.linalg.norm(emb) + 1e-8)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8))

def profile_path(uid:int) -> Path:
    return user_dir(uid) / "profile.json"

def load_profile(uid:int) -> Dict[str, Any]:
    p = profile_path(uid)
    if p.exists(): return json.loads(p.read_text())
    return {"embeds": [], "images": []}

def save_profile(uid:int, prof:Dict[str,Any]):
    profile_path(uid).write_text(json.dumps(prof))

def build_or_update_profile(uid:int):
    """Пересчитать эмбеддинги по сохранённым ref_*.jpg"""
    refs = list_ref_images(uid)
    prof = {"embeds": [], "images": []}
    for fp in refs:
        b = fp.read_bytes()
        emb = face_embed_from_bytes(b)
        if emb is not None:
            prof["embeds"].append(emb.tolist())
            prof["images"].append(fp.name)
    save_profile(uid, prof)
    return prof

def best_refs_for(uid:int, target_bytes:bytes, topk:int=3) -> List[Path]:
    """Выбрать topk референсов, ближайших к текущему лицу"""
    target = face_embed_from_bytes(target_bytes)
    refs = list_ref_images(uid)
    if target is None or not refs:
        return refs[:topk]
    prof = load_profile(uid)
    scores = []
    for name, emb_list in zip(prof.get("images", []), prof.get("embeds", [])):
        emb = np.array(emb_list, dtype=np.float32)
        score = cosine(target, emb)
        scores.append((score, name))
    scores.sort(reverse=True)
    chosen = [user_dir(uid)/name for _,name in scores[:topk]]
    return chosen if chosen else refs[:topk]

# ======================
# УТИЛИТЫ TELEGRAM/REPLICATE
# ======================
async def tg_public_url(message) -> str:
    f = await message.photo[-1].get_file()
    return f.file_path

async def tg_download_bytes(message) -> bytes:
    f = await message.photo[-1].get_file()
    b = await f.download_as_bytearray()
    return bytes(b)

def resolve_model_version(slug: str) -> str:
    if ":" in slug: return slug
    model = replicate.models.get(slug)
    versions = list(model.versions.list())
    if not versions: raise RuntimeError(f"Нет версий модели {slug}")
    return f"{slug}:{versions[0].id}"

def extract_any_url(output: Any) -> Optional[str]:
    if isinstance(output, str) and output.startswith(("http","https")): return output
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
            u = extract_any_url(out)
            if not u: raise RuntimeError("Empty output")
            return u
        except Exception as e:
            last = e
            logger.warning("Model %s rejected: %s", model, e)
    raise last or RuntimeError("Все варианты отклонены")

async def safe_send_image(update: Update, url: str, caption: str = ""):
    msg = update.message
    try:
        await msg.reply_photo(photo=url, caption=caption); return
    except Exception:
        try:
            r = requests.get(url, timeout=90); r.raise_for_status()
            bio = io.BytesIO(r.content); bio.name="result.jpg"
            await msg.reply_photo(photo=bio, caption=caption); return
        except Exception as e:
            await msg.reply_text(f"Готово, но ссылку не удалось вложить. URL:\n{url}\n({e})")

# ======================
# ГЕНЕРАЦИЯ (супер-похожесть, мульти-реф)
# ======================
def _seed_from(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16)

def instantid_once(image_url:str, face_ref_url:str, prompt:str, denoise:float, guidance:float, seed:int) -> Tuple[str, str]:
    resolved = resolve_model_version(INSTANTID_MODEL)
    payload = {
        "face_image": face_ref_url,            # основной ключ
        "input_image": face_ref_url,           # запасной
        "input_image_ref": face_ref_url,       # ещё один запасной
        "image": image_url,                    # сам кадр
        "prompt": prompt,
        "negative_prompt": NEGATIVE_PROMPT + ", enlarged nose, altered nose shape, flat hair, missing hair volume, bluish nose",
        "ip_adapter_scale": 1.0,               # держим лицо максимально
        "controlnet_conditioning_scale": 0.18, # почти не перерисовывать
        "strength": denoise,                   # 0.12–0.18
        "guidance_scale": guidance,            # ~2.3
        "num_inference_steps": 18,
        "seed": seed,
    }
    url = replicate_run_flexible(resolved, [payload])
    return url, resolved

def qwen_once(image_url:str, prompt:str, denoise:float, guidance:float, seed:int) -> Tuple[str,str]:
    resolved = resolve_model_version(QWEN_EDIT_MODEL)
    payload = {
        "image": image_url,
        "prompt": prompt + ", do not change geometry, only subtle color/contrast improvements",
        "negative_prompt": NEGATIVE_PROMPT,
        "strength": denoise,
        "guidance_scale": guidance,
        "num_inference_steps": 20,
        "seed": seed,
    }
    url = replicate_run_flexible(resolved, [payload])
    return url, resolved

def pick_most_similar(candidate_urls: List[str], target_face: np.ndarray) -> str:
    """Скачать кандидатов, посчитать схожесть лица, вернуть лучший URL"""
    _init_face_models()
    best_url, best_score = candidate_urls[0], -1.0
    for u in candidate_urls:
        try:
            b = requests.get(u, timeout=90).content
            emb = face_embed_from_bytes(b)
            if emb is None: continue
            sc = cosine(target_face, emb)
            if sc > best_score: best_score, best_url = sc, u
        except Exception:
            continue
    logger.info("Picked best candidate score=%.4f", best_score)
    return best_url

async def run_style_multiref(uid:int, photo_url:str, photo_bytes:bytes, preset_prompt:str) -> Tuple[str,str]:
    # готовим промпт
    base_prompt = (
        "highly realistic portrait, exact facial identity, preserve original facial proportions and features, "
        "keep natural hair volume and hairline, matched skin tone, balanced natural lighting, "
        "no stylization of anatomy, no geometry change, " + preset_prompt + AESTHETIC_SUFFIX
    )
    denoise = ULTRA_LOCK_STRENGTH
    guidance = ULTRA_LOCK_GUIDANCE
    seed = _seed_from(photo_url)

    # выбираем топ-референсы из профиля
    refs = best_refs_for(uid, photo_bytes, TOPK_REFS)
    if not refs:
        # нет профиля — работаем в «одиночном» режиме, реф = сам кадр
        logger.info("No profile: fallback to single-ref (input image)")
        url, model = instantid_once(photo_url, photo_url, base_prompt, denoise, guidance, seed)
        return url, model

    # для каждого референса гоняем один проход и собираем кандидатов
    cand_urls: List[str] = []
    for ref_path in refs:
        # делаем референс общедоступным через replicate.delivery: просто пихаем как URL нельзя,
        # поэтому используем Telegram file_url напрямую — работает.
        ref_bytes = ref_path.read_bytes()
        # трюк: кладём локальный реф в память Replicate через upload (если модель поддерживает) — но тут
        # используем Telegram/локальный URL: просто загрузим в memory.io на один шаг.
        # Для простоты — повторно используем исходный кадр, InstantID берёт лицо из face_image.
        # Чтоб был нормальный URL, положим реф как data URL:
        import base64
        data_url = "data:image/jpeg;base64," + base64.b64encode(ref_bytes).decode("ascii")

        try:
            u, _ = instantid_once(photo_url, data_url, base_prompt, denoise, guidance, seed)
            cand_urls.append(u)
        except Exception as e:
            logger.warning("InstantID with ref %s failed: %s", ref_path.name, e)

    # пункт безопасности: если кандидатов нет — fallback на qwen
    if not cand_urls:
        logger.info("No InstantID candidates, fallback to Qwen")
        u, model = qwen_once(photo_url, base_prompt, denoise, guidance, seed)
        return u, model

    # выбираем самый похожий по лицу
    target_emb = face_embed_from_bytes(photo_bytes)
    if target_emb is None:
        # если не смогли достать лицо из входа — вернём первый успешный
        return cand_urls[0], "instantid"

    best = pick_most_similar(cand_urls, target_emb)
    return best, "instantid"

# ======================
# ХЕНДЛЕРЫ
# ======================
def styles_keyboard() -> InlineKeyboardMarkup:
    rows, row = [], []
    for i, name in enumerate(STYLE_PRESETS.keys(), 1):
        row.append(InlineKeyboardButton(name.title(), callback_data=f"style:{name}"))
        if i % 3 == 0: rows.append(row); row=[]
    if row: rows.append(row)
    return InlineKeyboardMarkup(rows)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я делаю реалистичную стилизацию **с максимальной схожестью**.\n\n"
        "1) Сначала соберём твой ID-профиль (до 10 фото с разными ракурсами):\n"
        "   • /idenroll — включить набор, присылай фото подряд\n"
        "   • /iddone — завершить и сохранить профиль\n"
        "   • /idstatus — посмотреть, сколько фото в профиле\n"
        "   • /idreset — удалить профиль\n\n"
        "2) Затем пришли фото с подписью стиля, например: `boho`, `natural`, `vogue`.\n"
        "Бот выберет лучшие референсы и вернёт самый похожий результат."
    )

# ----- ID ENROLL -----
ENROLL_FLAG: Dict[int, bool] = {}  # user_id -> collecting?

async def id_enroll(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    ENROLL_FLAG[uid] = True
    await update.message.reply_text(
        "Режим набора ID включён. Пришли до **10** фото (разные ракурсы, без фильтров). "
        "Когда закончишь — отправь /iddone."
    )

async def id_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    ENROLL_FLAG[uid] = False
    prof = build_or_update_profile(uid)
    await update.message.reply_text(
        f"Готово. В профиле {len(prof['images'])} фото. Теперь присылай фото со стилем (например `boho`)."
    )

async def id_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    cnt = len(list_ref_images(uid))
    await update.message.reply_text(f"В профиле сохранено {cnt} фото.")

async def id_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    p = user_dir(uid)
    if p.exists(): shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    await update.message.reply_text("Профиль очищен.")

# ----- STYLE FLOW -----
async def callback_style(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    preset = q.data.split(":",1)[1]
    await q.message.reply_text(f"Окей, стиль: {preset}. Пришли фото с подписью `{preset}`")

async def style_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text("Выбери стиль:", reply_markup=styles_keyboard()); return
    await _run_style_flow(update, args[0].lower())

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Либо собираем ID, либо стилизуем, если в подписи найден стиль."""
    uid = update.effective_user.id
    m = update.message
    cap = (m.caption or "").lower()

    # если идёт набор
    if ENROLL_FLAG.get(uid):
        refs = list_ref_images(uid)
        if len(refs) >= 10:
            await m.reply_text("Уже 10 фото в профиле. Отправь /iddone.")
            return
        b = await tg_download_bytes(m)
        path = user_dir(uid) / f"ref_{int(time.time())}.jpg"
        save_image_bytes(path, b)
        await m.reply_text(f"Сохранила ({len(refs)+1}/10). Ещё?")
        return

    # иначе — стилизация
    preset = None
    for k in STYLE_PRESETS.keys():
        if re.search(rf"\b{k}\b", cap):
            preset = k; break
    if not preset:
        await m.reply_text("Укажи стиль в подписи (например `boho`) или нажми кнопку:", reply_markup=styles_keyboard())
        return

    await m.chat.send_action(ChatAction.UPLOAD_PHOTO)
    await m.reply_text(f"Стилизация (мульти-реф, ультра-похоже): {preset}… 🎨")

    try:
        photo_url = await tg_public_url(m)
        photo_bytes = await tg_download_bytes(m)
        url, model = await asyncio.to_thread(
            run_style_multiref, uid, photo_url, photo_bytes, STYLE_PRESETS[preset]
        )
        await safe_send_image(update, url, f"Готово ✨\nСтиль: {preset}\nBackend: {STYLE_BACKEND}\nМодель: {model}")
    except Exception as e:
        logger.exception("Ошибка стилизации")
        await m.reply_text(f"Упс, не вышло. Причина: {e}")

# ======================
# РЕСТАВРАЦИЯ (опционально)
# ======================
def run_restore_with_fallbacks(image_url: str) -> Tuple[str, str]:
    for slug in [PRIMARY_MODEL] + [m for m in FALLBACK_MODELS if m != PRIMARY_MODEL]:
        try:
            resolved = resolve_model_version(slug)
            inputs = [{"image": image_url}, {"img": image_url}]
            url = replicate_run_flexible(resolved, inputs)
            return url, resolved
        except Exception as e:
            logger.warning("Fallback %s failed: %s", slug, e)
    raise RuntimeError("Реставрация: все модели отклонили запрос")

# ======================
# SYSTEM
# ======================
async def _post_init(app):
    await app.bot.delete_webhook(drop_pending_updates=True)

async def _error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Unhandled error", exc_info=context.error)

def main():
    app = (ApplicationBuilder().token(TOKEN).post_init(_post_init).build())

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("idenroll", id_enroll))
    app.add_handler(CommandHandler("iddone", id_done))
    app.add_handler(CommandHandler("idstatus", id_status))
    app.add_handler(CommandHandler("idreset", id_reset))

    app.add_handler(CommandHandler("style", style_command))
    app.add_handler(CallbackQueryHandler(callback_style, pattern=r"^style:"))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    app.add_error_handler(_error_handler)

    logger.info("Bot ready (polling)")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()


