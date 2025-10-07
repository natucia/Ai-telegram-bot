# === Telegram LoRA Bot (Flux LoRA trainer + HARD styles + Redis persist
# + Identity/Gender locks + LOCKFACE fallback + MULTI-AVATARS + PRETTY MODE) ===
# Требования: python-telegram-bot==20.7, replicate==0.31.0, pillow==10.4.0, redis==5.0.1

from typing import Any, Dict, List, Optional, Tuple
Style = Dict[str, Any]

from styles import (  # словари лежат в styles.py
    STYLE_PRESETS, STYLE_CATEGORIES, THEME_BOOST,
    SCENE_GUIDANCE, RISKY_PRESETS
)

import os, re, io, json, time, asyncio, logging, shutil, random, contextlib
from pathlib import Path
from zipfile import ZipFile

import replicate
from replicate import Client
from replicate.exceptions import ReplicateError
from PIL import Image

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes,
    CallbackQueryHandler, filters
)

# ---------- ENV ----------
TOKEN = os.getenv("BOT_TOKEN", "")
os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN", "")

if not TOKEN or not re.match(r"^\d+:[A-Za-z0-9_-]{20,}$", TOKEN):
    raise RuntimeError("Некорректный BOT_TOKEN.")
if not os.getenv("REPLICATE_API_TOKEN"):
    raise RuntimeError("Нет REPLICATE_API_TOKEN.")

DEST_OWNER  = os.getenv("REPLICATE_DEST_OWNER", "").strip()
DEST_MODEL  = os.getenv("REPLICATE_DEST_MODEL", "yourtwin-lora").strip()

# Тренер LoRA
LORA_TRAINER_SLUG = os.getenv("LORA_TRAINER_SLUG", "replicate/flux-lora-trainer").strip()
LORA_INPUT_KEY    = os.getenv("LORA_INPUT_KEY", "input_images").strip()

# Классификатор пола (опц.)
GENDER_MODEL_SLUG = os.getenv("GENDER_MODEL_SLUG", "nateraw/vit-age-gender").strip()

# LOCKFACE (InstantID / FaceID adapter) — можно оставить пустым, тогда не используем
INSTANTID_SLUG = os.getenv("INSTANTID_SLUG", "").strip()

# --- Параметры обучения ---
LORA_MAX_STEPS     = int(os.getenv("LORA_MAX_STEPS", "1400"))
LORA_LR            = float(os.getenv("LORA_LR", "0.00006"))
LORA_USE_FACE_DET  = os.getenv("LORA_USE_FACE_DET", "true").lower() in ["1","true","yes","y"]
LORA_RESOLUTION    = int(os.getenv("LORA_RESOLUTION", "1024"))
LORA_CAPTION_PREF  = os.getenv(
    "LORA_CAPTION_PREFIX",
    "a high quality photo of the same person, neutral expression, gentle smile, "
    "balanced facial proportions, soft jawline, clear eyes"
).strip()

# --- Генерация (дефолты) ---
GEN_STEPS     = int(os.getenv("GEN_STEPS", "48"))
GEN_GUIDANCE  = float(os.getenv("GEN_GUIDANCE", "4.2"))
GEN_WIDTH     = int(os.getenv("GEN_WIDTH", "896"))
GEN_HEIGHT    = int(os.getenv("GEN_HEIGHT", "1152"))

# Верхний предел шагов у модели (чтобы не ловить 422)
MAX_STEPS = int(os.getenv("MAX_STEPS", "50"))

# ---- Anti-drift / aesthetics
NEGATIVE_PROMPT = (
    "cartoon, anime, 3d, cgi, plastic skin, overprocessed, oversharpen, "
    "lowres, blur, jpeg artifacts, text, watermark, logo, bad anatomy, extra fingers, short fingers, "
    "puffy face, swollen face, bulky masseter, wide jaw, clenched jaw, duckface, overfilled lips, "
    "cross-eye, misaligned pupils, double pupils, heterochromia, mismatched eye direction, "
    "identity drift, different person, ethnicity change, age change, face morph, face swap, "
    "eye spacing change, stretched face, narrowed eyes, exaggerated eyelid fold, "
    "fisheye, lens distortion, warped face, deformed skull, tiny head, giant head on small body, "
    "bodybuilder female, extreme makeup, heavy contouring, "
    "casual clothing, plain studio background, selfie, tourist photo"
)
AESTHETIC_SUFFIX = (
    ", photorealistic, visible skin texture, natural color, soft filmic contrast, gentle micro-sharpen, "
    "anatomically plausible facial landmarks, natural interocular distance"
)

# --- Pretty mode (мягкая красота без пластика) ---
PRETTY_MODE = os.getenv("PRETTY_MODE", "1").lower() in ("1","true","yes","y")
PRETTY_POS = (
    "subtle beauty retouch, even skin tone, faint under-eye smoothing, "
    "soft diffusion on highlights, slight glow, healthy complexion, "
    "delicate catchlights, softened nasolabial shadows, gentle friendly smile, "
    "micro-contrast on eyes and lips, tidy eyebrows"
)
PRETTY_NEG = (
    "harsh pores, deep nasolabial folds, eye bags, blotchy redness, "
    "oily hotspot shine, over-sharpened skin, beauty filter"
)
PRETTY_COMP_HINT = "camera slightly above eye level, flattering portrait angle"

def _inject_pretty(core_prompt: str, comp_text: str) -> str:
    if not PRETTY_MODE:
        return core_prompt
    parts = [core_prompt]
    if PRETTY_POS: parts.append(PRETTY_POS)
    if PRETTY_COMP_HINT and "camera at eye level" in comp_text:
        parts.append(PRETTY_COMP_HINT)
    return ", ".join(parts)

def _beauty_guardrail() -> str:
    return (
        "exact facial identity, identity preserved, "
        "balanced facial proportions, symmetrical face, natural oval, soft jawline, "
        "smooth cheek contour, relaxed neutral expression, subtle gentle smile, "
        "open expressive almond-shaped eyes, clean catchlights, clear irises, "
        "realistic body proportions, proportional shoulders and waist, natural posture"
    )

def _face_lock() -> str:
    return (
        "exact same face as the training photos, do not alter bone structure, "
        "natural interocular distance, consistent eyelid shape, pupils aligned, "
        "preserve cheekbone widths and lip fullness"
    )

def _anti_distort() -> str:
    return ("no fisheye, no lens distortion, no warping, no stretched face, "
            "natural perspective, proportional head size, realistic human anatomy")

def _gender_lock(gender:str) -> Tuple[str, str]:
    if gender == "female":
        pos = "female woman, feminine facial features"
        neg = "male, man, beard, stubble, mustache, adam's apple"
    else:
        pos = "male man, masculine facial features"
        neg = "female, woman, long eyelashes makeup, visible breasts"
    return pos, neg

# ---------- Композиция/линза/свет ----------
def _comp_text_and_size(comp: str) -> Tuple[str, Tuple[int,int]]:
    if comp == "closeup":
        return ("portrait framing from chest up, 85mm lens look, camera at eye level, subject distance 1.2m, "
                "no perspective distortion on face", (896, 1152))
    if comp == "half":
        return ("half body framing, 85mm lens look, camera at chest level, subject distance 2.5m, "
                "no perspective distortion on face", (896, 1344))
    return ("full body framing, 85mm lens look, camera at mid-torso level, head size natural for frame, "
            "no perspective distortion on face", (896, 1408))

def _tone_text(tone: str) -> str:
    return {
        "daylight": "soft natural daylight, neutral colors",
        "warm":     "golden hour warmth, gentle highlights",
        "cool":     "cool cinematic light, clean color balance",
        "noir":     "high contrast noir lighting, strong rim light, subtle haze",
        "neon":     "neon signs, wet reflections, cinematic backlight, vibrant saturation",
        "candle":   "warm candlelight, soft glow, volumetric rays",
    }.get(tone, "balanced soft lighting")

# ---------- logging ----------
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("bot")

# ---------- storage ----------
DATA_DIR = Path("profiles"); DATA_DIR.mkdir(exist_ok=True)

# === МУЛЬТИ-АВАТАРЫ: структура профиля ===
DEFAULT_AVATAR = {
    "images": [],
    "training_id": None,
    "finetuned_model": None,
    "finetuned_version": None,
    "status": None,
    "lockface": False
}
DEFAULT_PROFILE = {
    "gender": None,
    "current_avatar": "default",
    "avatars": {"default": DEFAULT_AVATAR.copy()}
}

def user_dir(uid:int) -> Path:
    p = DATA_DIR / str(uid); p.mkdir(parents=True, exist_ok=True); return p
def avatars_root(uid:int) -> Path:
    p = user_dir(uid) / "avatars"; p.mkdir(parents=True, exist_ok=True); return p
def avatar_dir(uid:int, avatar:str) -> Path:
    p = avatars_root(uid) / avatar; p.mkdir(parents=True, exist_ok=True); return p
def list_ref_images(uid:int, avatar:str) -> List[Path]:
    return sorted(avatar_dir(uid, avatar).glob("ref_*.jpg"))
def profile_path(uid:int) -> Path:
    return user_dir(uid) / "profile.json"

def _migrate_single_to_multi(uid:int, prof:Dict[str,Any]) -> Dict[str,Any]:
    if "avatars" in prof:
        return prof
    migrated = DEFAULT_PROFILE.copy()
    migrated["gender"] = prof.get("gender")
    default = DEFAULT_AVATAR.copy()
    default["training_id"] = prof.get("training_id")
    default["finetuned_model"] = prof.get("finetuned_model")
    default["finetuned_version"] = prof.get("finetuned_version")
    default["status"] = prof.get("status")
    default["lockface"] = prof.get("lockface", False)
    imgs = prof.get("images", [])
    if imgs:
        for name in imgs:
            src = user_dir(uid) / name
            if src.exists():
                dst = avatar_dir(uid, "default") / name
                if not dst.exists():
                    with contextlib.suppress(Exception):
                        shutil.move(str(src), str(dst))
        default["images"] = [p.name for p in list_ref_images(uid, "default")]
    migrated["avatars"]["default"] = default
    return migrated

_redis = None
REDIS_URL = os.getenv("REDIS_URL", "").strip()
if REDIS_URL:
    try:
        import redis  # redis==5.x
        _redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        _redis.ping()
        logger.info("Storage: Redis OK (%s)", REDIS_URL.rsplit("@",1)[-1])
    except Exception as e:
        logger.warning("Storage: Redis init failed (%s). Falling back to FS. Error: %s", REDIS_URL, e)
        _redis = None
else:
    logger.info("Storage: FS (no REDIS_URL)")

def load_profile(uid:int) -> Dict[str, Any]:
    if _redis:
        try:
            raw = _redis.get(f"profile:{uid}")
            if raw:
                prof = {**DEFAULT_PROFILE, **json.loads(raw)}
                prof = _migrate_single_to_multi(uid, prof)
                return prof
        except Exception as e:
            logger.warning("Redis load_profile failed: %s", e)
    p = profile_path(uid)
    if p.exists():
        with contextlib.suppress(Exception):
            prof = {**DEFAULT_PROFILE, **json.loads(p.read_text())}
            prof = _migrate_single_to_multi(uid, prof)
            return prof
    return DEFAULT_PROFILE.copy()

def save_profile(uid:int, prof:Dict[str,Any]):
    if _redis:
        try:
            _redis.set(f"profile:{uid}", json.dumps(prof, ensure_ascii=False))
        except Exception as e:
            logger.warning("Redis save_profile failed: %s", e)
    with contextlib.suppress(Exception):
        profile_path(uid).write_text(json.dumps(prof, ensure_ascii=False))

def delete_profile(uid:int):
    if _redis:
        with contextlib.suppress(Exception):
            _redis.delete(f"profile:{uid}")
    p = user_dir(uid)
    if p.exists(): shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def save_ref_downscaled(path: Path, raw: bytes, max_side=1024, quality=92):
    im = Image.open(io.BytesIO(raw)).convert("RGB"); im.thumbnail((max_side, max_side)); im.save(path, "JPEG", quality=quality)

# ---------- Аватарные утилиты ----------
def get_current_avatar_name(prof:Dict[str,Any]) -> str:
    name = prof.get("current_avatar") or "default"
    if name not in prof["avatars"]:
        name = "default"; prof["current_avatar"] = name
    return name

def get_avatar(prof:Dict[str,Any], name:Optional[str]=None) -> Dict[str,Any]:
    if not name:
        name = get_current_avatar_name(prof)
    if name not in prof["avatars"]:
        prof["avatars"][name] = DEFAULT_AVATAR.copy()
    return prof["avatars"][name]

def set_current_avatar(uid:int, name:str):
    prof = load_profile(uid)
    if name not in prof["avatars"]:
        prof["avatars"][name] = DEFAULT_AVATAR.copy()
    prof["current_avatar"] = name
    save_profile(uid, prof)

def ensure_avatar(uid:int, name:str):
    prof = load_profile(uid)
    if name not in prof["avatars"]:
        prof["avatars"][name] = DEFAULT_AVATAR.copy()
        save_profile(uid, prof)

def del_avatar(uid:int, name:str):
    prof = load_profile(uid)
    if name == "default":
        raise RuntimeError("Нельзя удалить аватар 'default'.")
    if name not in prof["avatars"]:
        return
    adir = avatar_dir(uid, name)
    with contextlib.suppress(Exception):
        if adir.exists(): shutil.rmtree(adir)
    prof["avatars"].pop(name, None)
    if prof["current_avatar"] == name:
        prof["current_avatar"] = "default"
    save_profile(uid, prof)

# ---------- Replicate helpers ----------
def resolve_model_version(slug: str) -> str:
    if ":" in slug: return slug
    model = replicate.models.get(slug); versions = list(model.versions.list())
    if not versions: raise RuntimeError(f"Нет версий модели {slug}")
    return f"{slug}:{versions[0].id}"

def extract_any_url(out: Any) -> Optional[str]:
    if isinstance(out, str) and out.startswith(("http","https")): return out
    if isinstance(out, list):
        for v in out:
            u = extract_any_url(v)
            if u: return u
    if isinstance(out, dict):
        for v in out.values():
            u = extract_any_url(v)
            if u: return u
    return None

def _check_slug(slug: str, label: str):
    try:
        mv = resolve_model_version(slug)
        logger.info("%s OK: %s", label, mv)
    except Exception as e:
        logger.warning("%s BAD ('%s'): %s", label, slug, e)

# ---------- авто-пол ----------
def _infer_gender_from_image(path: Path) -> Optional[str]:
    try:
        client = Client(api_token=os.environ["REPLICATE_API_TOKEN"])
        version_slug = resolve_model_version(GENDER_MODEL_SLUG)
        with open(path, "rb") as img_b:
            for key in ["image", "img", "input_image", "file"]:
                try:
                    pred = client.predictions.create(version=version_slug, input={key: img_b})
                    pred.wait()
                    out = pred.output
                    g = (out.get("gender") if isinstance(out, dict) else str(out)).lower()
                    if "female" in g or "woman" in g: return "female"
                    if "male" in g or "man" in g: return "male"
                except Exception as e:
                    logger.warning("Gender key '%s' failed: %s", key, e)
    except Exception as e:
        logger.warning("Gender inference error: %s", e)
    return None

def auto_detect_gender(uid:int) -> str:
    prof = load_profile(uid)
    av_name = get_current_avatar_name(prof)
    refs = list_ref_images(uid, av_name)
    if not refs: return "female"
    g = _infer_gender_from_image(refs[0])
    return g or "female"

# ---------- Генерация промпта ----------
def _style_lock(role:str, outfit:str, props:str, background:str, comp_hint:str) -> str:
    bits = [
        f"{role}" if role else "",
        f"wearing {outfit}" if outfit else "",
        f"with {props}" if props else "",
        f"background: {background}" if background else "",
        comp_hint,
        "unmistakable scene identity"
    ]
    return ", ".join([b for b in bits if b])

def build_prompt(meta: Style, gender: str, comp_text:str, tone_text:str, theme_boost:str) -> Tuple[str, str]:
    role = meta.get("role_f") if (gender=="female" and meta.get("role_f")) else meta.get("role","")
    if not role and meta.get("role_m") and gender=="male":
        role = meta.get("role_m","")
    outfit = meta.get("outfit_f") if (gender=="female" and meta.get("outfit_f")) else meta.get("outfit","")
    props = meta.get("props","")
    bg = meta.get("bg","")

    gpos, gneg = _gender_lock(gender)
    anti = _anti_distort()
    age_lock = "" if meta.get("allow_age_change") else "no age change, "

    if role or outfit or props or bg:
        core = ", ".join([
            _style_lock(role, outfit, props, bg, comp_text),
            tone_text,
            gpos,
            f"same person as the training photos, no ethnicity change, {age_lock}exact facial identity, identity preserved",
            "photorealistic, realistic body proportions, natural skin texture, filmic look",
            anti,
            _beauty_guardrail(),
            _face_lock(),
            theme_boost
        ])
        core += ", the costume and background must clearly communicate the role; avoid plain portrait"
        core = _inject_pretty(core, comp_text)
        gneg = (gneg + ", " + PRETTY_NEG) if PRETTY_MODE else gneg
        return core, gneg

    base_prompt = meta.get("p", "")
    core = ", ".join([
        f"{base_prompt}, {comp_text}, {tone_text}",
        gpos,
        f"same person as the training photos, {age_lock}exact facial identity, identity preserved",
        "cinematic key light and rim light, soft bounce fill, film grain subtle",
        "skin tone faithful",
        anti,
        _beauty_guardrail(),
        _face_lock(),
        theme_boost
    ])
    core = _inject_pretty(core, comp_text)
    gneg = (gneg + ", " + PRETTY_NEG) if PRETTY_MODE else gneg
    return core, gneg

# ---------- Инференс ----------
def generate_from_finetune(model_slug:str, prompt:str, steps:int, guidance:float, seed:int, w:int, h:int, negative_prompt:str) -> str:
    mv = resolve_model_version(model_slug)
    out = replicate.run(mv, input={
        "prompt": prompt + AESTHETIC_SUFFIX,
        "negative_prompt": negative_prompt,
        "width": w, "height": h,
        "num_inference_steps": min(MAX_STEPS, steps),
        "guidance_scale": guidance,
        "seed": seed,
    })
    url = extract_any_url(out)
    if not url: raise RuntimeError("Empty output")
    return url

def generate_with_instantid(face_path: Path, prompt:str, steps:int, guidance:float, seed:int, w:int, h:int, negative_prompt:str) -> str:
    mv = resolve_model_version(INSTANTID_SLUG)
    with open(face_path, "rb") as fb:
        out = replicate.run(mv, input={
            "face_image": fb,
            "prompt": prompt + AESTHETIC_SUFFIX,
            "negative_prompt": negative_prompt,
            "width": w, "height": h,
            "num_inference_steps": min(MAX_STEPS, steps),
            "guidance_scale": guidance,
            "seed": seed,
        })
    url = extract_any_url(out)
    if not url: raise RuntimeError("Empty output (InstantID)")
    return url

# ---------- UI ----------
def main_menu_kb() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("🧭 Выбрать стиль", callback_data="nav:styles")],
        [InlineKeyboardButton("📸 Набор фото", callback_data="nav:enroll"),
         InlineKeyboardButton("🧪 Обучение", callback_data="nav:train")],
        [InlineKeyboardButton("ℹ️ Мой статус", callback_data="nav:status"),
         InlineKeyboardButton("⚙️ Пол", callback_data="nav:gender")],
        [InlineKeyboardButton("🔒 LOCKFACE", callback_data="nav:lockface")],
        [InlineKeyboardButton("🤖 Аватары", callback_data="nav:avatars")]
    ]
    return InlineKeyboardMarkup(rows)

def categories_kb() -> InlineKeyboardMarkup:
    names = list(STYLE_CATEGORIES.keys())
    rows, row = [], []
    for i, name in enumerate(names, 1):
        row.append(InlineKeyboardButton(name, callback_data=f"cat:{name}"))
        if i % 2 == 0: rows.append(row); row=[]
    if row: rows.append(row)
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="nav:menu")])
    return InlineKeyboardMarkup(rows)

def styles_kb_for_category(cat: str) -> InlineKeyboardMarkup:
    names = STYLE_CATEGORIES.get(cat, [])
    rows = [[InlineKeyboardButton(name, callback_data=f"style:{name}")] for name in names]
    rows.append([InlineKeyboardButton("⬅️ Категории", callback_data="nav:styles")])
    return InlineKeyboardMarkup(rows)

def avatars_kb(uid:int) -> InlineKeyboardMarkup:
    prof = load_profile(uid)
    cur = get_current_avatar_name(prof)
    names = sorted(prof["avatars"].keys())
    rows = []
    for n in names:
        label = f"{'✅ ' if n==cur else ''}{n}"
        rows.append([InlineKeyboardButton(label, callback_data=f"avatar:set:{n}")])
    rows.append([
        InlineKeyboardButton("➕ Новый", callback_data="avatar:new"),
        InlineKeyboardButton("🗑 Удалить", callback_data="avatar:del")
    ])
    rows.append([InlineKeyboardButton("⬅️ Меню", callback_data="nav:menu")])
    return InlineKeyboardMarkup(rows)

# ---------- Handlers ----------
ENROLL_FLAG: Dict[Tuple[int,str],bool] = {}  # ключ: (uid, avatar)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я создам твою персональную фотомодель из 10 фото и буду генерировать тебя "
        "в узнаваемых кино-сценах.\n\n"
        "1) «📸 Набор фото» — загрузка до 10 снимков в активный аватар.\n"
        "2) «🧪 Обучение» — тренировка LoRA для активного аватара.\n"
        "3) «🧭 Выбрать стиль» — получи варианты.\n"
        "4) «🔒 LOCKFACE» — фиксация лица.\n"
        "5) «🤖 Аватары» — несколько моделей в одном профиле.",
        reply_markup=main_menu_kb()
    )

async def nav_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    key = q.data.split(":",1)[1]
    if key == "styles":
        await q.message.reply_text("Выбери категорию:", reply_markup=categories_kb())
    elif key == "menu":
        await q.message.reply_text("Главное меню:", reply_markup=main_menu_kb())
    elif key == "enroll":
        await id_enroll(update, context)
    elif key == "train":
        await trainid_cmd(update, context)
    elif key == "status":
        await id_status(update, context)
    elif key == "gender":
        await gender_cmd(update, context)
    elif key == "lockface":
        await lockface_cmd(update, context)
    elif key == "avatars":
        uid = update.effective_user.id
        await q.message.reply_text("Аватары:", reply_markup=avatars_kb(uid))

async def avatar_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    uid = update.effective_user.id
    parts = q.data.split(":")
    action = parts[1]
    if action == "set":
        name = parts[2]
        set_current_avatar(uid, name)
        await q.message.reply_text(f"Активный аватар: {name}", reply_markup=avatars_kb(uid))
    elif action == "new":
        await q.message.reply_text("Создай аватар командой: /avatarnew <имя>\nНапример: /avatarnew travel")
    elif action == "del":
        await q.message.reply_text("Удаление: /avatardel <имя> --force")

async def styles_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Выбери категорию:", reply_markup=categories_kb())

async def cb_category(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    cat = q.data.split(":",1)[1]
    await q.message.reply_text(f"Стиль — {cat}.\nВыбери сцену:", reply_markup=styles_kb_for_category(cat))

async def id_enroll(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    prof = load_profile(uid)
    av_name = get_current_avatar_name(prof)
    ENROLL_FLAG[(uid, av_name)] = True
    await update.effective_message.reply_text(
        f"Набор включён для аватара «{av_name}». Пришли подряд до 10 фото (фронтально, без фильтров). "
        "Когда закончишь — нажми /iddone."
    )

async def id_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    prof = load_profile(uid)
    av_name = get_current_avatar_name(prof)
    ENROLL_FLAG[(uid, av_name)] = False
    av = get_avatar(prof, av_name)
    av["images"] = [p.name for p in list_ref_images(uid, av_name)]
    try:
        if not prof.get("gender"):
            prof["gender"] = auto_detect_gender(uid)
    except Exception:
        prof["gender"] = prof.get("gender") or "female"
    save_profile(uid, prof)
    await update.message.reply_text(
        f"Готово ✅ В аватаре «{av_name}» {len(av['images'])} фото.\n"
        f"Определённый пол: {prof.get('gender') or '—'}.\n"
        "Далее — нажми «🧪 Обучение» или /trainid.",
        reply_markup=main_menu_kb()
    )

async def id_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id; prof = load_profile(uid)
    av_name = get_current_avatar_name(prof)
    av = get_avatar(prof, av_name)
    await update.effective_message.reply_text(
        f"Активный аватар: {av_name}\n"
        f"Фото: {len(list_ref_images(uid, av_name))}\n"
        f"Статус обучения: {av.get('status') or '—'}\n"
        f"Модель: {av.get('finetuned_model') or '—'}\n"
        f"Версия: {av.get('finetuned_version') or '—'}\n"
        f"Пол (общий): {prof.get('gender') or '—'}\n"
        f"LOCKFACE (для аватара): {'on' if av.get('lockface') else 'off'}\n"
        f"Pretty mode: {'ON' if PRETTY_MODE else 'OFF'}\n"
        f"MAX_STEPS: {MAX_STEPS}"
    )

async def id_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    delete_profile(uid)
    await update.message.reply_text("Профиль очищен. Жми «📸 Набор фото» и загрузи снимки заново.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    prof = load_profile(uid)
    av_name = get_current_avatar_name(prof)
    if ENROLL_FLAG.get((uid, av_name)):
        refs = list_ref_images(uid, av_name)
        if len(refs) >= 10:
            await update.message.reply_text("Уже 10/10. Нажми /iddone."); return
        f = await update.message.photo[-1].get_file()
        data = await f.download_as_bytearray()
        save_ref_downscaled(avatar_dir(uid, av_name) / f"ref_{int(time.time()*1000)}.jpg", bytes(data))
        await update.message.reply_text(f"Сохранила ({len(refs)+1}/10) для «{av_name}». Ещё?")
    else:
        await update.message.reply_text("Сначала включи набор: «📸 Набор фото» или /idenroll.")

async def setgender_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not context.args or context.args[0].lower() not in ["male","female"]:
        await update.message.reply_text("Используй: /setgender female | /setgender male"); return
    prof = load_profile(uid); prof["gender"] = context.args[0].lower(); save_profile(uid, prof)
    await update.message.reply_text(f"Ок! Пол установлен: {prof['gender']}")

async def gender_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id; prof = load_profile(uid)
    await update.effective_message.reply_text(
        f"Пол (общий для всех аватаров): {prof.get('gender') or '—'}\n"
        "Можно сменить командой: /setgender female | /setgender male"
    )

async def lockface_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    prof = load_profile(uid)
    av = get_avatar(prof)  # текущий
    av["lockface"] = not av.get("lockface", False)
    save_profile(uid, prof)
    state = "включён" if av["lockface"] else "выключен"
    await update.effective_message.reply_text(f"LOCKFACE {state} для активного аватара. В рискованных пресетах он всё равно включается автоматически.")

# ---- Аватарные команды ----
async def avatarnew_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not context.args:
        await update.message.reply_text("Использование: /avatarnew <имя>\nПример: /avatarnew travel"); return
    name = re.sub(r"[^\w\-\.\@]+", "_", " ".join(context.args)).strip()[:32] or "noname"
    ensure_avatar(uid, name)
    set_current_avatar(uid, name)
    await update.message.reply_text(f"Создан и выбран аватар: {name}\nТеперь «📸 Набор фото» запишет снимки в этот аватар.", reply_markup=avatars_kb(uid))

async def avatarset_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not context.args:
        await update.message.reply_text("Использование: /avatarset <имя>"); return
    name = " ".join(context.args).strip()
    prof = load_profile(uid)
    if name not in prof["avatars"]:
        await update.message.reply_text(f"Нет такого аватара: {name}. Посмотри /avatarlist или создай /avatarnew <имя>"); return
    set_current_avatar(uid, name)
    await update.message.reply_text(f"Ок, активный аватар: {name}", reply_markup=avatars_kb(uid))

async def avatarlist_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    prof = load_profile(uid)
    cur = get_current_avatar_name(prof)
    lines = ["Твои аватары:"]
    for n, av in prof["avatars"].items():
        refs = len(list_ref_images(uid, n))
        lines.append(f"{'▶️' if n==cur else '  '} {n}: фото {refs}, статус: {av.get('status') or '—'}, верс: {av.get('finetuned_version') or '—'}")
    await update.message.reply_text("\n".join(lines), reply_markup=avatars_kb(uid))

async def avatardel_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not context.args:
        await update.message.reply_text("Использование: /avatardel <имя> --force"); return
    args = context.args[:]
    force = False
    if "--force" in args:
        force = True
        args.remove("--force")
    name = " ".join(args).strip()
    if name == "default":
        await update.message.reply_text("«default» удалять нельзя — на нём дом держится 😅"); return
    prof = load_profile(uid)
    if name not in prof["avatars"]:
        await update.message.reply_text(f"Аватар «{name}» не найден."); return
    if not force:
        await update.message.reply_text("Добавь флаг --force для подтверждения: /avatardel <имя> --force"); return
    try:
        del_avatar(uid, name)
        await update.message.reply_text(f"Аватар «{name}» удалён.", reply_markup=avatars_kb(uid))
    except Exception as e:
        await update.message.reply_text(f"Не удалось удалить: {e}")

# ---- Обучение / Генерация ----
def _dest_model_slug(avatar:str) -> str:
    if not DEST_OWNER: raise RuntimeError("REPLICATE_DEST_OWNER не задан.")
    return f"{DEST_OWNER}/{DEST_MODEL}"

def _ensure_destination_exists(slug: str):
    try: replicate.models.get(slug)
    except Exception:
        o, name = slug.split("/",1)
        raise RuntimeError(f"Целевая модель '{slug}' не найдена. Создай на https://replicate.com/create (owner={o}, name='{name}').")

def _pack_refs_zip(uid:int, avatar:str) -> Path:
    refs = list_ref_images(uid, avatar)
    if len(refs) < 10: raise RuntimeError("Нужно 10 фото для обучения.")
    zpath = avatar_dir(uid, avatar) / "train.zip"
    with ZipFile(zpath, "w") as z:
        for i, p in enumerate(refs, 1):
            z.write(p, arcname=f"img_{i:02d}.jpg")
    return zpath

def start_lora_training(uid:int, avatar:str) -> str:
    dest_model = _dest_model_slug(avatar); _ensure_destination_exists(dest_model)
    trainer_version = resolve_model_version(LORA_TRAINER_SLUG)
    zip_path = _pack_refs_zip(uid, avatar)
    client = Client(api_token=os.environ["REPLICATE_API_TOKEN"])
    with open(zip_path, "rb") as f:
        training = client.trainings.create(
            version=trainer_version,
            input={
                LORA_INPUT_KEY: f,
                "max_train_steps": LORA_MAX_STEPS,
                "lora_lr": LORA_LR,
                "use_face_detection_instead": LORA_USE_FACE_DET,
                "caption_prefix": LORA_CAPTION_PREF,
                "resolution": LORA_RESOLUTION,
            },
            destination=dest_model
        )
    prof = load_profile(uid)
    av = get_avatar(prof, avatar)
    av["training_id"] = training.id
    av["status"] = "starting"
    av["finetuned_model"] = dest_model
    save_profile(uid, prof)
    return training.id

def check_training_status(uid:int, avatar:str) -> Tuple[str, Optional[str], Optional[str]]:
    prof = load_profile(uid)
    av = get_avatar(prof, avatar)
    tid = av.get("training_id")
    if not tid:
        return ("not_started", None, None)

    client = Client(api_token=os.environ["REPLICATE_API_TOKEN"])
    tr = client.trainings.get(tid)

    status = getattr(tr, "status", None) or (tr.get("status") if isinstance(tr, dict) else None) or "unknown"

    err = None
    for key in ("error", "failure", "message", "reason", "detail"):
        try:
            v = getattr(tr, key, None) if not isinstance(tr, dict) else tr.get(key)
            if isinstance(v, str) and v.strip():
                err = v.strip(); break
            if isinstance(v, dict):
                msg = v.get("message") or v.get("detail")
                if msg: err = str(msg); break
        except Exception:
            pass

    if status != "succeeded":
        av["status"] = status
        save_profile(uid, prof)
        return (status, None, err)

    destination = getattr(tr, "destination", None) or (isinstance(tr, dict) and tr.get("destination")) \
                  or av.get("finetuned_model") or _dest_model_slug(avatar)
    version_id = getattr(tr, "output", None) if not isinstance(tr, dict) else tr.get("output")
    if isinstance(version_id, dict): version_id = version_id.get("id") or version_id.get("version")

    slug_with_version = None
    try:
        if version_id:
            replicate.models.get(f"{destination}:{version_id}")
            slug_with_version = f"{destination}:{version_id}"
    except Exception:
        pass
    if not slug_with_version:
        try:
            model_obj = replicate.models.get(destination)
            versions = list(model_obj.versions.list())
            if versions: slug_with_version = f"{destination}:{versions[0].id}"
        except Exception:
            slug_with_version = destination

    av["status"] = status
    av["finetuned_model"] = destination
    if slug_with_version and ":" in slug_with_version:
        av["finetuned_version"] = slug_with_version.split(":",1)[1]
    save_profile(uid, prof)
    return (status, slug_with_version, None)

def _pinned_slug(av: Dict[str, Any]) -> str:
    base = av.get("finetuned_model") or ""
    ver  = av.get("finetuned_version")
    return f"{base}:{ver}" if (base and ver) else base

async def trainid_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    prof = load_profile(uid)
    av_name = get_current_avatar_name(prof)
    if len(list_ref_images(uid, av_name)) < 10:
        await update.effective_message.reply_text(f"Нужно 10 фото в аватаре «{av_name}». Сначала «📸 Набор фото» и затем /iddone."); return
    await update.effective_message.reply_text(f"Запускаю обучение LoRA для «{av_name}» на Replicate…")
    try:
        training_id = await asyncio.to_thread(start_lora_training, uid, av_name)
        await update.effective_message.reply_text(f"Стартанула. ID: `{training_id}`\nПроверяй /trainstatus время от времени.")
        # Ссылка на логи
        if DEST_OWNER and DEST_MODEL and training_id:
            await update.effective_message.reply_text(
                f"Логи тренировки: https://replicate.com/{DEST_OWNER}/{DEST_MODEL}/trainings/{training_id}"
            )
    except Exception as e:
        logging.exception("trainid failed")
        await update.effective_message.reply_text(f"Не удалось запустить обучение: {e}")

async def trainstatus_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    prof = load_profile(uid)
    av_name = get_current_avatar_name(prof)

    status, slug_with_ver, err = await asyncio.to_thread(check_training_status, uid, av_name)
    av = get_avatar(prof, av_name)
    tid = av.get("training_id")
    train_url = f"https://replicate.com/{DEST_OWNER}/{DEST_MODEL}/trainings/{tid}" if (DEST_OWNER and DEST_MODEL and tid) else None

    if status == "succeeded" and slug_with_ver:
        await update.effective_message.reply_text(
            f"Готово ✅\nАватар: {av_name}\nМодель: `{slug_with_ver}`\nТеперь — «🧭 Выбрать стиль».",
            reply_markup=categories_kb()
        )
        return

    if status in ("starting", "processing", "running", "queued", "pending"):
        await update.effective_message.reply_text(
            f"Статус «{av_name}»: {status}… В процессе{(' — см. логи: ' + train_url) if train_url else ''}."
        )
        return

    if status in ("failed", "canceled"):
        msg = f"⚠️ Тренировка «{av_name}»: {status.upper()}."
        if err:
            msg += f"\nПричина: {err}"
        if train_url:
            msg += f"\nЛоги: {train_url}"
        msg += (
            "\n\nЧто проверить:\n"
            "• Целевая модель существует (REPLICATE_DEST_OWNER/DEST_MODEL).\n"
            "• 10 фото, лицо крупно, без фильтров.\n"
            "• Кредиты Replicate не закончились.\n"
            "• Верны LORA_TRAINER_SLUG и LORA_INPUT_KEY."
        )
        await update.effective_message.reply_text(msg)
        return

    await update.effective_message.reply_text(f"Статус «{av_name}»: {status}. {('Логи: ' + train_url) if train_url else ''}")

async def cb_style(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    preset = q.data.split(":",1)[1]
    await start_generation_for_preset(update, context, preset)

def _neg_with_gender(neg_base:str, gender_negative:str) -> str:
    return neg_base + (", " + gender_negative if gender_negative else "")

async def start_generation_for_preset(update: Update, context: ContextTypes.DEFAULT_TYPE, preset: str):
    uid = update.effective_user.id
    prof = load_profile(uid)
    av_name = get_current_avatar_name(prof)
    av = get_avatar(prof, av_name)

    if av.get("status") != "succeeded":
        await update.effective_message.reply_text(f"Модель аватара «{av_name}» ещё не готова. Сначала /trainid и дождись /trainstatus = succeeded.")
        return

    meta = STYLE_PRESETS[preset]
    gender = (prof.get("gender") or "female").lower()
    comp_text, (w,h) = _comp_text_and_size(meta.get("comp","half"))
    tone_text = _tone_text(meta.get("tone","daylight"))
    theme_boost = THEME_BOOST.get(preset, "")

    prompt_core, gender_negative = build_prompt(meta, gender, comp_text, tone_text, theme_boost)
    model_slug = _pinned_slug(av)

    # guidance/steps (pretty + лимит)
    guidance = max(3.0, SCENE_GUIDANCE.get(preset, GEN_GUIDANCE))
    desired_steps = max(52, GEN_STEPS) if PRETTY_MODE else max(40, GEN_STEPS)
    steps = min(MAX_STEPS, desired_steps)

    await update.effective_message.chat.send_action(ChatAction.UPLOAD_PHOTO)
    desc = meta.get("desc", preset)
    await update.effective_message.reply_text(f"🎬 {preset}\nАватар: {av_name}\n{desc}\n\nГенерирую ({gender}, {w}×{h}) …")

    try:
        seeds = [int(time.time()) & 0xFFFFFFFF, random.randrange(2**32), random.randrange(2**32)]
        urls = []
        neg_base = _neg_with_gender(NEGATIVE_PROMPT, gender_negative)

        do_lock = (av.get("lockface") is True) or (preset in RISKY_PRESETS)
        face_refs = list_ref_images(uid, av_name)
        face_ref = face_refs[0] if face_refs else None

        for s in seeds:
            if do_lock and face_ref and INSTANTID_SLUG:
                try:
                    inst_steps = min(MAX_STEPS, max(36, steps))
                    url = await asyncio.to_thread(
                        generate_with_instantid,
                        face_path=face_ref,
                        prompt=prompt_core,
                        steps=inst_steps,
                        guidance=guidance,
                        seed=s, w=w, h=h,
                        negative_prompt=neg_base
                    )
                except ReplicateError as e:
                    if "Model not found" in str(e) or "404" in str(e):
                        logger.warning("InstantID BAD ('%s'): %s", INSTANTID_SLUG, e)
                        url = await asyncio.to_thread(
                            generate_from_finetune,
                            model_slug=model_slug,
                            prompt=prompt_core,
                            steps=steps,
                            guidance=guidance,
                            seed=s, w=w, h=h,
                            negative_prompt=neg_base
                        )
                    else:
                        raise
            else:
                url = await asyncio.to_thread(
                    generate_from_finetune,
                    model_slug=model_slug,
                    prompt=prompt_core,
                    steps=steps,
                    guidance=guidance,
                    seed=s, w=w, h=h,
                    negative_prompt=neg_base
                )
            urls.append(url)

        for i, u in enumerate(urls, 1):
            await update.effective_message.reply_photo(photo=u, caption=f"{preset} • {av_name} • вариант {i}{' • 🔒' if do_lock else ''}")

        await update.effective_message.reply_text(
            "Хочешь фиксировать лицо во всех стилях — переключай LOCKFACE для этого аватара. "
            "Нужно мягче/глянец — PRETTY_MODE уже включён; можно выключить командой /pretty."
        )
    except Exception as e:
        logging.exception("generation failed")
        await update.effective_message.reply_text(f"Ошибка генерации: {e}")

# --- Pretty mode toggle
async def pretty_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global PRETTY_MODE
    PRETTY_MODE = not PRETTY_MODE
    await update.message.reply_text(f"Pretty mode: {'ON' if PRETTY_MODE else 'OFF'}")

# ---------- System ----------
async def _post_init(app): await app.bot.delete_webhook(drop_pending_updates=True)

def main():
    app = ApplicationBuilder().token(TOKEN).post_init(_post_init).build()

    # Команды
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("menu", styles_cmd))
    app.add_handler(CommandHandler("idenroll", id_enroll))
    app.add_handler(CommandHandler("iddone", id_done))
    app.add_handler(CommandHandler("idstatus", id_status))
    app.add_handler(CommandHandler("idreset", id_reset))
    app.add_handler(CommandHandler("setgender", setgender_cmd))
    app.add_handler(CommandHandler("gender", gender_cmd))
    app.add_handler(CommandHandler("lockface", lockface_cmd))
    app.add_handler(CommandHandler("trainid", trainid_cmd))
    app.add_handler(CommandHandler("trainstatus", trainstatus_cmd))
    app.add_handler(CommandHandler("styles", styles_cmd))
    app.add_handler(CommandHandler("pretty", pretty_cmd))

    # Аватары
    app.add_handler(CommandHandler("avatarnew", avatarnew_cmd))
    app.add_handler(CommandHandler("avatarset", avatarset_cmd))
    app.add_handler(CommandHandler("avatarlist", avatarlist_cmd))
    app.add_handler(CommandHandler("avatardel", avatardel_cmd))

    # Кнопки
    app.add_handler(CallbackQueryHandler(nav_cb, pattern=r"^nav:"))
    app.add_handler(CallbackQueryHandler(cb_category, pattern=r"^cat:"))
    app.add_handler(CallbackQueryHandler(cb_style, pattern=r"^style:"))
    app.add_handler(CallbackQueryHandler(avatar_cb, pattern=r"^avatar:"))

    # Фото
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Пинг слагов (InstantID — только если задан)
    _check_slug(LORA_TRAINER_SLUG, "LoRA trainer")
    if INSTANTID_SLUG:
        _check_slug(INSTANTID_SLUG, "InstantID")
    else:
        logger.info("InstantID disabled (no INSTANTID_SLUG).")

    logger.info("Bot up. Trainer=%s DEST=%s GEN=%dx%d steps=%s guidance=%s Pretty=%s MAX_STEPS=%s",
                LORA_TRAINER_SLUG, f"{DEST_OWNER}/{DEST_MODEL}",
                GEN_WIDTH, GEN_HEIGHT, GEN_STEPS, GEN_GUIDANCE, PRETTY_MODE, MAX_STEPS)
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
