# === Telegram LoRA Bot (Flux LoRA trainer + pinned versions + THEMATIC RU styles + auto-gender + commercial UX) ===
# Команды:
# /start      — приветствие и меню
# /idenroll   — включить набор (до 10 фото)
# /iddone     — сохранить профиль (пол авто по первому фото)
# /trainid    — обучить LoRA на Replicate (replicate/flux-lora-trainer)
# /trainstatus— статус обучения; при success фиксируем version_id
# /menu       — главное меню
# /styles     — список категорий стилей (кнопки)
# /gender     — показать определённый пол
# /setgender  — вручную задать пол: /setgender female | /setgender male
# /idreset    — очистить профиль (набор фото заново)
# ================================================================================================

import os, re, io, json, time, asyncio, logging, shutil, random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zipfile import ZipFile

import requests
from PIL import Image
import replicate
from replicate import Client

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

# Тренер LoRA (Flux LoRA trainer)
LORA_TRAINER_SLUG = os.getenv("LORA_TRAINER_SLUG", "replicate/flux-lora-trainer").strip()
LORA_INPUT_KEY    = os.getenv("LORA_INPUT_KEY", "input_images").strip()

# Классификатор пола (можно заменить на свой)
GENDER_MODEL_SLUG = os.getenv("GENDER_MODEL_SLUG", "nateraw/vit-age-gender").strip()

# Твики обучения (бережные и устойчивые)
LORA_MAX_STEPS     = int(os.getenv("LORA_MAX_STEPS", "2000"))
LORA_LR            = float(os.getenv("LORA_LR", "0.00008"))
LORA_USE_FACE_DET  = os.getenv("LORA_USE_FACE_DET", "true").lower() in ["1","true","yes","y"]
LORA_CAPTION_PREF  = os.getenv(
    "LORA_CAPTION_PREFIX",
    "a photo of a person, relaxed neutral expression, gentle smile, balanced facial proportions, soft jawline, "
    "open expressive eyes, symmetrical eye shape, clear irises"
).strip()
LORA_RESOLUTION    = int(os.getenv("LORA_RESOLUTION", "1024"))

# Генерация — детальнее, мягче, без «пластика»
GEN_STEPS     = int(os.getenv("GEN_STEPS", "46"))
GEN_GUIDANCE  = float(os.getenv("GEN_GUIDANCE", "3.4"))
GEN_WIDTH     = int(os.getenv("GEN_WIDTH", "896"))
GEN_HEIGHT    = int(os.getenv("GEN_HEIGHT", "1152"))

NEGATIVE_PROMPT = (
    "cartoon, anime, 3d, cgi, beauty filter, skin smoothing, overprocessed, oversharpen, "
    "waxy skin, plastic skin, blur, lowres, distorted, bad anatomy, watermark, text, logo, "
    "puffy face, swollen face, hypertrophic masseter, wide jaw, clenched jaw, "
    "pursed lips, duckface, overfilled lips, deep nasolabial folds, "
    "squinting, narrow eyes, small eyes, asymmetrical eyes, cross-eyed, wall-eyed, lazy eye, "
    "droopy eyelids, misaligned pupils, extra pupils, fused eyes"
)
AESTHETIC_SUFFIX = (
    ", photo-realistic, visible skin pores, natural color, soft filmic contrast, no beautification"
)

# ---------- Тематика/композиция/тон ----------
# comp: closeup (крупный / грудной), half (по пояс), full (полный рост)
# tone: daylight, warm, cool, noir, neon, candle
def _comp_text_and_size(comp: str) -> Tuple[str, Tuple[int,int]]:
    if comp == "closeup":
        return ("from chest up, portrait composition", (896, 1152))
    if comp == "half":
        return ("half body, balanced composition", (896, 1344))
    return ("full body, natural perspective", (896, 1408))

def _tone_text(tone: str) -> str:
    return {
        "daylight": "soft natural daylight, neutral colors",
        "warm":     "golden hour warmth, gentle highlights",
        "cool":     "cool cinematic light, clean color balance",
        "noir":     "high contrast noir lighting, strong rim light, subtle haze",
        "neon":     "neon signs, wet reflections, cinematic backlight, vibrant saturation",
        "candle":   "warm candlelight, soft glow, volumetric rays",
    }.get(tone, "balanced soft lighting")

# ---------- Стили (русские названия, гендер-aware p_f / p_m / p, + comp/tone) ----------
Style = Dict[str, Any]
STYLE_PRESETS: Dict[str, Style] = {
    # Портрет / полупояс
    "Портрет у окна": {"p": "естественный портрет у большого окна, легкое боке", "comp": "closeup", "tone": "daylight"},
    "Портрет 85мм": {"p": "реалистичный портрет с эффектом 85мм объектива, малая глубина резкости", "comp": "closeup", "tone": "warm"},
    "Бьюти студия": {"p": "бьюти-портрет, чистый студийный свет, минимальный макияж", "comp": "closeup", "tone": "daylight"},
    "Кинопортрет": {"p": "кинематографичный портрет, рембрандтовский свет, мягкая пленочная зернистость", "comp": "closeup", "tone": "cool"},
    "Фильм-нуар (портрет)": {"p": "портрет в стиле кино-нуар, дым, высокая контрастность", "comp": "closeup", "tone": "noir"},

    # Современные сцены
    "Стритвэр город": {
        "p_f": "современный стритвэр, кроп-топ и джоггеры, городская улица",
        "p_m": "современный стритвэр, худи и джоггеры, городская улица",
        "comp": "half", "tone": "daylight"
    },
    "Вечерний выход": {
        "p_f": "элегантное вечернее платье на красной дорожке",
        "p_m": "классический смокинг на красной дорожке",
        "comp": "half", "tone": "warm"
    },
    "Бизнес": {
        "p_f": "деловой костюм, лобби современного офиса",
        "p_m": "деловой костюм, лобби современного офиса",
        "comp": "half", "tone": "daylight"
    },
    "Ночной неон": {"p": "улица в дождь, яркие неоновые вывески, отражения в лужах", "comp": "half", "tone": "neon"},

    # Профессии (выразительно-тематические)
    "Врач у палаты": {
        "p_f": "врач в белом халате и шапочке, стетоскоп; фон: больничная палата",
        "p_m": "врач в белом халате и шапочке, стетоскоп; фон: больничная палата",
        "comp": "half", "tone": "daylight"
    },
    "Хирург операционная": {"p": "хирург в шапочке и маске, хирургическая форма; фон: операционная с приборами", "comp": "half", "tone": "cool"},
    "Шеф-повар кухня": {"p": "шеф-повар в кителе; фон: профессиональная кухня, пламя и пар", "comp": "half", "tone": "warm"},
    "Учёный лаборатория": {"p": "лабораторный халат, пробирки и стеклянная посуда; фон: современная лаборатория", "comp": "half", "tone": "cool"},
    "Боксер на ринге": {
        "p_f": "боксерша в перчатках; фон: ринг, пот, жёсткий верхний свет",
        "p_m": "боксер в перчатках; фон: ринг, пот, жёсткий верхний свет",
        "comp": "half", "tone": "cool"
    },
    "Фитнес зал": {
        "p_f": "спортивный топ и легинсы; фон: тренажерный зал, драматичная контровая",
        "p_m": "майка и шорты; фон: тренажерный зал, драматичная контровая",
        "comp": "half", "tone": "cool"
    },

    # Приключения / Экшн
    "Приключенец (руины)": {
        "p_f": "исследовательница гробниц, тактический костюм, перчатки без пальцев; фон: древние руины",
        "p_m": "исследователь гробниц, тактический костюм, перчатки без пальцев; фон: древние руины",
        "comp": "full", "tone": "warm"
    },
    "Пустынный исследователь": {"p": "шарф, карго-экипировка; фон: песчаные дюны и каньон", "comp": "full", "tone": "warm"},
    "Горы снег": {"p": "альпинистская куртка, кошки/ледоруб; фон: заснеженный гребень и небо", "comp": "full", "tone": "cool"},
    "Серфер": {"p": "гидрокостюм, доска; фон: океанская волна и брызги", "comp": "full", "tone": "warm"},

    # Фэнтези / История
    "Эльфийская знать": {
        "p_f": "эльфийская королева в струящемся платье; фон: лесной храм и лучи света",
        "p_m": "эльфийский король в плаще и доспехах; фон: лесной храм и лучи света",
        "comp": "full", "tone": "candle"
    },
    "Самурай в храме": {"p": "самурайские доспехи и катана; фон: двор синтоистского храма с фонарями", "comp": "full", "tone": "warm"},
    "Средневековый рыцарь": {"p": "полный комплект доспехов и плащ; фон: замковый турнирный двор", "comp": "full", "tone": "daylight"},
    "Пират на палубе": {"p": "пиратская шляпа и сабля; фон: палуба корабля, штормовое море и туман", "comp": "full", "tone": "cool"},
    "Вестерн на коне": {
        "p_f": "ковбойская шляпа, кожаная куртка; сидит верхом на лошади; фон: пыльная улица Дикого Запада",
        "p_m": "ковбойская шляпа, кожаная куртка; сидит верхом на лошади; фон: пыльная улица Дикого Запада",
        "comp": "full", "tone": "warm"
    },

    # Sci-Fi / Киберпанк
    "Киберпанк улица": {"p": "кожаная куртка; фон: неоновые вывески, мокрый асфальт, голограммы", "comp": "full", "tone": "neon"},
    "Космический скафандр": {"p": "реалистичный EVA-скафандр; фон: звёздное небо, ангар корабля", "comp": "full", "tone": "cool"},
    "Космопилот на мостике": {"p": "лётный комбинезон, шлем под мышкой; фон: мостик звездолёта", "comp": "half", "tone": "cool"},
}

# Категории для UX
STYLE_CATEGORIES: Dict[str, List[str]] = {
    "Портреты": ["Портрет у окна", "Портрет 85мм", "Бьюти студия", "Кинопортрет", "Фильм-нуар (портрет)"],
    "Современные": ["Стритвэр город", "Вечерний выход", "Бизнес", "Ночной неон"],
    "Профессии": ["Врач у палаты", "Хирург операционная", "Шеф-повар кухня", "Учёный лаборатория", "Боксер на ринге", "Фитнес зал"],
    "Приключения": ["Приключенец (руины)", "Пустынный исследователь", "Горы снег", "Серфер"],
    "Фэнтези/История": ["Эльфийская знать", "Самурай в храме", "Средневековый рыцарь", "Пират на палубе", "Вестерн на коне"],
    "Sci-Fi": ["Киберпанк улица", "Космический скафандр", "Космопилот на мостике"],
}

# ---------- logging ----------
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("bot")

# ---------- storage ----------
DATA_DIR = Path("profiles"); DATA_DIR.mkdir(exist_ok=True)
def user_dir(uid:int) -> Path: p = DATA_DIR / str(uid); p.mkdir(parents=True, exist_ok=True); return p
def list_ref_images(uid:int) -> List[Path]: return sorted(user_dir(uid).glob("ref_*.jpg"))
def profile_path(uid:int) -> Path: return user_dir(uid) / "profile.json"
def load_profile(uid:int) -> Dict[str, Any]:
    p = profile_path(uid)
    if p.exists(): return json.loads(p.read_text())
    return {"images": [], "training_id": None, "finetuned_model": None, "finetuned_version": None, "status": None, "gender": None}
def save_profile(uid:int, prof:Dict[str,Any]): profile_path(uid).write_text(json.dumps(prof))

def save_ref_downscaled(path: Path, raw: bytes, max_side=1024, quality=92):
    im = Image.open(io.BytesIO(raw)).convert("RGB"); im.thumbnail((max_side, max_side)); im.save(path, "JPEG", quality=quality)

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
    refs = list_ref_images(uid)
    if not refs: return "female"
    g = _infer_gender_from_image(refs[0])
    return g or "female"

# ---------- LoRA training ----------
def _pack_refs_zip(uid:int) -> Path:
    refs = list_ref_images(uid)
    if len(refs) < 10: raise RuntimeError("Нужно 10 фото для обучения.")
    zpath = user_dir(uid) / "train.zip"
    with ZipFile(zpath, "w") as z:
        for i, p in enumerate(refs, 1):
            z.write(p, arcname=f"img_{i:02d}.jpg")
    return zpath

def _dest_model_slug() -> str:
    if not DEST_OWNER: raise RuntimeError("REPLICATE_DEST_OWNER не задан.")
    return f"{DEST_OWNER}/{DEST_MODEL}"

def _ensure_destination_exists(slug: str):
    try: replicate.models.get(slug)
    except Exception:
        o, name = slug.split("/",1)
        raise RuntimeError(f"Целевая модель '{slug}' не найдена. Создай на https://replicate.com/create (owner={o}, name='{name}').")

def start_lora_training(uid:int) -> str:
    dest_model = _dest_model_slug(); _ensure_destination_exists(dest_model)
    trainer_version = resolve_model_version(LORA_TRAINER_SLUG)
    zip_path = _pack_refs_zip(uid)
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
    prof["training_id"] = training.id
    prof["status"] = "starting"
    prof["finetuned_model"] = dest_model
    save_profile(uid, prof)
    return training.id

def check_training_status(uid:int) -> Tuple[str, Optional[str]]:
    prof = load_profile(uid); tid = prof.get("training_id")
    if not tid: return ("not_started", None)
    client = Client(api_token=os.environ["REPLICATE_API_TOKEN"])
    tr = client.trainings.get(tid)
    status = getattr(tr, "status", None) or (tr.get("status") if isinstance(tr, dict) else None) or "unknown"
    if status != "succeeded":
        prof["status"] = status; save_profile(uid, prof); return (status, None)

    destination = getattr(tr, "destination", None) or (isinstance(tr, dict) and tr.get("destination")) \
                  or prof.get("finetuned_model") or _dest_model_slug()
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

    prof["status"] = status
    prof["finetuned_model"] = destination
    if slug_with_version and ":" in slug_with_version:
        prof["finetuned_version"] = slug_with_version.split(":",1)[1]
    save_profile(uid, prof)
    return (status, slug_with_version)

def _pinned_slug(prof: Dict[str, Any]) -> str:
    base = prof.get("finetuned_model") or ""
    ver  = prof.get("finetuned_version")
    return f"{base}:{ver}" if (base and ver) else base

# ---------- Генерация ----------
def _prompt_for_gender(meta: Style, gender: str) -> str:
    if gender == "female" and meta.get("p_f"): return meta["p_f"]
    if gender == "male" and meta.get("p_m"): return meta["p_m"]
    return meta.get("p", "")

def _beauty_guardrail() -> str:
    # Умный «стабилизатор» лица/тела
    return ("balanced facial proportions, symmetrical face, no facial elongation, "
            "soft cheek contour, no cheek bulge, smooth jawline, "
            "realistic body proportions, proportional shoulders and waist, natural posture")

def generate_from_finetune(model_slug:str, prompt:str, steps:int, guidance:float, seed:int, w:int, h:int) -> str:
    mv = resolve_model_version(model_slug)
    out = replicate.run(mv, input={
        "prompt": prompt + AESTHETIC_SUFFIX,
        "negative_prompt": NEGATIVE_PROMPT,
        "width": w, "height": h,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "seed": seed,
    })
    url = extract_any_url(out)
    if not url: raise RuntimeError("Empty output")
    return url

# ---------- UI ----------
def main_menu_kb() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("🧭 Выбрать стиль", callback_data="nav:styles")],
        [InlineKeyboardButton("📸 Набор фото", callback_data="nav:enroll"),
         InlineKeyboardButton("🧪 Обучение", callback_data="nav:train")],
        [InlineKeyboardButton("ℹ️ Мой статус", callback_data="nav:status"),
         InlineKeyboardButton("⚙️ Пол", callback_data="nav:gender")]
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
    rows, row = [], []
    for i, name in enumerate(names, 1):
        row.append(InlineKeyboardButton(name, callback_data=f"style:{name}"))
        if i % 1 == 0: rows.append(row); row=[]
    rows.append([InlineKeyboardButton("⬅️ Категории", callback_data="nav:styles")])
    return InlineKeyboardMarkup(rows)

# ---------- Handlers ----------
ENROLL_FLAG: Dict[int,bool] = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я создам твою персональную фотомодель из 10 фото и буду генерировать тебя "
        "в **узнаваемых сценах** — от врача в палате до ковбоя на лошади.\n\n"
        "Как пользоваться:\n"
        "1) Нажми «📸 Набор фото», пришли до 10 снимков (фронтально, без фильтров).\n"
        "2) «🧪 Обучение» — запущу тренировку LoRA.\n"
        "3) «🧭 Выбрать стиль» — ткни сцену, и я сгенерирую.\n\n"
        "Красиво. Реалистично. Без «пластика».",
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

async def styles_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Выбери категорию:", reply_markup=categories_kb())

async def cb_category(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    cat = q.data.split(":",1)[1]
    await q.message.reply_text(f"Стиль — {cat}.\nВыбери сцену:", reply_markup=styles_kb_for_category(cat))

async def id_enroll(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id; ENROLL_FLAG[uid] = True
    await update.effective_message.reply_text(
        "Набор включён. Пришли подряд до 10 фото (фронтально, без фильтров, расслабленное лицо). "
        "Когда закончишь — нажми /iddone."
    )

async def id_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id; ENROLL_FLAG[uid] = False
    prof = load_profile(uid)
    prof["images"] = [p.name for p in list_ref_images(uid)]
    try:
        prof["gender"] = auto_detect_gender(uid)
    except Exception:
        prof["gender"] = prof.get("gender") or "female"
    save_profile(uid, prof)
    await update.message.reply_text(
        f"Готово ✅ В профиле {len(prof['images'])} фото.\n"
        f"Определённый пол: {prof['gender']}.\n"
        "Далее — нажми «🧪 Обучение» или команду /trainid.",
        reply_markup=main_menu_kb()
    )

async def id_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id; prof = load_profile(uid)
    await update.effective_message.reply_text(
        f"Фото: {len(list_ref_images(uid))}\n"
        f"Статус обучения: {prof.get('status') or '—'}\n"
        f"Модель: {prof.get('finetuned_model') or '—'}\n"
        f"Версия: {prof.get('finetuned_version') or '—'}\n"
        f"Пол: {prof.get('gender') or '—'}"
    )

async def id_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    p = user_dir(uid)
    if p.exists(): shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True); ENROLL_FLAG[uid] = False
    await update.message.reply_text("Профиль очищен. Жми «📸 Набор фото» и загрузи снимки заново.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if ENROLL_FLAG.get(uid):
        refs = list_ref_images(uid)
        if len(refs) >= 10:
            await update.message.reply_text("Уже 10/10. Нажми /iddone."); return
        f = await update.message.photo[-1].get_file()
        data = await f.download_as_bytearray()
        save_ref_downscaled(user_dir(uid) / f"ref_{int(time.time()*1000)}.jpg", bytes(data))
        await update.message.reply_text(f"Сохранила ({len(refs)+1}/10). Ещё?")
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
        f"Определённый пол: {prof.get('gender') or '—'}\n"
        "Можно сменить командой: /setgender female | /setgender male"
    )

async def trainid_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if len(list_ref_images(uid)) < 10:
        await update.effective_message.reply_text("Нужно 10 фото. Сначала «📸 Набор фото» и затем /iddone."); return
    await update.effective_message.reply_text("Запускаю обучение LoRA на Replicate…")
    try:
        training_id = await asyncio.to_thread(start_lora_training, uid)
        await update.effective_message.reply_text(f"Стартанула. ID: `{training_id}`\nПроверяй /trainstatus каждые 5–10 минут.")
    except Exception as e:
        logger.exception("trainid failed"); await update.effective_message.reply_text(f"Не удалось запустить обучение: {e}")

async def trainstatus_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    status, slug_with_ver = await asyncio.to_thread(check_training_status, uid)
    if slug_with_ver and status == "succeeded":
        await update.effective_message.reply_text(f"Готово ✅\nСтатус: {status}\nМодель: `{slug_with_ver}`\nТеперь — «🧭 Выбрать стиль».", reply_markup=categories_kb())
    else:
        await update.effective_message.reply_text(f"Статус: {status}. Ещё в процессе…")

async def cb_style(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    preset = q.data.split(":",1)[1]
    await start_generation_for_preset(update, context, preset)

async def start_generation_for_preset(update: Update, context: ContextTypes.DEFAULT_TYPE, preset: str):
    uid = update.effective_user.id
    prof = load_profile(uid)
    if prof.get("status") != "succeeded":
        await update.effective_message.reply_text("Модель ещё не готова. Сначала /trainid и дождись /trainstatus = succeeded.")
        return

    meta = STYLE_PRESETS[preset]
    gender = (prof.get("gender") or "female").lower()
    base_prompt = _prompt_for_gender(meta, gender)
    comp_text, (w,h) = _comp_text_and_size(meta.get("comp","half"))
    tone_text = _tone_text(meta.get("tone","daylight"))

    prompt_core = (
        f"{base_prompt}, {comp_text}, {tone_text}, "
        "exact facial identity, no geometry change, relaxed neutral expression, gentle smile, "
        "open expressive eyes, natural almond-shaped eyes, clear irises, "
        f"{_beauty_guardrail()}, cinematic tone"
    )

    model_slug = _pinned_slug(prof)

    await update.effective_message.chat.send_action(ChatAction.UPLOAD_PHOTO)
    await update.effective_message.reply_text(f"Генерирую: {preset} ({gender}, {w}×{h}) … 🎨")
    try:
        seed = int(time.time()) & 0xFFFFFFFF
        url = await asyncio.to_thread(generate_from_finetune, model_slug, prompt_core, GEN_STEPS, GEN_GUIDANCE, seed, w, h)
        await update.effective_message.reply_photo(photo=url, caption=f"Готово ✨\nСтиль: {preset}")
    except Exception as e:
        logging.exception("generation failed")
        await update.effective_message.reply_text(f"Ошибка генерации: {e}")

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
    app.add_handler(CommandHandler("trainid", trainid_cmd))
    app.add_handler(CommandHandler("trainstatus", trainstatus_cmd))
    app.add_handler(CommandHandler("styles", styles_cmd))

    # Кнопки
    app.add_handler(CallbackQueryHandler(nav_cb, pattern=r"^nav:"))
    app.add_handler(CallbackQueryHandler(cb_category, pattern=r"^cat:"))
    app.add_handler(CallbackQueryHandler(cb_style, pattern=r"^style:"))

    # Фото
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    logger.info("Bot up. Trainer=%s DEST=%s GEN=%dx%d steps=%s guidance=%s",
                LORA_TRAINER_SLUG, f"{DEST_OWNER}/{DEST_MODEL}", GEN_WIDTH, GEN_HEIGHT, GEN_STEPS, GEN_GUIDANCE)
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()






