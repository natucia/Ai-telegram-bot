# === Telegram LoRA Bot (Flux LoRA trainer + HARD styles + Redis persist + Identity/Gender locks) ===
# Требования: python-telegram-bot==20.7, replicate==0.31.0, pillow==10.4.0, redis==5.0.1

import os, re, io, json, time, asyncio, logging, shutil, random, contextlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zipfile import ZipFile

import replicate
from replicate import Client
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

# Тренер LoRA (Flux LoRA trainer)
LORA_TRAINER_SLUG = os.getenv("LORA_TRAINER_SLUG", "replicate/flux-lora-trainer").strip()
LORA_INPUT_KEY    = os.getenv("LORA_INPUT_KEY", "input_images").strip()

# Классификатор пола (опционально; можно заменить)
GENDER_MODEL_SLUG = os.getenv("GENDER_MODEL_SLUG", "nateraw/vit-age-gender").strip()

# --- Твики обучения ---
LORA_MAX_STEPS     = int(os.getenv("LORA_MAX_STEPS", "1400"))
LORA_LR            = float(os.getenv("LORA_LR", "0.00006"))
LORA_USE_FACE_DET  = os.getenv("LORA_USE_FACE_DET", "true").lower() in ["1","true","yes","y"]
LORA_RESOLUTION    = int(os.getenv("LORA_RESOLUTION", "1024"))
LORA_CAPTION_PREF  = os.getenv(
    "LORA_CAPTION_PREFIX",
    "a high quality photo of the same person, neutral expression, gentle smile, "
    "balanced facial proportions, soft jawline, clear eyes"
).strip()

# --- Генерация (жёстче слушает стиль) ---
GEN_STEPS     = int(os.getenv("GEN_STEPS", "48"))
GEN_GUIDANCE  = float(os.getenv("GEN_GUIDANCE", "4.2"))
GEN_WIDTH     = int(os.getenv("GEN_WIDTH", "896"))
GEN_HEIGHT    = int(os.getenv("GEN_HEIGHT", "1152"))

NEGATIVE_PROMPT = (
    "cartoon, anime, 3d, cgi, beauty filter, skin smoothing, waxy, overprocessed, oversharpen, "
    "lowres, blur, jpeg artifacts, text, watermark, logo, bad anatomy, extra fingers, short fingers, "
    "puffy face, swollen face, bulky masseter, wide jaw, clenched jaw, duckface, overfilled lips, "
    "narrow eyes, cross-eye, asymmetrical eyes, misaligned pupils, "
    "bodybuilder female, extreme makeup, heavy contouring, "
    "casual clothing, casual street background, plain studio background, selfie, tourist photo, "
    "identity drift, different person, ethnicity change, age change, "
    "fisheye, lens distortion, warped face, stretched face, deformed skull, "
    "double pupils, misaligned eyes"
)
AESTHETIC_SUFFIX = (
    ", photorealistic, visible skin texture, natural color, soft filmic contrast, gentle micro-sharpen, no beautification"
)

def _beauty_guardrail() -> str:
    return (
        "exact facial identity, identity preserved, "
        "balanced facial proportions, symmetrical face, natural oval, soft jawline, "
        "smooth cheek contour, relaxed neutral expression, subtle gentle smile, "
        "open expressive almond-shaped eyes, clean catchlights, clear irises, "
        "realistic body proportions, proportional shoulders and waist, natural posture"
    )

def _anti_distort() -> str:
    return ("no fisheye, no lens distortion, no warping, no stretched face, "
            "natural perspective, proportional head size, realistic human anatomy")

def _gender_lock(gender:str) -> Tuple[str, str]:
    if gender == "female":
        pos = "female woman, feminine facial features"
        neg = "male, man, masculine face, beard, stubble, mustache, adam's apple, broad jaw, thick neck"
    else:
        pos = "male man, masculine facial features"
        neg = "female, woman, feminine face, makeup eyelashes, narrow shoulders, breasts"
    return pos, neg

# ---------- Композиция/линза/свет ----------
def _comp_text_and_size(comp: str) -> Tuple[str, Tuple[int,int]]:
    if comp == "closeup":
        return ("portrait framing from chest up, 85mm lens look, subject distance 1.2m, shallow depth of field",
                (896, 1152))
    if comp == "half":
        return ("half body framing, 85mm lens look, subject distance 2.5m, shallow depth of field",
                (896, 1344))
    return ("full body framing, 85mm lens look, subject distance 5m, natural perspective",
            (896, 1408))

def _tone_text(tone: str) -> str:
    return {
        "daylight": "soft natural daylight, neutral colors",
        "warm":     "golden hour warmth, gentle highlights",
        "cool":     "cool cinematic light, clean color balance",
        "noir":     "high contrast noir lighting, strong rim light, subtle haze",
        "neon":     "neon signs, wet reflections, cinematic backlight, vibrant saturation",
        "candle":   "warm candlelight, soft glow, volumetric rays",
    }.get(tone, "balanced soft lighting")

# ---------- Стили (HARD STYLE-LOCK) ----------
Style = Dict[str, Any]
STYLE_PRESETS: Dict[str, Style] = {
    # Портреты
    "Портрет у окна": {
        "desc": "Крупный кинопортрет у большого окна; мягкая тень от рамы, живое боке.",
        "role": "cinematic window light portrait",
        "outfit": "neutral top",
        "props": "soft bokeh, window frame shadow on background",
        "bg": "large window with daylight glow, interior blur",
        "comp": "closeup", "tone": "daylight"
    },
    "Портрет 85мм": {
        "desc": "Классика 85мм — мизерная ГРИП.",
        "role": "85mm look beauty portrait",
        "outfit": "minimal elegant top",
        "props": "creamy bokeh, shallow depth of field",
        "bg": "neutral cinematic backdrop",
        "comp": "closeup", "tone": "warm"
    },
    "Бьюти студия": {
        "desc": "Чистый студийный свет, кожа без «пластика».",
        "role": "studio beauty portrait",
        "outfit": "clean minimal outfit",
        "props": "catchlights, controlled specular highlights",
        "bg": "seamless studio background with soft light gradients",
        "comp": "closeup", "tone": "daylight"
    },
    "Кинопортрет": {
        "desc": "Рембрандтовский свет и мягкая плёнка.",
        "role": "cinematic rembrandt light portrait",
        "outfit": "neutral film wardrobe",
        "props": "subtle film grain",
        "bg": "moody backdrop with soft falloff",
        "comp": "closeup", "tone": "cool"
    },
    "Фильм-нуар (портрет)": {
        "desc": "Дым, жёсткие тени, свет из жалюзи.",
        "role": "film noir portrait",
        "outfit": "vintage attire",
        "props": "venetian blinds light pattern, cigarette smoke curling",
        "bg": "high contrast noir backdrop",
        "comp": "closeup", "tone": "noir"
    },

    # Современные
    "Стритвэр город": {
        "desc": "Уличный лук, город дышит.",
        "role": "streetwear fashion look",
        "outfit_f": "crop top and joggers, sneakers",
        "outfit": "hoodie and joggers, sneakers",
        "props": "glass reflections, soft film grain",
        "bg": "daytime city street with graffiti and shop windows",
        "comp": "half", "tone": "daylight"
    },
    "Вечерний выход": {
        "desc": "Красная дорожка и блеск.",
        "role": "celebrity on red carpet",
        "outfit_f": "elegant evening gown",
        "outfit": "classic tuxedo",
        "props": "press lights, velvet ropes, photographers",
        "bg": "red carpet event entrance",
        "comp": "half", "tone": "warm"
    },
    "Бизнес": {
        "desc": "Лобби стеклянного офиса, строгая геометрия.",
        "role": "corporate executive portrait",
        "outfit_f": "tailored business suit",
        "outfit": "tailored business suit",
        "props": "tablet or folder",
        "bg": "modern glass office lobby with depth",
        "comp": "half", "tone": "daylight"
    },
    "Ночной неон": {
        "desc": "Кибернуар, мокрый асфальт.",
        "role": "urban night scene",
        "outfit_f": "long coat, boots",
        "outfit": "long coat, boots",
        "props": "colored reflections on wet asphalt, light rain droplets",
        "bg": "neon signs and steam from manholes",
        "comp": "half", "tone": "neon"
    },

    # Профессии
    "Врач у палаты": {
        "desc": "Белый халат, стетоскоп, палата за спиной.",
        "role": "medical doctor",
        "outfit_f": "white lab coat, scrub cap, stethoscope",
        "outfit": "white lab coat, scrub cap, stethoscope",
        "props": "ID badge, clipboard",
        "bg": "hospital ward interior with bed and monitors",
        "comp": "half", "tone": "daylight"
    },
    "Хирург операционная": {
        "desc": "Холодные приборы и блики.",
        "role": "surgeon in the operating room",
        "outfit": "surgical scrubs, mask, cap, gloves",
        "props": "surgical lights and instruments",
        "bg": "operating theatre with equipment",
        "comp": "half", "tone": "cool"
    },
    "Шеф-повар кухня": {
        "desc": "Огонь и пар, энергия ресторана.",
        "role": "head chef",
        "outfit": "white chef jacket and apron",
        "props": "pan with flames, stainless steel counters, copper pans",
        "bg": "professional restaurant kitchen",
        "comp": "half", "tone": "warm"
    },
    "Учёный лаборатория": {
        "desc": "Стекло, приборы, подсветки.",
        "role": "scientist in a lab",
        "outfit": "lab coat, safety glasses",
        "props": "flasks, pipettes, LED indicators",
        "bg": "modern laboratory benches and glassware",
        "comp": "half", "tone": "cool"
    },
    "Боксер на ринге": {
        "desc": "Жёсткий верхний свет, пот, канаты.",
        "role": "boxer on the ring",
        "outfit_f": "boxing sports bra and shorts, gloves",
        "outfit": "boxing shorts and gloves, mouthguard visible",
        "props": "ring ropes, sweat sheen, tape on wrists",
        "bg": "boxing ring under harsh top lights",
        "comp": "half", "tone": "cool"
    },
    "Фитнес зал": {
        "desc": "Контровый свет между тренажёрами.",
        "role": "fitness athlete training",
        "outfit_f": "sports bra and leggings",
        "outfit": "tank top and shorts",
        "props": "chalk dust, dumbbells or cable machine",
        "bg": "gym with machines and dramatic backlight",
        "comp": "half", "tone": "cool"
    },

    # Приключения / Экшн
    "Приключенец (руины)": {
        "desc": "Пыльные лучи, древние камни.",
        "role_f": "tomb raider explorer",
        "role": "tomb raider explorer",
        "outfit_f": "tactical outfit, fingerless gloves, utility belt",
        "outfit": "tactical outfit, fingerless gloves, utility belt",
        "props": "leather straps patina, map tube",
        "bg": "ancient sandstone ruins with sun rays and dust motes",
        "comp": "full", "tone": "warm"
    },
    "Пустынный исследователь": {
        "desc": "Дюны, песок и жар.",
        "role": "desert explorer",
        "outfit": "scarf, cargo outfit, boots",
        "props": "sand blowing in wind",
        "bg": "sand dunes and canyon under harsh sun",
        "comp": "full", "tone": "warm"
    },
    "Горы снег": {
        "desc": "Суровая красота высокогорья.",
        "role": "alpinist",
        "outfit": "mountain jacket, harness, crampons",
        "props": "ice axe in hand, spindrift",
        "bg": "snow ridge and blue shadows, cloudy sky",
        "comp": "full", "tone": "cool"
    },
    "Серфер": {
        "desc": "Брызги, солнечные блики, доска.",
        "role": "surfer athlete on a wave",
        "outfit_f": "black wetsuit",
        "outfit": "black wetsuit",
        "props": "a visible surfboard under the subject's arm or feet, water spray, droplets",
        "bg": "ocean wave breaking, golden backlight",
        "comp": "full", "tone": "warm"
    },

    # Фэнтези / История
    "Эльфийская знать": {
        "desc": "Лесной храм и лучи в тумане.",
        "role_f": "elven queen in a regal pose",
        "role": "elven king in a regal pose",
        "outfit_f": "flowing emerald gown with golden embroidery, delicate crown",
        "outfit": "ornate armor with emerald cloak, elegant crown",
        "props": "elven jewelry, filigree filigree patterns",
        "bg": "ancient forest temple, god rays in mist",
        "comp": "full", "tone": "candle"
    },
    "Самурай в храме": {
        "desc": "Лакированные доспехи, фонари, листья.",
        "role": "samurai warrior in a shrine courtyard",
        "outfit": "lacquered samurai armor, kabuto helmet",
        "props": "katana visible in hand",
        "bg": "Shinto shrine with lanterns, falling leaves",
        "comp": "full", "tone": "warm"
    },
    "Средневековый рыцарь": {
        "desc": "Полированный латный доспех, штандарты.",
        "role": "medieval knight",
        "outfit": "full plate armor with cloak",
        "props": "sword and shield",
        "bg": "castle tournament yard with banners and dust",
        "comp": "full", "tone": "daylight"
    },
    "Пират на палубе": {
        "desc": "Треуголка, сабля, мокрая палуба, шторм.",
        "role_f": "pirate captain",
        "role": "pirate captain",
        "outfit_f": "tricorn hat, leather corset, white shirt",
        "outfit": "tricorn hat, leather vest, white shirt",
        "props": "cutlass in hand, rope rigging, wet wood highlights",
        "bg": "ship deck in storm, sails and rigging, sea spray, gulls",
        "comp": "full", "tone": "cool"
    },
    "Древняя Греция": {
        "desc": "Белый мрамор и лазурь.",
        "role_f": "ancient Greek goddess",
        "role": "ancient Greek hero",
        "outfit_f": "white chiton with gold trim, diadem",
        "outfit": "white chiton with gold trim, laurel wreath",
        "props": "gold accessories",
        "bg": "white marble colonnade, statues, olive trees, turquoise pool",
        "comp": "half", "tone": "warm"
    },
    "Королева": {
        "desc": "Коронованная особа в тронном зале.",
        "role_f": "queen on a throne",
        "role": "king on a throne",
        "outfit_f": "royal gown with long train, jeweled crown, scepter",
        "outfit": "royal robe with golden embroidery, jeweled crown, scepter",
        "props": "ornate jewelry, velvet textures",
        "bg": "grand castle throne room with chandeliers and marble columns",
        "comp": "half", "tone": "warm"
    },

    # Sci-Fi
    "Киберпанк улица": {
        "desc": "Неон, мокрый асфальт, голограммы.",
        "role": "cyberpunk character walking in the street",
        "outfit_f": "leather jacket, high-waist pants, boots",
        "outfit": "leather jacket, techwear pants, boots",
        "props": "holographic billboards, overhead cables",
        "bg": "neon signs, wet asphalt reflections, steam from manholes",
        "comp": "full", "tone": "neon"
    },
    "Космический скафандр": {
        "desc": "Хард sci-fi.",
        "role": "astronaut",
        "outfit": "realistic EVA spacesuit",
        "props": "helmet reflections, suit details",
        "bg": "starfield and spaceship hangar",
        "comp": "full", "tone": "cool"
    },
    "Космопилот на мостике": {
        "desc": "Пульт, индикаторы, готовность к гиперпрыжку.",
        "role": "starship pilot on the bridge",
        "outfit": "flight suit, helmet under arm",
        "props": "control panels with glowing indicators",
        "bg": "spaceship bridge interior",
        "comp": "half", "tone": "cool"
    },
}

STYLE_CATEGORIES: Dict[str, List[str]] = {
    "Портреты": ["Портрет у окна", "Портрет 85мм", "Бьюти студия", "Кинопортрет", "Фильм-нуар (портрет)"],
    "Современные": ["Стритвэр город", "Вечерний выход", "Бизнес", "Ночной неон"],
    "Профессии": ["Врач у палаты", "Хирург операционная", "Шеф-повар кухня", "Учёный лаборатория", "Боксер на ринге", "Фитнес зал"],
    "Приключения": ["Приключенец (руины)", "Пустынный исследователь", "Горы снег", "Серфер"],
    "Фэнтези/История": ["Эльфийская знать", "Самурай в храме", "Средневековый рыцарь", "Пират на палубе", "Древняя Греция", "Королева"],
    "Sci-Fi": ["Киберпанк улица", "Космический скафандр", "Космопилот на мостике"],
}

THEME_BOOST = {
    "Пират на палубе": "rope rigging, storm clouds, wet highlights on wood, sea spray, gulls",
    "Древняя Греция": "ionic capitals, olive trees, turquoise water reflections, gold trim accents",
    "Ночной неон":     "rain droplets on lens, colored reflections on wet asphalt",
    "Фильм-нуар (портрет)": "venetian blinds light pattern, cigarette smoke curling, deep black shadows",
    "Приключенец (руины)": "floating dust motes in sunrays, chipped sandstone blocks, leather straps patina",
    "Горы снег":      "spindrift blown by wind, crampon scratches on ice, distant ridge line",
    "Киберпанк улица":"holographic billboards flicker, cable bundles overhead, neon kanji signs",
    "Серфер":         "rimlight on water droplets, sun flare",
    "Королева":       "subtle film grain, ceremonial ambience",
}

# Точечный буст послушности для сцен, где часто уводит
SCENE_GUIDANCE = {
    "Киберпанк улица": 4.8,
    "Космический скафандр": 4.6,
    "Самурай в храме": 4.6,
    "Средневековый рыцарь": 4.6,
}

# ---------- logging ----------
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("bot")

# ---------- storage (Redis persistent, fallback к FS если чего-то нет) ----------
DATA_DIR = Path("profiles"); DATA_DIR.mkdir(exist_ok=True)

def user_dir(uid:int) -> Path:
    p = DATA_DIR / str(uid); p.mkdir(parents=True, exist_ok=True); return p
def list_ref_images(uid:int) -> List[Path]:
    return sorted(user_dir(uid).glob("ref_*.jpg"))
def profile_path(uid:int) -> Path:
    return user_dir(uid) / "profile.json"

DEFAULT_PROFILE = {
    "images": [], "training_id": None, "finetuned_model": None,
    "finetuned_version": None, "status": None, "gender": None
}

REDIS_URL = os.getenv("REDIS_URL", "").strip()
_redis = None
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
                return {**DEFAULT_PROFILE, **json.loads(raw)}
        except Exception as e:
            logger.warning("Redis load_profile failed: %s", e)
    p = profile_path(uid)
    if p.exists():
        with contextlib.suppress(Exception):
            return {**DEFAULT_PROFILE, **json.loads(p.read_text())}
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

# ---------- авто-пол (опционально) ----------
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

    if role or outfit or props or bg:
        core = ", ".join([
            _style_lock(role, outfit, props, bg, comp_text),
            tone_text,
            gpos,
            "same person as the training photos, no ethnicity change, no age change, exact facial identity, identity preserved",
            "photorealistic, realistic body proportions, natural skin texture, filmic look",
            anti,
            _beauty_guardrail(),
            theme_boost
        ])
        core += ", the costume and background must clearly communicate the role; avoid plain portrait"
        return core, gneg

    base_prompt = _prompt_for_gender(meta, gender)
    core = ", ".join([
        f"{base_prompt}, {comp_text}, {tone_text}",
        gpos,
        "same person as the training photos, exact facial identity, identity preserved",
        "cinematic key light and rim light, soft bounce fill, film grain subtle",
        "skin tone faithful",
        anti,
        _beauty_guardrail(),
        theme_boost
    ])
    return core, gneg

def generate_from_finetune(model_slug:str, prompt:str, steps:int, guidance:float, seed:int, w:int, h:int, negative_prompt:str) -> str:
    mv = resolve_model_version(model_slug)
    out = replicate.run(mv, input={
        "prompt": prompt + AESTHETIC_SUFFIX,
        "negative_prompt": negative_prompt,
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
    rows = []
    for name in names:
        rows.append([InlineKeyboardButton(name, callback_data=f"style:{name}")])
    rows.append([InlineKeyboardButton("⬅️ Категории", callback_data="nav:styles")])
    return InlineKeyboardMarkup(rows)

# ---------- Handlers ----------
ENROLL_FLAG: Dict[int,bool] = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я создам твою персональную фотомодель из 10 фото и буду генерировать тебя "
        "в узнаваемых кино-сценах — от королевы в тронном зале до серфера на волне.\n\n"
        "1) «📸 Набор фото» — пришли до 10 снимков.\n"
        "2) «🧪 Обучение» — тренировка LoRA.\n"
        "3) «🧭 Выбрать стиль» — получи 3 варианта.",
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
        "Набор включён. Пришли подряд до 10 фото (фронтально, без фильтров). "
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
        "Далее — нажми «🧪 Обучение» или /trainid.",
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
    delete_profile(uid)
    ENROLL_FLAG[uid] = False
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
        await update.effective_message.reply_text(f"Стартанула. ID: `{training_id}`\nПроверяй /trainstatus время от времени.")
    except Exception as e:
        logger.exception("trainid failed"); await update.effective_message.reply_text(f"Не удалось запустить обучение: {e}")

async def trainstatus_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    status, slug_with_ver = await asyncio.to_thread(check_training_status, uid)
    if slug_with_ver and status == "succeeded":
        await update.effective_message.reply_text(
            f"Готово ✅\nСтатус: {status}\nМодель: `{slug_with_ver}`\nТеперь — «🧭 Выбрать стиль».",
            reply_markup=categories_kb()
        )
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
    comp_text, (w,h) = _comp_text_and_size(meta.get("comp","half"))
    tone_text = _tone_text(meta.get("tone","daylight"))
    theme_boost = THEME_BOOST.get(preset, "")

    prompt_core, gender_negative = build_prompt(meta, gender, comp_text, tone_text, theme_boost)
    model_slug = _pinned_slug(prof)

    # guidance с учётом сцены
    guidance = max(3.2, SCENE_GUIDANCE.get(preset, GEN_GUIDANCE))

    await update.effective_message.chat.send_action(ChatAction.UPLOAD_PHOTO)
    desc = meta.get("desc", preset)
    await update.effective_message.reply_text(f"🎬 {preset}\n{desc}\n\nГенерирую ({gender}, {w}×{h}) …")

    try:
        seeds = [int(time.time()) & 0xFFFFFFFF, random.randrange(2**32), random.randrange(2**32)]
        urls = []
        neg_base = NEGATIVE_PROMPT + (", " + gender_negative if gender_negative else "")
        for s in seeds:
            url = await asyncio.to_thread(
                generate_from_finetune,
                model_slug=model_slug,
                prompt=prompt_core,
                steps=max(40, GEN_STEPS),
                guidance=guidance,
                seed=s, w=w, h=h,
                negative_prompt=neg_base
            )
            urls.append(url)

        for i, u in enumerate(urls, 1):
            await update.effective_message.reply_photo(photo=u, caption=f"{preset} • вариант {i}")

        await update.effective_message.reply_text("Выбери лучший вариант — напиши «этот нрав», добавлю апскейл и вариации (по запросу).")
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
