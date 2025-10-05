# ================== Telegram LoRA Bot (Flux LoRA trainer + pinned versions + RU styles + auto-gender) ==================
# Фичи:
# - /idenroll -> загрузка до 10 фото
# - /iddone   -> сохранить профиль (авто-детект пола по первому фото)
# - /trainid  -> обучение LoRA на Replicate (replicate/flux-lora-trainer) в едином репозитории модели
# - /trainstatus -> статус; при успехе сохраняем конкретный version_id (pinned)
# - /styles   -> показать русские стили кнопками
# - (нажатие кнопки стиля) -> сразу генерация по зафиксированной версии юзера, авто-размер под стиль
# - /gender   -> показать определённый пол; /setgender male|female -> вручную задать
# =======================================================================================================================

import os, re, io, json, time, asyncio, logging, shutil
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

# -------------------- ENV --------------------
TOKEN = os.getenv("BOT_TOKEN", "")
os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN", "")

if not TOKEN or not re.match(r"^\d+:[A-Za-z0-9_-]{20,}$", TOKEN):
    raise RuntimeError("Некорректный BOT_TOKEN.")
if not os.getenv("REPLICATE_API_TOKEN"):
    raise RuntimeError("Нет REPLICATE_API_TOKEN.")

DEST_OWNER        = os.getenv("REPLICATE_DEST_OWNER", "").strip()               # напр. natucia
DEST_MODEL        = os.getenv("REPLICATE_DEST_MODEL", "yourtwin-lora").strip()  # единый репозиторий

# Тренер LoRA именно Flux LoRA trainer
LORA_TRAINER_SLUG = os.getenv("LORA_TRAINER_SLUG", "replicate/flux-lora-trainer").strip()
# Имя входного поля для архива с фото (у тренера это обычно "input_images")
LORA_INPUT_KEY    = os.getenv("LORA_INPUT_KEY", "input_images").strip()

# Классификатор пола (Replicate). Можно заменить на свой.
GENDER_MODEL_SLUG = os.getenv("GENDER_MODEL_SLUG", "nateraw/vit-age-gender").strip()

# Твики обучения (бережные, реалистичные)
LORA_MAX_STEPS     = int(os.getenv("LORA_MAX_STEPS", "2000"))
LORA_LR            = float(os.getenv("LORA_LR", "0.00008"))
LORA_USE_FACE_DET  = os.getenv("LORA_USE_FACE_DET", "true").lower() in ["1","true","yes","y"]
LORA_CAPTION_PREF  = os.getenv("LORA_CAPTION_PREFIX",
    "a photo of a person, relaxed neutral expression, gentle smile, soft jawline, balanced facial proportions, natural look, open expressive eyes, symmetrical eye shape, clear irises"
).strip()
LORA_RESOLUTION    = int(os.getenv("LORA_RESOLUTION", "1024"))

# Генерация — анти-пластик + «открытые глаза»
GEN_STEPS     = int(os.getenv("GEN_STEPS", "44"))
GEN_GUIDANCE  = float(os.getenv("GEN_GUIDANCE", "3.6"))
GEN_WIDTH     = int(os.getenv("GEN_WIDTH", "832"))
GEN_HEIGHT    = int(os.getenv("GEN_HEIGHT", "1216"))

NEGATIVE_PROMPT = (
    "cartoon, anime, 3d, cgi, overprocessed, oversharpen, beauty filter, skin smoothing, "
    "waxy skin, plastic skin, blur, lowres, deformed, distorted, bad anatomy, watermark, text, logo, "
    "puffy face, swollen face, chubby cheeks, hypertrophic masseter, wide jaw, clenched jaw, "
    "pursed lips, duckface, overfilled lips, nasolabial fold accent, "
    "squinting, narrow eyes, small eyes, asymmetrical eyes, cross-eyed, wall-eyed, lazy eye, "
    "droopy eyelids, heavy eyelids, misaligned pupils, extra pupils, fused eyes"
)
AESTHETIC_SUFFIX = (
    ", photo-realistic, visible skin pores, natural color, soft filmic contrast, "
    "balanced soft lighting, no beautification"
)

# -------------------- СТИЛИ (русские названия, гендер-aware p_f / p_m / p, + рекомендуемый размер) --------------------
Style = Dict[str, Any]
STYLE_PRESETS: Dict[str, Style] = {
    # ===== ПОРТРЕТЫ =====
    "Портрет 85мм": {
        "p": "ультра реалистичный портрет, эффект объектива 85мм, малая глубина резкости, мягкий ключевой свет, "
             "раскрытые выразительные глаза, естественная миндалевидная форма, четкие радужки, симметрия черт",
        "w": 896, "h": 1152
    },
    "Натуральный": { "p": "ультра реалистичный портрет, нейтральная цветокоррекция, расслабленное лицо, мягкая улыбка, мягкая линия челюсти", "w": 896, "h": 1152 },
    "Натуральный стройный": { "p": "реалистичный портрет, деликатные скулы, мягкая линия челюсти, сбалансированные пропорции, расслабленное лицо", "w": 896, "h": 1152 },
    "Бьюти мягкий свет": { "p": "бьюти-портрет, чистый студийный свет, мягкая диффузия, минимальный макияж", "w": 1024, "h": 1024 },
    "Vogue обложка": { "p": "обложечный бьюти-портрет, студийные софтбоксы, выверенные цвета" },
    "Окно мягкий свет": { "p": "портрет у окна, мягкая естественная диффузия, легкое боке" },
    "Кинопортрет": { "p": "кинематографичный портрет, рембрандтовский свет, легкая пленочная зернистость", "w": 960, "h": 1280 },
    "Муди свет": { "p": "мрачноватый портрет, контролируемые тени, деликатная контровая подсветка" },
    "Нуар крупно": { "p": "черно-белый фильм-нуар, разрезающий свет, дым, высокий контраст", "w": 896, "h": 1152 },
    "Ретро 50-е": { "p": "студийный портрет 1950-х, пленочный цвет, бабочка-лайт", "w": 896, "h": 1152 },
    "Глэм 80-е": { "p": "глам-портрет 1980-х, мягкая дымка, цветная контровая, журнальный вид", "w": 896, "h": 1152 },
    "Зимний портрет": { "p": "уличный зимний портрет, мягкий снег, уютное освещение, теплые тона кожи", "w": 896, "h": 1152 },
    "Пляж закат (портрет)": { "p": "портрет на пляже в золотой час, теплый контровый свет, волосы подсвечены", "w": 896, "h": 1152 },
    "Нуар мягкий цвет": { "p": "кинопортрет в духе нуара, цветной, мягкая контровая, дымка", "w": 896, "h": 1152 },

    # ===== FULL BODY — современное =====
    "Стритвэр": {
        "p_f": "полный рост, современный стритвэр, кроп-топ и джоггеры, городский переулок, пасмурный мягкий свет, аутентично",
        "p_m": "полный рост, современный стритвэр, худи и джоггеры, городский переулок, пасмурный мягкий свет, аутентично",
        "w": 832, "h": 1344
    },
    "Вечерний выход": {
        "p_f": "полный рост, элегантное вечернее платье, красная дорожка, мягкие софты, киношное боке",
        "p_m": "полный рост, классический смокинг, красная дорожка, мягкие софты, киношное боке",
        "w": 832, "h": 1344
    },
    "Бизнес": {
        "p_f": "полный рост, современный женский костюм, лобби офиса, мягкий дневной свет",
        "p_m": "полный рост, современный мужской костюм, лобби офиса, мягкий дневной свет",
        "w": 832, "h": 1344
    },
    "Фитнес зал": {
        "p_f": "полный рост, реалистичная фитнес-съемка в зале, топ и легинсы, легкий блеск пота, драматичная контровая",
        "p_m": "полный рост, реалистичная фитнес-съемка в зале, майка и шорты, легкий блеск пота, драматичная контровая",
        "w": 832, "h": 1344
    },
    "Ночной город неон": { "p": "полный рост, неоновая улица ночью, отражения в лужах, киношная контровая", "w": 832, "h": 1344 },
    "Фестиваль": { "p": "полный рост, образ для музыкального фестиваля, пыль заката, цветные огни, живой кадр", "w": 832, "h": 1344 },
    "Пляж закат (фулл)": { "p": "полный рост, пляж на закате, теплый контровый свет, естественная динамика позы", "w": 896, "h": 1408 },

    # ===== ПРИКЛЮЧЕНИЯ / ЭКШН =====
    "Приключенец": {
        "p_f": "полный рост, атлетичная исследовательница, тактический костюм, перчатки без пальцев, пояс, ботинки, динамичная поза, руины храма",
        "p_m": "полный рост, атлетичный рейдер гробниц, тактический костюм, перчатки без пальцев, пояс, ботинки, динамичная поза, руины храма",
        "w": 832, "h": 1344
    },
    "Пустынный исследователь": { "p": "полный рост, исследователь пустыни, шарф, карго-аутфит, каменный каньон, теплый закат", "w": 832, "h": 1344 },
    "Горы снег": { "p": "полный рост, альпинист(ка), пуховка, кошки, снежный гребень, драматичное небо", "w": 896, "h": 1408 },
    "Фридайвер": { "p": "полный рост, реалистичный фридайвер, длинные ласты, подводный голубой свет, лучи солнца, частицы", "w": 896, "h": 1408 },
    "Серфер": {
        "p_f": "полный рост, серфершa в гидрике, волна на фоне, блики воды, солнце",
        "p_m": "полный рост, серфер в гидрике, волна на фоне, блики воды, солнце",
        "w": 896, "h": 1408
    },
    "Байкер": { "p": "полный рост, кафе-рейсер мотоцикл, кожаная куртка, вечерний город, лампы накаливания", "w": 896, "h": 1408 },
    "Скейтер": { "p": "полный рост, трюк на скейте в воздухе, городской скейтпарк, золотой час", "w": 832, "h": 1344 },

    # ===== ФЭНТЕЗИ / ИСТОРИЯ =====
    "Эльфийская знать": {
        "p_f": "полный рост, эльфийская королева, струящееся платье, лесной храм, мягкие лучи",
        "p_m": "полный рост, эльфийский король, украшенные доспехи и плащ, лесной храм, мягкие лучи",
        "w": 960, "h": 1440
    },
    "Аркан маг": {
        "p_f": "полный рост, чародейка, темное струящееся платье, парящие руны, лунный свет",
        "p_m": "полный рост, чернокнижник, длинный плащ, парящие руны, лунный свет",
        "w": 896, "h": 1408
    },
    "Вампир готика": { "p": "полный рост, готический вампир, интерьер собора, свечи, кьяроскуро", "w": 896, "h": 1408 },
    "Самурай": { "p": "полный рост, реалистичные самурайские доспехи, катана, храмовый двор, сумерки и фонари", "w": 896, "h": 1408 },
    "Средневековый рыцарь": { "p": "полный рост, реалистичные доспехи, плащ, замковый двор, пасмурно", "w": 896, "h": 1408 },
    "Пират": { "p": "полный рост, пиратский костюм, треуголка, деревянный пирс, штормовое море, туман", "w": 896, "h": 1408 },
    "Древняя Греция": {
        "p_f": "полный рост, греческий хитон, мраморные колонны, мягкое солнце",
        "p_m": "полный рост, греческие доспехи/гиматий, мраморные колонны, мягкое солнце",
        "w": 896, "h": 1408
    },
    "Египет: царица/фараон": {
        "p_f": "полный рост, египетская царица, золотые украшения, храмовые рельефы, теплый свет факелов",
        "p_m": "полный рост, египетский фараон, золотые украшения, храмовые рельефы, теплый свет факелов",
        "w": 896, "h": 1408
    },
    "Рим: патриций/патрицианка": {
        "p_f": "полный рост, римская патрицианка, туника и стола, мраморный атрий",
        "p_m": "полный рост, римский патриций, туника и тога, мраморный атрий",
        "w": 896, "h": 1408
    },

    # ===== SCI-FI / КИБЕРПАНК =====
    "Киберпанк улица": {
        "p_f": "полный рост, неоновая киберпанк-улица, дождь, голограммы, отражения, кожаная куртка, контровой свет",
        "p_m": "полный рост, неоновая киберпанк-улица, дождь, голограммы, отражения, кожаная куртка, контровой свет",
        "w": 832, "h": 1344
    },
    "Космический скафандр": { "p": "полный рост, реалистичный EVA-скафандр, звездное поле, ангар корабля, hard-surface детали", "w": 960, "h": 1440 },
    "Космопилот": { "p": "полный рост, пилот космоистребителя, летный комбинезон, шлем под мышкой, подсветка ангара, объемная дымка", "w": 896, "h": 1408 },
    "Андроид": { "p": "полный рост, реалистичный андроид-гуманоид, минималистичная броня, неоновые отражения, дождь", "w": 832, "h": 1344 },
    "Космоопера командир": { "p": "полный рост, космический адмирал, глубокий черный мундир, мостик звездолёта, звездные поля", "w": 896, "h": 1408 },

    # ===== ПРОФЕССИИ / СПОРТ =====
    "Йога студия": { "p": "полный рост, поза йоги, дневной свет из окон, деревянный пол, спокойная атмосфера", "w": 832, "h": 1344 },
    "Бегун дорожка": {
        "p_f": "полный рост, бегунья на стадионе, шаг в движении, фон в смазе",
        "p_m": "полный рост, бегун на стадионе, шаг в движении, фон в смазе",
        "w": 832, "h": 1344
    },
    "Боксер ринг": {
        "p_f": "полный рост, боксерша, перчатки подняты, ринг, жесткий верхний свет, капли пота",
        "p_m": "полный рост, боксер, перчатки подняты, ринг, жесткий верхний свет, капли пота",
        "w": 896, "h": 1408
    },
    "Шеф-повар": { "p": "полный рост, китель повара, профессиональная кухня, пар и огонь, теплый свет", "w": 832, "h": 1344 },
    "Врач": { "p": "полный рост, коридор больницы, медицинская форма, стетоскоп, чистый мягкий свет", "w": 832, "h": 1344 },
    "Ученый лаборатория": { "p": "полный рост, лаборатория, белый халат, стеклянная посуда, мягкий холодный свет", "w": 832, "h": 1344 },

    # ===== СТИЛИЗАЦИИ / ЭПОХИ =====
    "Нуар детектив": { "p": "полный рост, детектив фильм-нуар, тренч, федора, дождливый переулок, жесткая контровая", "w": 832, "h": 1344 },
    "Вестерн": {
        "p_f": "полный рост, ковгерл, пыльная улица, деревянный салун, полуденное солнце",
        "p_m": "полный рост, ковбой, пыльная улица, деревянный салун, полуденное солнце",
        "w": 896, "h": 1408
    },
    "Балет": {
        "p_f": "полный рост, балерина в студии, пачка, арабеск, мягкий свет из окна",
        "p_m": "полный рост, артист балета, гран-жете, мягкий свет из окна",
        "w": 832, "h": 1344
    },
    "Ретро 20-е": {
        "p_f": "полный рост, стиль 1920-х, платье флапер, ар-деко клуб, легкий дым",
        "p_m": "полный рост, мужской костюм 1920-х, ар-деко клуб, легкий дым",
        "w": 832, "h": 1344
    },
    "Синтвейв 80-е": { "p": "полный рост, синтвейв-сцена, неоновая сетка, закатный горизонт, хром-элементы, легкая дымка", "w": 896, "h": 1408 },
    "Барокко бал": {
        "p_f": "полный рост, барочный бальный наряд, корсет, кринолин, свечи, зал с фресками",
        "p_m": "полный рост, барочный камзол и парик, свечи, зал с фресками",
        "w": 896, "h": 1408
    },
}

# -------------------- logging --------------------
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("bot")

# -------------------- storage --------------------
DATA_DIR = Path("profiles"); DATA_DIR.mkdir(exist_ok=True)
def user_dir(uid:int) -> Path:
    p = DATA_DIR / str(uid); p.mkdir(parents=True, exist_ok=True); return p
def list_ref_images(uid:int) -> List[Path]: return sorted(user_dir(uid)).glob("ref_*.jpg")
def list_ref_images(uid:int) -> List[Path]:
    return sorted(user_dir(uid).glob("ref_*.jpg"))
def profile_path(uid:int) -> Path: return user_dir(uid) / "profile.json"
def load_profile(uid:int) -> Dict[str, Any]:
    p = profile_path(uid)
    if p.exists(): return json.loads(p.read_text())
    return {"images": [], "training_id": None, "finetuned_model": None, "finetuned_version": None, "status": None, "gender": None}
def save_profile(uid:int, prof:Dict[str,Any]): profile_path(uid).write_text(json.dumps(prof))

def save_ref_downscaled(path: Path, raw: bytes, max_side=1024, quality=92):
    im = Image.open(io.BytesIO(raw)).convert("RGB"); im.thumbnail((max_side, max_side)); im.save(path, "JPEG", quality=quality)

# -------------------- replicate helpers --------------------
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

def replicate_run_flexible(model: str, inputs_list: Iterable[dict]) -> str:
    last = None
    for payload in inputs_list:
        try:
            out = replicate.run(model, input=payload)
            url = extract_any_url(out)
            if not url: raise RuntimeError("Empty output")
            return url
        except Exception as e:
            last = e; logger.warning("Replicate rejected payload: %s", e)
    raise last or RuntimeError("Все варианты отклонены")

# -------------------- Gender detection --------------------
def _infer_gender_from_image(path: Path) -> Optional[str]:
    """Через Replicate-классификатор: возвращаем 'female' / 'male' / None."""
    try:
        client = Client(api_token=os.environ["REPLICATE_API_TOKEN"])
        version_slug = resolve_model_version(GENDER_MODEL_SLUG)
        with open(path, "rb") as img_b:
            for key in ["image", "img", "input_image", "file"]:
                try:
                    pred = client.predictions.create(version=version_slug, input={key: img_b})
                    pred.wait()
                    out = pred.output
                    g = None
                    if isinstance(out, dict):
                        g = (out.get("gender") or out.get("label") or "").lower()
                    else:
                        g = str(out).lower()
                    if "female" in g or "woman" in g: return "female"
                    if "male" in g or "man" in g: return "male"
                except Exception as e:
                    logger.warning("Gender model input key '%s' failed: %s", key, e)
                    continue
    except Exception as e:
        logger.warning("Gender inference error: %s", e)
    return None

def auto_detect_gender(uid:int) -> str:
    """Берём первое фото; если не получилось — фоллбэк 'female'."""
    refs = list_ref_images(uid)
    if not refs: return "female"
    guess = _infer_gender_from_image(refs[0])
    return guess or "female"

# -------------------- LoRA training --------------------
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
    if isinstance(version_id, dict):
        version_id = version_id.get("id") or version_id.get("version")

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

def generate_from_finetune(model_slug:str, prompt:str, steps:int, guidance:float, seed:int, w:int, h:int) -> str:
    mv = resolve_model_version(model_slug)  # если вдруг без версии — latest
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

# -------------------- UI helpers --------------------
def styles_keyboard() -> InlineKeyboardMarkup:
    # делим на ряды по 2, чтобы русские названия не обрезались
    names = list(STYLE_PRESETS.keys())
    rows, row = [], []
    for i, name in enumerate(names, 1):
        row.append(InlineKeyboardButton(name, callback_data=f"style:{name}"))
        if i % 2 == 0:
            rows.append(row); row = []
    if row: rows.append(row)
    return InlineKeyboardMarkup(rows)

# -------------------- handlers --------------------
ENROLL_FLAG: Dict[int,bool] = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Обучаю твою персональную LoRA по 10 фото и генерю без новых фото.\n\n"
        "1) /idenroll — включить набор (до 10 фото)\n"
        "2) /iddone — сохранить профиль (пол автоматически по первому фото)\n"
        "3) /trainid — запустить обучение\n"
        "4) /trainstatus — проверить статус (после успеха зафиксируем твою версию)\n"
        "5) /styles — выбрать стиль кнопками (я сразу сгенерирую)\n"
        "6) /gender — показать определённый пол; /setgender male|female — вручную"
    )

async def id_enroll(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id; ENROLL_FLAG[uid] = True
    await update.message.reply_text("Набор включён. Пришли подряд до 10 фото. Когда закончишь — /iddone.")

async def id_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id; ENROLL_FLAG[uid] = False
    prof = load_profile(uid)
    prof["images"] = [p.name for p in list_ref_images(uid)]
    # авто-детект пола по первому фото (с фоллбэком)
    try:
        prof["gender"] = auto_detect_gender(uid)
    except Exception:
        prof["gender"] = prof.get("gender") or "female"
    save_profile(uid, prof)
    await update.message.reply_text(f"Готово. В профиле {len(prof['images'])} фото. Пол: {prof['gender']}. Далее: /trainid.")

async def id_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id; prof = load_profile(uid)
    await update.message.reply_text(
        f"Фото: {len(list_ref_images(uid))}\n"
        f"Статус: {prof.get('status') or '—'}\n"
        f"Модель: {prof.get('finetuned_model') or '—'}\n"
        f"Версия: {prof.get('finetuned_version') or '—'}\n"
        f"Пол: {prof.get('gender') or '—'}"
    )

async def id_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    p = user_dir(uid)
    if p.exists(): shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True); ENROLL_FLAG[uid] = False
    await update.message.reply_text("Профиль очищен. /idenroll чтобы собрать заново.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if ENROLL_FLAG.get(uid):
        refs = list_ref_images(uid)
        if len(refs) >= 10: await update.message.reply_text("Уже 10/10. Жми /iddone."); return
        f = await update.message.photo[-1].get_file()
        data = await f.download_as_bytearray()
        save_ref_downscaled(user_dir(uid) / f"ref_{int(time.time()*1000)}.jpg", bytes(data))
        await update.message.reply_text(f"Сохранила ({len(refs)+1}/10). Ещё?")
    else:
        await update.message.reply_text("Сначала /idenroll. После /iddone → /trainid.")

async def styles_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Выбери стиль — я сразу сгенерирую:", reply_markup=styles_keyboard())

def _prompt_for_gender(meta: Style, gender: str) -> str:
    if gender == "female" and meta.get("p_f"): return meta["p_f"]
    if gender == "male" and meta.get("p_m"): return meta["p_m"]
    return meta.get("p", "")

async def start_generation_for_preset(update: Update, context: ContextTypes.DEFAULT_TYPE, preset: str):
    uid = update.effective_user.id
    prof = load_profile(uid)
    if prof.get("status") != "succeeded":
        await update.effective_message.reply_text("Модель ещё не готова. Сначала /trainid и дождись /trainstatus = succeeded.")
        return

    meta = STYLE_PRESETS[preset]
    gender = (prof.get("gender") or "female").lower()
    base_prompt = _prompt_for_gender(meta, gender)

    prompt_core = (
        f"{base_prompt}, точная идентичность лица, без изменения геометрии, "
        "расслабленное нейтральное выражение, мягкая улыбка, "
        "раскрытые выразительные глаза, естественная миндалевидная форма, четкие радужки, "
        "симметричная форма глаз, корректный интервал между глазами, эффект 85мм портретного объектива"
    )

    w = int(meta.get("w") or GEN_WIDTH)
    h = int(meta.get("h") or GEN_HEIGHT)
    model_slug = _pinned_slug(prof)

    await update.effective_message.chat.send_action(ChatAction.UPLOAD_PHOTO)
    await update.effective_message.reply_text(f"Генерирую: {preset} ({gender}, {w}×{h}) … 🎨")
    try:
        seed = int(time.time()) & 0xFFFFFFFF
        url = await asyncio.to_thread(
            generate_from_finetune, model_slug, prompt_core, GEN_STEPS, GEN_GUIDANCE, seed, w, h
        )
        await update.effective_message.reply_photo(photo=url, caption=f"Готово ✨\nСтиль: {preset}")
    except Exception as e:
        logging.exception("generation failed")
        await update.effective_message.reply_text(f"Ошибка генерации: {e}")

async def cb_style(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    preset = q.data.split(":", 1)[1]
    if preset not in STYLE_PRESETS:
        await q.message.reply_text("Неизвестный стиль."); return
    await start_generation_for_preset(update, context, preset)

async def setgender_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not context.args or context.args[0].lower() not in ["male", "female"]:
        await update.message.reply_text("Используй: /setgender male | /setgender female"); return
    prof = load_profile(uid); prof["gender"] = context.args[0].lower(); save_profile(uid, prof)
    await update.message.reply_text(f"Ок. Пол установлен: {prof['gender']}")

async def gender_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id; prof = load_profile(uid)
    await update.message.reply_text(f"Определённый пол: {prof.get('gender') or '—'} (можно изменить /setgender)")

async def trainid_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if len(list_ref_images(uid)) < 10:
        await update.message.reply_text("Нужно 10 фото. Сначала /idenroll и пришли снимки."); return
    await update.message.reply_text("Запускаю обучение LoRA на Replicate…")
    try:
        training_id = await asyncio.to_thread(start_lora_training, uid)
        await update.message.reply_text(f"Стартанула. ID: `{training_id}`\nПроверяй /trainstatus каждые 5–10 минут.")
    except Exception as e:
        logger.exception("trainid failed"); await update.message.reply_text(f"Не удалось запустить обучение: {e}")

async def trainstatus_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    status, slug_with_ver = await asyncio.to_thread(check_training_status, uid)
    if slug_with_ver and status == "succeeded":
        await update.message.reply_text(f"Готово ✅\nСтатус: {status}\nМодель: `{slug_with_ver}`")
    else:
        await update.message.reply_text(f"Статус: {status}. Ещё в процессе…")

# -------------------- system --------------------
async def _post_init(app): await app.bot.delete_webhook(drop_pending_updates=True)

def main():
    app = ApplicationBuilder().token(TOKEN).post_init(_post_init).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("idenroll", id_enroll))
    app.add_handler(CommandHandler("iddone", id_done))
    app.add_handler(CommandHandler("idstatus", id_status))
    app.add_handler(CommandHandler("idreset", id_reset))
    app.add_handler(CommandHandler("styles", styles_cmd))
    app.add_handler(CallbackQueryHandler(cb_style, pattern=r"^style:"))
    app.add_handler(CommandHandler("setgender", setgender_cmd))
    app.add_handler(CommandHandler("gender", gender_cmd))
    app.add_handler(CommandHandler("trainid", trainid_cmd))
    app.add_handler(CommandHandler("trainstatus", trainstatus_cmd))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    logger.info("Bot up. Trainer=%s DEST=%s GEN=%dx%d steps=%s guidance=%s",
                LORA_TRAINER_SLUG, _dest_model_slug(), GEN_WIDTH, GEN_HEIGHT, GEN_STEPS, GEN_GUIDANCE)
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()






