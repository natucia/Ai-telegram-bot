# ================== Telegram LoRA Bot (one model repo + pinned versions) ==================
# /idenroll  -> собрать до 10 фото
# /iddone    -> сохранить профиль
# /trainid   -> обучить LoRA в едином репозитории модели (Replicate Trainings API)
# /trainstatus -> статус; при success сохраняем конкретный version_id
# /styles    -> список стилей
# /lstyle X  -> генерируем строго по pinned owner/model:version юзера
# ===========================================================================

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

# ЕДИНЫЙ репозиторий модели для всех обучений (создай один раз в Replicate)
DEST_OWNER        = os.getenv("REPLICATE_DEST_OWNER", "").strip()
DEST_MODEL        = os.getenv("REPLICATE_DEST_MODEL", "yourtwin-lora").strip()

# Тренер LoRA (public тренер)
LORA_TRAINER_SLUG = os.getenv("LORA_TRAINER_SLUG", "replicate/flux-lora-trainer").strip()
LORA_INPUT_KEY    = os.getenv("LORA_INPUT_KEY", "input_images").strip()

# Настройки обучения
LORA_MAX_STEPS     = int(os.getenv("LORA_MAX_STEPS", "2200"))
LORA_LR            = float(os.getenv("LORA_LR", "0.0001"))
LORA_USE_FACE_DET  = os.getenv("LORA_USE_FACE_DET", "true").lower() in ["1","true","yes","y"]
LORA_CAPTION_PREF  = os.getenv("LORA_CAPTION_PREFIX",
    "a photo of a young woman with fair skin, green eyes, ginger hair, natural look").strip()
LORA_RESOLUTION    = int(os.getenv("LORA_RESOLUTION", "1024"))

# Генерация
GEN_STEPS     = int(os.getenv("GEN_STEPS", "40"))
GEN_GUIDANCE  = float(os.getenv("GEN_GUIDANCE", "4.0"))
GEN_WIDTH     = int(os.getenv("GEN_WIDTH", "832"))
GEN_HEIGHT    = int(os.getenv("GEN_HEIGHT", "1216"))

NEGATIVE_PROMPT = (
    "cartoon, anime, 3d, cgi, overprocessed, oversharpen, waxy skin, plastic skin, "
    "beauty filter, unrealistic skin, blur, lowres, deformed, distorted, "
    "bad anatomy, watermark, text, logo"
)
AESTHETIC_SUFFIX = (
    ", photo-realistic, visible skin pores, natural color, filmic color grading, balanced lighting"
)

STYLE_PRESETS: Dict[str, str] = {
    "natural":     "ultra realistic portrait, subtle makeup, neutral color grading",
    "boho":        "boho portrait, earthy palette, soft daylight",
    "vogue":       "beauty cover shot, soft studio light, calibrated colors",
    "beauty_soft": "beauty portrait, clean studio light, controlled highlights, soft diffusion",
    "windowlight": "soft window light portrait, natural diffusion",
    "editorial":   "editorial fashion portrait, preserved natural imperfections",
    "moody":       "moody cinematic portrait, controlled shadows, subtle rim light",
}

# -------------------- logging --------------------
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("bot")

# -------------------- storage --------------------
DATA_DIR = Path("profiles"); DATA_DIR.mkdir(exist_ok=True)

def user_dir(uid:int) -> Path:
    p = DATA_DIR / str(uid)
    p.mkdir(parents=True, exist_ok=True)
    return p

def list_ref_images(uid:int) -> List[Path]:
    return sorted(user_dir(uid).glob("ref_*.jpg"))

def profile_path(uid:int) -> Path:
    return user_dir(uid) / "profile.json"

def load_profile(uid:int) -> Dict[str, Any]:
    p = profile_path(uid)
    if p.exists():
        return json.loads(p.read_text())
    return {
        "images": [],
        "training_id": None,
        "finetuned_model": None,
        "finetuned_version": None,
        "status": None,
    }

def save_profile(uid:int, prof:Dict[str,Any]):
    profile_path(uid).write_text(json.dumps(prof))

def save_ref_downscaled(path: Path, raw: bytes, max_side=1024, quality=92):
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    im.thumbnail((max_side, max_side))
    im.save(path, "JPEG", quality=quality)

# -------------------- replicate helpers --------------------
def resolve_model_version(slug: str) -> str:
    if ":" in slug:
        return slug
    model = replicate.models.get(slug)
    versions = list(model.versions.list())
    if not versions:
        raise RuntimeError(f"Нет версий модели {slug}")
    return f"{slug}:{versions[0].id}"

def extract_any_url(out: Any) -> Optional[str]:
    if isinstance(out, str) and out.startswith(("http", "https")):
        return out
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
            if not url:
                raise RuntimeError("Empty output")
            return url
        except Exception as e:
            last = e
            logger.warning("Replicate rejected payload: %s", e)
    raise last or RuntimeError("Все варианты отклонены")

# -------------------- LoRA training --------------------
def _pack_refs_zip(uid:int) -> Path:
    refs = list_ref_images(uid)
    if len(refs) < 10:
        raise RuntimeError("Нужно 10 фото для обучения.")
    zpath = user_dir(uid) / "train.zip"
    with ZipFile(zpath, "w") as z:
        for i, p in enumerate(refs, 1):
            z.write(p, arcname=f"img_{i:02d}.jpg")
    return zpath

def _dest_model_slug() -> str:
    if not DEST_OWNER:
        raise RuntimeError("REPLICATE_DEST_OWNER не задан.")
    return f"{DEST_OWNER}/{DEST_MODEL}"

def _ensure_destination_exists(slug: str):
    try:
        replicate.models.get(slug)
    except Exception:
        owner, name = slug.split("/", 1)
        raise RuntimeError(
            f"Целевая модель '{slug}' не найдена. Создай её вручную на https://replicate.com/create "
            f"(owner={owner}, name='{name}')."
        )

def start_lora_training(uid:int) -> str:
    dest_model = _dest_model_slug()
    _ensure_destination_exists(dest_model)
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

# === FIXED check_training_status ===
def check_training_status(uid:int) -> Tuple[str, Optional[str]]:
    prof = load_profile(uid)
    tid = prof.get("training_id")
    if not tid:
        return ("not_started", None)

    client = Client(api_token=os.environ["REPLICATE_API_TOKEN"])
    tr = client.trainings.get(tid)
    status = getattr(tr, "status", None) or (tr.get("status") if isinstance(tr, dict) else None) or "unknown"

    if status != "succeeded":
        prof["status"] = status
        save_profile(uid, prof)
        return (status, None)

    destination = getattr(tr, "destination", None) or (isinstance(tr, dict) and tr.get("destination")) \
                  or prof.get("finetuned_model") or _dest_model_slug()

    # берем version_id из output
    version_id = getattr(tr, "output", None) if not isinstance(tr, dict) else tr.get("output")
    if isinstance(version_id, dict):
        version_id = version_id.get("id") or version_id.get("version")

    slug_with_version = None
    try:
        if version_id:
            replicate.models.get(f"{destination}:{version_id}")
            slug_with_version = f"{destination}:{version_id}"
    except Exception:
        try:
            model_obj = replicate.models.get(destination)
            versions = list(model_obj.versions.list())
            if versions:
                slug_with_version = f"{destination}:{versions[0].id}"
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
    model_version = resolve_model_version(model_slug)
    inputs_list = [{
        "prompt": prompt + AESTHETIC_SUFFIX,
        "negative_prompt": NEGATIVE_PROMPT,
        "width": w, "height": h,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "seed": seed,
    }]
    return replicate_run_flexible(model_version, inputs_list)

# -------------------- Telegram handlers --------------------
ENROLL_FLAG: Dict[int,bool] = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я обучаю твою персональную LoRA по 10 фото.\n\n"
        "1) /idenroll — включить набор (до 10 фото)\n"
        "2) /iddone — сохранить профиль\n"
        "3) /trainid — запустить обучение\n"
        "4) /trainstatus — проверить статус\n"
        "5) /styles — стили\n"
        "6) /lstyle <preset> — генерация из твоей версии модели"
    )

async def id_enroll(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    ENROLL_FLAG[uid] = True
    await update.message.reply_text("Набор включён. Пришли до 10 фото, потом /iddone.")

async def id_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    ENROLL_FLAG[uid] = False
    prof = load_profile(uid)
    prof["images"] = [p.name for p in list_ref_images(uid)]
    save_profile(uid, prof)
    await update.message.reply_text(f"Готово. В профиле {len(prof['images'])} фото. Далее: /trainid.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if ENROLL_FLAG.get(uid):
        refs = list_ref_images(uid)
        if len(refs) >= 10:
            await update.message.reply_text("Уже 10/10. Жми /iddone."); return
        raw = await update.message.photo[-1].get_file()
        data = await raw.download_as_bytearray()
        save_ref_downscaled(user_dir(uid) / f"ref_{int(time.time()*1000)}.jpg", bytes(data))
        await update.message.reply_text(f"Сохранила ({len(refs)+1}/10). Ещё?")
    else:
        await update.message.reply_text("Сначала /idenroll, потом /iddone.")

async def trainid_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if len(list_ref_images(uid)) < 10:
        await update.message.reply_text("Нужно 10 фото.")
        return
    await update.message.reply_text("Запускаю обучение LoRA...")
    try:
        training_id = await asyncio.to_thread(start_lora_training, uid)
        await update.message.reply_text(f"Стартанула. ID: {training_id}\nПроверяй /trainstatus.")
    except Exception as e:
        logger.exception("trainid failed")
        await update.message.reply_text(f"Ошибка запуска: {e}")

async def trainstatus_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    status, slug = await asyncio.to_thread(check_training_status, uid)
    if slug and status == "succeeded":
        await update.message.reply_text(f"Готово ✅\n{slug}")
    else:
        await update.message.reply_text(f"Статус: {status}")

async def lstyle_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    args = context.args
    if not args:
        await update.message.reply_text("Укажи стиль, например /lstyle natural")
        return
    preset = args[0].lower()
    if preset not in STYLE_PRESETS:
        await update.message.reply_text("Нет такого стиля.")
        return

    prof = load_profile(uid)
    if prof.get("status") != "succeeded":
        await update.message.reply_text("Сначала дождись /trainstatus = succeeded.")
        return
    model_slug = _pinned_slug(prof)
    await update.message.reply_text(f"Генерирую из твоей модели: {model_slug}… 🎨")
    try:
        seed = int(time.time()) & 0xFFFFFFFF
        url = await asyncio.to_thread(
            generate_from_finetune,
            model_slug,
            f"{STYLE_PRESETS[preset]}, exact facial identity, no geometry change",
            GEN_STEPS, GEN_GUIDANCE, seed, GEN_WIDTH, GEN_HEIGHT
        )
        await update.message.reply_photo(photo=url, caption=f"Готово ✨\nСтиль: {preset}")
    except Exception as e:
        await update.message.reply_text(f"Ошибка генерации: {e}")

# -------------------- system --------------------
async def _post_init(app):
    await app.bot.delete_webhook(drop_pending_updates=True)

def main():
    app = ApplicationBuilder().token(TOKEN).post_init(_post_init).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("idenroll", id_enroll))
    app.add_handler(CommandHandler("iddone", id_done))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(CommandHandler("trainid", trainid_cmd))
    app.add_handler(CommandHandler("trainstatus", trainstatus_cmd))
    app.add_handler(CommandHandler("lstyle", lstyle_cmd))
    logger.info("Bot started. Trainer=%s DEST=%s", LORA_TRAINER_SLUG, _dest_model_slug())
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()



