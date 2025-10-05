# ------------------ Ai Telegram Bot with LoRA training ------------------
# Features:
# 1) /idenroll -> user uploads up to 10 photos (stored on disk)
# 2) /iddone   -> finalize profile
# 3) /trainid  -> start LoRA training on Replicate (GPU on their side)
# 4) /trainstatus -> poll training status; when 'succeeded' we store LoRA ref
# 5) /styles and /lstyle <preset> -> generate using the saved LoRA (no new photo)
# ------------------------------------------------------------------------

import os
import re
import io
import json
import time
import base64
import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from PIL import Image
import replicate

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes,
    CallbackQueryHandler, filters
)

# ------------------ ENV & sanity ------------------
TOKEN = os.getenv("BOT_TOKEN")
os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN", "")

if not TOKEN or not re.match(r"^\d+:[A-Za-z0-9_-]{20,}$", TOKEN):
    raise RuntimeError("Некорректный BOT_TOKEN.")
if not os.getenv("REPLICATE_API_TOKEN"):
    raise RuntimeError("Нет REPLICATE_API_TOKEN.")

# Replicate models (change to your preferred trainer/generator)
LORA_TRAINER_SLUG   = os.getenv("LORA_TRAINER_SLUG", "fofr/sdxl-lora-trainer").strip()
LORA_GENERATOR_SLUG = os.getenv("LORA_GENERATOR_SLUG", "stability-ai/sdxl").strip()

# Training params (tweak as you wish)
LORA_NETWORK_DIM   = int(os.getenv("LORA_NETWORK_DIM", "16"))
LORA_STEPS         = int(os.getenv("LORA_STEPS", "4000"))
LORA_LEARNING_RATE = float(os.getenv("LORA_LEARNING_RATE", "0.0001"))

# Generation defaults
GEN_STEPS     = int(os.getenv("GEN_STEPS", "28"))
GEN_GUIDANCE  = float(os.getenv("GEN_GUIDANCE", "6.5"))
GEN_WIDTH     = int(os.getenv("GEN_WIDTH", "1024"))
GEN_HEIGHT    = int(os.getenv("GEN_HEIGHT", "1024"))

# Styles (prompts)
STYLE_PRESETS: Dict[str, str] = {
    "natural":     "ultra realistic portrait, real skin texture, subtle makeup, neutral grading",
    "boho":        "boho portrait, earthy palette, soft daylight",
    "vogue":       "beauty cover shot, soft studio light, calibrated colors, photographic grain",
    "beauty_soft": "beauty portrait, clean studio light, controlled highlights, visible pores",
    "windowlight": "soft window light portrait, natural diffusion, balanced exposure",
    "editorial":   "editorial fashion portrait, preserved natural imperfections, pro color grading",
    "moody":       "moody cinematic portrait, controlled shadows, subtle rim light",
}

NEGATIVE_PROMPT = (
    "cartoon, anime, cgi, 3d, plastic skin, waxy skin, porcelain, airbrushed, beauty filter, smoothing, "
    "overprocessed, oversharpen, hdr effect, halo, neon skin, fake skin, watermark, text, logo, "
    "warped face, distorted face, changed facial proportions, geometry change, face reshape, exaggerated makeup"
)
AESTHETIC_SUFFIX = (
    ", natural healthy skin, preserved pores, balanced contrast, soft realistic light, "
    "no beauty filter, no plastic look"
)

# ------------------ Logging ------------------
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("bot")

# ------------------ Storage ------------------
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
    # fields for LoRA workflow
    return {"images": [], "lora_training_id": None, "lora_ref": None, "lora_status": None}

def save_profile(uid:int, prof:Dict[str,Any]):
    profile_path(uid).write_text(json.dumps(prof))

def save_ref_downscaled(path: Path, b: bytes, max_side=1024, quality=92):
    im = Image.open(io.BytesIO(b)).convert("RGB")
    im.thumbnail((max_side, max_side))
    im.save(path, "JPEG", quality=quality)

# ------------------ Telegram helpers ------------------
async def tg_download_bytes(message) -> bytes:
    f = await message.photo[-1].get_file()
    buf = await f.download_as_bytearray()
    return bytes(buf)

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

# ------------------ Replicate helpers ------------------
def resolve_model_version(slug: str) -> str:
    if ":" in slug: return slug
    model = replicate.models.get(slug)
    versions = list(model.versions.list())
    if not versions: raise RuntimeError(f"Нет версий модели {slug}")
    return f"{slug}:{versions[0].id}"

def extract_any_url(output: Any) -> Optional[str]:
    if isinstance(output, str) and output.startswith(("http","https")):
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
            logger.warning("Replicate rejected payload: %s", e)
    raise last or RuntimeError("Все варианты отклонены")

def _path_to_data_url(p: Path) -> str:
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return "data:image/jpeg;base64," + b64

# ------------------ LoRA training workflow ------------------
def start_lora_training(uid:int, network_dim:int, steps:int, lr:float) -> str:
    """Start LoRA training job on Replicate. Returns training_id (prediction id)."""
    if not LORA_TRAINER_SLUG:
        raise RuntimeError("Не задан LORA_TRAINER_SLUG")

    refs = list_ref_images(uid)
    if len(refs) < 10:
        raise RuntimeError("Нужно 10 фото для обучения LoRA.")

    # Most trainers accept data URLs; if your trainer expects HTTP URLs, host them and pass links.
    image_urls = [_path_to_data_url(p) for p in refs]

    # payload keys depend on a trainer; these work for fofr/sdxl-lora-trainer
    training_input = {
        "images": image_urls,
        "network_dim": network_dim,
        "max_train_steps": steps,
        "learning_rate": lr,
        "shuffle": True,
        "use_sdxl": True
    }

    # Kick off training (runs on Replicate GPUs)
    result = replicate.run(f"{LORA_TRAINER_SLUG}:latest", input=training_input)

    # normalize training id
    training_id = None
    if isinstance(result, dict) and "id" in result:
        training_id = result["id"]
    elif isinstance(result, str):
        training_id = result

    prof = load_profile(uid)
    prof["lora_training_id"] = training_id or "UNKNOWN"
    prof["lora_status"] = "starting"
    save_profile(uid, prof)

    return prof["lora_training_id"]

def check_lora_status(uid:int) -> Tuple[str, Optional[str]]:
    """Return (status, lora_ref|None) and persist into profile."""
    prof = load_profile(uid)
    training_id = prof.get("lora_training_id")
    if not training_id:
        return ("not_started", None)

    try:
        from replicate import Client
        client = Client(api_token=os.environ["REPLICATE_API_TOKEN"])
        pred = client.predictions.get(training_id)

        # Both object-like and dict-like access
        status = getattr(pred, "status", None) or (pred.get("status") if isinstance(pred, dict) else None) or "unknown"
        output = getattr(pred, "output", None) or (pred.get("output") if isinstance(pred, dict) else None)

        lora_ref = None
        if status == "succeeded" and output:
            if isinstance(output, str):
                lora_ref = output
            elif isinstance(output, dict):
                lora_ref = output.get("lora") or output.get("weights") or output.get("model")
            elif isinstance(output, list) and output:
                lora_ref = output[0]

        prof["lora_status"] = status
        if lora_ref:
            prof["lora_ref"] = lora_ref
        save_profile(uid, prof)

        return (status, prof.get("lora_ref"))
    except Exception as e:
        return (f"error: {e}", None)

def generate_with_lora(lora_ref:str, prompt:str, steps:int, guidance:float, seed:int, width:int, height:int) -> str:
    """Call SDXL (or your generator) with LoRA attached. Tries a few common arg names."""
    if not LORA_GENERATOR_SLUG:
        raise RuntimeError("Не задан LORA_GENERATOR_SLUG")
    resolved = resolve_model_version(LORA_GENERATOR_SLUG)

    inputs_list = [
        {
            "prompt": prompt + AESTHETIC_SUFFIX,
            "negative_prompt": NEGATIVE_PROMPT,
            "width": width, "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "seed": seed,
            "lora_urls": [lora_ref],  # variant 1
        },
        {
            "prompt": prompt + AESTHETIC_SUFFIX,
            "negative_prompt": NEGATIVE_PROMPT,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "seed": seed,
            "loras": [{"path": lora_ref, "scale": 0.85}],  # variant 2
        },
        {
            "prompt": prompt + AESTHETIC_SUFFIX,
            "negative_prompt": NEGATIVE_PROMPT,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "seed": seed,
            "adapter": lora_ref,  # variant 3
        },
    ]
    return replicate_run_flexible(resolved, inputs_list)

# ------------------ UI helpers ------------------
def styles_keyboard() -> InlineKeyboardMarkup:
    rows, row = [], []
    names = list(STYLE_PRESETS.keys())
    for i, name in enumerate(names, 1):
        row.append(InlineKeyboardButton(name.title(), callback_data=f"style:{name}"))
        if i % 3 == 0:
            rows.append(row); row=[]
    if row: rows.append(row)
    return InlineKeyboardMarkup(rows)

# ------------------ State ------------------
ENROLL_FLAG: Dict[int, bool] = {}  # user_id -> collecting refs?

# ------------------ Handlers ------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Бот обучает твою персональную LoRA по 10 фото и потом генерирует в выбранных стилях — без новых фото.\n\n"
        "Порядок:\n"
        "1) /idenroll — включить набор (пришли подряд до 10 фото)\n"
        "2) /iddone — сохранить профиль\n"
        "3) /trainid — запустить обучение LoRA (на Replicate)\n"
        "4) /trainstatus — проверить статус обучения\n"
        "5) /styles — список стилей\n"
        "6) /lstyle <preset> — сгенерировать со своей LoRA (без новых фото)"
    )

async def id_enroll(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    ENROLL_FLAG[uid] = True
    await update.message.reply_text("Набор включён. Пришли до 10 фото (крупные портреты, без фильтров). Когда закончишь — /iddone.")

async def id_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    ENROLL_FLAG[uid] = False
    # persist file names into profile
    prof = load_profile(uid)
    prof["images"] = [p.name for p in list_ref_images(uid)]
    save_profile(uid, prof)
    await update.message.reply_text(f"Готово. В профиле {len(prof['images'])} фото. Теперь /trainid.")

async def id_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    cnt = len(list_ref_images(uid))
    prof = load_profile(uid)
    lora_status = prof.get("lora_status")
    lora_ref = prof.get("lora_ref")
    msg = f"Фото в профиле: {cnt}.\nСтатус LoRA: {lora_status or '—'}"
    if lora_ref:
        msg += f"\nLoRA: {lora_ref}"
    await update.message.reply_text(msg)

async def id_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    p = user_dir(uid)
    if p.exists(): shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    ENROLL_FLAG[uid] = False
    await update.message.reply_text("Профиль очищен. Запусти /idenroll, чтобы собрать заново.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    m = update.message
    if ENROLL_FLAG.get(uid):
        refs = list_ref_images(uid)
        if len(refs) >= 10:
            await m.reply_text("Уже 10/10. Жми /iddone."); return
        b = await tg_download_bytes(m)
        save_ref_downscaled(user_dir(uid) / f"ref_{int(time.time()*1000)}.jpg", b, max_side=1024, quality=92)
        await m.reply_text(f"Сохранила ({len(refs)+1}/10). Ещё?")
        return
    await m.reply_text("Сначала включи набор: /idenroll. После /iddone → /trainid → /lstyle <preset>.")

async def styles_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Выбери стиль:", reply_markup=styles_keyboard())

async def cb_style(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    preset = q.data.split(":",1)[1]
    await q.message.reply_text(f"Стиль выбран: {preset}. Запусти `/lstyle {preset}` после обучения LoRA.")

# ----- LoRA commands -----
async def trainid_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if len(list_ref_images(uid)) < 10:
        await update.message.reply_text("Нужно 10 фото. Сначала /idenroll и пришли снимки."); return
    await update.message.reply_text("Запускаю обучение LoRA на Replicate…")
    try:
        training_id = await asyncio.to_thread(start_lora_training, uid, LORA_NETWORK_DIM, LORA_STEPS, LORA_LEARNING_RATE)
        await update.message.reply_text(f"Стартанула. ID: `{training_id}`\nПроверь через 5–10 минут: /trainstatus")
    except Exception as e:
        logger.exception("trainid failed")
        await update.message.reply_text(f"Не удалось запустить обучение: {e}")

async def trainstatus_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    status, lora_ref = await asyncio.to_thread(check_lora_status, uid)
    if lora_ref:
        await update.message.reply_text(f"Готово ✅\nСтатус: {status}\nLoRA: `{lora_ref}`")
    else:
        await update.message.reply_text(f"Статус: {status}. Ещё в процессе…")

async def lstyle_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    args = context.args
    if not args:
        await update.message.reply_text("Укажи стиль, например: `/lstyle natural`", reply_markup=styles_keyboard()); return
    preset = args[0].lower()
    if preset not in STYLE_PRESETS:
        await update.message.reply_text(f"Не знаю стиль '{preset}'. Смотри /styles"); return

    prof = load_profile(uid)
    lora_ref = prof.get("lora_ref")
    if not lora_ref:
        await update.message.reply_text("LoRA ещё не готова. Сначала /trainid и дождись /trainstatus = succeeded.")
        return

    await update.message.chat.send_action(ChatAction.UPLOAD_PHOTO)
    await update.message.reply_text(f"Генерирую со своей LoRA: {preset}… 🎨")
    try:
        seed = int(time.time()) & 0xFFFFFFFF
        url = await asyncio.to_thread(
            generate_with_lora,
            lora_ref,
            f"{STYLE_PRESETS[preset]}, exact facial identity, no geometry change",
            GEN_STEPS, GEN_GUIDANCE, seed, GEN_WIDTH, GEN_HEIGHT
        )
        await safe_send_image(update, url, f"Готово ✨\nСтиль: {preset}\n(LoRA)")
    except Exception as e:
        logger.exception("lstyle failed")
        await update.message.reply_text(f"Ошибка генерации: {e}")

# ------------------ System ------------------
async def _post_init(app):
    # drop old webhooks just in case
    await app.bot.delete_webhook(drop_pending_updates=True)

async def _error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Unhandled error", exc_info=context.error)

def main():
    app = (ApplicationBuilder().token(TOKEN).post_init(_post_init).build())

    # core flow
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("idenroll", id_enroll))
    app.add_handler(CommandHandler("iddone", id_done))
    app.add_handler(CommandHandler("idstatus", id_status))
    app.add_handler(CommandHandler("idreset", id_reset))

    # lora flow
    app.add_handler(CommandHandler("trainid", trainid_cmd))
    app.add_handler(CommandHandler("trainstatus", trainstatus_cmd))
    app.add_handler(CommandHandler("styles", styles_cmd))
    app.add_handler(CallbackQueryHandler(cb_style, pattern=r"^style:"))
    app.add_handler(CommandHandler("lstyle", lstyle_cmd))

    # photos only used during enrollment
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    app.add_error_handler(_error_handler)

    logger.info("Bot ready (polling) | LoRA trainer=%s | generator=%s", LORA_TRAINER_SLUG, LORA_GENERATOR_SLUG)
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()


