# ================== Telegram LoRA Bot (Train -> Generate, fixed) ==================
# Flow:
# /idenroll -> upload up to 10 photos
# /iddone   -> finalize profile
# /trainid  -> start LoRA training on Replicate (Trainings API, destination=model)
# /trainstatus -> poll status; when succeeded we can generate
# /styles   -> list presets
# /lstyle <preset> -> generate from your finetuned model (no new photo needed)
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

# Настройки обучения и генерации
DEST_OWNER        = os.getenv("REPLICATE_DEST_OWNER", "").strip()
DEST_MODEL        = os.getenv("REPLICATE_DEST_MODEL", "youtwin-lora").strip()
LORA_TRAINER_SLUG = os.getenv("LORA_TRAINER_SLUG", "replicate/flux-lora-trainer").strip()
LORA_INPUT_KEY    = os.getenv("LORA_INPUT_KEY", "input_images").strip()

GEN_STEPS     = int(os.getenv("GEN_STEPS", "28"))
GEN_GUIDANCE  = float(os.getenv("GEN_GUIDANCE", "6.5"))
GEN_WIDTH     = int(os.getenv("GEN_WIDTH", "1024"))
GEN_HEIGHT    = int(os.getenv("GEN_HEIGHT", "1024"))

NEGATIVE_PROMPT = (
    "cartoon, anime, cgi, 3d, plastic skin, waxy skin, porcelain, beauty filter, ai-artifacts, "
    "warped face, changed facial proportions, geometry change, text, watermark, logo"
)
AESTHETIC_SUFFIX = (
    ", highly realistic, preserved facial identity, natural skin texture, balanced lighting, no beauty filter"
)

STYLE_PRESETS: Dict[str, str] = {
    "natural":     "ultra realistic portrait, subtle makeup, neutral color grading",
    "boho":        "boho portrait, earthy palette, soft daylight",
    "vogue":       "beauty cover shot, soft studio light, calibrated colors",
    "beauty_soft": "beauty portrait, clean studio light, controlled highlights",
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
    return {"images": [], "training_id": None, "finetuned_model": None, "status": None}
def save_profile(uid:int, prof:Dict[str,Any]): profile_path(uid).write_text(json.dumps(prof))
def save_ref_downscaled(path: Path, raw: bytes, max_side=1024, quality=92):
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    im.thumbnail((max_side, max_side))
    im.save(path, "JPEG", quality=quality)

# -------------------- telegram helpers --------------------
async def tg_download_bytes(message) -> bytes:
    f = await message.photo[-1].get_file()
    ba = await f.download_as_bytearray()
    return bytes(ba)
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

# -------------------- replicate helpers --------------------
def resolve_model_version(slug: str) -> str:
    if ":" in slug: return slug
    model = replicate.models.get(slug)
    versions = list(model.versions.list())
    if not versions: raise RuntimeError(f"Нет версий модели {slug}")
    return f"{slug}:{versions[0].id}"

def extract_any_url(out: Any) -> Optional[str]:
    if isinstance(out, str) and out.startswith(("http", "https")): return out
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

def start_lora_training(uid:int) -> str:
    """Создаёт job обучения LoRA и пишет новую версию в {owner}/{model}."""
    if not DEST_OWNER:
        raise RuntimeError("REPLICATE_DEST_OWNER не задан (твой username на Replicate).")

    dest_model = f"{DEST_OWNER}/{DEST_MODEL}"

    # Проверим, что destination существует
    try:
        replicate.models.get(dest_model)
    except Exception:
        raise RuntimeError(
            f"Целевая модель '{dest_model}' не найдена. "
            f"Создай её на https://replicate.com/create (name='{DEST_MODEL}')."
        )

    trainer_version = resolve_model_version(LORA_TRAINER_SLUG)
    zip_path = _pack_refs_zip(uid)

    input_keys = []
    if LORA_INPUT_KEY: input_keys.append(LORA_INPUT_KEY)
    input_keys += ["input_images", "images_zip", "image_zip", "images"]
    used_keys, seen = [], set()
    for k in input_keys:
        if k not in seen:
            used_keys.append(k); seen.add(k)

    client = Client(api_token=os.environ["REPLICATE_API_TOKEN"])
    last_err = None
    for key in used_keys:
        try:
            with open(zip_path, "rb") as f:
                training = client.trainings.create(
                    version=trainer_version,
                    input={key: f},
                    destination=dest_model
                )
            prof = load_profile(uid)
            prof["training_id"] = training.id
            prof["status"] = "starting"
            prof["finetuned_model"] = dest_model
            save_profile(uid, prof)
            return training.id
        except Exception as e:
            last_err = e
            logger.warning("Trainer rejected key '%s': %s", key, e)
    raise RuntimeError(f"Не удалось создать обучение. Последняя ошибка: {last_err}")

def check_training_status(uid:int) -> Tuple[str, Optional[str]]:
    prof = load_profile(uid)
    tid = prof.get("training_id")
    if not tid: return ("not_started", None)
    client = Client(api_token=os.environ["REPLICATE_API_TOKEN"])
    tr = client.trainings.get(tid)
    status = getattr(tr, "status", None) or (tr.get("status") if isinstance(tr, dict) else None) or "unknown"
    prof["status"] = status
    save_profile(uid, prof)
    return (status, prof["finetuned_model"] if status == "succeeded" else None)

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

# -------------------- UI helpers --------------------
def styles_keyboard() -> InlineKeyboardMarkup:
    rows, row = [], []
    for i, name in enumerate(STYLE_PRESETS.keys(), 1):
        row.append(InlineKeyboardButton(name.title(), callback_data=f"style:{name}"))
        if i % 3 == 0:
            rows.append(row); row=[]
    if row: rows.append(row)
    return InlineKeyboardMarkup(rows)

# -------------------- handlers --------------------
ENROLL_FLAG: Dict[int,bool] = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я обучаю твою персональную LoRA по 10 фото, а затем генерирую кадры без новых фото.\n\n"
        "Шаги:\n"
        "1) /idenroll — включить набор (пришли до 10 фото)\n"
        "2) /iddone — сохранить профиль\n"
        "3) /trainid — запустить обучение LoRA\n"
        "4) /trainstatus — проверить статус\n"
        "5) /styles — список стилей\n"
        "6) /lstyle <preset> — сгенерировать из твоей модели"
    )

async def id_enroll(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    ENROLL_FLAG[uid] = True
    await update.message.reply_text("Набор включён. Пришли подряд до 10 фото. Когда закончишь — /iddone.")

async def id_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    ENROLL_FLAG[uid] = False
    prof = load_profile(uid)
    prof["images"] = [p.name for p in list_ref_images(uid)]
    save_profile(uid, prof)
    await update.message.reply_text(f"Готово. В профиле {len(prof['images'])} фото. Далее: /trainid.")

async def id_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    prof = load_profile(uid)
    await update.message.reply_text(
        f"Фото в профиле: {len(list_ref_images(uid))}\n"
        f"Статус обучения: {prof.get('status') or '—'}\n"
        f"Модель: {prof.get('finetuned_model') or '—'}"
    )

async def id_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    p = user_dir(uid)
    if p.exists(): shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    ENROLL_FLAG[uid] = False
    await update.message.reply_text("Профиль очищен. /idenroll чтобы собрать заново.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if ENROLL_FLAG.get(uid):
        refs = list_ref_images(uid)
        if len(refs) >= 10:
            await update.message.reply_text("Уже 10/10. Жми /iddone."); return
        raw = await tg_download_bytes(update.message)
        save_ref_downscaled(user_dir(uid) / f"ref_{int(time.time()*1000)}.jpg", raw)
        await update.message.reply_text(f"Сохранила ({len(refs)+1}/10). Ещё?")
    else:
        await update.message.reply_text("Сначала /idenroll. После /iddone → /trainid.")

async def styles_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Выбери стиль:", reply_markup=styles_keyboard())

async def cb_style(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    preset = q.data.split(":",1)[1]
    await q.message.reply_text(f"Стиль выбран: {preset}. Запусти `/lstyle {preset}` после обучения.")

async def trainid_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if len(list_ref_images(uid)) < 10:
        await update.message.reply_text("Нужно 10 фото.")
        return
    await update.message.reply_text("Запускаю обучение LoRA на Replicate…")
    try:
        training_id = await asyncio.to_thread(start_lora_training, uid)
        await update.message.reply_text(f"Стартанула. ID: `{training_id}`\nПроверяй /trainstatus каждые 5–10 минут.")
    except Exception as e:
        logger.exception("trainid failed")
        await update.message.reply_text(f"Не удалось запустить обучение: {e}")

async def trainstatus_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    status, model_slug = await asyncio.to_thread(check_training_status, uid)
    if model_slug and status == "succeeded":
        await update.message.reply_text(f"Готово ✅\nСтатус: {status}\nМодель: `{model_slug}`")
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
    model_slug = prof.get("finetuned_model")
    if not model_slug or prof.get("status") != "succeeded":
        await update.message.reply_text("Модель ещё не готова. Сначала /trainid и дождись /trainstatus = succeeded.")
        return
    await update.message.chat.send_action(ChatAction.UPLOAD_PHOTO)
    await update.message.reply_text(f"Генерирую из твоей модели: {preset}… 🎨")
    try:
        seed = int(time.time()) & 0xFFFFFFFF
        url = await asyncio.to_thread(
            generate_from_finetune,
            model_slug,
            f"{STYLE_PRESETS[preset]}, exact facial identity, no geometry change",
            GEN_STEPS, GEN_GUIDANCE, seed, GEN_WIDTH, GEN_HEIGHT
        )
        await safe_send_image(update, url, f"Готово ✨\nСтиль: {preset}\nМодель: {model_slug}")
    except Exception as e:
        logger.exception("lstyle failed")
        await update.message.reply_text(f"Ошибка генерации: {e}")

# -------------------- system --------------------
async def _post_init(app): await app.bot.delete_webhook(drop_pending_updates=True)
async def _error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Unhandled error", exc_info=context.error)
def main():
    app = ApplicationBuilder().token(TOKEN).post_init(_post_init).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("idenroll", id_enroll))
    app.add_handler(CommandHandler("iddone", id_done))
    app.add_handler(CommandHandler("idstatus", id_status))
    app.add_handler(CommandHandler("idreset", id_reset))
    app.add_handler(CommandHandler("trainid", trainid_cmd))
    app.add_handler(CommandHandler("trainstatus", trainstatus_cmd))
    app.add_handler(CommandHandler("styles", styles_cmd))
    app.add_handler(CallbackQueryHandler(cb_style, pattern=r"^style:"))
    app.add_handler(CommandHandler("lstyle", lstyle_cmd))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_error_handler(_error_handler)
    logger.info("Bot up. Trainer=%s DEST_OWNER=%s DEST_MODEL=%s", LORA_TRAINER_SLUG, DEST_OWNER, DEST_MODEL)
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()



