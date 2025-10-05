# ================== Telegram LoRA Bot (one model repo + pinned versions + rich styles) ==================
# /idenroll  -> собрать до 10 фото
# /iddone    -> сохранить профиль
# /trainid   -> обучить LoRA в едином репозитории модели (Replicate Trainings API)
# /trainstatus -> статус; при success сохраняем конкретный version_id
# /styles    -> показать пресеты
# /lstyle X  -> генерируем строго по pinned owner/model:version юзера, авто-размер под стиль
# =======================================================================================================

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

DEST_OWNER        = os.getenv("REPLICATE_DEST_OWNER", "").strip()             # напр. natucia
DEST_MODEL        = os.getenv("REPLICATE_DEST_MODEL", "yourtwin-lora").strip()# единый репо

LORA_TRAINER_SLUG = os.getenv("LORA_TRAINER_SLUG", "replicate/flux-lora-trainer").strip()
LORA_INPUT_KEY    = os.getenv("LORA_INPUT_KEY", "input_images").strip()

# Твики обучения для реализма
LORA_MAX_STEPS     = int(os.getenv("LORA_MAX_STEPS", "2000"))
LORA_LR            = float(os.getenv("LORA_LR", "0.00008"))
LORA_USE_FACE_DET  = os.getenv("LORA_USE_FACE_DET", "true").lower() in ["1","true","yes","y"]
LORA_CAPTION_PREF  = os.getenv("LORA_CAPTION_PREFIX",
    "a photo of a woman, relaxed neutral expression, gentle smile, soft jawline, balanced facial proportions, natural look"
).strip()
LORA_RESOLUTION    = int(os.getenv("LORA_RESOLUTION", "1024"))

# Генерация — анти-пластик
GEN_STEPS     = int(os.getenv("GEN_STEPS", "44"))
GEN_GUIDANCE  = float(os.getenv("GEN_GUIDANCE", "3.6"))
GEN_WIDTH     = int(os.getenv("GEN_WIDTH", "832"))     # дефолт, если стиль не задаёт свой
GEN_HEIGHT    = int(os.getenv("GEN_HEIGHT", "1216"))

NEGATIVE_PROMPT = (
    "cartoon, anime, 3d, cgi, overprocessed, oversharpen, beauty filter, skin smoothing, "
    "waxy skin, plastic skin, blur, lowres, deformed, distorted, bad anatomy, watermark, text, logo, "
    "puffy face, swollen face, chubby cheeks, hypertrophic masseter, wide jaw, clenched jaw, "
    "pursed lips, duckface, overfilled lips, nasolabial fold accent"
)
AESTHETIC_SUFFIX = (
    ", photo-realistic, visible skin pores, natural color, soft filmic contrast, "
    "balanced soft lighting, no beautification"
)

# -------------------- STYLES (prompt + recommended size) --------------------
# width/height можно не указывать — возьмутся из GEN_WIDTH/GEN_HEIGHT
STYLE_PRESETS: Dict[str, Dict[str, Any]] = {
    # Портреты
    "natural":       {"p": "ultra realistic portrait, neutral color grading, relaxed neutral expression, gentle smile, soft jawline", "w": 896, "h": 1152},
    "natural_slim":  {"p": "ultra realistic portrait, delicate cheekbones, soft jawline, balanced proportions, relaxed face", "w": 896, "h": 1152},
    "beauty_soft":   {"p": "beauty portrait, clean studio light, soft diffusion, subtle makeup", "w": 1024, "h": 1024},
    "vogue":         {"p": "editorial beauty cover shot, soft studio light, calibrated colors, shallow depth of field"},
    "windowlight":   {"p": "soft window light portrait, gentle bokeh background, natural diffusion"},
    "cinematic":     {"p": "cinematic portrait, shallow depth of field, Rembrandt lighting, film grain subtle", "w": 960, "h": 1280},
    "moody":         {"p": "moody cinematic portrait, controlled shadows, subtle rim light"},

    # Полный рост, разные миры
    "adventure_heroine": {  # vibe приключенческой героини (без прямого копирования тм)
        "p": "full body, athletic explorer heroine, tactical outfit, fingerless gloves, utility belt, boots, dynamic pose, ancient ruins background",
        "w": 832, "h": 1344
    },
    "cyberpunk_city": {
        "p": "full body, neon cyberpunk street, rain, holograms, reflective puddles, leather jacket, cinematic backlight",
        "w": 832, "h": 1344
    },
    "sci_fi_spacesuit": {
        "p": "full body, realistic EVA spacesuit, starfield, spaceship hangar lights, hard surface details",
        "w": 960, "h": 1440
    },
    "fantasy_elf": {
        "p": "full body, elegant fantasy elf queen, flowing gown, forest temple, soft god rays",
        "w": 960, "h": 1440
    },
    "medieval_knight": {
        "p": "full body, realistic medieval armor, cape, castle courtyard, overcast sky",
        "w": 896, "h": 1408
    },
    "desert_explorer": {
        "p": "full body, desert adventurer, scarf, cargo shorts, rocky canyon, warm sunset light",
        "w": 832, "h": 1344
    },
    "underwater_freediver": {
        "p": "full body, realistic freediver, long fins, underwater blue ambient light, sun rays, particles",
        "w": 896, "h": 1408
    },
    "snow_mountain": {
        "p": "full body, alpine mountaineer, down jacket, crampons, snowy ridge, dramatic sky",
        "w": 896, "h": 1408
    },
    "evening_gown": {
        "p": "full body, elegant evening gown, red carpet, soft spotlights, cinematic bokeh",
        "w": 832, "h": 1344
    },
    "streetwear": {
        "p": "full body, modern streetwear, urban alley, soft overcast light, authentic vibe",
        "w": 832, "h": 1344
    },
    "fitness_gym": {
        "p": "full body, realistic fitness shoot in gym, athleisure outfit, natural sweat sheen, dramatic rim light",
        "w": 832, "h": 1344
    },
    "bridal": {
        "p": "full body, bridal dress, soft natural light, delicate lace, airy veil, floral background",
        "w": 896, "h": 1408
    },
    "samurai": {
        "p": "full body, realistic samurai armor, katana, temple courtyard, dusk lanterns",
        "w": 896, "h": 1408
    },
    "steampunk": {
        "p": "full body, steampunk outfit, brass goggles, gears, steam pipes, warm tungsten light",
        "w": 832, "h": 1344
    },
}

# -------------------- logging --------------------
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("bot")

# -------------------- storage --------------------
DATA_DIR = Path("profiles"); DATA_DIR.mkdir(exist_ok=True)
def user_dir(uid:int) -> Path:
    p = DATA_DIR / str(uid); p.mkdir(parents=True, exist_ok=True); return p
def list_ref_images(uid:int) -> List[Path]: return sorted(user_dir(uid).glob("ref_*.jpg"))
def profile_path(uid:int) -> Path: return user_dir(uid) / "profile.json"
def load_profile(uid:int) -> Dict[str, Any]:
    p = profile_path(uid)
    if p.exists(): return json.loads(p.read_text())
    return {"images": [], "training_id": None, "finetuned_model": None, "finetuned_version": None, "status": None}
def save_profile(uid:int, prof:Dict[str,Any]): profile_path(uid).write_text(json.dumps(prof))

def save_ref_downscaled(path: Path, raw: bytes, max_side=1024, quality=92):
    im = Image.open(io.BytesIO(raw)).convert("RGB"); im.thumbnail((max_side, max_side))
    im.save(path, "JPEG", quality=quality)

# -------------------- replicate helpers --------------------
def resolve_model_version(slug: str) -> str:
    if ":" in slug: return slug
    model = replicate.models.get(slug)
    versions = list(model.versions.list())
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

# --- check_training_status: берём версию из output + валидация ---
def check_training_status(uid:int) -> Tuple[str, Optional[str]]:
    prof = load_profile(uid)
    tid = prof.get("training_id")
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
    model_version = resolve_model_version(model_slug)  # если вдруг без версии — latest
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
    names = list(STYLE_PRESETS.keys())
    rows, row = [], []
    for i, name in enumerate(names, 1):
        row.append(InlineKeyboardButton(name, callback_data=f"style:{name}"))
        if i % 3 == 0: rows.append(row); row=[]
    if row: rows.append(row)
    return InlineKeyboardMarkup(rows)

# -------------------- handlers --------------------
ENROLL_FLAG: Dict[int,bool] = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Обучаю твою персональную LoRA по 10 фото и генерю без новых фото.\n\n"
        "1) /idenroll — включить набор (до 10 фото)\n"
        "2) /iddone — сохранить профиль\n"
        "3) /trainid — запустить обучение\n"
        "4) /trainstatus — проверить статус\n"
        "5) /styles — список стилей\n"
        "6) /lstyle <preset> — генерация из твоей версии модели (авто-размер под стиль)"
    )

async def id_enroll(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id; ENROLL_FLAG[uid] = True
    await update.message.reply_text("Набор включён. Пришли подряд до 10 фото. Когда закончишь — /iddone.")

async def id_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id; ENROLL_FLAG[uid] = False
    prof = load_profile(uid); prof["images"] = [p.name for p in list_ref_images(uid)]
    save_profile(uid, prof)
    await update.message.reply_text(f"Готово. В профиле {len(prof['images'])} фото. Далее: /trainid.")

async def id_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id; prof = load_profile(uid)
    await update.message.reply_text(
        f"Фото: {len(list_ref_images(uid))}\n"
        f"Статус: {prof.get('status') or '—'}\n"
        f"Модель: {prof.get('finetuned_model') or '—'}\n"
        f"Версия: {prof.get('finetuned_version') or '—'}"
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
    await update.message.reply_text("Выбери стиль:", reply_markup=styles_keyboard())

async def cb_style(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    preset = q.data.split(":",1)[1]
    await q.message.reply_text(f"Стиль выбран: {preset}. Запусти `/lstyle {preset}` после обучения.")

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

async def lstyle_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    args = context.args
    if not args:
        await update.message.reply_text("Укажи стиль, напр.: `/lstyle natural`", reply_markup=styles_keyboard()); return
    preset = args[0].lower()
    if preset not in STYLE_PRESETS:
        await update.message.reply_text(f"Не знаю стиль '{preset}'. Смотри /styles"); return

    prof = load_profile(uid)
    if prof.get("status") != "succeeded":
        await update.message.reply_text("Модель ещё не готова. Сначала /trainid и дождись /trainstatus = succeeded."); return

    model_slug = _pinned_slug(prof)
    meta = STYLE_PRESETS[preset]
    prompt_core = f"{meta.get('p','')}, exact facial identity, no geometry change, relaxed neutral expression, gentle smile"

    w = int(meta.get("w") or GEN_WIDTH)
    h = int(meta.get("h") or GEN_HEIGHT)

    await update.message.chat.send_action(ChatAction.UPLOAD_PHOTO)
    await update.message.reply_text(f"Генерирую из: {model_slug}\nСтиль: {preset} ({w}x{h}) … 🎨")
    try:
        seed = int(time.time()) & 0xFFFFFFFF
        url = await asyncio.to_thread(generate_from_finetune, model_slug, prompt_core, GEN_STEPS, GEN_GUIDANCE, seed, w, h)
        await update.message.reply_photo(photo=url, caption=f"Готово ✨\nСтиль: {preset}")
    except Exception as e:
        logger.exception("lstyle failed")
        await update.message.reply_text(f"Ошибка генерации: {e}")

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
    app.add_handler(CommandHandler("trainid", trainid_cmd))
    app.add_handler(CommandHandler("trainstatus", trainstatus_cmd))
    app.add_handler(CommandHandler("lstyle", lstyle_cmd))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    logger.info("Bot up. Trainer=%s DEST=%s GEN=%dx%d steps=%s guidance=%s",
                LORA_TRAINER_SLUG, _dest_model_slug(), GEN_WIDTH, GEN_HEIGHT, GEN_STEPS, GEN_GUIDANCE)
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()




