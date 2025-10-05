import os
import re
import io
import asyncio
import logging
import requests
import replicate
from typing import Any, Iterable, Tuple, Dict, Optional
from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup,
)
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

PRIMARY_MODEL = os.getenv("REPLICATE_MODEL") or "tencentarc/gfpgan"
FALLBACK_MODELS = ["tencentarc/gfpgan", "xinntao/realesrgan"]

STYLE_BACKEND = "instantid"
INSTANTID_MODEL = os.getenv("INSTANTID_MODEL", "grandlineai/instant-id-photorealistic")
QWEN_EDIT_MODEL = os.getenv("QWEN_EDIT_MODEL", "qwen/qwen-image-edit-plus")

ULTRA_LOCK_STRENGTH = float(os.getenv("ULTRA_LOCK_STRENGTH") or 0.14)
ULTRA_LOCK_GUIDANCE = float(os.getenv("ULTRA_LOCK_GUIDANCE") or 2.3)

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

STYLE_PRESETS = {
    "boho": "boho portrait, natural fabrics, earthy palette, realistic fabric weave and skin pores, soft daylight",
    "natural": "ultra realistic portrait, real skin texture with pores and tiny vellus hair, subtle makeup, soft natural light, DSLR 85mm look",
    "editorial": "editorial fashion portrait, preserved natural imperfections, professional color grading",
    "beauty_soft": "beauty portrait, glossy lips yet real pores visible, clean studio light, controlled highlights",
    "vogue": "beauty cover shot, soft studio lighting, calibrated colors, glossy accents but visible pores",
}

# ======================
# ЛОГИ
# ======================
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================
def resolve_model_version(slug: str) -> str:
    if ":" in slug:
        return slug
    model = replicate.models.get(slug)
    versions = list(model.versions.list())
    if not versions:
        raise RuntimeError(f"Нет версий модели {slug}")
    return f"{slug}:{versions[0].id}"

async def tg_download_photo(message, path: str) -> bool:
    if not message.photo:
        return False
    f = await message.photo[-1].get_file()
    await f.download_to_drive(path)
    return True

async def tg_public_url(message) -> str:
    f = await message.photo[-1].get_file()
    return f.file_path

def _as_url(obj: Any) -> Optional[str]:
    if isinstance(obj, str) and obj.startswith(("http", "https")):
        return obj
    if hasattr(obj, "url") and getattr(obj, "url").startswith(("http", "https")):
        return obj.url
    return None

def extract_any_url(output: Any) -> Optional[str]:
    if isinstance(output, (list, tuple)):
        for v in output:
            r = extract_any_url(v)
            if r:
                return r
    if isinstance(output, dict):
        for v in output.values():
            r = extract_any_url(v)
            if r:
                return r
    return _as_url(output)

def normalize_output(output: Any) -> str:
    u = extract_any_url(output)
    if not u:
        raise RuntimeError("Модель вернула пустой ответ.")
    return u

def replicate_run_flexible(model: str, inputs_list: Iterable[dict]) -> str:
    last = None
    for payload in inputs_list:
        try:
            out = replicate.run(model, input=payload)
            logger.info("Raw output %s", model)
            return normalize_output(out)
        except Exception as e:
            last = e
            logger.warning("Model %s rejected payload: %s", model, e)
    raise last or RuntimeError("Не удалось получить результат.")

def run_restore_with_fallbacks(image_url: str) -> Tuple[str, str]:
    for slug in [PRIMARY_MODEL] + [m for m in FALLBACK_MODELS if m != PRIMARY_MODEL]:
        try:
            resolved = resolve_model_version(slug)
            inputs = [{"image": image_url}, {"img": image_url}]
            url = replicate_run_flexible(resolved, inputs)
            return url, resolved
        except Exception as e:
            logger.warning("Fallback model %s failed: %s", slug, e)
    raise RuntimeError("Все модели отклонили запрос.")

# ======================
# УЛЬТРА-ПОХОЖЕСТЬ
# ======================
def run_style_realistic(image_url: str, prompt: str, _strength: float, backend: str) -> Tuple[str, str]:
    denoise = max(0.10, min(0.18, ULTRA_LOCK_STRENGTH))
    guidance = ULTRA_LOCK_GUIDANCE

    base_prompt = (
        "highly realistic portrait, exact facial identity, preserve original facial proportions and features, "
        "keep natural hair volume and hairline, matched skin tone, balanced natural lighting, "
        "no stylization of anatomy, no geometry change, "
        + prompt + AESTHETIC_SUFFIX
    )

    negative = (
        NEGATIVE_PROMPT +
        ", enlarged nose, altered nose shape, changed lips, bluish nose, flat hair, missing hair volume"
    )

    try:
        import zlib
        seed = zlib.adler32(image_url.encode("utf-8")) % (2**31 - 1)
    except Exception:
        seed = 42

    if backend == "instantid":
        resolved = resolve_model_version(INSTANTID_MODEL)
        ip_scale = 1.0
        logger.info("InstantID input -> %s", image_url)
        inputs_try = [
            {
                "face_image": image_url,
                "input_image": image_url,
                "input_image_ref": image_url,
                "image": image_url,
                "prompt": base_prompt,
                "negative_prompt": negative,
                "ip_adapter_scale": ip_scale,
                "controlnet_conditioning_scale": 0.18,
                "strength": denoise,
                "guidance_scale": guidance,
                "num_inference_steps": 18,
                "seed": seed,
            }
        ]
        url = replicate_run_flexible(resolved, inputs_try)
        return url, resolved
    else:
        resolved = resolve_model_version(QWEN_EDIT_MODEL)
        inputs_try = [
            {
                "image": image_url,
                "prompt": base_prompt,
                "negative_prompt": negative,
                "strength": denoise,
                "guidance_scale": guidance,
                "num_inference_steps": 20,
                "seed": seed,
            }
        ]
        url = replicate_run_flexible(resolved, inputs_try)
        return url, resolved

# ======================
# ОТПРАВКА ФОТО
# ======================
async def safe_send_image(update: Update, url: str, caption: str = ""):
    msg = update.message
    try:
        await msg.reply_photo(photo=url, caption=caption)
    except Exception:
        content = requests.get(url).content
        bio = io.BytesIO(content)
        bio.name = "result.jpg"
        await msg.reply_photo(photo=bio, caption=caption)

# ======================
# ОБРАБОТЧИКИ
# ======================
def styles_keyboard():
    rows = []
    row = []
    for i, name in enumerate(STYLE_PRESETS.keys(), 1):
        row.append(InlineKeyboardButton(name.title(), callback_data=f"style:{name}"))
        if i % 3 == 0:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return InlineKeyboardMarkup(rows)

async def _run_style_flow(update: Update, preset: str):
    m = update.message
    if m.reply_to_message and m.reply_to_message.photo:
        src = m.reply_to_message
    elif m.photo:
        src = m
    else:
        await m.reply_text("Нужно прислать фото или ответить на него командой.")
        return

    await m.chat.send_action(ChatAction.UPLOAD_PHOTO)
    await m.reply_text(f"Стилизация (ультра-похоже): {preset}… 🎨")

    tmp = "input.jpg"
    try:
        await tg_download_photo(src, tmp)
        url = await tg_public_url(src)
        result, model = await asyncio.to_thread(
            run_style_realistic, url, STYLE_PRESETS[preset], ULTRA_LOCK_STRENGTH, STYLE_BACKEND
        )
        await safe_send_image(update, result, f"Готово ✨\nСтиль: {preset}\nМодель: {model}")
    except Exception as e:
        logger.exception(e)
        await m.reply_text(f"Ошибка: {e}")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

async def style_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text("Выбери стиль:", reply_markup=styles_keyboard())
        return
    await _run_style_flow(update, args[0].lower())

async def callback_style(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    preset = q.data.split(":")[1]
    await q.message.reply_text(f"Теперь пришли фото с подписью `/style {preset}`")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    m = update.message
    text = m.caption or ""
    for k in STYLE_PRESETS:
        if k.lower() in text.lower():
            await _run_style_flow(update, k)
            return
    await m.reply_text("Укажи стиль (например `boho` или `vogue`)", reply_markup=styles_keyboard())

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я делаю **реалистичную стилизацию фото** с сохранением лица.\n"
        "Просто пришли фото и стиль в подписи (например: `boho`, `vogue`, `natural`)."
    )

# ======================
# MAIN
# ======================
async def _post_init(app):
    await app.bot.delete_webhook(drop_pending_updates=True)

async def _error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Ошибка: %s", context.error)

def main():
    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .post_init(_post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("style", style_command))
    app.add_handler(CallbackQueryHandler(callback_style, pattern=r"^style:"))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_error_handler(_error_handler)

    logger.info("Бот запущен… (polling)")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()



