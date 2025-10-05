import os
import re
import io
import asyncio
import logging
import requests
import replicate
from typing import Any, Iterable, Tuple, Dict, Optional
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton, WebAppInfo
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes,
    CallbackQueryHandler, filters
)

# ======================
# ОКРУЖЕНИЕ И ПРОВЕРКИ
# ======================
TOKEN = os.getenv("BOT_TOKEN")
os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")

if not TOKEN or not re.match(r"^\d+:[A-Za-z0-9_-]{20,}$", TOKEN):
    raise RuntimeError("Некорректный BOT_TOKEN.")
if not os.getenv("REPLICATE_API_TOKEN"):
    raise RuntimeError("Нет REPLICATE_API_TOKEN.")

# ======================
# РЕСТАВРАЦИЯ (апскейл/ретушь) — вручную по /process
# ======================
PRIMARY_MODEL = os.getenv("REPLICATE_MODEL") or "tencentarc/gfpgan"
FALLBACK_MODELS = [
    "tencentarc/gfpgan",
    "xinntao/realesrgan",
]

# ======================
# СТИЛИ (максимальный реализм)
# ======================
STYLE_BACKEND = "instantid"  # для реализма держим только InstantID
INSTANTID_MODEL = os.getenv("INSTANTID_MODEL", "grandlineai/instant-id-photorealistic")
QWEN_EDIT_MODEL  = os.getenv("QWEN_EDIT_MODEL",  "qwen/qwen-image-edit-plus")

# Слабое вмешательство и низкий CFG => меньше пластика
STYLE_STRENGTH   = float(os.getenv("STYLE_STRENGTH") or 0.26)

NEGATIVE_PROMPT = (
    "cartoon, anime, cgi, 3d, plastic skin, waxy skin, porcelain, airbrushed, beauty filter, smoothing, "
    "overprocessed, oversharpen, hdr effect, halo, neon skin, garish, fake skin, cosplay wig, doll, "
    "ai-artifacts, deformed, bad anatomy, extra fingers, duplicated features, watermark, text, logo, "
    "overly saturated, extreme skin retouch, low detail, lowres, jpeg artifacts"
)

# === 20 реалистичных пресетов «как у CheeseAI», но без пластика
STYLE_PRESETS: Dict[str, str] = {
    # базовые портретные
    "natural":      "ultra realistic portrait, real skin texture with pores and tiny vellus hair, subtle makeup, "
                    "soft natural light, DSLR 85mm look, shallow depth of field, neutral color grading, photographic grain",
    "editorial":    "editorial fashion portrait, preserved natural imperfections, professional color grading, "
                    "soft studio key light + fill, realistic micro skin texture, calibrated tones",
    "headshot_pro": "premium corporate headshot, neutral seamless background, softbox lighting, crisp optics, "
                    "accurate skin tone, subtle film grain, lifelike detail",
    "beauty_soft":  "beauty portrait, glossy lips yet real pores visible, clean studio light, controlled highlights, "
                    "micro-contrast on skin, no beauty filter, neutral grading",
    "noir":         "cinematic noir portrait, deep yet clean contrast, soft rim light, film grain, "
                    "true skin detail and pores, realistic tone mapping",
    "street":       "candid street portrait, dusk city bokeh, available light, realistic color, subtle film grain, "
                    "skin texture preserved",
    "retro_film":   "1970s film portrait look, gentle film grain, organic contrast, natural skin texture, "
                    "neutral-warm color cast, real fabric texture",
    "hollywood":    "cinematic hollywood key light + kicker, realistic makeup, authentic skin microtexture, "
                    "balanced dynamic range, subtle grain, high fidelity detail",
    "vogue":        "beauty cover shot, soft studio lighting, calibrated colors, glossy accents but visible pores, "
                    "clean background, photographic grain",
    "windowlight":  "soft window light portrait, natural diffusion, gentle falloff, lifelike skin texture, "
                    "neutral color grading",
    "studio_softbox":"studio portrait with large softbox, wraparound light, clean background, neutral grading, "
                    "high microdetail of skin and hair",
    "moody":        "moody cinematic portrait, controlled shadows, subtle rim, realistic texture and pores, "
                    "neutral-cool grading, fine grain",
    "pinterest":    "lifestyle portrait, soft natural tones, gentle color palette, realistic skin texture, "
                    "minimal retouch, shallow depth",
    "boho":         "boho portrait, natural fabrics, earthy palette, realistic fabric weave and skin pores, soft daylight",
    "beach":        "beach portrait, golden hour, backlight rim, realistic skin texture, natural highlights, sand texture",
    "winter":       "winter portrait, cool neutral grading, soft overcast light, realistic skin and hair texture, "
                    "breath haze subtle",
    "fitness":      "fitness portrait, natural sheen (not plastic), defined but realistic skin detail, "
                    "studio rim lights, neutral color",
    "techwear":     "techwear portrait, matte fabrics with real weave, soft rim neon, realistic skin, city bokeh, "
                    "neutral saturation",
    # образы
    "princess":     "royal look portrait, elegant gown, subtle tiara, realistic skin texture with pores, "
                    "soft cinematic light, realistic fabric weave",
    "harley":       "Harley Quinn cosplay but real person, natural skin texture with visible pores, real hair texture "
                    "(not a wig), blonde pigtails with soft pink and blue tips, lived-in makeup, "
                    "cinematic key light, no plastic look",
    "superman":     "Superman cosplay photorealistic, authentic fabric texture, realistic proportions, rim light, "
                    "natural face detail and pores",
    "cyberpunk":    "photorealistic cyberpunk portrait, subtle neon rim, city bokeh, realistic speculars, "
                    "neutral color grading, skin texture preserved",
    "business":     "professional business portrait, neutral seamless background, softbox lighting, crisp detail, "
                    "real skin texture with pores, realistic color",
    "evening":      "evening glam portrait, smoky eyes, glossy lips, controlled highlights, fine skin detail preserved, "
                    "soft cinematic light",
}

# ======================
# ЛОГИРОВАНИЕ
# ======================
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ======================
# ВСПОМОГАТЕЛЬНОЕ — URL/файлы/Replicate
# ======================
def resolve_model_version(slug: str) -> str:
    if ":" in slug:
        return slug
    model = replicate.models.get(slug)
    versions = list(model.versions.list())
    if not versions:
        raise RuntimeError(f"У модели {slug} нет доступных версий.")
    return f"{slug}:{versions[0].id}"

async def tg_download_photo(message, path: str) -> bool:
    if not message or not message.photo:
        return False
    f = await message.photo[-1].get_file()
    await f.download_to_drive(path)
    return True

async def tg_public_url(message) -> str:
    f = await message.photo[-1].get_file()
    return f.file_path   # https://api.telegram.org/file/...

def _as_url(obj: Any) -> Optional[str]:
    if isinstance(obj, str) and obj.startswith(("http://", "https://")):
        return obj
    if hasattr(obj, "path"):
        p = getattr(obj, "path")
        if isinstance(p, str):
            if p.startswith(("http://", "https://")):
                return p
            if p.startswith("/"):
                return "https://replicate.delivery" + p
    if hasattr(obj, "url"):
        u = getattr(obj, "url")
        if isinstance(u, str) and u.startswith(("http://", "https://")):
            return u
    return None

def _walk(obj: Any) -> Iterable[Any]:
    url = _as_url(obj)
    if url is not None:
        yield url; return
    if isinstance(obj, dict):
        for v in obj.values(): yield from _walk(v)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj: yield from _walk(v)

def extract_any_url(output: Any) -> Optional[str]:
    for v in _walk(output): return v
    return None

def normalize_output(output: Any) -> str:
    direct = _as_url(output)
    if direct: return direct
    if isinstance(output, list) and output:
        direct = _as_url(output[0])
        if direct: return direct
        if isinstance(output[0], dict):
            for k in ("image","url","output","restored_image","output_image","result"):
                if k in output[0] and output[0][k]:
                    u = _as_url(output[0][k]) or extract_any_url(output[0][k])
                    if u: return u
    if isinstance(output, dict):
        for k in ("image","url","output","restored_image","output_image","result"):
            if k in output and output[k]:
                u = _as_url(output[k]) or extract_any_url(output[k])
                if u: return u
    any_url = extract_any_url(output)
    if any_url: return any_url
    raise RuntimeError("Модель вернула пустой или неожиданный ответ.")

def replicate_run_flexible(model: str, inputs_list: Iterable[dict]) -> str:
    last = None
    for payload in inputs_list:
        try:
            out = replicate.run(model, input=payload)
            logger.info("Raw output %s with %s: %r", model, payload, out)
            return normalize_output(out)
        except Exception as e:
            last = e
            logger.warning("Model %s rejected payload %s: %s", model, payload, e)
    raise last or RuntimeError("Не удалось получить результат.")

def run_restore_with_fallbacks(image_url: str) -> Tuple[str, str]:
    candidates = [PRIMARY_MODEL] + [m for m in FALLBACK_MODELS if m != PRIMARY_MODEL]
    last = None
    for slug in candidates:
        try:
            resolved = resolve_model_version(slug)
            low = resolved.lower()
            if "gfpgan" in low:
                inputs = [{"img": image_url}]
            elif "realesrgan" in low or "real-esrgan" in low:
                inputs = [{"image": image_url}]
            else:
                inputs = [{"image": image_url}, {"img": image_url}, {"input": image_url}]
            url = replicate_run_flexible(resolved, inputs)
            return url, resolved
        except Exception as e:
            last = e
            logger.warning("Fallback model %s failed: %s", slug, e)
    raise RuntimeError(f"Все модели отклонили запрос. Последняя ошибка: {last}")

# ===== Реалистичная стилизация (ID-сохранение)
def run_style_realistic(image_url: str, prompt: str, strength: float, backend: str) -> Tuple[str, str]:
    """
    Максимум реализма: сильное удержание лица, мягкая сила изменений, низкий CFG.
    """
    denoise  = max(0.18, min(0.30, strength))
    guidance = 3.4  # ещё ниже => меньше «нейропластика»

    if backend == "instantid":
        resolved = resolve_model_version(INSTANTID_MODEL)
        ip_scale = 0.93  # ещё крепче держим идентичность
        inputs_try = [
            {
                "face_image": image_url,
                "image": image_url,
                "prompt": prompt,
                "negative_prompt": NEGATIVE_PROMPT,
                "ip_adapter_scale": ip_scale,
                "controlnet_conditioning_scale": 0.50,  # меньше перерисовки
                "strength": denoise,
                "guidance_scale": guidance,
                "num_inference_steps": 26,  # чуть короче => меньше мыла
            },
            {
                "face_image": image_url,
                "prompt": prompt,
                "negative_prompt": NEGATIVE_PROMPT,
                "ip_adapter_scale": ip_scale,
                "num_inference_steps": 24,
                "guidance_scale": guidance,
            },
        ]
        url = replicate_run_flexible(resolved, inputs_try)
        return url, resolved

    elif backend == "qwen":
        resolved = resolve_model_version(QWEN_EDIT_MODEL)
        inputs_try = [
            {
                "image": image_url,
                "prompt": prompt,
                "negative_prompt": NEGATIVE_PROMPT,
                "strength": denoise,
                "guidance_scale": guidance,
                "num_inference_steps": 28,
            },
            {
                "image": image_url,
                "instruction": prompt + ". Avoid cartoon look. Keep natural skin texture.",
                "strength": denoise,
            },
        ]
        url = replicate_run_flexible(resolved, inputs_try)
        return url, resolved

    else:
        raise RuntimeError(f"Неизвестный backend '{backend}'.")

# ======================
# ОТПРАВКА РЕЗУЛЬТАТА (байтами, если надо)
# ======================
async def safe_send_image(update: Update, url: str, caption: str = ""):
    msg = update.message

    def _download_bytes(u: str) -> Optional[bytes]:
        try:
            r = requests.get(u, timeout=90, allow_redirects=True)
            r.raise_for_status()
            return r.content
        except Exception as e:
            logger.warning("Download failed: %s", e)
            return None

    if "replicate.delivery" in (url or ""):
        content = _download_bytes(url)
        if content:
            bio = io.BytesIO(content); bio.name = "result.jpg"
            try:
                await msg.reply_photo(photo=bio, caption=caption); return
            except Exception as e:
                logger.warning("Reply photo (bytes) failed: %s", e)
                bio.seek(0)
                try:
                    await msg.reply_document(document=bio, caption=caption or "Результат"); return
                except Exception as e2:
                    logger.warning("Reply document (bytes) failed: %s", e2)

    try:
        await msg.reply_photo(photo=url, caption=caption); return
    except Exception as e:
        logger.warning("Telegram refused URL as photo: %s", e)

    content = _download_bytes(url)
    if content:
        bio = io.BytesIO(content); bio.name = "result.jpg"
        try:
            await msg.reply_photo(photo=bio, caption=caption); return
        except Exception as e:
            logger.warning("Re-upload as bytes (photo) failed: %s", e)
            bio.seek(0)
            try:
                await msg.reply_document(document=bio, caption=caption or "Результат"); return
            except Exception as e2:
                logger.warning("Send as document failed: %s", e2)

    await msg.reply_text(f"Готово, но Телеграм не принял файл напрямую. Ссылка:\n{url}")

# ======================
# ПАРСЕР СТИЛЯ ИЗ ТЕКСТА (расширенные алиасы)
# ======================
RUS_PRESET_ALIASES = {
    "natural":  ["natural","натурал","естественно","натуральный","реализм","реалистично"],
    "editorial":["editorial","эдиториал","журнальный","fashion","фэшн","фешн"],
    "headshot_pro":["headshot","хедшот","деловой","business pro","профи"],
    "beauty_soft":["beauty","бьюти","мягкий","глянец","софт"],
    "noir":     ["noir","нуар","чб","ч/б","черно-белый"],
    "street":   ["street","стрит","улица","стритфото"],
    "retro_film":["retro","ретро","film","фильм","70s","винтаж"],
    "hollywood":["hollywood","голливуд","кино","cinema"],
    "vogue":    ["vogue","вог","обложка","cover"],
    "windowlight":["window","окно","естественный свет","дневной"],
    "studio_softbox":["studio","студия","софтбокс","softbox"],
    "moody":    ["moody","муди","кинематографично","кино"],
    "pinterest":["pinterest","лайфстайл","lifestyle"],
    "boho":     ["boho","бохо"],
    "beach":    ["beach","пляж","golden hour","закат"],
    "winter":   ["winter","зима","снег"],
    "fitness":  ["fitness","спорт","атлет"],
    "techwear": ["techwear","техвир","техстиль","урбан"],
    "princess": ["princess","принцесса","royal","королева"],
    "harley":   ["harley","харли","харли квин","quinn"],
    "superman": ["superman","супермен","кларк"],
    "cyberpunk":["cyberpunk","киберпанк","неон"],
    "business": ["business","деловой","офис","корп"],
    "evening":  ["evening","evening glam","вечер","вечерний"],
}

def detect_preset_from_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    t = text.lower().strip()
    tokens = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9#+/_-]+", t)
    joined = " ".join(tokens)

    m = re.search(r"(?:/style|стиль|style)[:\s]+([^\s#]+)", joined)
    if m:
        candidate = m.group(1).strip("#/ ").lower()
        if candidate in STYLE_PRESETS:
            return candidate
        for key, aliases in RUS_PRESET_ALIASES.items():
            if candidate in aliases:
                return key

    words = set(w.strip("#/ ").lower() for w in re.split(r"[\s,;]+", joined) if w.strip("#/ "))
    for key in STYLE_PRESETS:
        if key in words:
            return key
    for key, aliases in RUS_PRESET_ALIASES.items():
        if any(a in words for a in aliases):
            return key
    return None

# ======================
# СТИЛИЗАЦИЯ (реалистичная img2img)
# ======================
def styles_keyboard() -> InlineKeyboardMarkup:
    # строим клавиатуру динамически, 4 в ряд
    names = list(STYLE_PRESETS.keys())
    rows = []
    row = []
    for i, name in enumerate(names, 1):
        row.append(InlineKeyboardButton(name.replace("_"," ").title(), callback_data=f"style:{name}"))
        if i % 4 == 0:
            rows.append(row); row = []
    if row: rows.append(row)
    return InlineKeyboardMarkup(rows)

async def style_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text(
            "Выбери стиль и ответь этой командой на фото (или пришли фото с подписью `/style harley`).",
            reply_markup=styles_keyboard(),
        )
        return
    preset = args[0].strip().lower()
    await _run_style_flow(update, context, preset)

async def callback_style(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data or ""
    if not data.startswith("style:"): return
    preset = data.split(":", 1)[1]
    await q.message.reply_text(
        f"Окей, стиль: {preset}. Теперь ответь этой командой на нужное фото: `/style {preset}`",
        parse_mode=None
    )

async def _run_style_flow(update: Update, context: ContextTypes.DEFAULT_TYPE, preset: str):
    if preset not in STYLE_PRESETS:
        await update.message.reply_text(f"Неизвестный стиль '{preset}'. Смотри /styles", parse_mode=None)
        return

    m = update.message
    if m.reply_to_message and m.reply_to_message.photo:
        source = m.reply_to_message
    elif m.photo:
        source = m
    else:
        await m.reply_text("Нужно прислать фото (или ответить командой на фото). Пример: `/style harley`", parse_mode=None)
        return

    await m.chat.send_action(ChatAction.UPLOAD_PHOTO)
    await m.reply_text(f"Стилизация (реалистично): {preset}… 🎨")

    tmp_path = "style_input.jpg"
    try:
        ok = await tg_download_photo(source, tmp_path)
        if not ok:
            await m.reply_text("Не нашла фото для стилизации. Пришли изображение."); return

        public_url = await tg_public_url(source)

        use_strength = STYLE_STRENGTH  # 0.26 по умолчанию
        result_url, used_model = await asyncio.to_thread(
            run_style_realistic, public_url, STYLE_PRESETS[preset], use_strength, STYLE_BACKEND
        )
        await safe_send_image(update, result_url,
            caption=f"Готово ✨\nСтиль: {preset}\nBackend: {STYLE_BACKEND}\nМодель: {used_model}")

    except Exception as e:
        logger.exception("Ошибка стилизации")
        await m.reply_text(f"Упс, не вышло. Причина: {e}")
    finally:
        try:
            if os.path.exists(tmp_path): os.remove(tmp_path)
        except Exception:
            pass

# ======================
# ХЕНДЛЕРЫ
# ======================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ver = getattr(replicate, "__version__", "unknown")

    kb = [[InlineKeyboardButton("Открыть стили", callback_data="open_styles")]]
    await update.message.reply_text(
        "Привет! Я делаю **реалистичную** стилизацию фото.\n\n"
        "Как пользоваться:\n"
        "1) Пришли фото.\n"
        "2) В подписи к фото укажи стиль (например: `harley`, `natural`, `vogue`).\n"
        "— Фото сразу будет стилизовано (без пластика/сглаживания).\n\n"
        "Команды:\n"
        "• /styles — показать все стили\n"
        "• /style <preset> — стилизовать (если ответишь на фото)\n"
        "• /process — вручную запустить реставрацию/апскейл (опционально)\n\n"
        f"Стили backend: {STYLE_BACKEND} (InstantID={INSTANTID_MODEL})\n"
        f"replicate=={ver}",
        reply_markup=InlineKeyboardMarkup(kb)
    )

async def process_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    m = update.message
    if not m.reply_to_message:
        await m.reply_text("Сделай /process ответом на сообщение с фото 😉"); return
    if not m.reply_to_message.photo:
        await m.reply_text("В сообщении, на которое ты отвечаешь, нет фото."); return
    await _process_photo_and_reply(update, context, m.reply_to_message)

async def handle_direct_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Если в подписи к фото указан стиль — сразу стилизуем (без реставрации).
    Если стиля нет — предлагаем выбрать стиль.
    """
    m = update.message
    preset = detect_preset_from_text(m.caption)

    if preset and preset in STYLE_PRESETS:
        await m.chat.send_action(ChatAction.UPLOAD_PHOTO)
        await m.reply_text(f"Стилизация (реалистично): {preset}… 🎨")

        tmp_path = "style_input.jpg"
        try:
            ok = await tg_download_photo(m, tmp_path)
            if not ok:
                await m.reply_text("Не нашла фото для стилизации. Пришли изображение."); return

            public_url = await tg_public_url(m)

            use_strength = STYLE_STRENGTH
            result_url, used_model = await asyncio.to_thread(
                run_style_realistic, public_url, STYLE_PRESETS[preset], use_strength, STYLE_BACKEND
            )
            await safe_send_image(update, result_url,
                caption=f"Готово ✨\nСтиль: {preset}\nBackend: {STYLE_BACKEND}\nМодель: {used_model}")

        except Exception as e:
            logger.exception("Ошибка стилизации при прямой загрузке")
            await m.reply_text(f"Упс, не вышло. Причина: {e}")
        finally:
            try:
                if os.path.exists(tmp_path): os.remove(tmp_path)
            except Exception:
                pass
        return

    await m.reply_text(
        "Укажи стиль в подписи к фото (например: `harley`, `natural`, `vogue`) "
        "или выбери из списка ниже:",
        reply_markup=styles_keyboard(),
        parse_mode=None,
    )

async def styles_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Выбери стиль (кнопки ниже) и потом пришли фото с этим словом в подписи.\n"
        "Лайфхак: для максимального реализма держи лицо крупно, без фильтров.",
        reply_markup=styles_keyboard(),
        parse_mode=None,
    )

async def health(update: Update, context: ContextTypes.DEFAULT_TYPE):
    slug = os.getenv("REPLICATE_MODEL") or "tencentarc/gfpgan"
    try:
        resolved = resolve_model_version(slug)
    except Exception as e:
        resolved = f"не удалось получить версию: {e}"
    await update.message.reply_text(
        f"✅ Telegram OK\n"
        f"🔑 Replicate token: {'found' if os.getenv('REPLICATE_API_TOKEN') else 'missing'}\n"
        f"🧠 Restore model: {slug}\n"
        f"🎨 Style backend: {STYLE_BACKEND}\n"
        f"📦 InstantID: {INSTANTID_MODEL}\n"
        f"📦 QwenEdit: {QWEN_EDIT_MODEL}\n"
        f"🔢 Resolved restore version: {resolved}"
    )

async def open_styles_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    await q.message.reply_text(
        "Выбери стиль из списка:",
        reply_markup=styles_keyboard()
    )

# ======================
# ОСНОВНОЙ ПАЙПЛАЙН (реставрация по /process)
# ======================
async def _process_photo_and_reply(update: Update, context: ContextTypes.DEFAULT_TYPE, source_message):
    msg = update.message
    await msg.chat.send_action(ChatAction.UPLOAD_PHOTO)
    await msg.reply_text("Принято. Обрабатываю изображение… 🔧")

    tmp_path = "input.jpg"
    try:
        ok = await tg_download_photo(source_message, tmp_path)
        if not ok:
            await msg.reply_text("В ответе нет фото. Ответь /process на сообщение с картинкой.")
            return

        public_url = await tg_public_url(source_message)

        result_url, used_model = await asyncio.to_thread(run_restore_with_fallbacks, public_url)
        await safe_send_image(update, result_url, caption=f"Готово ✨\nМодель: {used_model}")

    except Exception as e:
        logger.exception("Ошибка обработки")
        await msg.reply_text(f"Упс, не вышло. Причина: {e}")
    finally:
        try:
            if os.path.exists(tmp_path): os.remove(tmp_path)
        except Exception:
            pass

# ======================
# MAIN
# ======================
def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("process", process_cmd))
    app.add_handler(CommandHandler("health", health))
    app.add_handler(CommandHandler("styles", styles_cmd))
    app.add_handler(CommandHandler("style", style_command))
    app.add_handler(CallbackQueryHandler(callback_style, pattern=r"^style:"))
    app.add_handler(CallbackQueryHandler(open_styles_cb, pattern=r"^open_styles$"))
    app.add_handler(MessageHandler(filters.PHOTO, handle_direct_photo))

    logger.info("Бот запущен…")
    app.run_polling(
        stop_signals=None,
        close_loop=False,
        allowed_updates=Update.ALL_TYPES,
    )

if __name__ == "__main__":
    main()


