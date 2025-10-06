    # === Telegram LoRA Bot — CINEMA SHOTS v3 + PERSIST ===
    # Flux LoRA trainer • кино-стили • must-include • anti-headshot • per-user Replicate model
    # Требования: python-telegram-bot==20.7, replicate==0.31.0, pillow==10.4.0

import os, re, io, json, time, asyncio, logging, shutil, random
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
    DEST_PREFIX = os.getenv("REPLICATE_DEST_MODEL_PREFIX", "yourtwin").strip()

    # Тренер LoRA (Flux LoRA trainer)
    LORA_TRAINER_SLUG = os.getenv("LORA_TRAINER_SLUG", "replicate/flux-lora-trainer").strip()
    LORA_INPUT_KEY    = os.getenv("LORA_INPUT_KEY", "input_images").strip()

    # Классификатор пола (опционально)
    GENDER_MODEL_SLUG = os.getenv("GENDER_MODEL_SLUG", "nateraw/vit-age-gender").strip()

    # --- Твики обучения (бережные) ---
    LORA_MAX_STEPS     = int(os.getenv("LORA_MAX_STEPS", "1400"))
    LORA_LR            = float(os.getenv("LORA_LR", "0.00006"))
    LORA_USE_FACE_DET  = os.getenv("LORA_USE_FACE_DET", "true").lower() in ["1","true","yes","y"]
    LORA_RESOLUTION    = int(os.getenv("LORA_RESOLUTION", "1024"))
    LORA_CAPTION_PREF  = os.getenv(
        "LORA_CAPTION_PREFIX",
        "a high quality photo of the same person, neutral expression, gentle smile, "
        "balanced facial proportions, soft jawline, clear eyes"
    ).strip()

    # --- Генерация (кинематографично, без пластика) ---
    GEN_STEPS     = int(os.getenv("GEN_STEPS", "46"))
    GEN_GUIDANCE  = float(os.getenv("GEN_GUIDANCE", "3.4"))
    GEN_WIDTH     = int(os.getenv("GEN_WIDTH", "896"))
    GEN_HEIGHT    = int(os.getenv("GEN_HEIGHT", "1152"))

    NEGATIVE_PROMPT = (
        "cartoon, anime, 3d, cgi, beauty filter, skin smoothing, waxy, overprocessed, oversharpen, "
        "lowres, blur, jpeg artifacts, text, watermark, logo, bad anatomy, extra fingers, short fingers, "
        "puffy face, swollen face, bulky masseter, wide jaw, clenched jaw, duckface, overfilled lips, "
        "narrow eyes, tiny eyes, cross-eye, lazy eye, asymmetrical eyes, misaligned pupils, "
        "overly muscular neck or shoulders, bodybuilder female, extreme makeup, heavy contouring"
    )
    AESTHETIC_SUFFIX = (
        ", photorealistic, visible skin texture, natural color, soft filmic contrast, gentle micro-sharpen, no beautification"
    )

    # Усиленный анти-портрет, чтобы сцены не превращались в селфи
    NEG_HEADSHOT = (
        "headshot, close-up, selfie, passport photo, cropped head, tight framing, "
        "studio backdrop, plain background, bokeh-only background, portrait crop"
    )

    def _beauty_guardrail() -> str:
        return (
            "exact facial identity, identity preserved, "
            "balanced facial proportions, symmetrical face, natural oval, soft jawline, "
            "smooth cheek contour, relaxed neutral expression, subtle gentle smile, "
            "open expressive almond-shaped eyes, clean catchlights, clear irises, "
            "realistic body proportions, proportional shoulders and waist, natural posture"
        )

    # ---------- Композиция / «линза» ----------
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

    # ---------- Стили «как кадр из кино» ----------
    Style = Dict[str, Any]
    STYLE_PRESETS: Dict[str, Style] = {
        # Портреты
        "Портрет у окна": {
            "desc": "Крупный кинопортрет у большого окна; мягкая тень от рамы, живое боке — кадр из европейской драмы.",
            "p": "естественный портрет у большого окна, легкое боке",
            "comp": "closeup", "tone": "daylight"
        },
        "Портрет 85мм": {
            "desc": "Классика 85мм — микроскопическая ГРИП и дорогой плёночный вайб.",
            "p": "реалистичный портрет с эффектом 85мм объектива, малая глубина резкости",
            "comp": "closeup", "tone": "warm"
        },
        "Бьюти студия": {
            "desc": "Чистый свет, аккуратные рефлексы, кожа без “пластика”.",
            "p": "бьюти-портрет, чистый студийный свет, минимальный макияж",
            "comp": "closeup", "tone": "daylight"
        },
        "Кинопортрет": {
            "desc": "Рембрандтовский свет и мягкая плёнка — фестивальное кино.",
            "p": "кинематографичный портрет, рембрандтовский свет, мягкая пленочная зернистость",
            "comp": "closeup", "tone": "cool"
        },
        "Фильм-нуар (портрет)": {
            "desc": "Жалюзи, дым, жёсткие тени — нуар 40-х.",
            "p": "портрет в стиле кино-нуар, дым, высокая контрастность",
            "comp": "closeup", "tone": "noir"
        },

        # Современные
        "Стритвэр город": {
            "desc": "Уличный лук, стекло и граффити, город дышит.",
            "p_f": "современный стритвэр, кроп-топ и джоггеры, городская улица",
            "p_m": "современный стритвэр, худи и джоггеры, городская улица",
            "comp": "half", "tone": "daylight"
        },
        "Вечерний выход": {
            "desc": "Красная дорожка и тёплый софтбокс — премьера.",
            "p_f": "элегантное вечернее платье на красной дорожке",
            "p_m": "классический смокинг на красной дорожке",
            "comp": "half", "tone": "warm"
        },
        "Бизнес": {
            "desc": "Стеклянное лобби, строгая геометрия — сериал про корпорации.",
            "p_f": "деловой костюм, лобби современного офиса",
            "p_m": "деловой костюм, лобби современного офиса",
            "comp": "half", "tone": "daylight"
        },
        "Ночной неон": {
            "desc": "Мокрый асфальт, пар и вывески — кибернуар.",
            "p": "улица в дождь, яркие неоновые вывески, отражения в лужах",
            "comp": "half", "tone": "neon"
        },

        # Профессии
        "Врач у палаты": {
            "desc": "Белый халат, стетоскоп, палата за спиной — документальный вайб.",
            "p_f": "врач в белом халате и шапочке, стетоскоп; фон: больничная палата",
            "p_m": "врач в белом халате и шапочке, стетоскоп; фон: больничная палата",
            "comp": "half", "tone": "daylight"
        },
        "Хирург операционная": {
            "desc": "Холодные приборы и блики на металле — триллер.",
            "p": "хирург в шапочке и маске, хирургическая форма; фон: операционная с приборами",
            "comp": "half", "tone": "cool"
        },
        "Шеф-повар кухня": {
            "desc": "Огонь, пар и шипение — гастрономический драйв.",
            "p": "шеф-повар в кителе; фон: профессиональная кухня, пламя и пар",
            "comp": "half", "tone": "warm"
        },
        "Учёный лаборатория": {
            "desc": "Подсветки приборов и стекло — популярная наука.",
            "p": "лабораторный халат, пробирки и стеклянная посуда; фон: современная лаборатория",
            "comp": "half", "tone": "cool"
        },
        "Боксер на ринге": {
            "desc": "Жёсткий верхний свет, пот, канаты — спортивная драма.",
            "p_f": "боксерша в перчатках; фон: ринг, пот, жёсткий верхний свет",
            "p_m": "боксер в перчатках; фон: ринг, пот, жёсткий верхний свет",
            "comp": "half", "tone": "cool"
        },
        "Фитнес зал": {
            "desc": "Контровой свет и пыль в лучах — сцена «над собой».",
            "p_f": "спортивный топ и легинсы; фон: тренажерный зал, драматичная контровая",
            "p_m": "майка и шорты; фон: тренажерный зал, драматичная контровая",
            "comp": "half", "tone": "cool"
        },

        # Приключения / Экшн
        "Приключенец (руины)": {
            "desc": "Пыльные лучи, древние блоки — приключенческий блокбастер.",
            "p_f": "исследовательница гробниц, тактический костюм, перчатки без пальцев; фон: древние руины",
            "p_m": "исследователь гробниц, тактический костюм, перчатки без пальцев; фон: древние руины",
            "comp": "full", "tone": "warm"
        },
        "Пустынный исследователь": {
            "desc": "Дюны, дрожащий жар — дух большой пустыни.",
            "p": "шарф, карго-экипировка; фон: песчаные дюны и каньон",
            "comp": "full", "tone": "warm"
        },
        "Горы снег": {
            "desc": "Синие тени, ледоруб, ветер — суровая красота альпинизма.",
            "p": "альпинистская куртка, кошки/ледоруб; фон: заснеженный гребень и небо",
            "comp": "full", "tone": "cool"
        },
        "Серфер": {
            "desc": "Брызги, блики, доска в руках и ломающееся море — летний спортфильм.",
            "p_f": "гидрокостюм или купальник для серфинга, доска под мышкой; фон: океанская волна и брызги, кромка прибоя",
            "p_m": "гидрокостюм, доска под мышкой; фон: океанская волна и брызги, кромка прибоя",
            "comp": "full", "tone": "warm"
        },

        # Фэнтези / История
        "Эльфийская знать": {
            "desc": "Изумрудный храм, лучи в тумане, драгоценные узоры — высокая фэнтези.",
            "p_f": "эльфийская королева в струящемся платье; фон: лесной храм и лучи света",
            "p_m": "эльфийский король в плаще и доспехах; фон: лесной храм и лучи света",
            "comp": "full", "tone": "candle"
        },
        "Самурай в храме": {
            "desc": "Фонари, лакированные доспехи — дзен и сталь.",
            "p": "самурайские доспехи и катана; фон: двор синтоистского храма с фонарями",
            "comp": "full", "tone": "warm"
        },
        "Средневековый рыцарь": {
            "desc": "Латы, штандарты, пыль турнира — исторический эпос.",
            "p": "полный комплект доспехов и плащ; фон: замковый турнирный двор",
            "comp": "full", "tone": "daylight"
        },
        "Пират на палубе": {
            "desc": "Треуголка, сабля, мокрая палуба, канаты, шторм — запах солёного приключения.",
            "p_f": "пиратская капитанша в треуголке, кожаный жилет, белая рубаха, корсет, сабля в руке; "
                   "фон: деревянная палуба корабля, штормовое море, такелаж и паруса, брызги и туман, чайки",
            "p_m": "пиратский капитан в треуголке, кожаный жилет, белая рубаха, сабля в руке; "
                   "фон: палуба корабля, штормовое море, такелаж, брызги и туман, чайки",
            "comp": "full", "tone": "cool"
        },
        "Древняя Греция": {
            "desc": "Белый мрамор, лазурная вода, золочёная отделка — античный миф оживает.",
            "p_f": "богиня в белой тунике (хитон) с золотой отделкой, диадема/венок; "
                   "фон: колоннада, мраморные статуи, кипарисы, лазурный бассейн, южное солнце",
            "p_m": "герой в хитоне с золотой отделкой, лавровый венец; "
                   "фон: колоннада, мраморные статуи, кипарисы, лазурный бассейн, южное солнце",
            "comp": "half", "tone": "warm"
        },

        # Sci-Fi
        "Киберпанк улица": {
            "desc": "Голограммы, пар из люков, мокрый асфальт — «Бегущий по лезвию».",
            "p": "кожаная куртка; фон: неоновые вывески, мокрый асфальт, голограммы",
            "comp": "full", "tone": "neon"
        },
        "Космический скафандр": {
            "desc": "Звёздный ангар и отражения в визоре — hard-sci-fi.",
            "p": "реалистичный EVA-скафандр; фон: звёздное небо, ангар корабля",
            "comp": "full", "tone": "cool"
        },
        "Космопилот на мостике": {
            "desc": "Пульт с индикаторами, свет приборов на лице — предвкушение гиперпрыжка.",
            "p": "лётный комбинезон, шлем под мышкой; фон: мостик звездолёта",
            "comp": "half", "tone": "cool"
        },
    }

    STYLE_CATEGORIES: Dict[str, List[str]] = {
        "Портреты": ["Портрет у окна", "Портрет 85мм", "Бьюти студия", "Кинопортрет", "Фильм-нуар (портрет)"],
        "Современные": ["Стритвэр город", "Вечерний выход", "Бизнес", "Ночной неон"],
        "Профессии": ["Врач у палаты", "Хирург операционная", "Шеф-повар кухня", "Учёный лаборатория", "Боксер на ринге", "Фитнес зал"],
        "Приключения": ["Приключенец (руины)", "Пустынный исследователь", "Горы снег", "Серфер"],
        "Фэнтези/История": ["Эльфийская знать", "Самурай в храме", "Средневековый рыцарь", "Пират на палубе", "Древняя Греция"],
        "Sci-Fi": ["Киберпанк улица", "Космический скафандр", "Космопилот на мостике"],
    }

    # Усилители фактуры/среды — добавляются в промпт
    THEME_BOOST = {
        "Пират на палубе": "rope rigging, wooden deck planks, storm clouds, wet highlights on wood, sea spray, gulls in distance",
        "Древняя Греция": "white marble columns, ionic capitals, olive trees, turquoise water reflections, gold trim accents",
        "Ночной неон":     "rain droplets on lens, steam from manholes, colored reflections on wet asphalt",
        "Фильм-нуар (портрет)": "venetian blinds light pattern, cigarette smoke curling, deep black shadows",
        "Приключенец (руины)": "floating dust motes in sunrays, chipped sandstone blocks, leather straps patina",
        "Горы снег":      "spindrift blown by wind, crampon scratches on ice, distant ridge line",
        "Киберпанк улица":"holographic billboards flicker, cable bundles overhead, neon kanji signs",
        "Вечерний выход": "red carpet stanchions, paparazzi flashes bokeh",
        "Бизнес":         "glass atrium reflections, escalator lines, marble floor sheen",
        "Шеф-повар кухня":"stainless steel counters, copper pans, gas flame flare",
        "Учёный лаборатория":"LED indicator lights, glassware refractions, soft blue fill",
        "Боксер на ринге":"ring ropes texture, chalk dust, sweat droplets catching light",
        "Фитнес зал":     "backlit equipment silhouettes, dust beams, rubber floor texture",
        "Серфер":         "sun glitter on water, backlit spray, footprints on wet sand",
        "Самурай в храме":"paper lantern glow, wooden beams lacquer sheen, falling maple leaves",
        "Средневековый рыцарь":"banners fluttering, straw ground, sunlight on polished steel",
        "Космический скафандр": "hangar catwalks, warning stripes, panel lights reflection",
        "Космопилот на мостике":"HUD glow, instrument reflections, starfield through canopy",
        "Стритвэр город": "graffiti walls, glass reflections, curb puddles",
        "Пустынный исследователь":"heat haze shimmer, wind-blown sand traces",
        "Портрет у окна":"window frame soft shadow, gentle background falloff",
        "Кинопортрет":"film grain subtle, classic key and fill balance",
        "Бьюти студия":"catchlight softbox, gradient backdrop subtle",
    }

    # Что ОБЯЗАТЕЛЬНО должно быть в кадре (ключевые объекты/среда)
    MUST_INCLUDE = {
        "Серфер": "with a surfboard clearly visible, breaking ocean wave behind, shoreline and wet sand, water spray",
        "Пират на палубе": "with a drawn cutlass/saber, visible ship rigging and sails, wooden deck, stormy sea",
        "Древняя Греция": "with marble columns and statues, turquoise pool reflections, golden trim details",
        "Киберпанк улица": "with neon signs, wet asphalt reflections, steam from vents, holographic billboards",
        "Горы снег": "with a visible ice axe, snow ridge line and distant peaks, wind-blown spindrift",
        "Приключенец (руины)": "with ancient stone blocks, sunbeams with dust, leather gear straps",
        "Пустынный исследователь": "with sand dunes and canyon walls, scarf and cargo gear, wind trails on sand",
        "Самурай в храме": "with a katana unsheathed, paper lanterns and temple courtyard",
        "Средневековый рыцарь": "with full plate armor and cloak, tournament yard with banners",
        "Вечерний выход": "with red carpet and flash bulbs, velvet ropes",
        "Бизнес": "with glass atrium or office lobby, escalator or elevator backdrop, marble floor",
        "Шеф-повар кухня": "with visible stove flame, stainless steel counters, pans",
        "Учёный лаборатория": "with glassware, racks and instruments in background",
        "Боксер на ринге": "with ring ropes and corner pads, gloves raised",
        "Фитнес зал": "with gym equipment silhouettes and backlight",
        "Космический скафандр": "with hangar interior or star backdrop, helmet visor reflections",
        "Космопилот на мостике": "with starship bridge consoles and indicators",
        "Стритвэр город": "with street scene, graffiti or glass reflections, pavement",
        "Портрет у окна": "with window edge shadow and interior hints",
        "Портрет 85мм": "with shallow depth background and outdoors/ambient light hints",
        "Бьюти студия": "with subtle gradient studio hint and catchlight",
        "Кинопортрет": "with cinematic key and fill balance hints",
        "Фильм-нуар (портрет)": "with venetian blind light pattern and smoke",
    }

    # Доп. негативы, чтобы не уводило в студию/пустой фон
    SCENE_NEGATIVE = {
        "Серфер": "indoor, studio, pool, lake, still water, plain background, portrait crop",
        "Пират на палубе": "studio, cosplay backdrop, plain background, portrait crop, indoor",
        "Древняя Греция": "modern interior, studio backdrop, portrait crop, indoor",
        "Киберпанк улица": "daylight city only, plain background, studio portrait, portrait crop",
        "Горы снег": "indoor, studio, forest path, beach, portrait crop",
        "Приключенец (руины)": "modern city, studio backdrop, portrait crop",
        "Пустынный исследователь": "beach resort, studio, portrait crop",
        "Самурай в храме": "modern dojo studio, plain backdrop, portrait crop",
        "Средневековый рыцарь": "modern cosplay studio, portrait crop, plain background",
        "Вечерний выход": "empty studio, office interior, daytime street",
        "Бизнес": "photo studio, home interior couch, portrait crop",
        "Шеф-повар кухня": "home kitchen, empty plain background, portrait crop",
        "Учёный лаборатория": "empty white wall, classroom, portrait crop",
        "Боксер на ринге": "fitness studio mirror selfie, portrait crop",
        "Фитнес зал": "empty white background, studio portrait, portrait crop",
        "Космический скафандр": "earth sidewalk, cosplay studio, portrait crop",
        "Космопилот на мостике": "plain background, empty corridor, portrait crop",
        "Стритвэр город": "plain studio backdrop, portrait crop",
        # портретам — мягкие исключения
        "Портрет у окна": "plain white studio, beauty filter",
        "Портрет 85мм": "plain studio backdrop, beauty filter",
        "Бьюти студия": "overly smoothed skin filter",
        "Кинопортрет": "overlit flat light",
        "Фильм-нуар (портрет)": "flat low-contrast light",
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

    # ---------- Replicate helpers & persistence ----------
    def resolve_model_version(slug: str) -> str:
        if ":" in slug: return slug
        model = replicate.models.get(slug); versions = list(model.versions.list())
        if not versions: raise RuntimeError(f"Нет версий модели {slug}")
        return f"{slug}:{versions[0].id}"

    def latest_version_slug(model_slug:str) -> str:
        try:
            model = replicate.models.get(model_slug)
            versions = list(model.versions.list())
            if versions:
                return f"{model_slug}:{versions[0].id}"
        except Exception:
            pass
        return model_slug

    def dest_model_for_uid(uid:int) -> str:
        if not DEST_OWNER:
            raise RuntimeError("REPLICATE_DEST_OWNER не задан.")
        return f"{DEST_OWNER}/{DEST_PREFIX}-{uid}"

    def ensure_model_exists(model_slug:str):
        try:
            replicate.models.get(model_slug)
        except Exception:
            owner, name = model_slug.split("/",1)
            raise RuntimeError(
                f"Целевая модель '{model_slug}' не найдена. "
                f"Создай её на https://replicate.com/create (owner={owner}, name='{name}')."
            )

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

    def start_lora_training(uid:int) -> str:
        dest_model = dest_model_for_uid(uid)
        ensure_model_exists(dest_model)
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
        prof = load_profile(uid)
        tid = prof.get("training_id")
        if not tid:
            slug_latest = latest_version_slug(dest_model_for_uid(uid))
            status = "succeeded" if ":" in slug_latest else "not_started"
            if status == "succeeded":
                prof.setdefault("finetuned_model", slug_latest.split(":",1)[0])
                prof["finetuned_version"] = slug_latest.split(":",1)[1]
                prof["status"] = status
                save_profile(uid, prof)
            return (status, slug_latest if status=="succeeded" else None)

        client = Client(api_token=os.environ["REPLICATE_API_TOKEN"])
        tr = client.trainings.get(tid)
        status = getattr(tr, "status", None) or (tr.get("status") if isinstance(tr, dict) else None) or "unknown"
        if status != "succeeded":
            prof["status"] = status; save_profile(uid, prof); return (status, None)

        destination = getattr(tr, "destination", None) or prof.get("finetuned_model") or dest_model_for_uid(uid)
        slug_with_version = latest_version_slug(destination)

        prof["status"] = status
        prof["finetuned_model"] = destination
        if ":" in slug_with_version:
            prof["finetuned_version"] = slug_with_version.split(":",1)[1]
        save_profile(uid, prof)
        return (status, slug_with_version)

    def _pinned_slug(prof: Dict[str, Any], uid:int) -> str:
        base = prof.get("finetuned_model") or dest_model_for_uid(uid)
        ver  = prof.get("finetuned_version")
        if ver:
            return f"{base}:{ver}"
        return latest_version_slug(base)

    # ---------- Генерация ----------
    def _prompt_for_gender(meta: Style, gender: str) -> str:
        if gender == "female" and meta.get("p_f"): return meta["p_f"]
        if gender == "male" and meta.get("p_m"): return meta["p_m"]
        return meta.get("p", "")

    def generate_from_finetune_hardened(model_slug:str, prompt:str, negative:str, steps:int, guidance:float, seed:int, w:int, h:int) -> str:
        mv = resolve_model_version(model_slug)
        out = replicate.run(mv, input={
            "prompt": prompt + AESTHETIC_SUFFIX,
            "negative_prompt": negative,
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
            "в **узнаваемых кино-сценах** — от пирата до космопилота.\n\n"
            "1) «📸 Набор фото» — пришли до 10 снимков (без фильтров).\n"
            "2) «🧪 Обучение» — запущу тренировку LoRA.\n"
            "3) «🧭 Выбрать стиль» — получи 4 варианта кадра.\n\n"
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
            "Далее — нажми «🧪 Обучение» или /trainid.",
            reply_markup=main_menu_kb()
        )

    async def id_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id; prof = load_profile(uid)
        await update.effective_message.reply_text(
            f"Фото: {len(list_ref_images(uid))}\n"
            f"Статус обучения: {prof.get('status') or '—'}\n"
            f"Модель: {prof.get('finetuned_model') or dest_model_for_uid(uid)}\n"
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
        # если профиля нет — всё равно генерируем по последней версии персональной модели
        if not prof.get("status") == "succeeded":
            # пробуем восстановиться молча
            slug_latest = latest_version_slug(dest_model_for_uid(uid))
            if ":" in slug_latest:
                prof["finetuned_model"] = slug_latest.split(":",1)[0]
                prof["finetuned_version"] = slug_latest.split(":",1)[1]
                prof["status"] = "succeeded"
                save_profile(uid, prof)
            else:
                await update.effective_message.reply_text("Модель ещё не готова. Сначала /trainid и дождись /trainstatus = succeeded.")
                return

        meta = STYLE_PRESETS[preset]
        gender = (prof.get("gender") or "female").lower()
        base_prompt = _prompt_for_gender(meta, gender)
        comp_text, (w,h) = _comp_text_and_size(meta.get("comp","half"))
        tone_text = _tone_text(meta.get("tone","daylight"))
        theme_boost = THEME_BOOST.get(preset, "")
        must = MUST_INCLUDE.get(preset, "")

        # Портрет vs сцена: разный guidance и анти-headshot
        is_portrait = meta.get("comp","half") == "closeup"
        steps = max(42, GEN_STEPS - (0 if is_portrait else 2))
        guidance = (GEN_GUIDANCE if is_portrait else max(3.8, GEN_GUIDANCE + 0.4))

        scene_negative = SCENE_NEGATIVE.get(preset, "")
        negative_full = NEGATIVE_PROMPT + (", " + NEG_HEADSHOT if not is_portrait else "") + (", " + scene_negative if scene_negative else "")

        wide_tags = "wide shot, long shot, full body, environment clearly visible" if not is_portrait else "portrait framing"
        prompt_core = (
            f"{base_prompt}, {comp_text}, {tone_text}, {wide_tags}, "
            "exact facial identity, identity preserved, identity preserved, "
            "cinematic key light and rim light, soft bounce fill, film grain subtle, "
            "skin tone faithful, "
            f"{_beauty_guardrail()}, {theme_boost}, {must}"
        )

        model_slug = _pinned_slug(prof, uid)
        await update.effective_message.chat.send_action(ChatAction.UPLOAD_PHOTO)
        desc = meta.get("desc", preset)
        await update.effective_message.reply_text(f"🎬 {preset}\n{desc}\n\nГенерирую ({gender}, {w}×{h}) …")

        try:
            seeds = [int(time.time()) & 0xFFFFFFFF, random.randrange(2**32), random.randrange(2**32), random.randrange(2**32)]
            urls = []
            for s in seeds:
                url = await asyncio.to_thread(
                    generate_from_finetune_hardened,
                    model_slug=model_slug,
                    prompt=prompt_core,
                    negative=negative_full,
                    steps=steps,
                    guidance=guidance,
                    seed=s, w=w, h=h
                )
                urls.append(url)

            for i, u in enumerate(urls, 1):
                await update.effective_message.reply_photo(photo=u, caption=f"{preset} • вариант {i}")

            await update.effective_message.reply_text("Хочешь апскейл/вариации — скажи «этот нрав» и номер.")
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

        logger.info("Bot up. Trainer=%s DEST=%s-* GEN=%dx%d steps=%s guidance=%s",
                    LORA_TRAINER_SLUG, f"{DEST_OWNER}/{DEST_PREFIX}", GEN_WIDTH, GEN_HEIGHT, GEN_STEPS, GEN_GUIDANCE)
        app.run_polling(drop_pending_updates=True)

    if __name__ == "__main__":
        main()








