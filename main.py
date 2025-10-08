# === Telegram LoRA Bot (Flux LoRA trainer + Redis persist + S3 storage)
# + InstantID LOCKFACE (2-step)
# + MULTI-AVATARS
# + NATURAL/Pretty per-user (cheese-like)
# + CONSISTENT FACE SCALE + anti-wide-face
# Требования: python-telegram-bot==20.7, replicate==0.31.0, pillow==10.4.0, redis==5.0.1, boto3==1.34.0+

from typing import Any, Dict, List, Optional, Tuple
Style = Dict[str, Any]

from styles import (  # твой styles.py
    STYLE_PRESETS, STYLE_CATEGORIES, THEME_BOOST,
    SCENE_GUIDANCE, RISKY_PRESETS
)

import os, re, io, json, time, asyncio, logging, shutil, random, contextlib, base64
from pathlib import Path
from zipfile import ZipFile

import requests
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

DEST_OWNER = os.getenv("REPLICATE_DEST_OWNER", "").strip()
DEST_MODEL = os.getenv("REPLICATE_DEST_MODEL", "yourtwin-lora").strip()

# LoRA trainer
LORA_TRAINER_SLUG = os.getenv("LORA_TRAINER_SLUG", "replicate/flux-lora-trainer").strip()
LORA_INPUT_KEY = os.getenv("LORA_INPUT_KEY", "input_images").strip()

# Пол (опц.)
GENDER_MODEL_SLUG = os.getenv("GENDER_MODEL_SLUG", "nateraw/vit-age-gender").strip()

# LOCKFACE (InstantID / FaceID adapter)
INSTANTID_SLUG = os.getenv("INSTANTID_SLUG", "").strip()
INSTANTID_STRENGTH = float(os.getenv("INSTANTID_STRENGTH", "0.88"))
INSTANTID_FACE_WEIGHT = float(os.getenv("INSTANTID_FACE_WEIGHT", "0.92"))

# --- Параметры обучения/генерации ---
LORA_MAX_STEPS = int(os.getenv("LORA_MAX_STEPS", "1400"))
LORA_LR = float(os.getenv("LORA_LR", "0.0006"))
LORA_USE_FACE_DET = os.getenv("LORA_USE_FACE_DET", "true").lower() in ("1","true","yes","y")
LORA_RESOLUTION = int(os.getenv("LORA_RESOLUTION", "1024"))

GEN_STEPS    = int(os.getenv("GEN_STEPS", "52"))
GEN_GUIDANCE = float(os.getenv("GEN_GUIDANCE", "4.4"))
GEN_WIDTH    = int(os.getenv("GEN_WIDTH", "1024"))
GEN_HEIGHT   = int(os.getenv("GEN_HEIGHT", "1344"))
MAX_STEPS    = int(os.getenv("MAX_STEPS", "56"))

# --- CONSISTENT FACE SCALE ---
CONSISTENT_SCALE = os.getenv("CONSISTENT_SCALE", "1").lower() in ("1","true","yes","y")
HEAD_HEIGHT_FRAC = float(os.getenv("HEAD_HEIGHT_FRAC", "0.36"))
HEAD_WIDTH_FRAC  = float(os.getenv("HEAD_WIDTH_FRAC", "0.28"))

# ---- Anti-drift / anti-wide-face ----
NEGATIVE_PROMPT_BASE = (
    "cartoon, anime, cgi, 3d, stylized, plastic skin, overprocessed, airbrushed, beauty-filter, "
    "lowres, blur, textureless skin, porcelain skin, waxy, gaussian blur, smoothing filter, "
    "text, watermark, logo, bad anatomy, identity drift, different person, face swap, face morph, "
    "ethnicity change, age change, hairline modification, beard reshaping, lip reshape, mouth corner lift, "
    "puffy face, swollen face, chubby cheeks, widened jaw, wide face, horizontally stretched face, "
    "tiny head, giant head, variable head scale, aspect distortion, fisheye, lens distortion, warping, "
    "plain selfie, tourist photo, plain studio backdrop, denoise artifacts, waxy highlight roll-off, "
    "excessive frequency separation, face slimming or widening, retouched pores"
)
AESTHETIC_SUFFIX = (
    ", photorealistic, visible fine skin texture, natural color, soft filmic contrast, "
    "micro-sharpen on eyes and lips only, anatomically plausible facial landmarks"
)

# ---------- NATURAL/PRETTY ----------
NATURAL_POS = "unretouched skin, realistic fine pores, gentle film grain, accurate skin undertones"
NATURAL_NEG = "skin smoothing, airbrush, beauty-filter, porcelain skin, waxy skin, HDR glam"
PRETTY_POS  = "subtle beauty retouch, even skin tone, micro-dodge-and-burn, cinematic color grading"
PRETTY_NEG  = "over-smoothing, plastic look, face slimming, heavy makeup, excessive sharpening"
PRETTY_COMP_HINT = "camera slightly above eye level, flattering portrait angle"

def _beauty_guardrail() -> str:
    return ("exact facial identity, identity preserved, balanced facial proportions, "
            "keep original zygomatic width and jaw width, do not widen face, catchlights in eyes")
def _beauty_lock() -> str:
    return "preserve bone structure; no face slimming/widening; no jaw/cheek/nose/lip reshape"
def _face_lock() -> str:
    return "keep interocular distance, eyelid shape and lip fullness consistent"
def _anti_distort() -> str:
    return "no fisheye, no lens distortion, natural perspective"

def _gender_lock(gender:str) -> Tuple[str, str]:
    if gender == "male":
        return "male man, masculine facial features, light stubble allowed", "female, heavy makeup"
    return "female woman, feminine facial features", "male, beard, mustache, adam's apple"

# ---------- logging ----------
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("bot")

# ---------- storage ----------
DATA_DIR = Path("profiles"); DATA_DIR.mkdir(exist_ok=True)

DEFAULT_AVATAR = {
    "images": [],
    "s3_refs": [],              # NEW
    "training_id": None,
    "finetuned_model": None,
    "finetuned_version": None,
    "status": None,
    "lockface": True
}
DEFAULT_PROFILE = {
    "gender": None,
    "natural": True,
    "pretty": False,
    "current_avatar": "default",
    "avatars": {"default": DEFAULT_AVATAR.copy()}
}

def user_dir(uid:int) -> Path:
    p = DATA_DIR / str(uid); p.mkdir(parents=True, exist_ok=True); return p
def avatars_root(uid:int) -> Path:
    p = user_dir(uid) / "avatars"; p.mkdir(parents=True, exist_ok=True); return p
def avatar_dir(uid:int, avatar:str) -> Path:
    p = avatars_root(uid) / avatar; p.mkdir(parents=True, exist_ok=True); return p
def list_ref_images(uid:int, avatar:str) -> List[Path]:
    return sorted(avatar_dir(uid, avatar).glob("ref_*.jpg"))
def profile_path(uid:int) -> Path:
    return user_dir(uid) / "profile.json"

def _migrate_single_to_multi(uid:int, prof:Dict[str,Any]) -> Dict[str,Any]:
    if "avatars" in prof: return prof
    migrated = DEFAULT_PROFILE.copy()
    migrated["gender"] = prof.get("gender")
    default = DEFAULT_AVATAR.copy()
    for k in ("training_id","finetuned_model","finetuned_version","status","lockface"):
        default[k] = prof.get(k) if k in prof else default.get(k)
    imgs = prof.get("images", [])
    if imgs:
        for name in imgs:
            src = user_dir(uid) / name
            if src.exists():
                dst = avatar_dir(uid, "default") / name
                if not dst.exists():
                    with contextlib.suppress(Exception):
                        shutil.move(str(src), str(dst))
        default["images"] = [p.name for p in list_ref_images(uid, "default")]
    migrated["avatars"]["default"] = default
    return migrated

# ---------- Redis ----------
_redis = None
REDIS_URL = os.getenv("REDIS_URL", "").strip()
if REDIS_URL:
    try:
        import redis
        _redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        _redis.ping()
        logger.info("Storage: Redis OK (%s)", REDIS_URL.rsplit("@",1)[-1])
    except Exception as e:
        logger.warning("Storage: Redis init failed (%s). FS fallback. Error: %s", REDIS_URL, e)
        _redis = None
else:
    logger.info("Storage: FS (no REDIS_URL)")

# ---------- S3 ----------
import boto3
_S3 = None
def _s3():
    global _S3
    if _S3: return _S3
    ep = os.getenv("S3_ENDPOINT", "") or None
    reg = os.getenv("S3_REGION", "us-east-1")
    _S3 = boto3.client(
        "s3",
        endpoint_url=ep,
        region_name=reg,
        aws_access_key_id=os.getenv("S3_KEY"),
        aws_secret_access_key=os.getenv("S3_SECRET"),
    )
    return _S3

S3_BUCKET = os.getenv("S3_BUCKET", "").strip()
S3_PUBLIC_BASE = os.getenv("S3_PUBLIC_BASE", "").rstrip("/")

def s3_put_bytes(key:str, data:bytes, content_type:str="image/jpeg") -> str:
    if not S3_BUCKET: raise RuntimeError("S3_BUCKET не задан")
    _s3().put_object(Bucket=S3_BUCKET, Key=key, Body=data, ContentType=content_type, ACL="private")
    return f"{S3_PUBLIC_BASE}/{key}" if S3_PUBLIC_BASE else ""

def s3_get_bytes(key:str) -> bytes:
    r = _s3().get_object(Bucket=S3_BUCKET, Key=key)
    return r["Body"].read()

def s3_presigned_url(key:str, expires=86400) -> str:
    return _s3().generate_presigned_url("get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=expires)

def _avatar_s3_key(uid:int, avatar:str, filename:str) -> str:
    return f"refs/{uid}/{avatar}/{filename}"

def _trainzip_s3_key(uid:int, avatar:str) -> str:
    return f"trainzips/{uid}/{avatar}/train_{int(time.time())}.zip"

# ---------- Лицо в Redis (кэш) ----------
def _face_cache_key(uid:int, avatar:str) -> str:
    return f"face:{uid}:{avatar}"

def save_face_cache(uid:int, avatar:str, jpeg_bytes:bytes):
    if _redis:
        _redis.set(_face_cache_key(uid, avatar), base64.b64encode(jpeg_bytes).decode("ascii"))

def load_face_cache(uid:int, avatar:str) -> Optional[bytes]:
    if not _redis: return None
    b64 = _redis.get(_face_cache_key(uid, avatar))
    return base64.b64decode(b64) if b64 else None

# ---------- профили ----------
def load_profile(uid:int) -> Dict[str, Any]:
    if _redis:
        try:
            raw = _redis.get(f"profile:{uid}")
            if raw:
                prof = {**DEFAULT_PROFILE, **json.loads(raw)}
                prof = _migrate_single_to_multi(uid, prof)
                return prof
        except Exception as e:
            logger.warning("Redis load_profile failed: %s", e)
    p = profile_path(uid)
    if p.exists():
        with contextlib.suppress(Exception):
            prof = {**DEFAULT_PROFILE, **json.loads(p.read_text())}
            prof = _migrate_single_to_multi(uid, prof)
            return prof
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
            # чистим кэш лица
            _redis.delete(_face_cache_key(uid, "default"))
    p = user_dir(uid)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

# ---------- Рефы ----------
def save_ref_downscaled(path: Path, raw: bytes, max_side=1152, quality=95, uid:int=None, avatar:str=None):
    """Сохраняем JPEG локально, кэшируем лицо в Redis и грузим копию в S3."""
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = im.size
    side = int(min(w, h) * 0.85)
    cx, cy = w // 2, h // 2
    left = max(0, cx - side // 2); top = max(0, cy - side // 2)
    im = im.crop((left, top, left + side, top + side))
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True, progressive=True)
    jpeg = buf.getvalue()
    path.write_bytes(jpeg)
    if uid is not None and avatar is not None:
        save_face_cache(uid, avatar, jpeg)
        # S3
        try:
            key = _avatar_s3_key(uid, avatar, path.name)
            s3_put_bytes(key, jpeg, "image/jpeg")
            prof = load_profile(uid)
            av = prof["avatars"].get(avatar) or DEFAULT_AVATAR.copy()
            s3_list = set(av.get("s3_refs", []))
            s3_list.add(key)
            av["s3_refs"] = list(s3_list)
            prof["avatars"][avatar] = av
            save_profile(uid, prof)
        except Exception as e:
            logger.warning("S3 upload failed: %s", e)

def ensure_local_refs_from_s3(uid:int, avatar:str, need:int=10):
    """Если локально пусто — пытаемся скачать из S3 хотя бы need файлов."""
    refs = list_ref_images(uid, avatar)
    if len(refs) >= need: return
    prof = load_profile(uid); av = prof["avatars"].get(avatar) or {}
    ok = 0
    for key in av.get("s3_refs", []):
        fname = key.split("/")[-1]
        dst = avatar_dir(uid, avatar) / fname
        if not dst.exists():
            try:
                data = s3_get_bytes(key)
                dst.write_bytes(data)
                ok += 1
                if ok >= need: break
            except Exception:
                continue

def load_face_from_s3(uid:int, avatar:str) -> Optional[bytes]:
    prof = load_profile(uid); av = prof["avatars"].get(avatar) or {}
    for key in av.get("s3_refs", []):
        try:
            return s3_get_bytes(key)
        except Exception:
            continue
    return None

# ---------- Аватарные утилиты ----------
def get_current_avatar_name(prof:Dict[str,Any]) -> str:
    name = prof.get("current_avatar") or "default"
    if name not in prof["avatars"]:
        name = "default"; prof["current_avatar"] = name
    return name

def get_avatar(prof:Dict[str,Any], name:Optional[str]=None) -> Dict[str,Any]:
    if not name: name = get_current_avatar_name(prof)
    if name not in prof["avatars"]:
        prof["avatars"][name] = DEFAULT_AVATAR.copy()
    return prof["avatars"][name]

def set_current_avatar(uid:int, name:str):
    prof = load_profile(uid)
    if name not in prof["avatars"]:
        prof["avatars"][name] = DEFAULT_AVATAR.copy()
    prof["current_avatar"] = name
    save_profile(uid, prof)

def ensure_avatar(uid:int, name:str):
    prof = load_profile(uid)
    if name not in prof["avatars"]:
        prof["avatars"][name] = DEFAULT_AVATAR.copy()
    save_profile(uid, prof)

def del_avatar(uid:int, name:str):
    prof = load_profile(uid)
    if name == "default": raise RuntimeError("Нельзя удалить аватар 'default'.")
    if name not in prof["avatars"]: return
    adir = avatar_dir(uid, name)
    with contextlib.suppress(Exception):
        if adir.exists(): shutil.rmtree(adir)
    prof["avatars"].pop(name, None)
    if prof["current_avatar"] == name: prof["current_avatar"] = "default"
    save_profile(uid, prof)

# ---------- Replicate helpers ----------
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

def _check_slug(slug: str, label: str):
    try:
        mv = resolve_model_version(slug)
        logger.info("%s OK: %s", label, mv)
    except Exception as e:
        logger.warning("%s BAD ('%s'): %s", label, slug, e)

# ---------- авто-пол ----------
def _infer_gender_from_image(path: Path) -> Optional[str]:
    try:
        client = Client(api_token=os.environ["REPLICATE_API_TOKEN"])
        version_slug = resolve_model_version(GENDER_MODEL_SLUG)
        with open(path, "rb") as img_b:
            pred = client.predictions.create(version=version_slug, input={"image": img_b})
            pred.wait()
            out = pred.output
            g = (out.get("gender") if isinstance(out, dict) else str(out)).lower()
            if "female" in g or "woman" in g: return "female"
            if "male" in g or "man" in g: return "male"
    except Exception as e:
        logger.warning("Gender inference error: %s", e)
    return None

def auto_detect_gender(uid:int) -> str:
    prof = load_profile(uid)
    av_name = get_current_avatar_name(prof)
    refs = list_ref_images(uid, av_name)
    if not refs: return "female"
    g = _infer_gender_from_image(refs[0])
    return g or "female"

DEFAULT_FEMALE_CAPTION = "a high quality photo of the same woman, natural brows, neutral expression, true-to-life skin texture"
DEFAULT_MALE_CAPTION   = "a high quality photo of the same man, natural brows, neutral expression, true-to-life skin texture"

def _caption_for_gender(g: str) -> str:
    env_override = os.getenv("LORA_CAPTION_PREFIX", "").strip()
    if env_override:
        return env_override
    return DEFAULT_MALE_CAPTION if (g or "").lower() == "male" else DEFAULT_FEMALE_CAPTION

# ---------- Генерация промпта ----------
def _face_scale_hint() -> str:
    if not CONSISTENT_SCALE: return ""
    hh = int(HEAD_HEIGHT_FRAC * 100); hw = int(HEAD_WIDTH_FRAC * 100)
    return (f"keep constant head scale, head height ~{hh}% of frame, head width ~{hw}% of frame, "
            "subject centered, do not zoom")

def _safe_portrait_size(w:int, h:int) -> Tuple[int,int]:
    ar = w / max(1, h)
    if ar >= 0.75: return int(h*0.66), h
    return w, h

def _comp_text_and_size(comp: str) -> Tuple[str, Tuple[int,int]]:
    scale_txt = _face_scale_hint()
    if comp == "closeup":
        w, h = _safe_portrait_size(896, 1152)
        return (f"portrait from chest up, 85mm look, eye-level camera, no distortion, {scale_txt}", (w,h))
    if comp == "half":
        w, h = _safe_portrait_size(GEN_WIDTH, max(GEN_HEIGHT, 1344))
        return (f"half body, 85mm look, chest-level camera, no distortion, {scale_txt}", (w,h))
    w, h = _safe_portrait_size(GEN_WIDTH, 1408)
    return (f"full body, 85mm look, mid-torso camera, head size natural, no distortion, {scale_txt}", (w,h))

def _tone_text(tone: str) -> str:
    return {
        "daylight": "soft daylight at 45°, gentle fill, cinematic contrast",
        "beauty":   "large softbox key, subtle hair light, clean WB",
        "warm":     "golden hour warmth, soft rim",
        "cool":     "cool cinematic key with neutral fill",
        "noir":     "high contrast noir with rim",
        "neon":     "neon signs, cinematic backlight",
        "candle":   "warm candlelight, volumetric glow",
    }.get(tone, "balanced soft lighting")

def _style_lock(role:str, outfit:str, props:str, background:str, comp_hint:str) -> str:
    bits = [role or "", f"wearing {outfit}" if outfit else "", f"with {props}" if props else "",
            f"background: {background}" if background else "", comp_hint, "unmistakable scene identity"]
    return ", ".join([b for b in bits if b])

def _inject_beauty(core_prompt: str, comp_text: str, natural: bool, pretty: bool) -> str:
    parts = [core_prompt]
    if natural: parts.append(NATURAL_POS)
    if pretty:
        parts.append(PRETTY_POS)
        if PRETTY_COMP_HINT and ("eye" in comp_text or "chest" in comp_text):
            parts.append(PRETTY_COMP_HINT)
    return ", ".join(parts)

def build_prompt(meta: Style, gender: str, comp_text:str, tone_text:str,
                 theme_boost:str, natural:bool, pretty:bool) -> Tuple[str, str]:
    role = meta.get("role_f") if (gender=="female" and meta.get("role_f")) else meta.get("role","")
    if not role and meta.get("role_m") and gender=="male": role = meta.get("role_m","")
    outfit = meta.get("outfit_f") if (gender=="female" and meta.get("outfit_f")) else meta.get("outfit","")
    props = meta.get("props",""); bg = meta.get("bg","")

    gpos, gneg = _gender_lock(gender)
    anti = _anti_distort()

    common_bits = [
        tone_text, gpos,
        "same person as training photos, no ethnicity or age change, exact facial identity +++",
        "photorealistic, realistic body proportions, fine skin texture, filmic look",
        "do not widen face; keep cheekbone and jaw width; preserve lips",
        "85mm portrait look", _face_scale_hint(), anti, _beauty_guardrail(), _beauty_lock(), _face_lock(), theme_boost
    ]

    if role or outfit or props or bg:
        core = ", ".join([_style_lock(role, outfit, props, bg, comp_text)] + common_bits)
        core += ", the costume and background must clearly communicate the role; avoid plain portrait"
    else:
        base_prompt = meta.get("p", "")
        core = ", ".join([f"{base_prompt}, {comp_text}"] + common_bits)

    core = _inject_beauty(core, comp_text, natural, pretty)

    neg = gneg
    if natural: neg = (neg + ", " + NATURAL_NEG) if neg else NATURAL_NEG
    if pretty:  neg = (neg + ", " + PRETTY_NEG)  if neg else PRETTY_NEG
    return core, neg

# ---------- Инференс ----------
def generate_from_finetune(model_slug:str, prompt:str, steps:int, guidance:float, seed:int, w:int, h:int, negative_prompt:str) -> str:
    mv = resolve_model_version(model_slug)
    out = replicate.run(mv, input={
        "prompt": prompt + AESTHETIC_SUFFIX,
        "negative_prompt": negative_prompt,
        "width": w, "height": h,
        "num_inference_steps": min(MAX_STEPS, steps),
        "guidance_scale": guidance,
        "seed": seed,
    })
    url = extract_any_url(out)
    if not url: raise RuntimeError("Empty output")
    return url

    import io  # убедись, что импорт есть наверху файла

def generate_with_instantid(
        face_bytes: bytes,
        prompt: str,
        steps: int,
        guidance: float,
        seed: int,
        w: int,
        h: int,
        negative_prompt: str,
        natural: bool = True,
        pretty: bool = False,
        content_image_bytes: Optional[bytes] = None,
    ) -> str:
        mv = resolve_model_version(INSTANTID_SLUG)

        # --- настройки силы замка лица (ВОТ ЭТИ СТРОКИ ДОЛЖНЫ БЫТЬ!) ---
        base = INSTANTID_STRENGTH
        base_face = INSTANTID_FACE_WEIGHT
        if natural and not pretty:
            strength, face_w = base * 0.9, base_face * 0.9
        elif pretty:
            strength, face_w = min(0.98, base * 1.05), min(0.98, base_face * 1.05)
        else:
            strength, face_w = base, base_face
        # ----------------------------------------------------------------

        # Replicate ждёт file-like объекты, а не «сырые» bytes
        inputs: Dict[str, Any] = {
            "prompt": prompt + AESTHETIC_SUFFIX,
            "negative_prompt": negative_prompt,
            "width": w,
            "height": h,
            "num_inference_steps": min(MAX_STEPS, steps),
            "guidance_scale": guidance,
            "seed": seed,
            "id_strength": strength,
            "image_identity": strength,
            "face_strength": face_w,
            "adapter_strength": strength,
            "face_image": io.BytesIO(face_bytes),
        }
        if content_image_bytes is not None:
            inputs["image"] = io.BytesIO(content_image_bytes)

        out = replicate.run(mv, input=inputs)
        url = extract_any_url(out)
        if not url:
            raise RuntimeError("Empty output (InstantID)")
        return url

# ---------- UI ----------
def main_menu_kb() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("🧭 Выбрать стиль", callback_data="nav:styles")],
        [InlineKeyboardButton("📸 Набор фото", callback_data="nav:enroll"),
         InlineKeyboardButton("🧪 Обучение", callback_data="nav:train")],
        [InlineKeyboardButton("ℹ️ Мой статус", callback_data="nav:status"),
         InlineKeyboardButton("⚙️ Пол", callback_data="nav:gender")],
        [InlineKeyboardButton("🔒 LOCKFACE", callback_data="nav:lockface")],
        [InlineKeyboardButton("✨ Natural/Pretty", callback_data="nav:beauty")],
        [InlineKeyboardButton("🤖 Аватары", callback_data="nav:avatars")]
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
    rows = [[InlineKeyboardButton(name, callback_data=f"style:{name}")] for name in names]
    rows.append([InlineKeyboardButton("⬅️ Категории", callback_data="nav:styles")])
    return InlineKeyboardMarkup(rows)

def avatars_kb(uid:int) -> InlineKeyboardMarkup:
    prof = load_profile(uid); cur = get_current_avatar_name(prof)
    names = sorted(prof["avatars"].keys())
    rows = []
    for n in names:
        label = f"{'✅ ' if n==cur else ''}{n}"
        rows.append([InlineKeyboardButton(label, callback_data=f"avatar:set:{n}")])
    rows.append([InlineKeyboardButton("➕ Новый", callback_data="avatar:new"),
                 InlineKeyboardButton("🗑 Удалить", callback_data="avatar:del")])
    rows.append([InlineKeyboardButton("⬅️ Меню", callback_data="nav:menu")])
    return InlineKeyboardMarkup(rows)

# ----- Callback для «Аватары»
async def avatar_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    uid = update.effective_user.id
    parts = q.data.split(":")
    if len(parts) < 2: return
    action = parts[1]
    if action == "set":
        if len(parts) < 3:
            await q.message.reply_text("Не указан аватар. Используй /avatarlist."); return
        name = parts[2]; set_current_avatar(uid, name)
        await q.message.reply_text(f"Активный аватар: {name}", reply_markup=avatars_kb(uid))
    elif action == "new":
        await q.message.reply_text("Создай новый: /avatarnew <имя> (пример: /avatarnew travel)")
    elif action == "del":
        await q.message.reply_text("Удаление: /avatardel <имя> --force")
    else:
        await q.message.reply_text("Неизвестное действие.")

# ---------- Handlers ----------
ENROLL_FLAG: Dict[Tuple[int,str],bool] = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Загружай до 10 фото → обучим твой аватар и будем генерировать твои сцены 😎",
        reply_markup=main_menu_kb()
    )

async def nav_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    key = q.data.split(":",1)[1]
    if key == "styles": await q.message.reply_text("Выбери категорию:", reply_markup=categories_kb())
    elif key == "menu": await q.message.reply_text("Главное меню:", reply_markup=main_menu_kb())
    elif key == "enroll": await id_enroll(update, context)
    elif key == "train": await trainid_cmd(update, context)
    elif key == "status": await id_status(update, context)
    elif key == "gender": await gender_cmd(update, context)
    elif key == "lockface": await lockface_cmd(update, context)
    elif key == "avatars": await q.message.reply_text("Аватары:", reply_markup=avatars_kb(update.effective_user.id))
    elif key == "beauty":
        uid = update.effective_user.id
        prof = load_profile(uid)
        prof["natural"] = not prof.get("natural", True) if not prof.get("pretty", False) else prof["natural"]
        save_profile(uid, prof)
        await q.message.reply_text(f"Natural: {'ON' if prof['natural'] else 'OFF'} • Pretty: {'ON' if prof.get('pretty') else 'OFF'}")

async def styles_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Выбери категорию:", reply_markup=categories_kb())

async def cb_category(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    cat = q.data.split(":",1)[1]
    await q.message.reply_text(f"Стиль — {cat}. Выбери сцену:", reply_markup=styles_kb_for_category(cat))

async def id_enroll(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    prof = load_profile(uid); av_name = get_current_avatar_name(prof)
    ENROLL_FLAG[(uid, av_name)] = True
    await update.effective_message.reply_text(
        f"Набор включён для «{av_name}». Пришли подряд до 10 фото (фронтально, без фильтров). "
        "Когда закончишь — нажми /iddone."
    )

async def id_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    prof = load_profile(uid); av_name = get_current_avatar_name(prof)
    ENROLL_FLAG[(uid, av_name)] = False
    av = get_avatar(prof, av_name)
    av["images"] = [p.name for p in list_ref_images(uid, av_name)]
    if not prof.get("gender"):
        try: prof["gender"] = auto_detect_gender(uid)
        except Exception: prof["gender"] = "female"
    save_profile(uid, prof)
    await update.message.reply_text(
        f"Готово ✅ В «{av_name}» {len(av['images'])} фото.\nДалее — «🧪 Обучение» или /trainid.",
        reply_markup=main_menu_kb()
    )

async def id_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id; prof = load_profile(uid)
    av_name = get_current_avatar_name(prof); av = get_avatar(prof, av_name)
    await update.effective_message.reply_text(
        f"Активный аватар: {av_name}\n"
        f"Фото (локально): {len(list_ref_images(uid, av_name))}\n"
        f"S3 рефов: {len(av.get('s3_refs', []))}\n"
        f"Статус: {av.get('status') or '—'}\n"
        f"Модель: {av.get('finetuned_model') or '—'}\n"
        f"Версия: {av.get('finetuned_version') or '—'}\n"
        f"Пол: {prof.get('gender') or '—'}\n"
        f"LOCKFACE: {'on' if av.get('lockface') else 'off'}\n"
        f"Natural: {'ON' if prof.get('natural', True) else 'OFF'} • Pretty: {'ON' if prof.get('pretty', False) else 'OFF'}"
    )

async def id_reset(update: Update, Context):
    uid = update.effective_user.id
    delete_profile(uid)
    await update.message.reply_text("Профиль очищен. Жми «📸 Набор фото» и загрузи снимки заново.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    prof = load_profile(uid); av_name = get_current_avatar_name(prof)
    if ENROLL_FLAG.get((uid, av_name)):
        refs = list_ref_images(uid, av_name)
        if len(refs) >= 10:
            await update.message.reply_text("Уже 10/10. Нажми /iddone."); return
        f = await update.message.photo[-1].get_file()
        data = await f.download_as_bytearray()
        save_ref_downscaled(avatar_dir(uid, av_name) / f"ref_{int(time.time()*1000)}.jpg", bytes(data), uid=uid, avatar=av_name)
        await update.message.reply_text(f"Сохранила ({len(refs)+1}/10) для «{av_name}». Ещё?")
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
        f"Пол: {prof.get('gender') or '—'}\nСменить: /setgender female | /setgender male"
    )

async def lockface_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    prof = load_profile(uid); av = get_avatar(prof)
    av["lockface"] = not av.get("lockface", True); save_profile(uid, prof)
    await update.effective_message.reply_text(f"LOCKFACE {'включён' if av['lockface'] else 'выключен'} для активного аватара.")

# ---- Аватарные команды ----
async def avatarnew_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not context.args:
        await update.message.reply_text("Использование: /avatarnew <имя>"); return
    name = re.sub(r"[^\w\-\.\@]+", "_", " ".join(context.args)).strip()[:32] or "noname"
    ensure_avatar(uid, name); set_current_avatar(uid, name)
    await update.message.reply_text(f"Создан и выбран аватар: {name}", reply_markup=avatars_kb(uid))

async def avatarset_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not context.args:
        await update.message.reply_text("Использование: /avatarset <имя>"); return
    name = " ".join(context.args).strip()
    prof = load_profile(uid)
    if name not in prof["avatars"]:
        await update.message.reply_text(f"Аватар «{name}» не найден."); return
    set_current_avatar(uid, name)
    await update.message.reply_text(f"Ок, активный аватар: {name}", reply_markup=avatars_kb(uid))

async def avatarlist_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    prof = load_profile(uid); cur = get_current_avatar_name(prof)
    lines = ["Твои аватары:"]
    for n, av in prof["avatars"].items():
        refs = len(list_ref_images(uid, n))
        lines.append(f"{'▶️' if n==cur else ' '} {n}: лок.фото {refs}, S3 {len(av.get('s3_refs', []))}, статус: {av.get('status') or '—'}")
    await update.message.reply_text("\n".join(lines), reply_markup=avatars_kb(uid))

async def avatardel_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not context.args:
        await update.message.reply_text("Использование: /avatardel <имя> --force"); return
    args = context.args[:]; force = False
    if "--force" in args: force = True; args.remove("--force")
    name = " ".join(args).strip()
    if name == "default":
        await update.message.reply_text("«default» удалять нельзя."); return
    prof = load_profile(uid)
    if name not in prof["avatars"]:
        await update.message.reply_text(f"Аватар «{name}» не найден."); return
    if not force:
        await update.message.reply_text("Добавь флаг --force: /avatardel <имя> --force"); return
    try:
        del_avatar(uid, name)
        await update.message.reply_text(f"Аватар «{name}» удалён.", reply_markup=avatars_kb(uid))
    except Exception as e:
        await update.message.reply_text(f"Не удалось удалить: {e}")

# ---- Обучение / Генерация ----
def _dest_model_slug(avatar:str) -> str:
    if not DEST_OWNER: raise RuntimeError("REPLICATE_DEST_OWNER не задан.")
    return f"{DEST_OWNER}/{DEST_MODEL}"

def _ensure_destination_exists(slug: str):
    try:
        replicate.models.get(slug)
    except Exception:
        o, name = slug.split("/",1)
        raise RuntimeError(f"Целевая модель '{slug}' не найдена. Создай на https://replicate.com/create (owner={o}, name='{name}').")

def _pack_refs_zip(uid:int, avatar:str) -> Path:
    ensure_local_refs_from_s3(uid, avatar, need=10)
    refs = list_ref_images(uid, avatar)
    if len(refs) < 10: raise RuntimeError("Нужно 10 фото для обучения.")
    zpath = avatar_dir(uid, avatar) / "train.zip"
    with ZipFile(zpath, "w") as z:
        for i, p in enumerate(refs, 1):
            z.write(p, arcname=f"img_{i:02d}.jpg")
    # зальём zip в S3 (необязательно для работы, но полезно)
    try:
        key = _trainzip_s3_key(uid, avatar)
        s3_put_bytes(key, Path(zpath).read_bytes(), "application/zip")
    except Exception as e:
        logger.warning("S3 train.zip upload failed: %s", e)
    return zpath

def start_lora_training(uid:int, avatar:str) -> str:
    dest_model = _dest_model_slug(avatar); _ensure_destination_exists(dest_model)
    trainer_version = resolve_model_version(LORA_TRAINER_SLUG)
    zip_path = _pack_refs_zip(uid, avatar)
    client = Client(api_token=os.environ["REPLICATE_API_TOKEN"])

    prof = load_profile(uid)
    g = (prof.get("gender") or auto_detect_gender(uid) or "female").lower()
    caption_prefix = _caption_for_gender(g)

    with open(zip_path, "rb") as f:
        training = client.trainings.create(
            version=trainer_version,
            input={
                LORA_INPUT_KEY: f,
                "max_train_steps": LORA_MAX_STEPS,
                "lora_lr": LORA_LR,
                "use_face_detection_instead": LORA_USE_FACE_DET,
                "caption_prefix": caption_prefix,
                "resolution": LORA_RESOLUTION,
            },
            destination=dest_model
        )
    prof = load_profile(uid); av = get_avatar(prof, avatar)
    av["training_id"] = training.id; av["status"] = "starting"; av["finetuned_model"] = dest_model
    save_profile(uid, prof)
    return training.id

def check_training_status(uid:int, avatar:str) -> Tuple[str, Optional[str], Optional[str]]:
    prof = load_profile(uid); av = get_avatar(prof, avatar); tid = av.get("training_id")
    if not tid: return ("not_started", None, None)
    client = Client(api_token=os.environ["REPLICATE_API_TOKEN"])
    tr = client.trainings.get(tid)
    status = getattr(tr, "status", None) or (tr.get("status") if isinstance(tr, dict) else None) or "unknown"
    err = None
    try:
        err = (tr.get("error") or tr.get("detail")) if isinstance(tr, dict) else (getattr(tr, "error", None) or getattr(tr, "detail", None))
    except Exception:
        pass
    if status != "succeeded":
        av["status"] = status; save_profile(uid, prof); return (status, None, err)

    destination = getattr(tr, "destination", None) or (isinstance(tr, dict) and tr.get("destination")) \
        or av.get("finetuned_model") or _dest_model_slug(avatar)
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

    av["status"] = status; av["finetuned_model"] = destination
    if slug_with_version and ":" in slug_with_version:
        av["finetuned_version"] = slug_with_version.split(":",1)[1]
    save_profile(uid, prof)
    return (status, slug_with_version, None)

def _pinned_slug(av: Dict[str, Any]) -> str:
    base = av.get("finetuned_model") or ""; ver = av.get("finetuned_version")
    return f"{base}:{ver}" if (base and ver) else base

async def trainid_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id; prof = load_profile(uid)
    av_name = get_current_avatar_name(prof)
    ensure_local_refs_from_s3(uid, av_name, need=10)
    if len(list_ref_images(uid, av_name)) < 10:
        await update.effective_message.reply_text(f"Нужно 10 фото в «{av_name}». Сначала «📸 Набор фото» и затем /iddone."); return
    await update.effective_message.reply_text(f"Запускаю обучение LoRA для «{av_name}»…")
    try:
        training_id = await asyncio.to_thread(start_lora_training, uid, av_name)
        await update.effective_message.reply_text(f"Стартанула. ID: {training_id}\nПроверяй /trainstatus.")
        if DEST_OWNER and DEST_MODEL and training_id:
            await update.effective_message.reply_text(f"Логи: https://replicate.com/{DEST_OWNER}/{DEST_MODEL}/trainings/{training_id}")
    except Exception as e:
        logging.exception("trainid failed")
        await update.effective_message.reply_text(f"Не удалось запустить обучение: {e}")

async def trainstatus_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id; prof = load_profile(uid)
    av_name = get_current_avatar_name(prof)
    status, slug_with_ver, err = await asyncio.to_thread(check_training_status, uid, av_name)
    av = get_avatar(prof, av_name); tid = av.get("training_id")
    train_url = f"https://replicate.com/{DEST_OWNER}/{DEST_MODEL}/trainings/{tid}" if (DEST_OWNER and DEST_MODEL and tid) else None

    if status == "succeeded" and slug_with_ver:
        await update.effective_message.reply_text(
            f"Готово ✅\nАватар: {av_name}\nМодель: {slug_with_ver}\nТеперь — «🧭 Выбрать стиль».",
            reply_markup=categories_kb()
        ); return
    if status in ("starting","processing","running","queued","pending"):
        await update.effective_message.reply_text(f"Статус «{av_name}»: {status}… {('Логи: ' + train_url) if train_url else ''}"); return
    if status in ("failed","canceled"):
        msg = f"⚠️ Тренировка «{av_name}»: {status.upper()}."
        if err: msg += f"\nПричина: {err}"
        if train_url: msg += f"\nЛоги: {train_url}"
        msg += ("\n\nПроверь: целевая модель существует; 10 фото без фильтров; кредиты Replicate; LORA_* env.")
        await update.effective_message.reply_text(msg); return
    await update.effective_message.reply_text(f"Статус «{av_name}»: {status}. {('Логи: ' + train_url) if train_url else ''}")

def _neg_with_gender(neg_base:str, gender_negative:str) -> str:
    return (neg_base + (", " + gender_negative if gender_negative else "")).strip(", ")

async def cb_style(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    preset = q.data.split(":",1)[1]
    await start_generation_for_preset(update, context, preset)

async def start_generation_for_preset(update: Update, context: ContextTypes.DEFAULT_TYPE, preset: str):
    uid = update.effective_user.id
    prof = load_profile(uid); av_name = get_current_avatar_name(prof)
    av = get_avatar(prof, av_name)
    if av.get("status") != "succeeded":
        await update.effective_message.reply_text(f"Модель «{av_name}» ещё не готова. /trainid → /trainstatus = succeeded.")
        return

    meta = STYLE_PRESETS[preset]
    gender  = (prof.get("gender") or "female").lower()
    natural = prof.get("natural", True)
    pretty  = prof.get("pretty", False)

    desired_comp = meta.get("comp","half")
    if CONSISTENT_SCALE and desired_comp not in ("half","closeup"):
        desired_comp = "half"

    comp_text, (w,h) = _comp_text_and_size(desired_comp)
    tone_text = _tone_text(meta.get("tone","beauty" if "portrait" in preset.lower() else "daylight"))
    theme_boost = THEME_BOOST.get(preset, "")
    prompt_core, gender_negative = build_prompt(meta, gender, comp_text, tone_text, theme_boost, natural, pretty)
    model_slug = _pinned_slug(av)

    guidance = max(3.8, min(4.4, SCENE_GUIDANCE.get(preset, GEN_GUIDANCE)))
    steps = min(MAX_STEPS, 46 if natural else max(48, GEN_STEPS))
    if pretty: guidance, steps = min(4.8, guidance + 0.4), min(MAX_STEPS, max(50, steps))

    await update.effective_message.chat.send_action(ChatAction.UPLOAD_PHOTO)
    desc = meta.get("desc", preset)
    await update.effective_message.reply_text(f"🎬 {preset}\nАватар: {av_name}\n{desc}\n\nГенерирую (InstantID lock, {gender}, {w}×{h}) …")

    try:
        face_refs = list_ref_images(uid, av_name)
        face_bytes = None
        if face_refs:
            face_bytes = Path(face_refs[0]).read_bytes()
        if face_bytes is None:
            face_bytes = load_face_cache(uid, av_name)
        if face_bytes is None:
            face_bytes = load_face_from_s3(uid, av_name)

        if not (INSTANTID_SLUG and face_bytes):
            raise RuntimeError("InstantID не настроен или нет реф-фото. Заполни INSTANTID_SLUG и загрузи фото.")

        seeds = [random.randrange(2**32) for _ in range(3)]
        neg_base = _neg_with_gender(NEGATIVE_PROMPT_BASE, gender_negative)

        base_steps = steps
        inst_steps = min(MAX_STEPS, max(40, steps))

        for s in seeds:
            base_url = await asyncio.to_thread(
                generate_from_finetune, model_slug=model_slug, prompt=prompt_core,
                steps=base_steps, guidance=guidance, seed=random.randrange(2**32),
                w=w, h=h, negative_prompt=neg_base
            )
            base_bytes = requests.get(base_url, timeout=60).content

            url = await asyncio.to_thread(
                generate_with_instantid, face_bytes=face_bytes, prompt=prompt_core,
                steps=inst_steps, guidance=guidance, seed=s, w=w, h=h,
                negative_prompt=neg_base, natural=natural, pretty=pretty,
                content_image_bytes=base_bytes
            )

            await update.effective_message.reply_photo(photo=url, caption=f"{preset} • {av_name} • 🔒")

    except Exception as e:
        logging.exception("generation failed")
        await update.effective_message.reply_text(f"Ошибка генерации: {e}")

# --- Toggles ---
async def pretty_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    prof = load_profile(uid)
    prof["pretty"] = not prof.get("pretty", False)
    if prof["pretty"]: prof["natural"] = True
    save_profile(uid, prof)
    await update.message.reply_text(f"Pretty: {'ON' if prof['pretty'] else 'OFF'} (Natural: {'ON' if prof['natural'] else 'OFF'})")

async def natural_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    prof = load_profile(uid)
    prof["natural"] = not prof.get("natural", True)
    save_profile(uid, prof)
    await update.message.reply_text(f"Natural: {'ON' if prof['natural'] else 'OFF'} (Pretty: {'ON' if prof.get('pretty') else 'OFF'})")

# ---------- System ----------
async def _post_init(app):
    await app.bot.delete_webhook(drop_pending_updates=True)

def main():
    app = ApplicationBuilder().token(TOKEN).post_init(_post_init).build()

    # Команды
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("menu", start))
    app.add_handler(CommandHandler("idenroll", id_enroll))
    app.add_handler(CommandHandler("iddone", id_done))
    app.add_handler(CommandHandler("idstatus", id_status))
    app.add_handler(CommandHandler("idreset", id_reset))
    app.add_handler(CommandHandler("setgender", setgender_cmd))
    app.add_handler(CommandHandler("gender", gender_cmd))
    app.add_handler(CommandHandler("lockface", lockface_cmd))
    app.add_handler(CommandHandler("trainid", trainid_cmd))
    app.add_handler(CommandHandler("trainstatus", trainstatus_cmd))
    app.add_handler(CommandHandler("styles", styles_cmd))
    app.add_handler(CommandHandler("pretty", pretty_cmd))
    app.add_handler(CommandHandler("natural", natural_cmd))

    # Аватары
    app.add_handler(CommandHandler("avatarnew", avatarnew_cmd))
    app.add_handler(CommandHandler("avatarset", avatarset_cmd))
    app.add_handler(CommandHandler("avatarlist", avatarlist_cmd))
    app.add_handler(CommandHandler("avatardel", avatardel_cmd))

    # Кнопки
    app.add_handler(CallbackQueryHandler(nav_cb, pattern=r"^nav:"))
    app.add_handler(CallbackQueryHandler(cb_category, pattern=r"^cat:"))
    app.add_handler(CallbackQueryHandler(cb_style, pattern=r"^style:"))
    app.add_handler(CallbackQueryHandler(avatar_cb, pattern=r"^avatar:"))

    # Фото
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Проверки
    _check_slug(LORA_TRAINER_SLUG, "LoRA trainer")
    if INSTANTID_SLUG: _check_slug(INSTANTID_SLUG, "InstantID")
    else: logger.info("InstantID disabled (no INSTANTID_SLUG).")

    logger.info("Bot up. DEST=%s/%s GEN=%dx%d", DEST_OWNER, DEST_MODEL, GEN_WIDTH, GEN_HEIGHT)
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
