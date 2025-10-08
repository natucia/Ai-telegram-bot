# === Telegram LoRA Bot (Flux LoRA trainer + Redis persist
# + Identity/Gender locks + InstantID LOCKFACE (1/2-step fallback)
# + MULTI-AVATARS + NATURAL/Pretty per-user + CONSISTENT FACE SCALE
# + S3 storage + concurrency limits + retries) ===
# Требования:
# python-telegram-bot==20.7, replicate==0.31.0, pillow==10.4.0, redis==5.0.1, boto3==1.34.0+

from typing import Any, Dict, List, Optional, Tuple, Iterable
Style = Dict[str, Any]

from styles import (  # твой styles.py
    STYLE_PRESETS, STYLE_CATEGORIES, THEME_BOOST,
    SCENE_GUIDANCE, RISKY_PRESETS
)

import os, re, io, json, time, asyncio, logging, shutil, random, contextlib, tempfile
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED
import tempfile, os
import requests
import replicate
from replicate import Client
from PIL import Image, ImageOps

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
INSTANTID_FORCE_TWOSTEP = os.getenv("INSTANTID_FORCE_TWOSTEP", "0").lower() in ("1","true","yes","y")

# --- Параметры обучения ---
LORA_MAX_STEPS = int(os.getenv("LORA_MAX_STEPS", "1400"))
LORA_LR = float(os.getenv("LORA_LR", "0.0006"))
LORA_USE_FACE_DET = os.getenv("LORA_USE_FACE_DET", "true").lower() in ("1","true","yes","y")
LORA_RESOLUTION = int(os.getenv("LORA_RESOLUTION", "1024"))

DEFAULT_FEMALE_CAPTION = (
    "a high quality photo of the same woman, natural brows, medium-length hair with natural hairline, "
    "brown eyes, neutral expression, balanced facial proportions, natural lip shape, "
    "no retouch, true-to-life skin texture"
)
DEFAULT_MALE_CAPTION = (
    "a high quality photo of the same man, natural brows, short hair or neat hairstyle, "
    "brown eyes, neutral expression, balanced facial proportions, natural lip shape, "
    "no retouch, true-to-life skin texture"
)

# --- Генерация ---
GEN_STEPS = int(os.getenv("GEN_STEPS", "48"))
GEN_GUIDANCE = float(os.getenv("GEN_GUIDANCE", "4.2"))
GEN_WIDTH = int(os.getenv("GEN_WIDTH", "896"))
GEN_HEIGHT = int(os.getenv("GEN_HEIGHT", "1152"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "50"))

# --- CONSISTENT FACE SCALE ---
CONSISTENT_SCALE = os.getenv("CONSISTENT_SCALE", "1").lower() in ("1","true","yes","y")
HEAD_HEIGHT_FRAC = float(os.getenv("HEAD_HEIGHT_FRAC", "0.36"))
HEAD_WIDTH_FRAC  = float(os.getenv("HEAD_WIDTH_FRAC", "0.28"))

# ---- Anti-drift / anti-wide-face ----
NEGATIVE_PROMPT_BASE = (
    "cartoon, anime, cgi, 3d, stylized, plastic skin, overprocessed, airbrushed, beauty-filter, "
    "lowres, blur, textureless skin, porcelain skin, waxy, gaussian blur, smoothing filter, "
    "text, watermark, logo, bad anatomy, extra fingers, short fingers, "
    "identity drift, different person, face swap, face morph, ethnicity change, age change, "
    "hairline modification, beard reshaping, lip reshape, mouth corner lift, "
    "puffy face, swollen face, chubby cheeks, bloated cheeks, widened jaw, broad zygomatic width, "
    "wide face, horizontally stretched face, tiny head, giant head, variable head scale, "
    "zoomed-in extreme close-up, distant tiny face, aspect distortion, fisheye, lens distortion, warping, "
    "plain selfie, tourist photo, plain studio backdrop, denoise artifacts, waxy highlight roll-off, "
    "excessive frequency separation, face slimming or widening, retouched pores"
)

AESTHETIC_SUFFIX = (
    ", photorealistic, visible fine skin texture, natural color, soft filmic contrast, "
    "micro-sharpen on eyes and lips only, anatomically plausible facial landmarks"
)

# ---------- NATURAL/PRETTY ----------
NATURAL_POS = (
    "unretouched skin, realistic fine pores, subtle skin texture variations, "
    "natural micro-contrast, gentle film grain, true-to-life color, accurate skin undertones"
)
NATURAL_NEG = (
    "skin smoothing, airbrush, beauty-filter, porcelain skin, waxy skin, HDR glam, "
    "overprocessed clarity, excessive de-noise, plastic texture, glam retouch"
)

PRETTY_POS = (
    "subtle beauty retouch, even skin tone, faint under-eye smoothing, slight glow, tidy eyebrows"
)
PRETTY_NEG = (
    "over-smoothing, harsh pores, deep nasolabial folds, oily hotspot shine, oversharpened skin, beauty filter"
)
PRETTY_COMP_HINT = "camera slightly above eye level, flattering portrait angle"

# ---------- logging ----------
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("bot")

# ---------- Concurrency & retries ----------
# Лимиты на процесс: не спамим Replicate/S3, чтобы бот жил под нагрузкой
GEN_SEMAPHORE = asyncio.Semaphore(int(os.getenv("GEN_CONCURRENCY", "6")))
TRAIN_SEMAPHORE = asyncio.Semaphore(int(os.getenv("TRAIN_CONCURRENCY", "2")))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "60"))

def _retry(fn, *args, tries=3, backoff=1.8, label="op", **kwargs):
    last = None
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last = e
            if i < tries-1:
                time.sleep(backoff**i)
            else:
                raise RuntimeError(f"{label} failed after {tries} tries: {e}") from e

# ---------- storage (FS/S3) ----------
DATA_DIR = Path("profiles"); DATA_DIR.mkdir(exist_ok=True)

USE_S3 = os.getenv("USE_S3", "0").lower() in ("1","true","yes","y")
S3_BUCKET = os.getenv("S3_BUCKET", "").strip()
S3_REGION = os.getenv("S3_REGION", "").strip()
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID", "").strip()
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY", "").strip()
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "").strip() or None
S3_PREFIX = os.getenv("S3_PREFIX", "").strip().strip("/")

s3_client = None
if USE_S3:
    import boto3
    session = boto3.session.Session(
        aws_access_key_id=S3_ACCESS_KEY_ID or None,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY or None,
        region_name=S3_REGION or None
    )
    s3_client = session.client("s3", endpoint_url=S3_ENDPOINT_URL)  # endpoint_url поддерживает S3-совместимые хранилища

def _s3_key(*parts: str) -> str:
    p = "/".join(str(x).strip("/").replace("//","/") for x in parts if x is not None)
    return f"{S3_PREFIX}/{p}".strip("/") if S3_PREFIX else p

def tmp_path(suffix=".jpg") -> Path:
    return Path(tempfile.mkstemp(prefix="bot_", suffix=suffix)[1])

# --- FS utils (локальный fallback) ---
def user_dir(uid:int) -> Path:
    p = DATA_DIR / str(uid); p.mkdir(parents=True, exist_ok=True); return p
def avatars_root(uid:int) -> Path:
    p = user_dir(uid) / "avatars"; p.mkdir(parents=True, exist_ok=True); return p
def avatar_dir(uid:int, avatar:str) -> Path:
    p = avatars_root(uid) / avatar; p.mkdir(parents=True, exist_ok=True); return p
def profile_path(uid:int) -> Path:
    return user_dir(uid) / "profile.json"

# --- Storage Abstraction ---
class Storage:
    def save_ref_image(self, uid:int, avatar:str, raw:bytes) -> str: ...
    def list_ref_images(self, uid:int, avatar:str) -> List[str]: ...
    def delete_avatar(self, uid:int, avatar:str): ...
    def get_local_copy(self, key:str) -> Path: ...
    def pack_refs_zip(self, uid:int, avatar:str) -> Path: ...

class FSStorage(Storage):
    def save_ref_image(self, uid:int, avatar:str, raw:bytes) -> str:
        path = avatar_dir(uid, avatar) / f"ref_{int(time.time()*1000)}.jpg"
        _save_ref_downscaled_local(path, raw)
        return str(path)

    def list_ref_images(self, uid:int, avatar:str) -> List[str]:
        return sorted(str(p) for p in avatar_dir(uid, avatar).glob("ref_*.jpg"))

    def delete_avatar(self, uid:int, avatar:str):
        adir = avatar_dir(uid, avatar)
        with contextlib.suppress(Exception):
            if adir.exists(): shutil.rmtree(adir)

    def get_local_copy(self, key:str) -> Path:
        # уже локальный путь
        return Path(key)

    def pack_refs_zip(self, uid:int, avatar:str) -> Path:
        refs = self.list_ref_images(uid, avatar)
        if len(refs) < 10: raise RuntimeError("Нужно 10 фото для обучения.")
        zpath = avatar_dir(uid, avatar) / "train.zip"
        with ZipFile(zpath, "w", compression=ZIP_STORED) as z:
            for i, kp in enumerate(refs, 1):
                z.write(Path(kp), arcname=f"img_{i:02d}.jpg")
        return zpath

class S3Storage(Storage):
    def save_ref_image(self, uid:int, avatar:str, raw:bytes) -> str:
        if not s3_client: raise RuntimeError("S3 не инициализирован.")
        # даунскейл в памяти
        buf = io.BytesIO()
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        im = _center_crop80(im)
        im.thumbnail((1024,1024))
        im.save(buf, "JPEG", quality=92); buf.seek(0)
        key = _s3_key("profiles", str(uid), "avatars", avatar, f"ref_{int(time.time()*1000)}.jpg")
        _retry(s3_client.put_object, Bucket=S3_BUCKET, Key=key, Body=buf.getvalue(), ContentType="image/jpeg", label="s3_put")
        return f"s3://{S3_BUCKET}/{key}"

    def list_ref_images(self, uid:int, avatar:str) -> List[str]:
        if not s3_client: return []
        prefix = _s3_key("profiles", str(uid), "avatars", avatar, "ref_")
        keys: List[str] = []
        cont = None
        while True:
            resp = _retry(s3_client.list_objects_v2, Bucket=S3_BUCKET, Prefix=prefix, ContinuationToken=cont, label="s3_list") \
                   if cont else _retry(s3_client.list_objects_v2, Bucket=S3_BUCKET, Prefix=prefix, label="s3_list")
            for it in resp.get("Contents", []):
                keys.append(f"s3://{S3_BUCKET}/{it['Key']}")
            if resp.get("IsTruncated"):
                cont = resp.get("NextContinuationToken")
            else:
                break
        return sorted(keys)

    def delete_avatar(self, uid:int, avatar:str):
        prefix = _s3_key("profiles", str(uid), "avatars", avatar)
        # собираем и удаляем пачкой
        to_del = []
        cont = None
        while True:
            resp = _retry(s3_client.list_objects_v2, Bucket=S3_BUCKET, Prefix=prefix, ContinuationToken=cont, label="s3_list") \
                   if cont else _retry(s3_client.list_objects_v2, Bucket=S3_BUCKET, Prefix=prefix, label="s3_list")
            keys = [ {"Key": o["Key"]} for o in resp.get("Contents", []) ]
            if keys:
                _retry(s3_client.delete_objects, Bucket=S3_BUCKET, Delete={"Objects": keys}, label="s3_del")
            if not resp.get("IsTruncated"): break
            cont = resp.get("NextContinuationToken")

    def get_local_copy(self, key:str) -> Path:
        # key формата s3://bucket/key
        if not key.startswith("s3://"): raise RuntimeError("Ожидался s3:// ключ")
        _, _, bucket_and_key = key.partition("s3://")
        bucket, _, obj_key = bucket_and_key.partition("/")
        path = tmp_path(".jpg")
        _retry(s3_client.download_file, bucket, obj_key, str(path), label="s3_download")
        return path

    def pack_refs_zip(self, uid:int, avatar:str) -> Path:
        refs = self.list_ref_images(uid, avatar)
        if len(refs) < 10: raise RuntimeError("Нужно 10 фото для обучения.")
        zpath = Path(tempfile.mkstemp(prefix="train_", suffix=".zip")[1])
        with ZipFile(zpath, "w", compression=ZIP_STORED) as z:
            for i, key in enumerate(refs, 1):
                lp = self.get_local_copy(key)
                z.write(lp, arcname=f"img_{i:02d}.jpg")
        # опционально можно залить ZIP в S3 (закомментировано):
        # zip_key = _s3_key("profiles", str(uid), "avatars", avatar, "train.zip")
        # _retry(s3_client.upload_file, str(zpath), S3_BUCKET, zip_key, label="s3_upload_zip")
        return zpath

# выбор стораджа
STORAGE: Storage = S3Storage() if USE_S3 else FSStorage()
logger.info("Storage backend: %s", "S3" if USE_S3 else "FS")

# ---------- профили (Redis/FS) ----------
DEFAULT_AVATAR = {
    "images": [],  # список ключей (FS-путь или s3://…)
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

def _center_crop80(im: Image.Image) -> Image.Image:
    w, h = im.size
    side = int(min(w, h) * 0.8)
    cx, cy = w // 2, h // 2
    left = max(0, cx - side // 2)
    top = max(0, cy - side // 2)
    return im.crop((left, top, left + side, top + side))

def _save_ref_downscaled_local(path: Path, raw: bytes, max_side=1024, quality=92):
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    im = _center_crop80(im)
    im.thumbnail((max_side, max_side))
    im.save(path, "JPEG", quality=quality)

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

def list_ref_images(uid:int, avatar:str) -> List[str]:
    return STORAGE.list_ref_images(uid, avatar)

def profile_path(uid:int) -> Path:
    return user_dir(uid) / "profile.json"

def _migrate_single_to_multi(uid:int, prof:Dict[str,Any]) -> Dict[str,Any]:
    if "avatars" in prof: return prof
    migrated = DEFAULT_PROFILE.copy()
    migrated["gender"] = prof.get("gender")
    default = DEFAULT_AVATAR.copy()
    default["training_id"] = prof.get("training_id")
    default["finetuned_model"] = prof.get("finetuned_model")
    default["finetuned_version"] = prof.get("finetuned_version")
    default["status"] = prof.get("status")
    default["lockface"] = prof.get("lockface", True)
    imgs = prof.get("images", [])
    if imgs:
        default["images"] = STORAGE.list_ref_images(uid, "default")
    migrated["avatars"]["default"] = default
    return migrated

_redis = None
REDIS_URL = os.getenv("REDIS_URL", "").strip()
if REDIS_URL:
    try:
        import redis
        _redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        _redis.ping()
        logger.info("Storage: Redis OK (%s)", REDIS_URL.rsplit("@",1)[-1])
    except Exception as e:
        logger.warning("Storage: Redis init failed (%s). Falling back to FS. Error: %s", REDIS_URL, e)
        _redis = None
else:
    logger.info("Storage: FS for profiles (no REDIS_URL)")

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
        user_dir(uid).mkdir(parents=True, exist_ok=True)
        profile_path(uid).write_text(json.dumps(prof, ensure_ascii=False))

def delete_profile(uid:int):
    if _redis:
        with contextlib.suppress(Exception):
            _redis.delete(f"profile:{uid}")
    # удаляем все аватары из хранилища
    prof = load_profile(uid)
    for name in list(prof.get("avatars", {}).keys()):
        if name == "default": continue
        STORAGE.delete_avatar(uid, name)
    # default тоже чистим
    STORAGE.delete_avatar(uid, "default")
    # чистим локальную папку профиля
    p = user_dir(uid)
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)

# ---------- авто-пол ----------
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

def _infer_gender_from_image_local(local_path: Path) -> Optional[str]:
    try:
        client = Client(api_token=os.environ["REPLICATE_API_TOKEN"])
        version_slug = resolve_model_version(GENDER_MODEL_SLUG)
        with open(local_path, "rb") as img_b:
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
    face_key = refs[0]
    local = STORAGE.get_local_copy(face_key)
    g = _infer_gender_from_image_local(local)
    with contextlib.suppress(Exception):
        if local.exists() and str(local).startswith(tempfile.gettempdir()):
            local.unlink()
    return g or "female"

def _caption_for_gender(g: str) -> str:
    env_override = os.getenv("LORA_CAPTION_PREFIX", "").strip()
    if env_override:
        return env_override
    return DEFAULT_MALE_CAPTION if (g or "").lower() == "male" else DEFAULT_FEMALE_CAPTION

# ---------- Композиция/линза/свет (без изменений значимых) ----------
def _beauty_guardrail() -> str:
    return (
        "exact facial identity, identity preserved, "
        "balanced facial proportions, symmetrical face, natural oval, soft jawline, "
        "keep original zygomatic width and jaw width, do not widen face, "
        "open expressive eyes with clean catchlights"
    )

def _face_lock() -> str:
    return (
        "keep same bone structure, natural interocular distance, consistent eyelid shape, "
        "aligned pupils, preserve cheekbone width and lip fullness"
    )

def _anti_distort() -> str:
    return "no fisheye, no lens distortion, no warping, natural perspective, proportional head size"

def _gender_lock(gender:str) -> Tuple[str, str]:
    if gender == "male":
        pos = "male man, masculine facial features, light stubble allowed"
        neg = "female, woman heavy makeup"
    else:
        pos = "female woman, feminine facial features"
        neg = "male, man, beard, stubble, mustache, adam's apple"
    return pos, neg

def _safe_portrait_size(w:int, h:int) -> Tuple[int,int]:
    ar = w / max(1, h)
    if ar >= 0.75:
        return int(h*0.66), h  # ~2:3
    return w, h

def _face_scale_hint() -> str:
    if not CONSISTENT_SCALE:
        return ""
    hh = int(HEAD_HEIGHT_FRAC * 100)
    hw = int(HEAD_WIDTH_FRAC * 100)
    return (
        f"keep constant head scale across all images, head height ~{hh}% of frame from chin to top of head, "
        f"head width ~{hw}% of frame width, subject centered, do not zoom in or out"
    )

def _comp_text_and_size(comp: str) -> Tuple[str, Tuple[int,int]]:
    scale_txt = _face_scale_hint()
    if comp == "closeup":
        w, h = _safe_portrait_size(896, 1152)
        return (
            f"portrait framing from chest up, 85mm lens look, camera at eye level, subject distance ~1.2m, "
            f"no perspective distortion on face, head width proportional, natural perspective, {scale_txt}", (w, h)
        )
    if comp == "half":
        w, h = _safe_portrait_size(GEN_WIDTH, max(GEN_HEIGHT, 1344))
        return (
            f"half body framing, 85mm lens look, camera at chest level, subject distance ~2.0m, "
            f"no perspective distortion on face, head width proportional, natural perspective, {scale_txt}", (w, h)
        )
    w, h = _safe_portrait_size(GEN_WIDTH, 1408)
    return (
        f"full body framing, 85mm lens look, camera at mid-torso level, head size natural for frame, "
        f"no perspective distortion on face, {scale_txt}", (w, h)
    )

def _tone_text(tone: str) -> str:
    return {
        "daylight": "soft natural daylight, neutral colors",
        "warm": "golden hour warmth, gentle highlights",
        "cool": "cool cinematic light, clean color balance",
        "noir": "high contrast noir lighting, subtle rim light",
        "neon": "neon signs, wet reflections, cinematic backlight, vibrant saturation",
        "candle": "warm candlelight, soft glow, volumetric rays",
    }.get(tone, "balanced soft lighting")

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

def _inject_beauty(core_prompt: str, comp_text: str, natural: bool, pretty: bool) -> str:
    parts = [core_prompt]
    if natural: parts.append(NATURAL_POS)
    if pretty:
        parts.append(PRETTY_POS)
        if "eye level" in comp_text or "chest level" in comp_text:
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
    age_lock = "" if meta.get("allow_age_change") else "no age change, "

    common_bits = [
        tone_text, gpos,
        "same person as the training photos, no ethnicity change, " + age_lock + "exact facial identity, identity preserved +++",
        "photorealistic, realistic body proportions, natural fine skin texture, filmic look",
        "do not widen face, keep original cheekbone width and jaw width, preserve lips shape",
        "85mm lens portrait look",
        _face_scale_hint(),
        anti, _beauty_guardrail(), _face_lock(), theme_boost
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
    if pretty:  neg = (neg + ", " + PRETTY_NEG) if neg else PRETTY_NEG
    return core, neg

# ---------- Инференс/генерация с ретраями ----------
def generate_from_finetune(model_slug:str, prompt:str, steps:int, guidance:float, seed:int, w:int, h:int, negative_prompt:str) -> str:
    mv = resolve_model_version(model_slug)
    def _run():
        return replicate.run(mv, input={
            "prompt": prompt + AESTHETIC_SUFFIX,
            "negative_prompt": negative_prompt,
            "width": w, "height": h,
            "num_inference_steps": min(MAX_STEPS, steps),
            "guidance_scale": guidance,
            "seed": seed,
        })
    out = _retry(_run, label="replicate_gen")
    url = extract_any_url(out)
    if not url: raise RuntimeError("Empty output")
    return url

def _bytes_to_tempfile(data: bytes, suffix: str) -> str:
                            fd, path = tempfile.mkstemp(prefix="inst_", suffix=suffix)
                            with os.fdopen(fd, "wb") as f:
                                f.write(data)
                            return path

def generate_with_instantid(face_path: Path, prompt: str, steps: int, guidance: float,
                                                    seed: int, w: int, h: int, negative_prompt: str,
                                                    natural: bool = True, content_image_bytes: Optional[bytes] = None) -> str:
                            mv = resolve_model_version(INSTANTID_SLUG)

                            # не ослабляем на natural
                            strength = INSTANTID_STRENGTH
                            face_w   = INSTANTID_FACE_WEIGHT

                            tmp_base = None
                            if content_image_bytes:
                                tmp_base = _bytes_to_tempfile(content_image_bytes, ".jpg")

                            # file-like дескрипторы — НЕ bytes
                            face_f = open(face_path, "rb")
                            image_f = open(tmp_base, "rb") if tmp_base else None

                            # минимально совместимый набор полей
                            inputs: Dict[str, Any] = {
                                "prompt": prompt + AESTHETIC_SUFFIX,
                                "negative_prompt": negative_prompt,
                                "width": w, "height": h,
                                "num_inference_steps": min(MAX_STEPS, steps),
                                "guidance_scale": guidance,
                                "seed": seed,

                                # сила локфейса
                                "id_strength": strength,
                                "face_strength": face_w,
                            }

                            # разные форки ждут разные ключи — дублируем
                            inputs["face_image"] = face_f
                            inputs["reference_image"] = face_f
                            inputs["id_image"] = face_f
                            inputs["face"] = face_f

                            if image_f:
                                inputs["image"] = image_f      # 2-шаг: подмешиваем базовый кадр LoRA

                            try:
                                out = replicate.run(mv, input=inputs)
                                url = extract_any_url(out)
                                if not url:
                                    raise RuntimeError("Empty output (InstantID)")
                                return url
                            finally:
                                with contextlib.suppress(Exception):
                                    face_f.close()
                                if image_f:
                                    with contextlib.suppress(Exception):
                                        image_f.close()
                                if tmp_base:
                                    with contextlib.suppress(Exception):
                                        os.remove(tmp_base)



# ---------- UI/KB (как было) ----------
def main_menu_kb() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("🧭 Выбрать стиль", callback_data="nav:styles")],
        [InlineKeyboardButton("📸 Набор фото", callback_data="nav:enroll"), InlineKeyboardButton("🧪 Обучение", callback_data="nav:train")],
        [InlineKeyboardButton("ℹ️ Мой статус", callback_data="nav:status"), InlineKeyboardButton("⚙️ Пол", callback_data="nav:gender")],
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
    prof = load_profile(uid)
    cur = get_current_avatar_name(prof)
    names = sorted(prof["avatars"].keys())
    rows = []
    for n in names:
        label = f"{'✅ ' if n==cur else ''}{n}"
        rows.append([InlineKeyboardButton(label, callback_data=f"avatar:set:{n}")])
    rows.append([InlineKeyboardButton("➕ Новый", callback_data="avatar:new"),
                 InlineKeyboardButton("🗑 Удалить", callback_data="avatar:del")])
    rows.append([InlineKeyboardButton("⬅️ Меню", callback_data="nav:menu")])
    return InlineKeyboardMarkup(rows)

# ----- Callback для кнопок "Аватары" -----
async def avatar_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    uid = update.effective_user.id
    parts = q.data.split(":")
    if len(parts) < 2:
        return
    action = parts[1]
    if action == "set":
        if len(parts) < 3:
            await q.message.reply_text("Не указан аватар. Используй /avatarlist.")
            return
        name = parts[2]
        set_current_avatar(uid, name)
        await q.message.reply_text(f"Активный аватар: {name}", reply_markup=avatars_kb(uid))
    elif action == "new":
        await q.message.reply_text("Создай новый: /avatarnew <имя> (пример: /avatarnew travel)")
    elif action == "del":
        await q.message.reply_text("Удаление: /avatardel <имя> --force")
    else:
        await q.message.reply_text("Неизвестное действие. Открой «🤖 Аватары» ещё раз.")

# ---------- Handlers ----------
ENROLL_FLAG: Dict[Tuple[int,str],bool] = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я создам твою персональную фотомодель из 10 фото и буду генерировать тебя в узнаваемых сценах.\n\n"
        "1) «📸 Набор фото» — загрузка до 10 снимков в активный аватар.\n"
        "2) «🧪 Обучение» — тренировка LoRA.\n"
        "3) «🧭 Выбрать стиль» — варианты.\n"
        "4) «🔒 LOCKFACE» — фиксация лица.\n"
        "5) «✨ Natural/Pretty» — натуральность или лёгкая ретушь.\n"
        "6) «🤖 Аватары» — несколько моделей.",
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
    elif key == "avatars":
        uid = update.effective_user.id
        await q.message.reply_text("Аватары:", reply_markup=avatars_kb(uid))
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
    prof = load_profile(uid)
    av_name = get_current_avatar_name(prof)
    ENROLL_FLAG[(uid, av_name)] = True
    await update.effective_message.reply_text(
        f"Набор включён для «{av_name}». Пришли подряд до 10 фото (фронтально, без фильтров). "
        "Когда закончишь — нажми /iddone."
    )

async def id_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    prof = load_profile(uid)
    av_name = get_current_avatar_name(prof)
    ENROLL_FLAG[(uid, av_name)] = False
    av = get_avatar(prof, av_name)
    av["images"] = list_ref_images(uid, av_name)
    try:
        if not prof.get("gender"):
            prof["gender"] = auto_detect_gender(uid)
    except Exception:
        prof["gender"] = prof.get("gender") or "female"
    save_profile(uid, prof)
    await update.message.reply_text(
        f"Готово ✅ В «{av_name}» {len(av['images'])} фото.\n"
        f"Пол: {prof.get('gender') or '—'}.\n"
        "Далее — «🧪 Обучение» или /trainid.",
        reply_markup=main_menu_kb()
    )

async def id_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id; prof = load_profile(uid)
    av_name = get_current_avatar_name(prof); av = get_avatar(prof, av_name)
    await update.effective_message.reply_text(
        f"Активный аватар: {av_name}\n"
        f"Фото: {len(list_ref_images(uid, av_name))}\n"
        f"Статус: {av.get('status') or '—'}\n"
        f"Модель: {av.get('finetuned_model') or '—'}\n"
        f"Версия: {av.get('finetuned_version') or '—'}\n"
        f"Пол: {prof.get('gender') or '—'}\n"
        f"LOCKFACE: {'on' if av.get('lockface') else 'off'}\n"
        f"Natural: {'ON' if prof.get('natural', True) else 'OFF'} • Pretty: {'ON' if prof.get('pretty', False) else 'OFF'}"
    )

async def id_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    delete_profile(uid)
    await update.message.reply_text("Профиль очищен. Жми «📸 Набор фото» и загрузи снимки заново.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    prof = load_profile(uid)
    av_name = get_current_avatar_name(prof)
    if ENROLL_FLAG.get((uid, av_name)):
        refs = list_ref_images(uid, av_name)
        if len(refs) >= 10:
            await update.message.reply_text("Уже 10/10. Нажми /iddone."); return
        f = await update.message.photo[-1].get_file()
        data = await f.download_as_bytearray()
        key = STORAGE.save_ref_image(uid, av_name, bytes(data))
        # подхватим новый список в профиле
        prof = load_profile(uid); av = get_avatar(prof, av_name)
        av["images"] = list_ref_images(uid, av_name)
        save_profile(uid, prof)
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
        f"Пол (общий): {prof.get('gender') or '—'}\n"
        "Сменить: /setgender female | /setgender male"
    )

async def lockface_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    prof = load_profile(uid)
    av = get_avatar(prof)
    av["lockface"] = not av.get("lockface", True)
    save_profile(uid, prof)
    state = "включён" if av["lockface"] else "выключен"
    await update.effective_message.reply_text(f"LOCKFACE {state} для активного аватара.")

# ---- Аватарные команды ----
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
    STORAGE.delete_avatar(uid, name)
    prof["avatars"].pop(name, None)
    if prof["current_avatar"] == name: prof["current_avatar"] = "default"
    save_profile(uid, prof)

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
        lines.append(f"{'▶️' if n==cur else ' '} {n}: фото {refs}, статус: {av.get('status') or '—'}, верс: {av.get('finetuned_version') or '—'}")
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
    return STORAGE.pack_refs_zip(uid, avatar)

def start_lora_training(uid:int, avatar:str) -> str:
    dest_model = _dest_model_slug(avatar); _ensure_destination_exists(dest_model)
    trainer_version = resolve_model_version(LORA_TRAINER_SLUG)
    zip_path = _pack_refs_zip(uid, avatar)
    client = Client(api_token=os.environ["REPLICATE_API_TOKEN"])

    prof = load_profile(uid)
    g = (prof.get("gender") or auto_detect_gender(uid) or "female").lower()
    caption_prefix = _caption_for_gender(g)

    with open(zip_path, "rb") as f:
        training = _retry(
            client.trainings.create,
            version=trainer_version,
            input={
                LORA_INPUT_KEY: f,
                "max_train_steps": LORA_MAX_STEPS,
                "lora_lr": LORA_LR,
                "use_face_detection_instead": LORA_USE_FACE_DET,
                "caption_prefix": caption_prefix,
                "resolution": LORA_RESOLUTION,
            },
            destination=dest_model,
            label="replicate_train"
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
        if isinstance(tr, dict):
            err = tr.get("error") or tr.get("detail")
        else:
            err = getattr(tr, "error", None) or getattr(tr, "detail", None)
    except Exception:
        pass

    if status != "succeeded":
        av["status"] = status; save_profile(uid, prof)
        return (status, None, err)

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
    if len(list_ref_images(uid, av_name)) < 10:
        await update.effective_message.reply_text(f"Нужно 10 фото в «{av_name}». Сначала «📸 Набор фото» и затем /iddone."); return
    await update.effective_message.reply_text(f"Запускаю обучение LoRA для «{av_name}»…")
    try:
        async with TRAIN_SEMAPHORE:
            training_id = await asyncio.to_thread(start_lora_training, uid, av_name)
        await update.effective_message.reply_text(f"Стартанула. ID: {training_id}\nПроверяй /trainstatus.")
        if DEST_OWNER and DEST_MODEL and training_id:
            await update.effective_message.reply_text(
                f"Логи: https://replicate.com/{DEST_OWNER}/{DEST_MODEL}/trainings/{training_id}"
            )
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
        await update.effective_message.reply_text(
            f"Статус «{av_name}»: {status}… {('Логи: ' + train_url) if train_url else ''}"
        ); return

    if status in ("failed","canceled"):
        msg = f"⚠️ Тренировка «{av_name}»: {status.upper()}."
        if err: msg += f"\nПричина: {err}"
        if train_url: msg += f"\nЛоги: {train_url}"
        msg += ("\n\nПроверь: целевая модель существует; 10 фото без фильтров; кредиты Replicate; правильные LORA_* env.")
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
    gender = (prof.get("gender") or "female").lower()
    natural = prof.get("natural", True)
    pretty  = prof.get("pretty", False)

    desired_comp = meta.get("comp","half")
    if CONSISTENT_SCALE and desired_comp not in ("half","closeup"):
        desired_comp = "half"

    comp_text, (w,h) = _comp_text_and_size(desired_comp)
    tone_text = _tone_text(meta.get("tone","daylight"))
    theme_boost = THEME_BOOST.get(preset, "")
    prompt_core, gender_negative = build_prompt(meta, gender, comp_text, tone_text, theme_boost, natural, pretty)
    model_slug = _pinned_slug(av)

    guidance = max(3.8, min(4.4, SCENE_GUIDANCE.get(preset, GEN_GUIDANCE)))
    steps = min(MAX_STEPS, 46 if natural else max(48, GEN_STEPS))

    await update.effective_message.chat.send_action(ChatAction.UPLOAD_PHOTO)
    desc = meta.get("desc", preset)
    await update.effective_message.reply_text(f"🎬 {preset}\nАватар: {av_name}\n{desc}\n\nГенерирую ({gender}, {w}×{h}) …")

    try:
        refs = list_ref_images(uid, av_name)
        if not refs: raise RuntimeError("Нет рефов.")
        face_key = refs[0]
        face_local = STORAGE.get_local_copy(face_key)

        neg_base = _neg_with_gender(NEGATIVE_PROMPT_BASE, gender_negative)
        variants = [("lock", random.randrange(2**32)), ("lock", random.randrange(2**32)), ("plain", random.randrange(2**32))]

        async with GEN_SEMAPHORE:
            for mode, s in variants:
                use_lock = (mode == "lock") and (av.get("lockface") is not False) and INSTANTID_SLUG
                if use_lock:
                    inst_steps = min(MAX_STEPS, max(38, steps))
                    if INSTANTID_FORCE_TWOSTEP:
                        base_url = await asyncio.to_thread(
                            generate_from_finetune, model_slug=model_slug, prompt=prompt_core,
                            steps=steps, guidance=guidance, seed=random.randrange(2**32),
                            w=w, h=h, negative_prompt=neg_base
                        )
                        base_bytes = _retry(requests.get, base_url, timeout=HTTP_TIMEOUT, label="get_base").content
                        url = await asyncio.to_thread(
                            generate_with_instantid, face_local_path=face_local, prompt=prompt_core,
                            steps=inst_steps, guidance=guidance, seed=s, w=w, h=h,
                            negative_prompt=neg_base, natural=natural,
                            content_image_bytes=base_bytes
                        )
                    else:
                        try:
                            url = await asyncio.to_thread(
                                generate_with_instantid, face_local_path=face_local, prompt=prompt_core,
                                steps=inst_steps, guidance=guidance, seed=s, w=w, h=h,
                                negative_prompt=neg_base, natural=natural
                            )
                        except Exception as e:
                            if "INSTANTID_NEEDS_IMAGE" in str(e):
                                base_url = await asyncio.to_thread(
                                    generate_from_finetune, model_slug=model_slug, prompt=prompt_core,
                                    steps=steps, guidance=guidance, seed=random.randrange(2**32),
                                    w=w, h=h, negative_prompt=neg_base
                                )
                                base_bytes = _retry(requests.get, base_url, timeout=HTTP_TIMEOUT, label="get_base").content
                                url = await asyncio.to_thread(
                                    generate_with_instantid, face_local_path=face_local, prompt=prompt_core,
                                    steps=inst_steps, guidance=guidance, seed=s, w=w, h=h,
                                    negative_prompt=neg_base, natural=natural,
                                    content_image_bytes=base_bytes
                                )
                            else:
                                raise
                else:
                    url = await asyncio.to_thread(
                        generate_from_finetune, model_slug=model_slug, prompt=prompt_core,
                        steps=steps, guidance=guidance, seed=s, w=w, h=h, negative_prompt=neg_base
                    )
                tag = "🔒" if use_lock else "◻️"
                await update.effective_message.reply_photo(photo=url, caption=f"{preset} • {av_name} • {tag}")

        # чистим temp при S3
        with contextlib.suppress(Exception):
            if face_local.exists() and str(face_local).startswith(tempfile.gettempdir()):
                face_local.unlink()

        await update.effective_message.reply_text("Готово. Если какой-то пресет «плывёт», скажи его имя — притяну гайки именно для него.")

    except Exception as e:
        logging.exception("generation failed")
        await update.effective_message.reply_text(f"Ошибка генерации: {e}")

# --- Toggles (пер-юзер) ---
async def pretty_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    prof = load_profile(uid)
    prof["pretty"] = not prof.get("pretty", False)
    if prof["pretty"]:
        prof["natural"] = True
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

    # Пинг слагов
    _check_slug(LORA_TRAINER_SLUG, "LoRA trainer")
    if INSTANTID_SLUG:
        _check_slug(INSTANTID_SLUG, "InstantID")
    else:
        logger.info("InstantID disabled (no INSTANTID_SLUG).")

    logger.info(
        "Bot up. Trainer=%s DEST=%s GEN=%dx%d steps=%s guidance=%s ConsistentScale=%s Storage=%s",
        LORA_TRAINER_SLUG, f"{DEST_OWNER}/{DEST_MODEL}", GEN_WIDTH, GEN_HEIGHT,
        GEN_STEPS, GEN_GUIDANCE, CONSISTENT_SCALE, "S3" if USE_S3 else "FS"
    )
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
