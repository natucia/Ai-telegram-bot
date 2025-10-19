# === Telegram LoRA Bot (Flux LoRA trainer + Redis persist 
# + Identity/Gender locks — NO InstantID, pure LoRA 
# + MULTI-AVATARS + NATURAL/Pretty per-user + CONSISTENT FACE SCALE 
# + S3 storage + concurrency limits + retries 
# + STRICT WAIST-UP ONLY (no full body) 
# + SUBJECT TOKEN + THEME BOOST SANITIZER + IDENTITY-SAFE MODE 
# + AVATAR-SCOPED GENDER + NO-COMMAND UX FOR AVATARS 
# + FACE ID ADAPTER (cheese.ai style) ===

from typing import Any, Dict, List, Optional, Tuple, Iterable
from typing import Union, IO
Style = Dict[str, Any]

from styles import (
    STYLE_PRESETS, STYLE_CATEGORIES, THEME_BOOST, SCENE_GUIDANCE, RISKY_PRESETS
)
import os, re, io, json, time, asyncio, logging, shutil, random, contextlib, tempfile, hashlib, base64
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED
import requests
import replicate
from replicate import Client
from PIL import Image, ImageOps, ImageFilter
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, CallbackQueryHandler, filters
)
import struct
from telegram import ReplyKeyboardMarkup

import logging
logger = logging.getLogger()


# === FACEID WORKFLOW: robust import + adapter ===
try:
    from faceid_workflow_integration import start_generation_for_preset as _wf_start_generation_for_preset
    FACEID_WORKFLOW_AVAILABLE = True
except Exception as _imp_err:
    _wf_start_generation_for_preset = None
    FACEID_WORKFLOW_AVAILABLE = False
    logger.warning("FaceID workflow integration unavailable: %s", _imp_err)


async def start_generation_via_workflow(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        preset: str,
        show_intro: bool = False,
        avatar_name: Optional[str] = None,   # можно не передавать; если передали — фиксируем аватар
    ):
        """
        Workflow-ветка с шимом lora_url:
        - если у аватара нет HTTPS .safetensors, но есть slug версии модели,
          кладём в av['lora_url'] строку 'slug:<owner/model:version>' и считаем это валидным.
        - face_ref (presigned HTTPS / локальный путь) обновляем при необходимости.
        - при желании можно фиксировать конкретный аватар через avatar_name.
        """
        # 0) наличие WF
        if not (_wf_start_generation_for_preset and FACEID_WORKFLOW_AVAILABLE):
            raise RuntimeError("FaceID workflow не подключён или не экспортирует start_generation_for_preset")

        uid = update.effective_user.id
        prof = load_profile(uid)
        av_name = avatar_name or get_current_avatar_name(prof)

        # 1) подтянем lora_url/face_url из профиля/Replicate
        lora_url, face_url = await asyncio.to_thread(recover_lora_and_face_urls, uid, av_name)

        # 1a) ШИМ: если нет валидного .safetensors, но есть slug -> записываем 'slug:<...>'
        prof2 = load_profile(uid)
        av = get_avatar(prof2, av_name)

        def _has_https_weights(u: Optional[str]) -> bool:
            return isinstance(u, str) and u.startswith(("http://", "https://")) and u.endswith(".safetensors")

        if not _has_https_weights(av.get("lora_url")):
            base = av.get("finetuned_model") or ""
            ver  = av.get("finetuned_version")
            # аккуратно соберём slug (если ver уже в формате <model:ver-id> — не дублируем двоеточие)
            model_slug = f"{base}:{ver}" if (base and ver and ":" not in (ver or "")) else (base or "")
            if model_slug:
                av["lora_url"] = f"slug:{model_slug}"
                save_profile(uid, prof2)  # сохраним, чтобы WF увидел

        # 2) если face_url пуст/протух — обновим
        def _is_https(u: str) -> bool:
            return isinstance(u, str) and u.startswith(("http://", "https://"))

        def _alive(u: str) -> bool:
            try:
                r = requests.head(u, timeout=8, allow_redirects=True)
                if r.status_code == 200:
                    return True
                if r.status_code in (401, 403, 405):
                    r2 = requests.get(u, timeout=8, stream=True)
                    return r2.status_code == 200
            except Exception:
                pass
            return False

        if not face_url or (_is_https(face_url) and not _alive(face_url)):
            try:
                face_url = await asyncio.to_thread(prepare_face_embedding, uid, av_name)
                prof = load_profile(uid)  # перечитаем профиль на всякий
            except Exception as e:
                logger.warning("refresh face embedding failed: %s", e)

        # 3) финальная валидация источников для WF
        av = get_avatar(load_profile(uid), av_name)
        lu = av.get("lora_url") or ""
        lora_source = (
            "weights" if _has_https_weights(lu)
            else ("slug" if (isinstance(lu, str) and lu.startswith("slug:")) else "none")
        )
        face_kind = (
            "https" if (_is_https(face_url or "")) else
            ("file" if (isinstance(face_url, str) and os.path.exists(face_url)) else "none")
        )
        ok_lora = lora_source in ("weights", "slug")
        ok_face = face_kind in ("https", "file")

        logger.info(
            "WORKFLOW DISPATCH: preset=%s avatar=%s lora_ok=%s(%s) face_ref=%s",
            preset, av_name, ok_lora, lora_source, face_kind
        )

        if not ok_lora:
            raise RuntimeError("Не найдена LoRA: ни HTTPS .safetensors, ни slug версии модели.")
        if not ok_face:
            raise RuntimeError("Нет валидного FaceID-референса (ни presigned HTTPS, ни локального файла).")

        # 4) запуск WF — передаём колбэки профиля; фиксируем аватар, если его передали в аргумент
        return await _wf_start_generation_for_preset(
            update, context, preset,
            STYLE_PRESETS,
            load_profile,
            (lambda _prof: av_name) if avatar_name else get_current_avatar_name,
            get_avatar,
        )






def _stable_seed(*parts:str) -> int:
    h = hashlib.sha1(("::".join(parts)).encode("utf-8")).digest()
    return struct.unpack(">Q", h[:8])[0] & 0xFFFFFFFF

# ---------- ENV ----------
TOKEN = os.getenv("BOT_TOKEN", "")
PHOTO_COUNTER_MSG_ID: Dict[Tuple[int, str], int] = {}  # (uid, avatar) -> msg_id последнего сообщения-счётчика

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

FACEID_WORKFLOW_FORCE_OFF = os.getenv("FACEID_WORKFLOW_FORCE_OFF", "0").lower() in ("1", "true", "yes", "y")

# Пол (опц.) — автоинференс
GENDER_MODEL_SLUG = os.getenv("GENDER_MODEL_SLUG", "nateraw/vit-age-gender").strip()

# --- Face ID Adapter ---
FACE_ID_ADAPTER_ENABLED = os.getenv("FACE_ID_ADAPTER_ENABLED", "true").lower() in ("1","true","yes","y")
FACE_ID_MODEL_SLUG = os.getenv("FACE_ID_MODEL_SLUG", "lucataco/ip-adapter-faceid").strip()
FACE_ID_WEIGHT = float(os.getenv("FACE_ID_WEIGHT", "0.8"))
FACE_ID_NOISE = float(os.getenv("FACE_ID_NOISE", "0.1"))
FACE_ID_SCALE = float(os.getenv("FACE_ID_SCALE", "0.9"))

# --- Параметры обучения ---
LORA_MAX_STEPS = int(os.getenv("LORA_MAX_STEPS", "1400"))
LORA_LR = float(os.getenv("LORA_LR", "0.0006"))
LORA_USE_FACE_DET = os.getenv("LORA_USE_FACE_DET", "true").lower() in ("1","true","yes","y")
LORA_RESOLUTION = int(os.getenv("LORA_RESOLUTION", "1024"))

DEFAULT_FEMALE_CAPTION = (
    "a high quality photo of the same woman, natural brows, medium-length hair with natural hairline, "
    "brown eyes, neutral relaxed expression, balanced facial proportions, natural lip shape, "
    "no retouch, true-to-life skin texture"
)

DEFAULT_MALE_CAPTION = (
    "a high quality photo of the same man, natural brows, short hair or neat hairstyle, "
    "brown eyes, neutral relaxed expression, balanced facial proportions, natural lip shape, "
    "no retouch, true-to-life skin texture"
)

# --- Генерация ---
GEN_STEPS = int(os.getenv("GEN_STEPS", "44"))
GEN_GUIDANCE = float(os.getenv("GEN_GUIDANCE", "4.6"))
GEN_WIDTH = int(os.getenv("GEN_WIDTH", "896"))
GEN_HEIGHT = int(os.getenv("GEN_HEIGHT", "1152"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "50"))

# --- CONSISTENT FACE SCALE ---
CONSISTENT_SCALE = os.getenv("CONSISTENT_SCALE", "1").lower() in ("1","true","yes","y")
HEAD_HEIGHT_FRAC = float(os.getenv("HEAD_HEIGHT_FRAC", "0.42"))
HEAD_WIDTH_FRAC = float(os.getenv("HEAD_WIDTH_FRAC", "0.32"))

# --- Composition policy ---
FORCE_WAIST_UP = os.getenv("FORCE_WAIST_UP", "1").lower() in ("1","true","yes","y")
ALLOW_SEATED = os.getenv("ALLOW_SEATED", "1").lower() in ("1","true","yes","y")

# ---- Anti-drift / anti-wide-face ----
NEGATIVE_PROMPT_BASE = (
    "cartoon, anime, cgi, 3d render, stylized, illustration, digital painting, painterly, brush strokes, "
    "vector art, smooth shading, plastic skin, overprocessed, airbrushed, beauty-filter, "
    "lowres, blurry, textureless skin, porcelain skin, waxy, gaussian blur, smoothing filter, "
    "text, watermark, logo, bad anatomy, extra fingers, different person, identity drift, face swap, "
    "ethnicity change, skin tone change, undertone shift, tanning effect, bleaching, age change, hairline change, "
    "distorted proportions, vertical face elongation, face slimming, stretched chin, narrow jaw, "
    "lens distortion, fisheye, warping, stretched face, perspective distortion, "
    "plain selfie, flash photo, harsh shadows, denoise artifacts, over-sharpened, waxy highlight roll-off, "
    "skin smoothing, porcelain texture, HDR glamour, excessive clarity"
)

NO_FULL_BODY_NEG = (
    "no full body, no head-to-toe, no feet, no shoes in frame, "
    "no legs below waist, no knees visible, avoid distant camera, "
    "avoid tiny face, avoid wide background composition"
)

AESTHETIC_SUFFIX = (
    ", RAW photograph, DSLR photo, 85mm lens, shallow depth of field, true-to-life color, "
    "soft filmic contrast, natural white balance, visible fine skin pores and micro-texture, "
    "subtle film grain, micro-sharpen on eyes only, realistic lens response"
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
    "subtle beauty retouch, even skin tone, faint under-eye smoothing, "
    "gentle softening around nasolabial area, slight glow, tidy eyebrows"
)

PRETTY_NEG = (
    "over-smoothing, harsh pores, deep nasolabial folds, oily hotspot shine, oversharpened skin, beauty filter"
)

PRETTY_COMP_HINT = "camera slightly above eye level, flattering portrait angle"

# ---------- FACIAL RELAX ----------
FACIAL_RELAX_POS = (
    "relaxed facial muscles, no jaw clenching, relaxed masseter, "
    "subtle nasolabial area, softened smile lines, gentle mouth corners, "
    "no frown lines between eyebrows"
)

FACIAL_RELAX_NEG = (
    "jaw clenching, tensed masseter muscles, deep nasolabial folds, "
    "marionette lines, harsh smile lines, emphasized wrinkles, grimace"
)

# ---------- logging ----------
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("bot")

# ---------- Concurrency & retries ----------
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

def _download_image_bytes(url: str, tries: int = 3, timeout: float = HTTP_TIMEOUT) -> bytes:
    headers = {"User-Agent": "TelegramLoRABot/1.0"}
    last_exc = None
    for i in range(tries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, stream=True)
            r.raise_for_status()
            return r.content if r.content else r.raw.read()
        except Exception as e:
            last_exc = e
            time.sleep(1.6 ** i)
    raise RuntimeError(f"download failed for {url}: {last_exc}")

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
    s3_client = session.client("s3", endpoint_url=S3_ENDPOINT_URL)

def _s3_key(*parts: str) -> str:
    p = "/".join(str(x).strip("/").replace("//","/") for x in parts if x is not None)
    return f"{S3_PREFIX}/{p}".strip("/") if S3_PREFIX else p

def tmp_path(suffix=".jpg") -> Path:
    return Path(tempfile.mkstemp(prefix="bot_", suffix=suffix)[1])

def _downscale_like_camera(im: Image.Image, target_max=1152) -> Image.Image:
    w, h = im.size
    if max(w, h) <= target_max:
        return im
    scale = target_max / float(max(w, h))
    nw, nh = int(w * scale), int(h * scale)
    return im.resize((nw, nh), Image.Resampling.LANCZOS)

def _photo_look(im: Image.Image) -> Image.Image:
    im = _downscale_like_camera(im, 1152)
    im = im.filter(ImageFilter.UnsharpMask(radius=1.0, percent=60, threshold=6))
    noise = Image.effect_noise(im.size, 3).convert("L").point(lambda p: int(p*0.08))
    im = Image.blend(im, Image.merge("RGB", (noise, noise, noise)), 0.08)
    return im

# --- FS utils (локальный fallback) ---
def _find_first_https_safetensors(data: Any) -> Optional[str]:
    def rec(x: Any) -> Optional[str]:
        if isinstance(x, str):
            if x.startswith(("http://", "https://")) and x.lower().endswith(".safetensors"):
                return x
            return None
        if isinstance(x, dict):
            # самые частые ключи
            for k in ("safetensors", "safetensors_url", "weights_url", "url"):
                v = x.get(k)
                got = rec(v)
                if got:
                    return got
            for v in x.values():
                got = rec(v)
                if got:
                    return got
        if isinstance(x, (list, tuple)):
            for v in x:
                got = rec(v)
                if got:
                    return got
        return None
    return rec(data)


def recover_lora_and_face_urls(uid: int, av_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Пытается заполнить av['lora_url'] (HTTPS .safetensors) и av['face_image_url'].
    Возвращает (lora_url, face_image_url), даже если они уже были.
    """
    prof = load_profile(uid); av = get_avatar(prof, av_name)
    lora_url = av.get("lora_url")
    face_url = av.get("face_image_url") or av.get("face_embedding")

    # 1) LORA URL из тренировки Replicate
    if not lora_url:
        tid = av.get("training_id")
        try:
            client = Client(api_token=os.environ["REPLICATE_API_TOKEN"])
            tr = client.trainings.get(tid) if tid else None
            # достаём из output любую https .safetensors
            if tr is not None:
                output = getattr(tr, "output", None)
                if isinstance(tr, dict):
                    output = tr.get("output")
                cand = _find_first_https_safetensors(output)
                if cand:
                    lora_url = cand
        except Exception as e:
            logger.warning("recover_lora_and_face_urls: fetch training output failed: %s", e)

        # Если всё ещё нет — попробуем из destination модели (иногда trainer кладёт ссылку в model card output)
        if not lora_url:
            try:
                dest = av.get("finetuned_model")
                if dest:
                    model_obj = replicate.models.get(dest)
                    versions = list(model_obj.versions.list())
                    if versions:
                        v0 = versions[0]
                        # Некоторые тренеры дублируют ссылки в полях версии; проверим как строку/словарь
                        lora_url = _find_first_https_safetensors(getattr(v0, "openapi_schema", {}) or {})
                        if not lora_url:
                            lora_url = _find_first_https_safetensors(getattr(v0, "__dict__", {}))
            except Exception as e:
                logger.warning("recover_lora_and_face_urls: scan model version failed: %s", e)

        if lora_url:
            av["lora_url"] = lora_url

    # 2) FACE IMAGE URL: если нет — используем уже подготовленный presigned из main.prepare_face_embedding
    if not face_url and av.get("face_embedding"):
        face_url = av["face_embedding"]
        av["face_image_url"] = face_url

    save_profile(uid, prof)
    return lora_url, face_url




def user_dir(uid:int) -> Path:
    p = DATA_DIR / str(uid); p.mkdir(parents=True, exist_ok=True); return p

def avatars_root(uid:int) -> Path:
    p = user_dir(uid) / "avatars"; p.mkdir(parents=True, exist_ok=True); return p

def avatar_dir(uid:int, avatar:str) -> Path:
    p = avatars_root(uid) / avatar; p.mkdir(parents=True, exist_ok=True); return p

def profile_path(uid:int) -> Path:
    return user_dir(uid) / "profile.json"

# ---------- Память/Состояния для UX без команд ----------
PENDING_NEW_AVATAR: Dict[int, bool] = {}
PENDING_DELETE_AVATAR: Dict[int, Optional[str]] = {}
ENROLL_FLAG: Dict[Tuple[int,str],bool] = {}

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
            if adir.exists():
                shutil.rmtree(adir)

    def get_local_copy(self, key:str) -> Path:
        return Path(key)

    def pack_refs_zip(self, uid:int, avatar:str) -> Path:
        refs = self.list_ref_images(uid, avatar)
        if len(refs) < 10:
            raise RuntimeError("Нужно 10 фото для обучения.")
        zpath = avatar_dir(uid, avatar) / "train.zip"
        with ZipFile(zpath, "w", compression=ZIP_STORED) as z:
            for i, kp in enumerate(refs, 1):
                z.write(Path(kp), arcname=f"img_{i:02d}.jpg")
        return zpath

class S3Storage(Storage):
    def save_ref_image(self, uid:int, avatar:str, raw:bytes) -> str:
        if not s3_client:
            raise RuntimeError("S3 не инициализирован.")
        buf = io.BytesIO()
        im = Image.open(io.BytesIO(raw))
        im = ImageOps.exif_transpose(im).convert("RGB")
        im = ImageOps.contain(im, (1024,1024))
        im.save(buf, "JPEG", quality=92); buf.seek(0)
        key = _s3_key("profiles", str(uid), "avatars", avatar, f"ref_{int(time.time()*1000)}.jpg")
        _retry(s3_client.put_object, Bucket=S3_BUCKET, Key=key, Body=buf.getvalue(), ContentType="image/jpeg", label="s3_put")
        return f"s3://{S3_BUCKET}/{key}"

    def list_ref_images(self, uid:int, avatar:str) -> List[str]:
        if not s3_client:
            return []
        prefix = _s3_key("profiles", str(uid), "avatars", avatar, "ref_")
        keys: List[str] = []
        cont = None
        while True:
            resp = _retry(s3_client.list_objects_v2, Bucket=S3_BUCKET, Prefix=prefix, ContinuationToken=cont, label="s3_list") \
                if cont else _retry(s3_client.list_objects_v2, Bucket=S3_BUCKET, Prefix=prefix, label="s3_list")
            for it in resp.get("Contents", []):
                keys.append(f"s3://{S3_BUCKET}/{it['Key']}")
            if not resp.get("IsTruncated"):
                break
            cont = resp.get("NextContinuationToken")
        return sorted(keys)

    def delete_avatar(self, uid:int, avatar:str):
        prefix = _s3_key("profiles", str(uid), "avatars", avatar)
        cont = None
        while True:
            resp = _retry(s3_client.list_objects_v2, Bucket=S3_BUCKET, Prefix=prefix, ContinuationToken=cont, label="s3_list") \
                if cont else _retry(s3_client.list_objects_v2, Bucket=S3_BUCKET, Prefix=prefix, label="s3_list")
            keys = [ {"Key": o["Key"]} for o in resp.get("Contents", []) ]
            if keys:
                _retry(s3_client.delete_objects, Bucket=S3_BUCKET, Delete={"Objects": keys}, label="s3_del")
            if not resp.get("IsTruncated"):
                break
            cont = resp.get("NextContinuationToken")

    def get_local_copy(self, key:str) -> Path:
        if not key.startswith("s3://"):
            raise RuntimeError("Ожидался s3:// ключ")
        _, _, bucket_and_key = key.partition("s3://")
        bucket, _, obj_key = bucket_and_key.partition("/")
        path = tmp_path(".jpg")
        _retry(s3_client.download_file, bucket, obj_key, str(path), label="s3_download")
        return path

    def pack_refs_zip(self, uid:int, avatar:str) -> Path:
        refs = self.list_ref_images(uid, avatar)
        if len(refs) < 10:
            raise RuntimeError("Нужно 10 фото для обучения.")
        zpath = Path(tempfile.mkstemp(prefix="train_", suffix=".zip")[1])
        with ZipFile(zpath, "w", compression=ZIP_STORED) as z:
            for i, key in enumerate(refs, 1):
                lp = self.get_local_copy(key)
                z.write(lp, arcname=f"img_{i:02d}.jpg")
        return zpath

STORAGE: Storage = S3Storage() if USE_S3 else FSStorage()
logger.info("Storage backend: %s", "S3" if USE_S3 else "FS")

# ---------- профили (Redis/FS) ----------
DEFAULT_AVATAR = {
    "images": [],
    "training_id": None,
    "finetuned_model": None,
    "finetuned_version": None,
    "status": None,
    "lockface": True,
    "token": None,
    "gender": None, # <-- пол хранится на уровне аватара
    "face_embedding": None, # <-- Face ID embedding
}

DEFAULT_PROFILE = {
    "gender": None, # глобальный (бэкап), но не обязателен
    "natural": True,
    "pretty": False,
    "current_avatar": "default",
    "avatars": {"default": DEFAULT_AVATAR.copy()}
}

def _save_ref_downscaled_local(path: Path, raw: bytes, max_side=1024, quality=92):
    im = Image.open(io.BytesIO(raw))
    im = ImageOps.exif_transpose(im).convert("RGB")
    im = ImageOps.contain(im, (max_side, max_side))
    im.save(path, "JPEG", quality=quality)

def _avatar_token(uid:int, avatar:str) -> str:
    raw = f"{uid}:{avatar}".encode("utf-8")
    h = hashlib.sha1(raw).digest()
    return "yt_" + base64.b32encode(h)[:6].decode("ascii").lower()

def get_current_avatar_name(prof:Dict[str,Any]) -> str:
    name = prof.get("current_avatar") or "default"
    if name not in prof["avatars"]:
        name = "default"; prof["current_avatar"] = name
    return name

def get_avatar(prof:Dict[str,Any], name:Optional[str]=None) -> Dict[str,Any]:
    if not name:
        name = get_current_avatar_name(prof)
    if name not in prof["avatars"]:
        prof["avatars"][name] = DEFAULT_AVATAR.copy()
    av = prof["avatars"][name]
    if not av.get("token"):
        uid_hint = prof.get("_uid_hint", 0)
        av["token"] = _avatar_token(uid_hint, name)
    return av

def list_ref_images(uid:int, avatar:str) -> List[str]:
    return STORAGE.list_ref_images(uid, avatar)

def _migrate_single_to_multi(uid:int, prof:Dict[str,Any]) -> Dict[str,Any]:
    if "avatars" in prof:
        return prof
    migrated = DEFAULT_PROFILE.copy()
    migrated["gender"] = prof.get("gender")
    default = DEFAULT_AVATAR.copy()
    default["training_id"] = prof.get("training_id")
    default["finetuned_model"] = prof.get("finetuned_model")
    default["finetuned_version"] = prof.get("finetuned_version")
    default["status"] = prof.get("status")
    default["lockface"] = prof.get("lockface", True)
    default["gender"] = prof.get("gender") # переносим в аватар как начальное значение
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

# --- Итерация по всем профилям (для фонового поллера) ---


def load_all_profiles() -> Dict[int, Dict[str, Any]]:
        """
        Возвращает {uid: profile} — именно dict, не список.
        Работает и с Redis, и с файловым хранилищем.
        """
        out: Dict[int, Dict[str, Any]] = {}

        if _redis:
            try:
                for key in _redis.scan_iter(match="profile:*", count=100):
                    try:
                        uid = int(str(key).split(":", 1)[1])
                    except Exception:
                        continue
                    raw = _redis.get(key)
                    if not raw:
                        continue
                    try:
                        prof = {**DEFAULT_PROFILE, **json.loads(raw)}
                        prof = _migrate_single_to_multi(uid, prof)
                        out[uid] = prof
                    except Exception as e:
                        logger.warning("load_all_profiles[redis] skip %r: %s", key, e)
            except Exception as e:
                logger.warning("load_all_profiles[redis] failed: %s", e)
        else:
            try:
                if DATA_DIR.exists():
                    for p in DATA_DIR.iterdir():
                        if not p.is_dir():
                            continue
                        try:
                            uid = int(p.name)
                        except Exception:
                            continue
                        prof_path = p / "profile.json"
                        if not prof_path.exists():
                            continue
                        try:
                            prof = {**DEFAULT_PROFILE, **json.loads(prof_path.read_text())}
                            prof = _migrate_single_to_multi(uid, prof)
                            out[uid] = prof
                        except Exception as e:
                            logger.warning("load_all_profiles[fs] skip %s: %s", prof_path, e)
            except Exception as e:
                logger.warning("load_all_profiles[fs] failed: %s", e)

        return out

def delete_profile(uid:int):
    if _redis:
        with contextlib.suppress(Exception):
            _redis.delete(f"profile:{uid}")
    prof = load_profile(uid)
    for name in list(prof.get("avatars", {}).keys()):
        if name == "default":
            continue
        STORAGE.delete_avatar(uid, name)
    STORAGE.delete_avatar(uid, "default")
    p = user_dir(uid)
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)

# ---------- авто-пол + кэш версий ----------
_MODEL_VER_CACHE: Dict[str, str] = {}

def resolve_model_version(slug: str) -> str:
    if ":" in slug:
        return slug
    if slug in _MODEL_VER_CACHE:
        return f"{slug}:{_MODEL_VER_CACHE[slug]}"
    model = replicate.models.get(slug)
    versions = list(model.versions.list())
    if not versions:
        raise RuntimeError(f"Нет версий модели {slug}")
    _MODEL_VER_CACHE[slug] = versions[0].id
    return f"{slug}:{versions[0].id}"

def extract_any_url(out: Any) -> Optional[str]:
    if isinstance(out, str) and out.startswith(("http","https")):
        return out
    if isinstance(out, list):
        for v in out:
            u = extract_any_url(v)
            if u:
                return u
    if isinstance(out, dict):
        for v in out.values():
            u = extract_any_url(v)
            if u:
                return u
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
        if "female" in g or "woman" in g:
            return "female"
        if "male" in g or "man" in g:
            return "male"
    except Exception as e:
        logger.warning("Gender inference error: %s", e)
    return None

def auto_detect_gender(uid:int, avatar: Optional[str]=None) -> str:
    prof = load_profile(uid)
    av_name = avatar or get_current_avatar_name(prof)
    refs = list_ref_images(uid, av_name)
    if not refs:
        return "female"
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

# ---------- Face ID Adapter ----------
def extract_face_embedding(image_path: Path) -> Optional[str]:
    """Извлекает Face ID embedding с помощью модели IP-Adapter FaceID"""
    try:
        client = Client(api_token=os.environ["REPLICATE_API_TOKEN"])
        version_slug = resolve_model_version(FACE_ID_MODEL_SLUG)

        with open(image_path, "rb") as img_file:
            prediction = client.predictions.create(
                version=version_slug,
                input={
                    "image": img_file,
                    "scale": FACE_ID_SCALE,
                    "prompt": "face",  # Минимальный промпт для извлечения эмбеддинга
                    "num_outputs": 1,
                    "num_inference_steps": 20,
                    "seed": 42
                }
            )

        prediction.wait()

        # IP-Adapter FaceID обычно возвращает эмбеддинг в output
        if prediction.output and isinstance(prediction.output, list):
            # Сохраняем URL первого изображения как репрезентацию эмбеддинга
            return prediction.output[0]
        elif prediction.output and isinstance(prediction.output, str):
            return prediction.output

    except Exception as e:
        logger.warning("Face ID embedding extraction failed: %s", e)

    return None

def prepare_face_embedding(uid: int, avatar: str) -> Optional[str]:
        """Кладём в профиль ссылку на ЛУЧШЕЕ фото лица.
        Для S3 — presigned HTTPS, для FS — путь к файлу (мы его потом откроем)."""
        refs = list_ref_images(uid, avatar)
        if not refs:
            logger.warning("prepare_face_embedding: нет референсов")
            return None

        key = refs[0]
        url: Optional[str] = None

        if key.startswith("s3://"):
            if not s3_client:
                logger.warning("prepare_face_embedding: нет s3_client")
                return None
            _, _, bucket_and_key = key.partition("s3://")
            bucket, _, obj_key = bucket_and_key.partition("/")
            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": obj_key},
                ExpiresIn=int(os.getenv("FACE_ID_URL_TTL", "86400"))
            )
        else:
            # FS — просто путь на диск
            url = key

        prof = load_profile(uid)
        av = get_avatar(prof, avatar)
        av["face_embedding"] = url
        save_profile(uid, prof)
        logger.info("prepare_face_embedding: set face_embedding=%r", url)
        return url



# ---------- Промпт-замки ----------
def _beauty_guardrail() -> str:
    return ("exact facial identity, identity preserved, balanced facial proportions, symmetrical face, natural oval, soft jawline, "
            "keep original cheekbone width and jaw width, do not widen face, "
            "style must only affect clothing, background and lighting, not facial features")

def _face_lock() -> str:
    return ("keep same bone structure, natural interocular distance, consistent eyelid shape, "
            "aligned pupils, preserve cheekbone width and lip fullness")

def _oval_lock() -> str:
    return ("keep the same facial oval as in the training photos, "
            "no vertical face elongation, no face slimming, "
            "no stretched or lengthened chin, no narrowed or widened jaw, "
            "preserve original jawline curvature and cheekbone width")

def _ethnicity_lock() -> str:
    return (
        "preserve the same ethnicity as in the training photos, "
        "do not alter eye shape or eyelid crease presence, "
        "preserve inner/outer canthus angle (no canthal tilt change), "
        "preserve nasal bridge shape and width, "
        "preserve lip thickness proportions and philtrum length"
    )

def _anti_distort() -> str:
    return "no fisheye, no lens distortion, no warping, natural perspective, proportional head size"

def _frontal_lock() -> str:
    return ("frontal face, facing camera, eyes looking into the lens, "
            "head tilt under 3 degrees, no three-quarter, no profile, "
            "neutral relaxed expression, no exaggerated smile, no pursed lips, "
            "ears symmetric, pupils aligned")

def _head_scale_lock() -> str:
    hh = int(HEAD_HEIGHT_FRAC * 100); hw = int(HEAD_WIDTH_FRAC * 100)
    return (f"keep constant head scale, head height about {hh}% of frame and head width about {hw}% of frame, "
            "do not zoom, do not crop the forehead or chin, stable camera distance")

def _gender_lock(gender:str) -> Tuple[str, str]:
    if gender == "male":
        return "male man, masculine facial features, light stubble allowed", "female, woman heavy makeup"
    return "female woman, feminine facial features", "male, man, beard, stubble, mustache, adam's apple"

def _safe_portrait_size(w:int, h:int) -> Tuple[int,int]:
    ar = w / max(1, h)
    if ar >= 0.75:
        return int(h*0.66), h # ~2:3
    return w, h

def _face_scale_hint() -> str:
    if not CONSISTENT_SCALE:
        return ""
    hh = int(HEAD_HEIGHT_FRAC * 100); hw = int(HEAD_WIDTH_FRAC * 100)
    return (f"keep constant head scale across all images, head height ~{hh}% of frame from chin to top of head, "
            f"head width ~{hw}% of frame width, subject centered, do not zoom in or out")

# === Композиции (строго без full-body) ===
def _variants_for_preset(meta: Style) -> List[str]:
    comps = meta.get("comps")
    if isinstance(comps, list) and comps:
        comps = [("half" if c == "full" else c) for c in comps]
        comps = [c for c in comps if c in ("half","closeup")]
        if comps:
            return comps
    return ["half", "half", "closeup"]

def _maybe_seated_hint() -> str:
    return "subject seated on a chair or sofa, relaxed posture" if ALLOW_SEATED and random.random() < 0.6 else ""

def _comp_text_and_size(comp: str) -> Tuple[str, Tuple[int,int]]:
    scale_txt = _face_scale_hint()
    if comp == "closeup":
        w, h = _safe_portrait_size(896, 1152); seated = _maybe_seated_hint()
        return (f"portrait from chest up (no waist), shoulders fully in frame, {seated} "
                f"camera at eye level, 85mm look, {scale_txt}".strip(), (w, h))
    w, h = _safe_portrait_size(GEN_WIDTH, max(GEN_HEIGHT, 1344)); seated = _maybe_seated_hint()
    return (f"half body from waist up (include waist), hands may appear near frame edges, {seated} "
            f"camera at chest level, slight downward angle, 85mm look, {scale_txt}".strip(), (w, h))

def _comp_negatives(kind: str) -> str:
    base = NO_FULL_BODY_NEG if FORCE_WAIST_UP else ""
    if kind == "half":
        return (base + ", no chest-up tight crop, avoid cropping above waist").strip(", ")
    if kind == "closeup":
        return (base + ", no waist-up framing, no hands in frame, avoid showing torso").strip(", ")
    return (base + ", no chest-up crop, no waist-up crop").strip(", ")

def _tone_text(tone: str) -> str:
    return {
        "daylight": "soft natural daylight, neutral colors",
        "warm": "golden hour warmth, gentle highlights", 
        "cool": "cool cinematic light, clean color balance",
        "noir": "high contrast noir lighting, subtle rim light",
        "neon": "neon signs, wet reflections, cinematic backlight, vibrant saturation",
        "candle": "warm candlelight, soft glow, volumetric rays",
    }.get(tone, "balanced soft lighting")

_FACE_KEYS = ("face","skin","jaw","cheek","nose","lips","eyes","eyelid","eyelash","makeup","freckles")

def _safe_theme_boost(txt:str) -> str:
    t = (txt or "").lower()
    if any(k in t for k in _FACE_KEYS):
        return ""
    return txt

def _style_lock(role:str, outfit:str, props:str, background:str, comp_hint:str) -> str:
    bits = [role or "", f"wearing {outfit}" if outfit else "", f"with {props}" if props else "", 
            f"background: {background}" if background else "", comp_hint, "unmistakable scene identity"]
    return ", ".join([b for b in bits if b])

def _inject_beauty(core_prompt: str, comp_text: str, natural: bool, pretty: bool) -> str:
    parts = [core_prompt]
    if natural:
        parts.append(NATURAL_POS)
    if pretty:
        parts.append(PRETTY_POS)
    if "eye level" in comp_text or "chest level" in comp_text:
        parts.append(PRETTY_COMP_HINT)
    return ", ".join(parts)

def build_prompt(meta: Style, gender: str, comp_text:str, tone_text:str, theme_boost:str, natural:bool, pretty:bool, avatar_token:str="") -> Tuple[str, str]:
    role = meta.get("role_f") if (gender=="female" and meta.get("role_f")) else meta.get("role","")
    if not role and meta.get("role_m") and gender=="male":
        role = meta.get("role_m","")
    outfit = meta.get("outfit_f") if (gender=="female" and meta.get("outfit_f")) else meta.get("outfit","")
    props = meta.get("props",""); bg = meta.get("bg","")
    subject_tag = (f"photo of person {avatar_token}" if avatar_token else "").strip()

    gpos, gneg = _gender_lock(gender)
    anti = _anti_distort()
    hair_lock = "keep original hair color tone and hairstyle family"

    common_bits = [
        subject_tag,
        tone_text,
        gpos,
        "same person as the training photos, no ethnicity change, exact facial identity +++",
        "photorealistic, realistic body proportions, natural fine skin texture, filmic look",
        "keep original facial proportions, same interocular distance and cheekbone width, preserve lip shape and beard density",
        "85mm lens portrait look",
        hair_lock,
        _frontal_lock(),
        _oval_lock(),
        _head_scale_lock(),
        _face_scale_hint(),
        anti,
        _beauty_guardrail(),
        _face_lock(),
        _ethnicity_lock(),
        FACIAL_RELAX_POS,
        theme_boost
    ]

    if role or outfit or props or bg:
        core = ", ".join([_style_lock(role, outfit, props, bg, comp_text)] + common_bits)
        core += ", the costume and background must clearly communicate the role; avoid plain portrait"
    else:
        base_prompt = meta.get("p", "")
        core = ", ".join([f"{base_prompt}, {comp_text}"] + common_bits)

    core = _inject_beauty(core, comp_text, natural, pretty)

    neg = gneg
    if natural:
        neg = (neg + ", " + NATURAL_NEG) if neg else NATURAL_NEG
    if pretty:
        neg = (neg + ", " + PRETTY_NEG) if neg else PRETTY_NEG
    neg = (neg + ", " + FACIAL_RELAX_NEG) if neg else FACIAL_RELAX_NEG

    return core, neg

IDENTITY_SAFE_NEG = (
    "no makeup change, no lip reshaping, no nose reshaping, "
    "no jawline reshaping, no cheekbone reshaping, no eyebrow reshaping"
)

def _identity_safe_tune(preset_key:str, guidance:float, comps:List[str]) -> Tuple[float, List[str], str]:
    if preset_key not in RISKY_PRESETS:
        return guidance, comps, ""
    g = min(guidance, 4.6)
    cc = ["closeup", "half", "closeup"]
    return g, cc, IDENTITY_SAFE_NEG

    # ---------- Инференс/генерация (исправлено) ----------
def _resolve_face_ref(uid: int, avatar: str) -> Optional[Union[str, IO[bytes]]]:
        """
        Возвращает ИЛИ presigned HTTPS URL (для S3), ИЛИ локальный путь (FS).
        Никаких bytes. Никаких "s3://".
        """
        prof = load_profile(uid)
        av = get_avatar(prof, avatar)

        # 1) Если ранее подготовили embedding — используем его (там уже presigned https или локальный путь)
        ref = av.get("face_embedding")
        if isinstance(ref, str):
            if ref.startswith(("http://", "https://")) and "X-Amz-Signature" in ref:
                return ref
            if os.path.exists(ref):
                return ref

        # 2) Иначе берём первое реф-фото из набора
        refs = list_ref_images(uid, avatar)
        if not refs:
            return None
        key = refs[0]

        # FS — путь
        if os.path.exists(key):
            return key

        # S3 — делаем presigned HTTPS
        if isinstance(key, str) and key.startswith("s3://"):
            if not s3_client:
                return None
            _, _, bucket_and_key = key.partition("s3://")
            bucket, _, obj_key = bucket_and_key.partition("/")
            return s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": obj_key},
                ExpiresIn=int(os.getenv("FACE_ID_URL_TTL", "86400"))
            )

        # Уже HTTPS (редкий случай) — прокатывает
        if isinstance(key, str) and key.startswith(("http://","https://")):
            return key

        return None

def _get_or_refresh_face_ref(uid: int, avatar: str) -> Optional[Union[str, IO[bytes]]]:
    """
    Возвращает рабочий источник для FaceID:
    - presigned HTTPS на лицо из av['face_embedding'], если живой
    - локальный путь (FS)
    - если нет/протух: пытается пересоздать через prepare_face_embedding(...)
    Возвращает URL/путь/файловый объект — то, что принимает Replicate SDK.
    """
    # 1) пробуем существующее значение (embedding)
    prof = load_profile(uid)
    av = get_avatar(prof, avatar)
    ref = av.get("face_embedding")

    def _is_ok_https(u: str) -> bool:
        return isinstance(u, str) and u.startswith(("http://", "https://"))

    # Проверяем жив ли presigned (HEAD/GET)
    def _alive(u: str) -> bool:
        try:
            r = requests.head(u, timeout=10, allow_redirects=True)
            if r.status_code == 200:
                return True
            # AWS иногда 403 на HEAD — попробуем GET с небольшим range
            if r.status_code in (401, 403, 405):
                r2 = requests.get(u, timeout=10, stream=True)
                return r2.status_code == 200
        except Exception:
            pass
        return False

    # a) HTTPS → жив?
    if _is_ok_https(ref) and _alive(ref):
        return ref

    # b) локальный путь существует?
    if isinstance(ref, str) and os.path.exists(ref):
        return ref

    # 2) нет валидного embedding → пробуем пересоздать
    try:
        new_ref = prepare_face_embedding(uid, avatar)
        if _is_ok_https(new_ref) and _alive(new_ref):
            return new_ref
        if isinstance(new_ref, str) and os.path.exists(new_ref):
            return new_ref
    except Exception as e:
        logger.warning("auto refresh face_embedding failed: %s", e)

    # 3) fallback: возьмём первое референс-фото и подпишем при необходимости
    try:
        refs = list_ref_images(uid, avatar)
        if not refs:
            return None
        key = refs[0]
        if os.path.exists(key):           # FS
            return key
        if isinstance(key, str) and key.startswith("s3://") and s3_client:
            _, _, bucket_and_key = key.partition("s3://")
            bucket, _, obj_key = bucket_and_key.partition("/")
            url = s3_client.generate_presigned_url(
                "get_object", Params={"Bucket": bucket, "Key": obj_key},
                ExpiresIn=int(os.getenv("FACE_ID_URL_TTL", "86400"))
            )
            # заодно сохраним как новый embedding
            av["face_embedding"] = url
            save_profile(uid, prof)
            return url
        if isinstance(key, str) and key.startswith(("http://","https://")):
            return key
    except Exception as e:
        logger.warning("fallback face ref failed: %s", e)

    return None

    # ---------- Инференс с FaceID (без bytes в JSON) ----------
def generate_with_face_id_adapter(
        model_slug: str,
        prompt: str,
        steps: int,
        guidance: float,
        seed: int,
        w: int,
        h: int,
        negative_prompt: str,
        face_embedding_url: Union[str, IO[bytes]],
    ) -> str:
        """
        Replicate принимает либо файловый хендл (rb), либо HTTPS URL.
        Никаких сырых bytes — иначе «Object of type bytes is not JSON serializable».
        """
        mv = resolve_model_version(model_slug)

        # подготавливаем корректный источник
        face_source: Any
        if isinstance(face_embedding_url, str):
            if os.path.exists(face_embedding_url):
                face_source = open(face_embedding_url, "rb")          # file handle
            elif face_embedding_url.startswith(("http://", "https://")):
                face_source = face_embedding_url                      # URL
            else:
                raise RuntimeError(f"FACE-ID: неподдерживаемый ref: {face_embedding_url!r}")
        elif hasattr(face_embedding_url, "read"):
            face_source = face_embedding_url                          # IO[bytes]
        else:
            raise RuntimeError("FACE-ID: нужен путь/URL/файловый объект")

        extra_inputs = {
            "face_image": face_source,
            "ip_adapter_scale": FACE_ID_WEIGHT,
            "face_id_noise": FACE_ID_NOISE,
        }

        try:
            out = _retry(
                lambda: replicate.run(
                    mv,
                    input={
                        "prompt": prompt + AESTHETIC_SUFFIX,
                        "negative_prompt": negative_prompt,
                        "width": w,
                        "height": h,
                        "num_inference_steps": min(MAX_STEPS, steps),
                        "guidance_scale": guidance,
                        "seed": seed,
                        **extra_inputs,
                    },
                ),
                label="replicate_gen_faceid",
            )
        finally:
            try:
                if hasattr(face_source, "close"):
                    face_source.close()
            except Exception:
                pass

        url = extract_any_url(out)
        if not url:
            raise RuntimeError(f"Empty output with FACE-ID (model={model_slug})")
        return url


    # ---------- Диспетчер генерации ----------
def generate_from_finetune(
        model_slug: str,
        prompt: str,
        steps: int,
        guidance: float,
        seed: int,
        w: int,
        h: int,
        negative_prompt: str,
        face_embedding_url: Optional[Union[str, IO[bytes]]] = None,
    ) -> str:
        use_faceid = bool(FACE_ID_ADAPTER_ENABLED and face_embedding_url)
        logger.info("GEN DISPATCH: model=%s use_faceid=%s", model_slug, use_faceid)

        if use_faceid:
            return generate_with_face_id_adapter(
                model_slug=model_slug,
                prompt=prompt,
                steps=steps,
                guidance=guidance,
                seed=seed,
                w=w,
                h=h,
                negative_prompt=negative_prompt,
                face_embedding_url=face_embedding_url,  # URL/путь/IO[bytes]
            )

        mv = resolve_model_version(model_slug)
        out = _retry(
            lambda: replicate.run(
                mv,
                input={
                    "prompt": prompt + AESTHETIC_SUFFIX,
                    "negative_prompt": negative_prompt,
                    "width": w,
                    "height": h,
                    "num_inference_steps": min(MAX_STEPS, steps),
                    "guidance_scale": guidance,
                    "seed": seed,
                },
            ),
            label="replicate_gen_plain",
        )
        url = extract_any_url(out)
        if not url:
            raise RuntimeError("Empty output (plain)")
        return url



# ---------- UI/KB ----------
def main_menu_kb() -> InlineKeyboardMarkup:
        rows = [
            [InlineKeyboardButton("🧭 Выбрать стиль", callback_data="nav:styles")],
            [InlineKeyboardButton("📸 Набор фото", callback_data="nav:enroll"),
             InlineKeyboardButton("🧪 Обучение", callback_data="nav:train")],
            [InlineKeyboardButton("🤖 Аватары", callback_data="nav:avatars")],
        ]
        return InlineKeyboardMarkup(rows)

def categories_kb() -> InlineKeyboardMarkup:
    names = list(STYLE_CATEGORIES.keys())
    rows, row = [], []
    for i, name in enumerate(names, 1):
        row.append(InlineKeyboardButton(name, callback_data=f"cat:{name}"))
        if i % 2 == 0:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    # ❌ убрали кнопку назад в главное меню
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

        # Новые кнопки для активного аватара
        rows.append([
            InlineKeyboardButton("⚧ Сменить пол", callback_data="avatar:genderchange"),
            InlineKeyboardButton("✏️ Переименовать", callback_data="avatar:rename")
        ])
        rows.append([
            InlineKeyboardButton("➕ Новый", callback_data="avatar:new"),
            InlineKeyboardButton("🗑 Удалить…", callback_data="avatar:del")
        ])
        return InlineKeyboardMarkup(rows)


def avatar_gender_kb(name:str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("♀ Женский", callback_data=f"avatar:gender:{name}:female"),
        InlineKeyboardButton("♂ Менский", callback_data=f"avatar:gender:{name}:male")
    ]])

def enroll_done_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("Готово ✅", callback_data="enroll:done")]])

def delete_pick_kb(uid:int) -> InlineKeyboardMarkup:
    prof = load_profile(uid)
    rows = []
    for n in sorted([x for x in prof["avatars"].keys() if x != "default"]):
        rows.append([InlineKeyboardButton(f"Удалить «{n}»", callback_data=f"avatar:delpick:{n}")])
    if not rows:
        rows = [[InlineKeyboardButton("Нет ничего для удаления", callback_data="nav:avatars")]]
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="nav:avatars")])
    return InlineKeyboardMarkup(rows)

def delete_confirm_kb(name:str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Да, удалить", callback_data=f"avatar:delyes:{name}")],
        [InlineKeyboardButton("Нет", callback_data="nav:avatars")]
    ])

def face_id_toggle_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Включить Face ID", callback_data="faceid:enable"),
            InlineKeyboardButton("❌ Выключить Face ID", callback_data="faceid:disable")
        ],
        [InlineKeyboardButton("🔄 Обновить Embedding", callback_data="faceid:refresh")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="nav:menu")]
    ])

# ----- Callback для «Аватары» и связанных действий -----
    # ожидание переименования
PENDING_RENAME_AVATAR: Dict[int, Optional[str]] = {}

async def avatar_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
        q = update.callback_query
        await q.answer()

        uid = update.effective_user.id
        prof = load_profile(uid); prof["_uid_hint"] = uid; save_profile(uid, prof)
        parts = q.data.split(":")
        action = parts[1] if len(parts) > 1 else ""

        if action == "set":
            name = parts[2] if len(parts) > 2 else None
            if not name or name not in prof["avatars"]:
                await _replace_with_new_below(q.message, "Аватар не найден. Выбери из списка:", reply_markup=avatars_kb(uid))
                return
            set_current_avatar(uid, name)
            av = get_avatar(prof, name)

            if not av.get("gender"):
                await _replace_with_new_below(q.message, f"Выбран «{name}». Укажи пол:", reply_markup=avatar_gender_kb(name))
                return

            # пол есть — просто обновим меню аватаров (галочка переедет)
            await _replace_with_new_below(q.message, f"Активный: «{name}» • Пол: {av.get('gender','—')}", reply_markup=avatars_kb(uid))
            return

        elif action == "gender":  # avatar:gender:<name>:female|male
            if len(parts) >= 4:
                name, g = parts[2], parts[3]
                prof = load_profile(uid)
                if name in prof["avatars"]:
                    prof["avatars"][name]["gender"] = "female" if g == "female" else "male"
                    save_profile(uid, prof)

                    # после выбора пола — просим 10 фото и включаем набор
                    with contextlib.suppress(Exception):
                        await q.message.delete()
                    await update.effective_chat.send_message(
                        f"Пол для «{name}»: {prof['avatars'][name]['gender']}. "
                        "Теперь пришли подряд 10 фронтальных фото без фильтров."
                    )
                    # включаем режим набора и показываем кнопку «Готово» на всякий случай
                    av_name = name
                    ENROLL_FLAG[(uid, av_name)] = True
                    await update.effective_chat.send_message("📸 Набор фото активирован.", reply_markup=enroll_done_kb())
                else:
                    await _replace_with_new_below(q.message, "Аватар не найден.", reply_markup=avatars_kb(uid))
            return

        elif action == "genderchange":
            # смена пола для ТЕКУЩЕГО
            cur = get_current_avatar_name(prof)
            await _replace_with_new_below(q.message, f"Сменить пол для «{cur}». Выбери:", reply_markup=avatar_gender_kb(cur))
            return

        elif action == "rename":
            # ждём новое имя для текущего
            cur = get_current_avatar_name(prof)
            PENDING_RENAME_AVATAR[uid] = cur
            await _replace_with_new_below(q.message, f"Переименование «{cur}». Введи новое имя одним сообщением:")
            return

        elif action == "new":
            PENDING_NEW_AVATAR[uid] = True
            await _replace_with_new_below(q.message, "Введи имя нового аватара одним сообщением (например: travel, work, glam).")
            return

        elif action == "del":
            await _replace_with_new_below(q.message, "Выбери, что удалить:", reply_markup=delete_pick_kb(uid))
            return

        elif action == "delpick":
            name = parts[2] if len(parts) > 2 else None
            if not name:
                await _replace_with_new_below(q.message, "Не понял, что удалять.", reply_markup=avatars_kb(uid))
                return
            await _replace_with_new_below(q.message, f"Удалить «{name}» безвозвратно?", reply_markup=delete_confirm_kb(name))
            return

        elif action == "delyes":
            name = parts[2] if len(parts) > 2 else None
            if not name:
                await _replace_with_new_below(q.message, "Не указан аватар.", reply_markup=avatars_kb(uid))
                return
            try:
                del_avatar(uid, name)
                await _replace_with_new_below(q.message, f"«{name}» удалён.", reply_markup=avatars_kb(uid))
            except Exception as e:
                await _replace_with_new_below(q.message, f"Не удалось удалить: {e}", reply_markup=avatars_kb(uid))
            return

        else:
            await _replace_with_new_below(q.message, "Аватары:", reply_markup=avatars_kb(uid))
            return








# ---------- Face ID Callback ---------   
async def face_id_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
                    q = update.callback_query
                    await q.answer()
                    uid = update.effective_user.id
                    prof = load_profile(uid)
                    av_name = get_current_avatar_name(prof)
                    av = get_avatar(prof, av_name)

                    parts = q.data.split(":")
                    action = parts[1] if len(parts) > 1 else ""

                    if action == "enable":
                        global FACE_ID_ADAPTER_ENABLED
                        FACE_ID_ADAPTER_ENABLED = True
                        await _replace_with_new_below(q.message, "✅ Face ID adapter включен", reply_markup=face_id_toggle_kb())

                    elif action == "disable":
                        FACE_ID_ADAPTER_ENABLED = False
                        await _replace_with_new_below(q.message, "❌ Face ID adapter выключен", reply_markup=face_id_toggle_kb())

                    elif action == "refresh":
                        msg = await _replace_with_new_below(q.message, "🔄 Обновляю Face ID embedding…", reply_markup=None)
                        try:
                            embedding = await asyncio.to_thread(prepare_face_embedding, uid, av_name)
                            text = "✅ Face ID embedding обновлён" if embedding else "❌ Не удалось обновить Face ID embedding"
                            # обновлённый статус выводим НОВЫМ сообщением снизу, без редактирования
                            await msg.reply_text(text, reply_markup=face_id_toggle_kb(), disable_web_page_preview=True)
                        except Exception as e:
                            await msg.reply_text(f"❌ Ошибка: {e}", reply_markup=face_id_toggle_kb(), disable_web_page_preview=True)

                    else:
                        status = "включен" if FACE_ID_ADAPTER_ENABLED else "выключен"
                        has_embedding = "есть" if av.get("face_embedding") else "нет"
                        text = (
                            "👤 Face ID Adapter\n\n"
                            f"Статус: {status}\n"
                            f"Embedding: {has_embedding}\n"
                            f"Вес: {FACE_ID_WEIGHT}\n"
                            f"Шум: {FACE_ID_NOISE}\n"
                            f"Масштаб: {FACE_ID_SCALE}"
                        )
                        await _replace_with_new_below(q.message, text, reply_markup=face_id_toggle_kb())



# ---------- Handlers ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        prof = load_profile(uid)
        prof["_uid_hint"] = uid
        save_profile(uid, prof)

        await update.message.reply_text(
            "Привет! Я создам твою персональную фотомодель из 10 фото и буду генерировать тебя в узнаваемых сценах.\n\n"
            "— «📸 Набор фото» — загрузи до 10 снимков.\n"
            "— «🧪 Обучение» — тренируем твою LoRA.\n"
            "— «🧭 Стили» — выбирай сцены и жанры.\n"
            "— «🤖 Аватары» — несколько моделей с отдельным полом.\n"
            "Face ID включён автоматически; Natural=ON, Pretty=OFF, LockFace=OFF.",
            reply_markup=bottom_reply_kb()
        )


# ---------- UI utils ----------

        # ---------- UI utils: удалить старое, прислать новое снизу ----------
# --- Persistent main menu (не удаляем) ---
    # --- "Вечное" Главное меню ---
MAIN_MENU_MSG_ID: Dict[int, int] = {}  # uid -> message_id

def _is_main_menu_msg(uid: int, msg_id: Optional[int]) -> bool:
        return msg_id is not None and MAIN_MENU_MSG_ID.get(uid) == msg_id

# --- Трекер, чтобы не было двух «Главных меню» подряд ---
LAST_MAIN_MENU_MSG_ID: Dict[int, int] = {}  # uid -> message_id

def bottom_reply_kb() -> ReplyKeyboardMarkup:
    rows = [
        ["Стили", "Аватар"],
        ["Набор фото", "Обучение"],
        # можно добавить: ["Меню"]
    ]
    return ReplyKeyboardMarkup(
        rows,
        resize_keyboard=True,
        is_persistent=True,       # держим всегда
        one_time_keyboard=False,
        selective=False
    )


async def ensure_main_menu(bot, chat_id: int, uid: int, text: str = "Главное меню:"):
        """
        Гарантирует, что у пользователя есть ОДНО "вечное" главное меню.
        Если уже есть — НИЧЕГО НЕ ДЕЛАЕМ (не редактируем).
        Если нет — отправляем и запоминаем message_id.
        """
        if MAIN_MENU_MSG_ID.get(uid):
            return
        new_msg = await bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=main_menu_kb(),
            disable_web_page_preview=True
        )
        MAIN_MENU_MSG_ID[uid] = new_msg.message_id

async def _send_below_preserving(qmsg, text: str, reply_markup=None):
        """
        Всегда отправляет НОВОЕ сообщение ниже (НЕ удаляя источник).
        Нужно, чтобы кнопки в Главном меню просто "рожали" новое сообщение.
        """
        return await qmsg.reply_text(text, reply_markup=reply_markup, disable_web_page_preview=True)


async def _replace_with_new_below(qmsg, text: str, reply_markup=None):
        """
        Твой старый хелпер: отправляет НОВОЕ сообщение и удаляет старое.
        Оставляем для эфемерных меню (аватары и т.п.).
        """
        new_msg = await qmsg.reply_text(text, reply_markup=reply_markup, disable_web_page_preview=True)
        with contextlib.suppress(Exception):
            await qmsg.delete()
        return new_msg



async def nav_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
            q = update.callback_query
            await q.answer()

            uid = update.effective_user.id
            prof = load_profile(uid)
            prof["_uid_hint"] = uid
            save_profile(uid, prof)
            key = q.data.split(":", 1)[1]

            # вспомогательный локальный хелпер
            async def replace_card(text: str, kb=None):
                return await _replace_with_new_below(q.message, text, reply_markup=kb)

            # определяем, откуда пришло — из главного окна или карточки
            def _is_from_main() -> bool:
                return bool(q.message.reply_markup and q.message.reply_markup.keyboard)

            def _show_below(text: str, kb=None):
                return context.bot.send_message(q.message.chat_id, text, reply_markup=kb)

            # === навигация ===
            if key == "styles":
                await replace_card("Выбери категорию:", categories_kb())
                return

            elif key == "enroll":
                if _is_from_main():
                    await _show_below("📸 Набор фото…")
                    await id_enroll(update, context)
                else:
                    await replace_card("📸 Набор фото…")
                    await id_enroll(update, context)
                return

            elif key == "train":
                if _is_from_main():
                    await _show_below("Запускаю подготовку модели…")
                    await trainid_cmd(update, context)
                else:
                    await replace_card("Запускаю подготовку модели…")
                    await trainid_cmd(update, context)
                return

            elif key == "avatars":
                await replace_card("Аватары:", avatars_kb(uid))
                return

            elif key == "status":
                # оставляем на случай старых кнопок
                await replace_card("Обновляю статус…")
                await id_status(update, context)
                return

            elif key == "menu":
                # вместо «главного меню» — просто подсказка
                with contextlib.suppress(Exception):
                    await q.message.delete()
                await context.bot.send_message(
                    chat_id=q.message.chat_id,
                    text="Меню всегда снизу 👇",
                    reply_markup=bottom_reply_kb()
                )
                return

            else:
                await replace_card("Выбери категорию:", categories_kb())
                return







async def styles_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
        # если пришла команда, создаём одно сообщение с меню
        await update.message.reply_text("Выбери категорию:", reply_markup=categories_kb())

async def cb_category(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    cat = q.data.split(":", 1)[1]
    await _replace_with_new_below(q.message, f"Стиль — {cat}. Выбери сцену:", reply_markup=styles_kb_for_category(cat))


    async def id_enroll(update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        prof = load_profile(uid); prof["_uid_hint"] = uid; save_profile(uid, prof)
        av_name = get_current_avatar_name(prof)
        ENROLL_FLAG[(uid, av_name)] = True
        await update.effective_message.reply_text(
            f"Набор включён для «{av_name}». Пришли подряд 10 фронтальных фото без фильтров."
        )


async def cb_enroll_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # callback enroll:done → та же логика, что и /iddone
    await id_done(update, context)


async def id_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        prof = load_profile(uid); prof["_uid_hint"] = uid; save_profile(uid, prof)
        av_name = get_current_avatar_name(prof)
        ENROLL_FLAG[(uid, av_name)] = False
        av = get_avatar(prof, av_name)
        av["images"] = list_ref_images(uid, av_name)

        # автоопределение пола (если не задан)
        if not av.get("gender"):
            try:
                av["gender"] = auto_detect_gender(uid, av_name)
            except Exception:
                av["gender"] = av.get("gender") or (prof.get("gender") or "female")

        # Face ID embedding — готовим молча, без уведомлений
        try:
            await asyncio.to_thread(prepare_face_embedding, uid, av_name)
        except Exception as e:
            logger.warning("Face ID embedding preparation (silent) failed: %s", e)

        save_profile(uid, prof)
        g = av.get("gender") or "—"
        await update.effective_message.reply_text(
            f"Готово ✅ В «{av_name}» {len(av['images'])} фото.\nПол аватара: {g}\nДалее — запусти подготовку модели.",
            reply_markup=main_menu_kb()
        )


async def id_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    # 1) читаем профиль и активный аватар
    prof = load_profile(uid); prof["_uid_hint"] = uid; save_profile(uid, prof)
    av_name = get_current_avatar_name(prof); av = get_avatar(prof, av_name)

    # 2) если есть активная тренировка — ОБНОВИ статус через Replicate
    tid = av.get("training_id")
    if tid:
        # если ещё не конечный статус — проверим через API
        if av.get("status") not in {"succeeded", "failed", "canceled"}:
            try:
                status, slug_with_ver, err = await asyncio.to_thread(check_training_status, uid, av_name)
                # перечитываем профиль после обновления
                prof = load_profile(uid); av = get_avatar(prof, av_name)
            except Exception as e:
                logging.warning("id_status: check failed: %s", e)

    # 3) готовим вывод
    model = av.get("finetuned_model") or "—"
    ver_id = av.get("finetuned_version") or "—"
    st = av.get("status") or "—"
    token = av.get("token") or "—"
    g = av.get("gender") or "—"
    face_id_status = "✅" if av.get("face_embedding") else "❌"

    # универсальная ссылка на логи
    log_btn = None
    if tid:
        log_btn = InlineKeyboardMarkup([[InlineKeyboardButton("Открыть логи Replicate", url=f"https://replicate.com/trainings/{tid}")]])

    msg = (
        f"Активный аватар: {av_name}\n"
        f"Фото: {len(list_ref_images(uid, av_name))}\n"
        f"Статус: {st}\n"
        f"Модель: {model}\n"
        f"Версия: {ver_id}\n"
        f"Токен: {token}\n"
        f"Пол (аватар): {g}\n"
        f"LOCKFACE: {'on' if av.get('lockface') else 'off'}\n"
        f"Face ID: {face_id_status}\n"
        f"Natural: {'ON' if prof.get('natural', True) else 'OFF'} • Pretty: {'ON' if prof.get('pretty', False) else 'OFF'}"
    )
    await update.effective_message.reply_text(msg, reply_markup=log_btn)

async def id_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    delete_profile(uid)
    await update.message.reply_text("Профиль очищен. Жми «📸 Набор фото» и загрузи снимки заново.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        prof = load_profile(uid); prof["_uid_hint"] = uid; save_profile(uid, prof)
        av_name = get_current_avatar_name(prof)

        # если режим набора не включён — мягко включим (на случай, если юзер просто шлёт фото)
        if not ENROLL_FLAG.get((uid, av_name), False):
            ENROLL_FLAG[(uid, av_name)] = True

        # 1) скачиваем и сохраняем фото
        try:
            f = await update.effective_message.photo[-1].get_file()
            data = await f.download_as_bytearray()
            _ = STORAGE.save_ref_image(uid, av_name, data)
        except Exception as e:
            await update.effective_message.reply_text(f"⚠️ Не удалось сохранить фото: {e}")
            return

        # 2) считаем, показываем прогресс (и удаляем старый счётчик)
        count = len(list_ref_images(uid, av_name))
        prev_id = PHOTO_COUNTER_MSG_ID.get((uid, av_name))
        if prev_id:
            with contextlib.suppress(Exception):
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=prev_id)

        msg = await update.effective_message.reply_text(f"📸 Фото {min(count,10)}/10")
        PHOTO_COUNTER_MSG_ID[(uid, av_name)] = msg.message_id

        # 3) если ещё не 10 — ждём следующие
        if count < 10:
            return

        # 4) ровно/более 10 — фиксируем набор и запускаем пайплайн
        ENROLL_FLAG[(uid, av_name)] = False

        av = get_avatar(prof, av_name)
        av["images"] = list_ref_images(uid, av_name)

        # если нет пола — определим
        if not av.get("gender"):
            try:
                av["gender"] = auto_detect_gender(uid, av_name)
            except Exception:
                av["gender"] = av.get("gender") or (prof.get("gender") or "female")

        save_profile(uid, prof)

        # 5) готовим Face ID embedding (до обучения)
        try:
            await update.effective_message.reply_text("🔄 Подготавливаю Face ID embedding…")
            embedding = await asyncio.to_thread(prepare_face_embedding, uid, av_name)
            if embedding:
                await update.effective_message.reply_text("✅ Face ID embedding готов")
            else:
                await update.effective_message.reply_text("⚠️ Не удалось подготовить Face ID embedding — продолжу без него.")
        except Exception as e:
            logger.warning("Face ID embedding preparation failed: %s", e)

        # 6) стартуем обучение
        await update.effective_message.reply_text("🚀 Запускаю обучение LoRA…")
        try:
            async with TRAIN_SEMAPHORE:
                training_id = await asyncio.to_thread(start_lora_training, uid, av_name)
            await update.effective_message.reply_text(
                f"🧪 Создаём твою цифровую копию… это займёт немного времени.\n"
                f"ID обучения: {training_id}"
            )
            # мониторим статус и сообщаем результат
            asyncio.create_task(_wait_training_and_notify(context.bot, update.effective_chat.id, uid, av_name))
        except Exception as e:
            logger.exception("train start failed")
            await update.effective_message.reply_text(f"⚠️ Не удалось запустить обучение: {e}")

    # резолвер: байты из TG → STORAGE.save_ref_image → HTTPS URL
async def _resolve_direct_photo_url(update, context) -> str:
        global STORAGE
        f = await update.effective_message.photo[-1].get_file()
        data = await f.download_as_bytearray()
        url = STORAGE.save_ref_image(uid, av_name, data)
        return url
    

# === Хелпер переименования аватара (FS / S3) ===
def rename_avatar(uid:int, old:str, new:str):
    if old == new:
        return
    prof = load_profile(uid)

    if old not in prof["avatars"]:
        raise RuntimeError("Исходный аватар не найден.")
    if new in prof["avatars"]:
        raise RuntimeError("Аватар с таким именем уже существует.")

    # 1) перенос хранилища
    if USE_S3 and s3_client:
        old_prefix = _s3_key("profiles", str(uid), "avatars", old)
        new_prefix = _s3_key("profiles", str(uid), "avatars", new)
        cont = None
        to_delete = []
        while True:
            resp = _retry(
                s3_client.list_objects_v2,
                Bucket=S3_BUCKET,
                Prefix=old_prefix,
                ContinuationToken=cont,
                label="s3_list_for_rename"
            ) if cont else _retry(
                s3_client.list_objects_v2,
                Bucket=S3_BUCKET,
                Prefix=old_prefix,
                label="s3_list_for_rename"
            )

            for it in resp.get("Contents", []):
                key_old = it["Key"]
                key_rel = key_old[len(old_prefix):].lstrip("/")
                key_new = f"{new_prefix}/{key_rel}"
                _retry(
                    s3_client.copy_object,
                    Bucket=S3_BUCKET,
                    CopySource={"Bucket": S3_BUCKET, "Key": key_old},
                    Key=key_new,
                    label="s3_copy_for_rename"
                )
                to_delete.append({"Key": key_old})

            if not resp.get("IsTruncated"):
                break
            cont = resp.get("NextContinuationToken")

        if to_delete:
            _retry(
                s3_client.delete_objects,
                Bucket=S3_BUCKET,
                Delete={"Objects": to_delete},
                label="s3_del_old_after_rename"
            )

    else:
        # FS
        op = avatar_dir(uid, old)
        np = avatars_root(uid) / new
        if np.exists():
            raise RuntimeError("Папка для нового имени уже существует.")
        if op.exists():
            shutil.move(str(op), str(np))

    # 2) перенос записи профиля
    prof["avatars"][new] = prof["avatars"].pop(old)
    if prof.get("current_avatar") == old:
        prof["current_avatar"] = new
    save_profile(uid, prof)


# ---- Аватарные команды/утилиты ----
def set_current_avatar(uid:int, name:str):
    prof = load_profile(uid)
    if name not in prof["avatars"]:
        prof["avatars"][name] = DEFAULT_AVATAR.copy()
    prof["current_avatar"] = name
    prof["_uid_hint"] = uid
    _ = get_avatar(prof, name) # гарантируем токен
    save_profile(uid, prof)

def ensure_avatar(uid:int, name:str):
    prof = load_profile(uid)
    if name not in prof["avatars"]:
        prof["avatars"][name] = DEFAULT_AVATAR.copy()
    prof["_uid_hint"] = uid
    _ = get_avatar(prof, name)
    save_profile(uid, prof)

def del_avatar(uid:int, name:str):
    prof = load_profile(uid)
    if name == "default":
        raise RuntimeError("Нельзя удалить аватар 'default'.")
    if name not in prof["avatars"]:
        return
    STORAGE.delete_avatar(uid, name)
    prof["avatars"].pop(name, None)
    if prof["current_avatar"] == name:
        prof["current_avatar"] = "default"
    save_profile(uid, prof)

# ---- Текстовый обработчик для имени нового аватара (без команд) ----
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        text = (update.message.text or "").strip()
        if not text:
            return

        # --- переименование активного аватара ---
        if PENDING_RENAME_AVATAR.get(uid):
            old = PENDING_RENAME_AVATAR.pop(uid)
            new = re.sub(r"[^\w\-\.\@]+", "_", text)[:32] or "noname"
            try:
                rename_avatar(uid, old, new)
                await update.message.reply_text(f"Готово. «{old}» → «{new}».", reply_markup=avatars_kb(uid))
            except Exception as e:
                await update.message.reply_text(f"⚠️ Не удалось переименовать: {e}", reply_markup=avatars_kb(uid))
            return

        # --- создание нового аватара (теперь фото просим ПОСЛЕ выбора пола) ---
        if PENDING_NEW_AVATAR.get(uid):
            name = re.sub(r"[^\w\-\.\@]+", "_", text)[:32] or "noname"
            ensure_avatar(uid, name)
            set_current_avatar(uid, name)
            PENDING_NEW_AVATAR.pop(uid, None)

            await update.message.reply_text(
                f"Создан и выбран аватар: «{name}». Укажи пол:",
                reply_markup=avatar_gender_kb(name)
            )
            # фото НЕ просим здесь — дождёмся выбора пола
            return

        # --- reply-клавиатура ---
        t = text.lower()

        if t in ("стили", "style", "styles"):
            await update.message.reply_text("Выбери категорию:", reply_markup=categories_kb())
            return

        if t in ("аватар", "аватары", "avatar"):
            await update.message.reply_text("Аватары:", reply_markup=avatars_kb(uid))
            return

        if t in ("набор фото", "фото", "enroll", "добавить фото"):
            await update.message.reply_text("📸 Набор фото активирован. Пришли до 10 фото подряд.")
            await id_enroll(update, context)
            return

        if t in ("обучение", "train", "тренировка"):
            await update.message.reply_text("🧪 Обучение…")
            await trainid_cmd(update, context)
            return

        if t in ("меню", "главное меню", "menu"):
            await update.message.reply_text("Меню всегда снизу 👇", reply_markup=bottom_reply_kb())
            return

        if t in ("статус", "мой статус", "status"):
            await update.message.reply_text("ℹ️ Обновляю статус…")
            await id_status(update, context)
            return

        # иначе — молчим, чтобы не ломать UX
        return



    # можно добавить другие «ожидания» при необходимости
    # иначе — игнорируем текст, чтобы не ломать UX кнопками

# ---- Обучение / Генерация ----
def _dest_model_slug(avatar:str) -> str:
    if not DEST_OWNER:
        raise RuntimeError("REPLICATE_DEST_OWNER не задан.")
    return f"{DEST_OWNER}/{DEST_MODEL}"

def _ensure_destination_exists(slug: str):
    try:
        replicate.models.get(slug)
    except Exception:
        o, name = slug.split("/",1)
        raise RuntimeError(f"Целевая модель '{slug}' не найдена. Создай на replicate.com/create (owner={o}, name='{name}').")

def _pack_refs_zip(uid:int, avatar:str) -> Path:
    return STORAGE.pack_refs_zip(uid, avatar)

def start_lora_training(uid: int, avatar: str) -> str:
    """Стартует обучение LoRA на Replicate."""
    # --- целевая модель и тренер
    dest_model = _dest_model_slug(avatar)
    _ensure_destination_exists(dest_model)
    trainer_slug = os.getenv("LORA_TRAINER_SLUG", "replicate/fast-flux-trainer").strip()
    trainer_version = resolve_model_version(trainer_slug)

    # --- упаковка фото
    zip_path = _pack_refs_zip(uid, avatar)

    # --- presigned URL (или явный URL из ENV для диагностики)
    direct_override = os.getenv("TRAIN_ZIP_DIRECT_URL", "").strip()
    if direct_override:
        presigned_url = direct_override
    else:
        if not s3_client or not S3_BUCKET:
            raise RuntimeError("Нужен S3: установи USE_S3=1 и S3_* переменные окружения.")
        key = _s3_key("profiles", str(uid), "avatars", avatar, "train.zip")
        with open(zip_path, "rb") as fz:
            _retry(
                s3_client.put_object, Bucket=S3_BUCKET, Key=key, Body=fz, 
                ContentType="application/zip", label="s3_put_trainzip"
            )
        presigned_url = s3_client.generate_presigned_url(
            "get_object", 
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=int(os.getenv("TRAIN_ZIP_URL_TTL", "21600")) # 6 часов
        )

    # --- проверка доступности ZIP
    if not presigned_url.lower().startswith("https://"):
        raise RuntimeError("Presigned URL должен быть HTTPS.")
    try:
        r = requests.get(presigned_url, timeout=25, stream=True)
        r.raise_for_status()
        sig = r.raw.read(4) if hasattr(r, "raw") else r.content[:4]
        if sig[:2] != b"PK":
            raise RuntimeError("Presigned URL не отдаёт валидный ZIP (нет сигнатуры PK).")
    except Exception as e:
        raise RuntimeError(f"ZIP недоступен по presigned URL: {e}")

    # --- токен/пол/подпись
    prof = load_profile(uid)
    av = get_avatar(prof, avatar)
    gender = (av.get("gender") or prof.get("gender") or auto_detect_gender(uid, avatar) or "female").lower()
    token = av.get("token") or _avatar_token(uid, avatar)
    caption_prefix = (
        f"photo of person {token}. " + _caption_for_gender(gender) + 
        ", frontal face, neutral relaxed expression, natural light"
    )

    # --- поля под конкретный тренер
    payload: Dict[str, Any] = {
        "input_images": presigned_url, # ZIP-URL обязательно
        "resolution": LORA_RESOLUTION,
        "use_face_detection_instead": LORA_USE_FACE_DET,
        "max_train_steps": LORA_MAX_STEPS,
    }

    lower_slug = trainer_slug.lower()
    if "fast-flux-trainer" in lower_slug:
        payload.update({
            "trigger_word": token,
            "lora_type": "subject",
            "caption_prefix": caption_prefix, # понимается тренером, не критично
        })
    else: # replicate/flux-lora-trainer
        payload.update({
            "caption_prefix": caption_prefix,
            "autocaption": False,
            "captioner": "none", 
            "caption_model": "none",
            "use_llava": False,
            "use_blip": False,
            "network_rank": int(os.getenv("LORA_RANK", "16")),
            "network_alpha": int(os.getenv("LORA_ALPHA", "16")),
            "lora_lr": LORA_LR,
        })

    # --- запуск тренировки
    client = Client(api_token=os.environ["REPLICATE_API_TOKEN"])
    training = _retry(
        client.trainings.create,
        version=trainer_version,
        input=payload,
        destination=dest_model
    )

    # --- безопасно достаём id (SDK может вернуть dict)
    tid = getattr(training, "id", None)
    if tid is None and isinstance(training, dict):
        tid = training.get("id")
    if not tid:
        raise RuntimeError(f"Replicate не вернул training id: {training!r}")

    # --- записываем профиль
    prof = load_profile(uid)
    av = get_avatar(prof, avatar)
    av["training_id"] = tid
    av["status"] = "starting"
    av["finetuned_model"] = dest_model
    save_profile(uid, prof)

    return tid

def check_training_status(uid: int, avatar: str) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Возвращает (status, slug_with_version_if_ready, error_text_or_None)
        и по пути обновляет профиль: status, finetuned_model, finetuned_version, lora_url.
        Супер-терпеливый парсер .safetensors — умеет список/словарь/вложенность и fallback к версии модели.
        """
        prof = load_profile(uid)
        av = get_avatar(prof, avatar)
        tid = av.get("training_id")
        if not tid:
            return ("not_started", None, None)

        client = Client(api_token=os.environ["REPLICATE_API_TOKEN"])

        # --- 1) тянем объект тренировки ---
        try:
            tr = client.trainings.get(tid)
        except Exception as e:
            return ("unknown", None, str(e))

        # Безопасно достаём поля из объекта/словаря
        if isinstance(tr, dict):
            status      = tr.get("status", "unknown")
            destination = tr.get("destination") or av.get("finetuned_model")
            err         = tr.get("error") or tr.get("detail")
            output      = tr.get("output")
        else:
            status      = getattr(tr, "status", "unknown")
            destination = getattr(tr, "destination", None) or av.get("finetuned_model")
            err         = getattr(tr, "error", None) or getattr(tr, "detail", None)
            output      = getattr(tr, "output", None)

        av["status"] = status
        if destination:
            av["finetuned_model"] = destination

        # --- helper: универсальный поиск первой HTTPS-ссылки на .safetensors ---
        def _find_first_safetensors_url(data: Any) -> Optional[str]:
            def _is_url(s: Any) -> bool:
                return isinstance(s, str) and s.startswith(("http://", "https://"))
            try:
                # строка
                if isinstance(data, str):
                    return data if _is_url(data) and ".safetensors" in data else None
                # список / кортеж
                if isinstance(data, (list, tuple)):
                    for v in data:
                        got = _find_first_safetensors_url(v)
                        if got:
                            return got
                    return None
                # словарь
                if isinstance(data, dict):
                    # частые ключи у разных тренеров
                    for k in ("safetensors", "safetensors_url", "weights_url", "url", "model_url", "lora_url"):
                        v = data.get(k)
                        if _is_url(v) and ".safetensors" in v:
                            return v
                    # вложенные контейнеры
                    for v in data.values():
                        got = _find_first_safetensors_url(v)
                        if got:
                            return got
                    return None
            except Exception:
                return None
            return None

        # чуть подсветим, что же пришло
        try:
            sample = None
            if isinstance(output, (list, tuple)) and output:
                sample = output[0]
            elif isinstance(output, dict):
                sample = {k: type(v).__name__ for k, v in list(output.items())[:5]}
            logger.info("TRAIN OUT sample=%r", sample)
        except Exception:
            pass

        # --- 2) если succeeded — закрепляем версию + достаём ссылку на веса ---
        slug_with_ver: Optional[str] = None
        if status == "succeeded" and destination:
            # 2.1) закрепим версию модели
            try:
                model_obj = replicate.models.get(destination)
                versions = list(model_obj.versions.list())
                if versions:
                    ver_id = versions[0].id
                    av["finetuned_version"] = ver_id
                    slug_with_ver = f"{destination}:{ver_id}"
                else:
                    slug_with_ver = destination
            except Exception as e:
                logger.warning("Не удалось получить версию модели %s: %s", destination, e)
                slug_with_ver = destination

            # 2.2) пытаемся вытащить HTTPS .safetensors из output тренировки
            weights_url: Optional[str] = None
            try:
                if output is not None:
                    weights_url = _find_first_safetensors_url(output)

                # 2.3) fallback: иногда тренер не кладёт ссылку в output → пробуем в описании версии
                if not weights_url:
                    try:
                        model_obj = replicate.models.get(destination)
                        versions = list(model_obj.versions.list())
                        if versions:
                            v0 = versions[0]
                            cand = getattr(v0, "openapi_schema", {}) or {}
                            weights_url = _find_first_safetensors_url(cand)
                            if not weights_url:
                                weights_url = _find_first_safetensors_url(getattr(v0, "__dict__", {}))
                    except Exception as e:
                        logger.warning("Скан версии модели на .safetensors провалился: %s", e)

                if weights_url and weights_url.startswith(("http://","https://")) and ".safetensors" in weights_url:
                    av["lora_url"] = weights_url
                    logger.info("Saved lora_url for avatar '%s': %s", avatar, weights_url)
                else:
                    logger.warning("Не удалось извлечь lora_url (.safetensors) из training.output")
            except Exception as e:
                logger.warning("Парсинг lora_url из output провалился: %s", e)

        save_profile(uid, prof)
        return (status, slug_with_ver, err)



def _pinned_slug(av: Dict[str, Any]) -> str:
    base = av.get("finetuned_model") or ""
    ver = av.get("finetuned_version")

    # Если есть и модель и версия
    if base and ver:
        # Убедимся, что версия не содержит лишних символов
        clean_ver = ver.strip()
        if ":" in clean_ver:
            clean_ver = clean_ver.split(":")[-1]
        return f"{base}:{clean_ver}"

    # Если есть только модель, попробуем получить последнюю версию
    elif base:
        try:
            model_obj = replicate.models.get(base)
            versions = list(model_obj.versions.list())
            if versions:
                latest_ver = versions[0].id
                av["finetuned_version"] = latest_ver
                # Сохраняем обновленную версию в профиль
                uid = av.get("_uid_hint", 0)
                prof = load_profile(uid)
                current_av_name = get_current_avatar_name(prof)
                if current_av_name in prof["avatars"]:
                    prof["avatars"][current_av_name]["finetuned_version"] = latest_ver
                    save_profile(uid, prof)
                return f"{base}:{latest_ver}"
        except Exception as e:
            logger.error(f"Failed to get latest version for {base}: {e}")

    return base or ""

async def trainid_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
            uid = update.effective_user.id
            prof = load_profile(uid); prof["_uid_hint"] = uid; save_profile(uid, prof)
            av_name = get_current_avatar_name(prof)

            if len(list_ref_images(uid, av_name)) < 10:
                await update.effective_message.reply_text(
                    f"Нужно 10 фото в «{av_name}». Сначала загрузите снимки и нажмите «Готово ✅»."
                )
                return

            await update.effective_message.reply_text("Запускаю подготовку модели…")

            try:
                async with TRAIN_SEMAPHORE:
                    _ = await asyncio.to_thread(start_lora_training, uid, av_name)
                # Никакого ID и ссылок — лаконично:
                await update.effective_message.reply_text(
                    "Отлично, подготовка модели началась. Я сообщу, когда всё будет готово."
                )
            except Exception as e:
                logging.exception("trainid failed")
                await update.effective_message.reply_text(f"Не удалось запустить подготовку: {e}")


async def trainstatus_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    prof = load_profile(uid)
    av_name = get_current_avatar_name(prof)

    # тянем свежий статус из Replicate
    status, slug_with_ver, err = await asyncio.to_thread(check_training_status, uid, av_name)

    # перечитываем профиль на всякий
    prof = load_profile(uid)
    av = get_avatar(prof, av_name)
    tid = av.get("training_id")

    display = {
        "starting":   "starting (готовится к запуску)",
        "queued":     "queued (в очереди)",
        "running":    "running (обучается)",
        "processing": "processing (публикую версию)",
        "succeeded":  "succeeded",
        "failed":     "failed",
        "canceled":   "canceled",
        "unknown":    "unknown",
        None:         "unknown",
    }.get(status, status or "unknown")

    train_url = (
        f"https://replicate.com/{DEST_OWNER}/{DEST_MODEL}/trainings/{tid}"
        if (DEST_OWNER and DEST_MODEL and tid) else None
    )

    # === успех: сразу ведём в выбор стилей ===
    if status == "succeeded" and slug_with_ver:
        await update.effective_message.reply_text(
            f"✅ Обучение завершено!\n"
            f"Аватар: {av_name}\n"
            f"Модель: {slug_with_ver}\n\n"
            f"Выбирай категорию стилей👇"
        )
        await update.effective_message.reply_text("Выбери категорию:", reply_markup=categories_kb())
        return

    # === всё ещё идёт ===
    if status in ("starting", "queued", "running", "processing"):
        extra = f"\nЛоги: {train_url}" if train_url else ""
        await update.effective_message.reply_text(f"Статус «{av_name}»: {display}{extra}")
        return

    # === неудача/отмена ===
    if status in ("failed", "canceled"):
        msg = f"⚠️ Тренировка «{av_name}»: {status.upper()}."
        if err:
            msg += f"\nПричина: {err}"
        if train_url:
            msg += f"\nЛоги: {train_url}"
        await update.effective_message.reply_text(msg)
        return

    # === прочее/неизвестно ===
    await update.effective_message.reply_text(f"Статус «{av_name}»: {display}.")


async def _wait_training_and_notify(bot, chat_id: int, uid: int, av_name: str):
    """Пулим статус до терминального и уведомляем пользователя."""
    try:
        while True:
            status, slug_with_ver, err = await asyncio.to_thread(check_training_status, uid, av_name)
            if status in ("succeeded", "failed", "canceled"):
                break
            await asyncio.sleep(20)  # мягкий пуллинг

        if status == "succeeded" and slug_with_ver:
            # на всякий — можно обновить/перегенерить embedding после обучения
            with contextlib.suppress(Exception):
                await asyncio.to_thread(prepare_face_embedding, uid, av_name)

            await bot.send_message(
                chat_id=chat_id,
                text=(
                    "✅ Цифровая копия создана! Можно выбирать стили.\n"
                    "Выбери категорию:"
                ),
                reply_markup=categories_kb()
            )
        else:
            msg = f"⚠️ Обучение «{av_name}»: {status.upper() if status else 'UNKNOWN'}."
            if err:
                msg += f"\nПричина: {err}"
            await bot.send_message(chat_id=chat_id, text=msg)
    except Exception as e:
        await bot.send_message(chat_id=chat_id, text=f"⚠️ Ошибка мониторинга обучения: {e}")



# === Принудительное обновление статуса с Replicate ===
async def refreshstatus_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    prof = load_profile(uid); prof["_uid_hint"] = uid; save_profile(uid, prof)
    av_name = get_current_avatar_name(prof)

    try:
        status, slug_with_ver, err = await asyncio.to_thread(check_training_status, uid, av_name)
    except Exception as e:
        await update.effective_message.reply_text(f"check_training_status error: {e}")
        return

    prof = load_profile(uid); av = get_avatar(prof, av_name)
    msg = (
        f"REFRESHED\n"
        f"avatar={av_name}\n"
        f"status={av.get('status')}\n"
        f"model={av.get('finetuned_model')}\n"
        f"version={av.get('finetuned_version')}\n"
        f"err={err or '—'}\n"
        f"slug_with_ver={slug_with_ver or '—'}"
    )
    await update.effective_message.reply_text(msg)

def _neg_with_gender(neg_base:str, gender_negative:str) -> str:
    return (neg_base + (", " + gender_negative if gender_negative else "")).strip(", ")


async def wf_toggle_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global FACEID_WORKFLOW_FORCE_OFF
    FACEID_WORKFLOW_FORCE_OFF = not FACEID_WORKFLOW_FORCE_OFF
    await update.message.reply_text(f"Workflow FaceID: {'OFF' if FACEID_WORKFLOW_FORCE_OFF else 'ON'}")


async def cb_style(update: Update, context: ContextTypes.DEFAULT_TYPE):
        q = update.callback_query
        await q.answer()
        preset = q.data.split(":", 1)[1]

        uid = update.effective_user.id
        prof = load_profile(uid); prof["_uid_hint"] = uid; save_profile(uid, prof)
        av_name = get_current_avatar_name(prof)
        av = get_avatar(prof, av_name)

        if not av.get("gender"):
            await _replace_with_new_below(
                q.message,
                f"Для «{av_name}» не указан пол. Выбери пол, а потом вернись к стилям:",
                reply_markup=avatar_gender_kb(av_name)
            )
            return

        await _replace_with_new_below(q.message, f"Генерирую «{preset}» (аватар: {av_name})…", reply_markup=None)

        force_plain = FACEID_WORKFLOW_FORCE_OFF

        if not force_plain and FACE_ID_ADAPTER_ENABLED and FACEID_WORKFLOW_AVAILABLE:
            try:
                await start_generation_via_workflow(update, context, preset, show_intro=False, avatar_name=av_name)
                return
            except Exception as e:
                logger.warning("WF failed; fallback to plain: %s", e)

        await start_generation_for_preset(update, context, preset, show_intro=False, avatar_name=av_name, force_plain=True)





# === ПРЯМАЯ ГЕНЕРАЦИЯ БЕЗ workflow И БЕЗ lora_url (фикс дублей) ==
async def start_generation_for_preset(
                            update: Update,
                            context: ContextTypes.DEFAULT_TYPE,
                            preset: str,
                            show_intro: bool = True,
                            avatar_name: Optional[str] = None,
                            force_plain: bool = False,
                        ):
                            """
                            Генерация по пресету.
                            - Жёстко фиксируем имя аватара на момент клика (avatar_name).
                            - Если workflow доступен и не запрещён, пробуем его; при ошибке — fallback в прямую генерацию.
                            - В прямой генерации используем FaceID-референс через _resolve_face_ref (если включён).
                            """
                            uid = update.effective_user.id
                            prof = load_profile(uid); prof["_uid_hint"] = uid; save_profile(uid, prof)
                            av_name = avatar_name or get_current_avatar_name(prof)  # фиксируем активный аватар
                            av = get_avatar(prof, av_name)

                            # 1) сначала пробуем WF-ветку (если не принудительно plain)
                            if (not force_plain) and FACE_ID_ADAPTER_ENABLED and FACEID_WORKFLOW_AVAILABLE:
                                try:
                                    await start_generation_via_workflow(
                                        update, context, preset, show_intro=show_intro, avatar_name=av_name
                                    )
                                    return
                                except Exception as e:
                                    logger.warning("FaceID workflow path failed; fallback to direct: %s", e)

                            # 2) прямая генерация финетюном
                            if av.get("status") != "succeeded":
                                await update.effective_message.reply_text(
                                    f"Модель «{av_name}» ещё не готова. /trainid → /trainstatus = succeeded."
                                )
                                return

                            model_slug = _pinned_slug(av) or av.get("finetuned_model")
                            if not model_slug:
                                await update.effective_message.reply_text("Не найден финетюн модели у аватара.")
                                return

                            if preset not in STYLE_PRESETS:
                                await update.effective_message.reply_text(f"Стиль «{preset}» не найден.")
                                return
                            meta = STYLE_PRESETS[preset]

                            gender  = (av.get("gender") or prof.get("gender") or "female").lower()
                            natural = bool(prof.get("natural", True))
                            pretty  = bool(prof.get("pretty", False))
                            avatar_token = av.get("token", "")

                            # FaceID-референс именно для этого аватара (если включён)
                            face_ref = None
                            if FACE_ID_ADAPTER_ENABLED:
                                try:
                                    face_ref = await asyncio.to_thread(_resolve_face_ref, uid, av_name)
                                except Exception as e:
                                    logger.warning("resolve_face_ref error: %s", e)
                                    face_ref = None

                            logger.info(
                                "GEN PREP (PLAIN): avatar=%s gender=%s preset=%s use_faceid=%s",
                                av_name, gender, preset, bool(face_ref)
                            )

                            # Композиции и identity-safe твики
                            comps = _variants_for_preset(meta)
                            guidance, comps, extra_neg = _identity_safe_tune(preset, GEN_GUIDANCE, comps)

                            if show_intro:
                                await update.effective_message.reply_text(f"Генерирую «{preset}» (аватар: {av_name})…")

                            sent = 0
                            for i, comp in enumerate(comps):
                                try:
                                    comp_text, (w, h) = _comp_text_and_size(comp)

                                    # лёгкая вариация кадра
                                    if comp == "half" and i % 2 == 1:
                                        comp_text += ", camera slightly closer, gentle 5° head turn"
                                    elif comp == "closeup" and i % 2 == 1:
                                        comp_text += ", micro-reframe, eyes focus a touch brighter"

                                    tone_text   = _tone_text(meta.get("tone", ""))
                                    theme_boost = _safe_theme_boost(THEME_BOOST.get(preset, ""))

                                    prompt, neg = build_prompt(
                                        meta, gender, comp_text, tone_text, theme_boost,
                                        natural, pretty, avatar_token
                                    )

                                    # спец-правка для некоторых пресетов (если есть у тебя в стилях)
                                    if preset == "Харли-Квинн":
                                        if gender.startswith("f"):
                                            prompt += ", " + ", ".join(meta.get("force_keywords_f", []))
                                        else:
                                            prompt += ", " + ", ".join(meta.get("force_keywords_m", []))

                                    if extra_neg:
                                        neg = _neg_with_gender(neg, extra_neg)

                                    seed = _stable_seed(str(uid), av_name, preset, f"{comp}:{i}")

                                    url = await asyncio.to_thread(
                                        generate_from_finetune,
                                        model_slug, prompt, GEN_STEPS, guidance, seed, w, h, neg, face_ref
                                    )
                                    await update.effective_message.reply_photo(url)
                                    sent += 1

                                except Exception as e:
                                    logger.exception("gen failed for comp=%s", comp)
                                    await update.effective_message.reply_text(f"⚠️ Ошибка генерации ({comp}): {e}")

                            if sent == 0:
                                await update.effective_message.reply_text("Не удалось сгенерировать ни одного изображения.")






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
    await update.message.reply_text(f"Natural: {'ON' if prof['natural'] else 'OFF'} (Pretty: {'ON' if prof.get('pretty', False) else 'OFF'})")

# ---------- System ----------
# --- ФОНОВЫЙ ПОЛЛЕР БЕЗ JobQueue ---
        # ---------- System ----------
        # --- ФОНОВЫЙ ПОЛЛЕР БЕЗ JobQueue ---
async def _poller_loop(app):
            while True:
                try:
                    # обходим всех пользователей и просто дёргаем наш check_training_status
                    profiles = load_all_profiles()
                    for uid, prof in profiles.items():
                        try:
                            av_name = get_current_avatar_name(prof)
                            # check_training_status сам обновляет профиль внутри
                            await asyncio.to_thread(check_training_status, uid, av_name)
                        except Exception as e:
                            logger.warning("poller: user %s avatar %s failed: %s", uid, av_name, e)
                except Exception as e:
                    logger.warning("poller loop outer failed: %s", e)

                # опрашиваем раз в 5 минут
                await asyncio.sleep(300)

async def _post_init(app):
    await app.bot.delete_webhook(drop_pending_updates=True)
    # запускаем фоновую корутину вместо job_queue
    app.create_task(_poller_loop(app))
    

def main():
    app = ApplicationBuilder().token(TOKEN).post_init(_post_init).build()

    # Команды (оставлены для совместимости; UX — кнопками)
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("menu", start))
    app.add_handler(CommandHandler("wf", wf_toggle_cmd))
    app.add_handler(CommandHandler("idstatus", id_status))
    app.add_handler(CommandHandler("trainid", trainid_cmd))
    app.add_handler(CommandHandler("trainstatus", trainstatus_cmd))
    app.add_handler(CommandHandler("refreshstatus", refreshstatus_cmd))

    # Кнопки
    app.add_handler(CallbackQueryHandler(nav_cb, pattern=r"^nav:"))
    app.add_handler(CallbackQueryHandler(cb_category, pattern=r"^cat:"))
    app.add_handler(CallbackQueryHandler(cb_style, pattern=r"^style:"))
    app.add_handler(CallbackQueryHandler(avatar_cb, pattern=r"^avatar:"))
    app.add_handler(CallbackQueryHandler(face_id_cb, pattern=r"^faceid:"))
    app.add_handler(CallbackQueryHandler(cb_enroll_done, pattern=r"^enroll:done$"))

    # Фото и текст
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Пинг слагов
    _check_slug(LORA_TRAINER_SLUG, "LoRA trainer")
    if FACE_ID_ADAPTER_ENABLED:
        _check_slug(FACE_ID_MODEL_SLUG, "Face ID model")

    logger.info("InstantID removed: pure LoRA mode.")
    logger.info(
        "Bot up. Trainer=%s DEST=%s GEN=%dx%d steps=%s guidance=%s ConsistentScale=%s WaistUp=%s Storage=%s FaceID=%s",
        LORA_TRAINER_SLUG, f"{DEST_OWNER}/{DEST_MODEL}", GEN_WIDTH, GEN_HEIGHT, GEN_STEPS, GEN_GUIDANCE,
        CONSISTENT_SCALE, FORCE_WAIST_UP, "S3" if USE_S3 else "FS", FACE_ID_ADAPTER_ENABLED
    )
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()