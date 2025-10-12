# -*- coding: utf-8 -*-
"""
faceid_workflow_integration.py
Drop‑in интеграция для большого main.py:
- Генерация через Replicate: fofr/any-comfyui-workflow + flux_pulid_lora_workflow_api.json
- Комбо: Flux LoRA + PuLID (FaceID) + кадр waist‑up
- Автосбор фото (первые 10) → выбор фронтального → старт обучения LoRA
- Пол (male/female) влияет на промпт
Что нужно в окружении: REPLICATE_API_TOKEN
Файл workflow: flux_pulid_lora_workflow_api.json (рядом с вашим main.py)
"""
from __future__ import annotations
import os, json, asyncio, logging, time, random
from typing import Dict, Any, Optional, Tuple, List

import aiohttp
import replicate
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

# === Константы/настройки ===
REPLICATE_MODEL = "fofr/any-comfyui-workflow:latest"
WORKFLOW_JSON_PATH = "flux_pulid_lora_workflow_api.json"
MAX_UPLOADS_FOR_TRAIN = 10

# === Глобальные кэши ===
_replicate_client: Optional[replicate.Client] = None
_workflow_json_cache: Optional[str] = None

def get_replicate_client() -> replicate.Client:
    global _replicate_client
    if _replicate_client is None:
        api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not api_token:
            raise RuntimeError("REPLICATE_API_TOKEN не задан")
        _replicate_client = replicate.Client(api_token=api_token)
    return _replicate_client

def load_workflow_json() -> str:
    global _workflow_json_cache
    if _workflow_json_cache is None:
        with open(WORKFLOW_JSON_PATH, "r", encoding="utf-8") as f:
            _workflow_json_cache = f.read()
    return _workflow_json_cache

# === Вспомогалки, используют ваш профиль/аватар ===
def get_avatar_gender(av: Dict[str, Any]) -> str:
    g = (av.get("gender") or av.get("sex") or "").strip().lower()
    if g in ("male", "m", "man", "парень", "мужчина"):
        return "male"
    return "female"

def choose_dimensions_by_comp(_: str) -> Tuple[int, int]:
    return (1024, 1536)  # вертикаль «по пояс/по грудь»

def sanitize_base_prompt_from_preset(meta: Dict[str, Any]) -> str:
    parts = []
    for k in ("role", "bg", "props", "tone", "desc"):
        if meta.get(k):
            parts.append(str(meta[k]))
    parts.append("waist-up, no full body")
    return ", ".join(parts)

def build_gender_prompts(base_prompt: str, gender: str) -> Tuple[str, str]:
    if gender == "male":
        outfit = "well-fitted shirt or t-shirt, clean look"
        look = "handsome, masculine facial proportions"
    else:
        outfit = "neutral elegant top, soft makeup"
        look = "beautiful, feminine facial proportions"
    positive = (
        f"{base_prompt}, waist-up portrait, looking at camera, {look}, "
        f"{outfit}, soft flattering light, professional photo, detailed skin, natural tones"
    )
    negative = (
        "lowres, blurry, deformed, extra fingers, extra limbs, mutation, artifacts, "
        "full body, body full shot, feet, cut off face, bad anatomy, over-saturated, "
        "asian ethnicity, slanted eyes, caricature"
    )
    return positive, negative

def resolve_lora_url(av: Dict[str, Any]) -> str:
    candidates = [
        av.get("lora_url"),
        (av.get("lora") or {}).get("url"),
        (av.get("trained") or {}).get("safetensors_url"),
        (av.get("trained") or {}).get("weights_url"),
        (av.get("replicate") or {}).get("lora_url"),
    ]
    for url in candidates:
        if url and isinstance(url, str) and url.startswith("http"):
            return url
    raise RuntimeError("Не найден lora_url у аватара (HTTPS .safetensors)")

def resolve_face_image_url(av: Dict[str, Any]) -> str:
    candidates = [
        av.get("face_image_url"),
        av.get("cover_face"),
        (av.get("images") or {}).get("face"),
        (av.get("refs") or {}).get("face"),
    ]
    for url in candidates:
        if url and isinstance(url, str) and url.startswith("http"):
            return url
    raise RuntimeError("Не найден face_image_url у аватара (HTTPS на фронтальное фото)")

# === Выбор фронтального фото из первых 10 ===
async def head_content_length(url: str) -> int:
    try:
        async with aiohttp.ClientSession() as s:
            async with s.head(url, timeout=10) as r:
                return int(r.headers.get("Content-Length") or 0)
    except Exception:
        return 0

async def fetch_image_size(url: str) -> tuple[int,int]:
    return (0, 0)

async def select_front_face_from_first_10(urls: List[str]) -> str:
    candidates = []
    for u in urls:
        w, h = await fetch_image_size(u)
        clen = await head_content_length(u)
        portrait_bonus = 1 if (h and w and h >= w) else 0
        score = (portrait_bonus * 10_000_000) + clen
        candidates.append((score, u))
    if not candidates:
        return urls[0]
    candidates.sort(reverse=True)
    return candidates[0][1]

# === Сбор inputs и запуск Replicate ===
def build_replicate_inputs_for_workflow(
    workflow_json: str,
    prompt: str,
    negative: str,
    seed: int,
    width: int,
    height: int,
    lora_url: str,
    face_image_url: str,
    lora_strength: float = 0.8,
    pulid_weight: float = 0.8,
    pulid_start: float = 4.0,
    steps: int = 28,
) -> Dict[str, Any]:
    return {
        "workflow": workflow_json,
        "inputs": {
            "prompt": prompt,
            "negative": negative,
            "seed": seed,
            "steps": steps,
            "width": width,
            "height": height,
            "lora_url": lora_url,
            "lora_strength": lora_strength,
            "pulid_weight": pulid_weight,
            "pulid_start": pulid_start,
        },
        "input_images": {
            "face_image": face_image_url
        }
    }

async def run_replicate_any_comfyui(inputs: Dict[str, Any]) -> Any:
    client = get_replicate_client()
    loop = asyncio.get_running_loop()
    def _call():
        return client.run(REPLICATE_MODEL, input=inputs)
    return await loop.run_in_executor(None, _call)

# === Публичные хелперы для интеграции в ваш main ===
async def start_generation_for_preset(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    preset: str,
    STYLE_PRESETS: Dict[str, Dict[str, Any]],
    load_profile,
    get_current_avatar_name,
    get_avatar,
):
    uid = update.effective_user.id
    msg = update.effective_message

    try:
        prof = load_profile(uid)
        av_name = get_current_avatar_name(prof)
        av = get_avatar(prof, av_name)

        if av.get("status") != "succeeded":
            await msg.reply_text(f"Модель «{av_name}» ещё не готова. Статус: {av.get('status','unknown')}")
            return

        gender = get_avatar_gender(av)
        lora_url = resolve_lora_url(av)
        face_image_url = resolve_face_image_url(av)

        meta = STYLE_PRESETS.get(preset, {})
        base_prompt = sanitize_base_prompt_from_preset(meta)
        positive, negative = build_gender_prompts(base_prompt, gender)

        width, height = choose_dimensions_by_comp(meta.get("comp") or "waistup")
        seed = random.randint(1, 2_147_483_647)

        workflow_json = load_workflow_json()
        inputs = build_replicate_inputs_for_workflow(
            workflow_json=workflow_json,
            prompt=positive,
            negative=negative,
            seed=seed,
            width=width, height=height,
            lora_url=lora_url,
            face_image_url=face_image_url,
            lora_strength=0.8,
            pulid_weight=0.8,
            pulid_start=4.0,
            steps=28
        )

        await context.bot.send_chat_action(chat_id=msg.chat_id, action=ChatAction.UPLOAD_PHOTO)
        start_ts = time.time()
        output = await run_replicate_any_comfyui(inputs)

        if not output:
            raise RuntimeError("Пустой ответ от Replicate")
        urls = list(output) if isinstance(output, (list, tuple)) else [str(output)]
        if not urls:
            raise RuntimeError("Нет URL изображений")

        elapsed = time.time() - start_ts
        cap = f"Готово ({elapsed:.1f} c). Пресет: {preset} · Пол: {gender}"
        await msg.reply_photo(photo=urls[0], caption=cap)

    except Exception as e:
        logging.exception("Ошибка генерации: %s", e)
        await msg.reply_text(
            f"Не смогла сгенерировать: {e}\n"
            f"Проверь REPLICATE_API_TOKEN, доступность lora_url/face_image_url по HTTPS, "
            f"и что файл {WORKFLOW_JSON_PATH} рядом с main.py."
        )

async def on_user_upload_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    load_profile,
    save_profile,
    get_current_avatar_name,
    get_avatar,
    resolve_direct_photo_url,
    start_lora_training,           # ваша функция
):
    uid = update.effective_user.id
    msg = update.effective_message

    try:
        photo_url = await resolve_direct_photo_url(update, context)
        prof = load_profile(uid)
        av_name = get_current_avatar_name(prof)
        av = get_avatar(prof, av_name)
        if "uploads" not in av:
            av["uploads"] = []
        if len(av["uploads"]) < MAX_UPLOADS_FOR_TRAIN:
            av["uploads"].append(photo_url)
        save_profile(prof)

        await msg.reply_text(f"Фото загружено ({len(av['uploads'])}/{MAX_UPLOADS_FOR_TRAIN}).")

        if len(av["uploads"]) >= MAX_UPLOADS_FOR_TRAIN and av.get("status") in (None, "idle", "failed"):
            face_url = await select_front_face_from_first_10(av["uploads"][:MAX_UPLOADS_FOR_TRAIN])
            av["face_image_url"] = face_url
            av["status"] = "training"
            job_id = start_lora_training(av["uploads"][:MAX_UPLOADS_FOR_TRAIN], avatar_name=av["name"])
            av["train_job_id"] = job_id
            save_profile(prof)
            await msg.reply_text("Начала обучение вашей LoRA. Сообщу, когда будет готово.")
    except Exception as e:
        logging.exception("Ошибка приёма фото: %s", e)
        await msg.reply_text(f"Не удалось принять фото: {e}")

async def training_poller_tick(
    load_all_profiles, save_profile, get_current_avatar_name, get_avatar, check_lora_training_status
):
    """Один тик поллера: обойдёт всех юзеров и обновит статусы."""
    profiles = load_all_profiles()
    for uid, prof in profiles.items():
        av_name = get_current_avatar_name(prof)
        av = get_avatar(prof, av_name)
        if av.get("status") == "training" and av.get("train_job_id"):
            status, lora_url = check_lora_training_status(av["train_job_id"])
            if status == "succeeded" and lora_url:
                av["status"] = "succeeded"
                av["lora_url"] = lora_url
                save_profile(prof)
            elif status == "failed":
                av["status"] = "failed"
                save_profile(prof)
