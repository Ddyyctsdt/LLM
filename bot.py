import asyncio
import json
import os
import re
import threading
import time
from collections import deque
from typing import List, Optional, Tuple, Dict, Any

import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, filters, ContextTypes
)
from llama_cpp import Llama

# -------------------- تنظیمات اولیه --------------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
MODEL_URL = "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-UD-Q4_K_XL.gguf?download=true"
MODEL_PATH = "model.gguf"
SETTINGS_DIR = "user_settings"
CHAT_SESSIONS_DIR = "chat_sessions"
LOGS_DIR = "logs"

STATUS_DOWNLOADING = "downloading"
STATUS_READY = "ready"
STATUS_ERROR = "error"

current_status = STATUS_DOWNLOADING
download_progress = 0
download_error_msg = ""
llm: Optional[Llama] = None

user_request_lock: Dict[int, bool] = {}
request_queue = deque()
queue_processing = False

active_jobs: Dict[str, Dict[str, Any]] = {}

DEFAULT_SETTINGS = {
    "max_tokens": 500,
    "temperature": 0.7,
    "top_p": 0.95,
    "n_ctx": 4096,  # افزایش برای Cache Reuse بهتر
    "streaming": True,
    "reply": True,
    "developer_mode": False,
    "system_prompt_enabled": False,
    "system_prompt": "تو یک دستیار مفید و بی‌سازشکاری.",
    "show_thinking_timer": True,
    "enable_thinking": True  # گزینه جدید: فعال/غیرفعال کردن Think
}

# -------------------- توابع کمکی --------------------
def ensure_dirs():
    os.makedirs(SETTINGS_DIR, exist_ok=True)
    os.makedirs(CHAT_SESSIONS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

def load_user_settings(user_id: int) -> dict:
    path = os.path.join(SETTINGS_DIR, f"{user_id}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
            for key in DEFAULT_SETTINGS:
                if key not in data:
                    data[key] = DEFAULT_SETTINGS[key]
            return data
    return DEFAULT_SETTINGS.copy()

def save_user_settings(user_id: int, settings: dict):
    path = os.path.join(SETTINGS_DIR, f"{user_id}.json")
    with open(path, "w") as f:
        json.dump(settings, f)

def remove_think_tags(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def clean_response(text: str) -> str:
    text = remove_think_tags(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def load_chat_history(user_id: int, chat_id: str) -> List[dict]:
    filename = os.path.join(CHAT_SESSIONS_DIR, f"{user_id}_{chat_id}.json")
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return []

def save_chat_history(user_id: int, chat_id: str, history: List[dict]):
    filename = os.path.join(CHAT_SESSIONS_DIR, f"{user_id}_{chat_id}.json")
    with open(filename, "w") as f:
        json.dump(history, f)

def get_chat_name(user_id: int, chat_id: str) -> str:
    name_file = os.path.join(CHAT_SESSIONS_DIR, f"{user_id}_{chat_id}_name.txt")
    if os.path.exists(name_file):
        with open(name_file, "r") as nf:
            return nf.read().strip()
    return chat_id

def set_chat_name(user_id: int, chat_id: str, name: str):
    name_file = os.path.join(CHAT_SESSIONS_DIR, f"{user_id}_{chat_id}_name.txt")
    with open(name_file, "w") as nf:
        nf.write(name)

def list_user_chats(user_id: int) -> List[Tuple[str, str]]:
    chats = []
    for f in os.listdir(CHAT_SESSIONS_DIR):
        if f.startswith(f"{user_id}_") and f.endswith(".json") and not f.endswith("_name.txt"):
            chat_id = f.replace(f"{user_id}_", "").replace(".json", "")
            chats.append((chat_id, get_chat_name(user_id, chat_id)))
    return chats

def generate_chat_name_from_history(user_id: int, chat_id: str) -> Optional[str]:
    history = load_chat_history(user_id, chat_id)
    user_msgs = [msg["content"] for msg in history if msg["role"] == "user"]
    if len(user_msgs) < 3:
        return None
    prompt = f"Based on the following user messages, generate a short 3-word title (in Persian) that summarizes the main topic:\n" + "\n".join(user_msgs[:3])
    try:
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.3
        )
        summary = response['choices'][0]['message']['content'].strip()
        words = summary.split()[:3]
        return " ".join(words)
    except:
        return None

def update_chat_name_if_needed(user_id: int, chat_id: str):
    history = load_chat_history(user_id, chat_id)
    user_msg_count = sum(1 for msg in history if msg["role"] == "user")
    if user_msg_count % 3 == 0 and user_msg_count > 0:
        current_name = get_chat_name(user_id, chat_id)
        if current_name == chat_id or not current_name:
            new_name = generate_chat_name_from_history(user_id, chat_id)
            if new_name:
                set_chat_name(user_id, chat_id, new_name)

def build_history_with_system(user_id: int, history: List[dict]) -> List[dict]:
    settings = load_user_settings(user_id)
    if settings.get("system_prompt_enabled", False):
        system_prompt = settings.get("system_prompt", DEFAULT_SETTINGS["system_prompt"])
        if not history or history[0].get("role") != "system":
            return [{"role": "system", "content": system_prompt}] + history
    return history

# -------------------- دانلود مدل --------------------
def download_model():
    global download_progress, current_status, download_error_msg
    try:
        headers = {}
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        response = requests.get(MODEL_URL, stream=True, timeout=60, headers=headers)
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    download_progress = (downloaded / total_size) * 100
        global llm
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=DEFAULT_SETTINGS["n_ctx"],
            n_threads=os.cpu_count() or 2,  # استفاده از تمام هسته‌های CPU
            chat_format="qwen",
            verbose=False
        )
        current_status = STATUS_READY
        download_progress = 100
    except Exception as e:
        current_status = STATUS_ERROR
        download_error_msg = str(e)

# -------------------- توابع کمکی توکن --------------------
def count_tokens(text: str) -> int:
    return len(text) // 4

# -------------------- تولید پاسخ غیراستریم --------------------
def get_response_non_streaming(user_id: int, chat_id: str, prompt: str) -> Tuple[str, int, int]:
    settings = load_user_settings(user_id)
    raw_history = load_chat_history(user_id, chat_id)
    history = build_history_with_system(user_id, raw_history)
    history.append({"role": "user", "content": prompt})
    prompt_text = json.dumps(history)
    prompt_tokens = count_tokens(prompt_text)

    response = llm.create_chat_completion(
        messages=history,
        max_tokens=settings["max_tokens"],
        temperature=settings["temperature"],
        top_p=settings["top_p"],
        stream=False,
        chat_template_kwargs={"enable_thinking": settings.get("enable_thinking", True)}
    )
    raw_response = response['choices'][0]['message']['content']
    cleaned_response = clean_response(raw_response)
    completion_tokens = count_tokens(cleaned_response)
    raw_history.append({"role": "user", "content": prompt})
    raw_history.append({"role": "assistant", "content": cleaned_response})
    save_chat_history(user_id, chat_id, raw_history)
    update_chat_name_if_needed(user_id, chat_id)
    return cleaned_response, prompt_tokens, completion_tokens

# -------------------- تولید پاسخ با استریم --------------------
def generate_response_stream(user_id: int, chat_id: str, prompt: str):
    settings = load_user_settings(user_id)
    raw_history = load_chat_history(user_id, chat_id)
    history = build_history_with_system(user_id, raw_history)
    history.append({"role": "user", "content": prompt})
    prompt_text = json.dumps(history)
    prompt_tokens = count_tokens(prompt_text)

    stream = llm.create_chat_completion(
        messages=history,
        max_tokens=settings["max_tokens"],
        temperature=settings["temperature"],
        top_p=settings["top_p"],
        stream=True,
        chat_template_kwargs={"enable_thinking": settings.get("enable_thinking", True)}
    )
    full_response = ""
    for chunk in stream:
        if "choices" in chunk and len(chunk["choices"]) > 0:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                full_response += content
                yield content, None, None
    cleaned = clean_response(full_response)
    completion_tokens = count_tokens(cleaned)
    raw_history.append({"role": "user", "content": prompt})
    raw_history.append({"role": "assistant", "content": cleaned})
    save_chat_history(user_id, chat_id, raw_history)
    update_chat_name_if_needed(user_id, chat_id)
    yield None, prompt_tokens, completion_tokens

# -------------------- تابع تایمر و ارسال پاسخ --------------------
async def send_response(update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int, chat_id: str, prompt: str):
    settings = load_user_settings(user_id)
    dev_mode = settings.get("developer_mode", False)
    show_timer = settings.get("show_thinking_timer", True) and not dev_mode

    job_id = f"{user_id}_{chat_id}_{int(time.time()*1000)}"
    stop_flag = False
    timer_message = None
    timer_task = None
    response_message = None
    sent_messages = []

    active_jobs[job_id] = {
        "stop_flag": False,
        "user_id": user_id,
        "chat_id": update.effective_chat.id,
        "sent_messages": [],
        "timer_message_id": None,
        "timer_task": None
    }

    try:
        if dev_mode:
            full_response, prompt_tokens, completion_tokens = get_response_non_streaming(user_id, chat_id, prompt)
            timestamp = int(time.time())
            filename = f"response_{user_id}_{chat_id}_{timestamp}.txt"
            filepath = os.path.join(LOGS_DIR, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"User: {prompt}\n\nAssistant:\n{full_response}\n\n---\nPrompt tokens: {prompt_tokens}\nCompletion tokens: {completion_tokens}")
            with open(filepath, "rb") as doc:
                await update.message.reply_document(
                    document=InputFile(doc, filename=filename),
                    reply_to_message_id=update.message.message_id if settings.get("reply", True) else None
                )
            return

        streaming = settings.get("streaming", True)

        if show_timer:
            timer_msg = await update.message.reply_text(
                "🧠 در حال فکر کردن... 0 ثانیه\n(برای لغو، دکمه زیر را بزنید)",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("❌ لغو", callback_data=f"cancel_{job_id}")]])
            )
            timer_message = timer_msg
            active_jobs[job_id]["timer_message_id"] = timer_msg.message_id
            active_jobs[job_id]["sent_messages"].append((timer_msg.chat_id, timer_msg.message_id))
            start_time = time.time()
            async def timer_loop():
                while not stop_flag:
                    elapsed = int(time.time() - start_time)
                    try:
                        await context.bot.edit_message_text(
                            chat_id=timer_msg.chat_id,
                            message_id=timer_msg.message_id,
                            text=f"🧠 در حال فکر کردن... {elapsed} ثانیه\n(برای لغو، دکمه زیر را بزنید)",
                            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("❌ لغو", callback_data=f"cancel_{job_id}")]])
                        )
                    except Exception:
                        pass
                    await asyncio.sleep(5)
            timer_task = asyncio.create_task(timer_loop())
            active_jobs[job_id]["timer_task"] = timer_task

        if not streaming:
            full_response, prompt_tokens, completion_tokens = get_response_non_streaming(user_id, chat_id, prompt)
            parts = [full_response[i:i+3500] for i in range(0, len(full_response), 3500)]
            for idx, part in enumerate(parts):
                prefix = f"(بخش {idx+1}/{len(parts)})\n" if len(parts) > 1 else ""
                text = prefix + part
                msg = await update.message.reply_text(text, reply_to_message_id=update.message.message_id if settings.get("reply", True) else None)
                sent_messages.append((msg.chat_id, msg.message_id))
            await update.message.reply_text(f"📊 آمار توکن: ورودی={prompt_tokens} | خروجی={completion_tokens} | مجموع={prompt_tokens+completion_tokens}")
        else:
            generator = generate_response_stream(user_id, chat_id, prompt)
            first_chunk = True
            current_text = ""
            last_edit_time = 0
            part_counter = 1
            total_prompt_tokens = 0
            total_completion_tokens = 0

            for token, pt, ct in generator:
                if stop_flag:
                    break
                if token:
                    current_text += token
                    now = time.time()
                    if now - last_edit_time >= 7 or len(current_text) > 3500:
                        if len(current_text) > 3500:
                            part_text = current_text[:3500]
                            remainder = current_text[3500:]
                            if response_message:
                                await context.bot.edit_message_text(
                                    chat_id=response_message.chat_id,
                                    message_id=response_message.message_id,
                                    text=part_text
                                )
                            prefix = f"(ادامه {part_counter+1})\n"
                            new_msg = await update.message.reply_text(
                                prefix + "⏳ در حال تولید...",
                                reply_to_message_id=update.message.message_id if settings.get("reply", True) else None
                            )
                            sent_messages.append((new_msg.chat_id, new_msg.message_id))
                            response_message = new_msg
                            current_text = remainder
                            part_counter += 1
                        else:
                            if not first_chunk and response_message:
                                try:
                                    await context.bot.edit_message_text(
                                        chat_id=response_message.chat_id,
                                        message_id=response_message.message_id,
                                        text=current_text
                                    )
                                except Exception:
                                    pass
                            else:
                                sent = await update.message.reply_text(
                                    "⏳ در حال تولید...",
                                    reply_to_message_id=update.message.message_id if settings.get("reply", True) else None
                                )
                                sent_messages.append((sent.chat_id, sent.message_id))
                                response_message = sent
                                first_chunk = False
                        last_edit_time = now
                else:
                    total_prompt_tokens = pt or 0
                    total_completion_tokens = ct or 0
                    break

            if not stop_flag and response_message and current_text:
                await context.bot.edit_message_text(
                    chat_id=response_message.chat_id,
                    message_id=response_message.message_id,
                    text=current_text
                )
            if not stop_flag:
                await update.message.reply_text(f"📊 آمار توکن: ورودی={total_prompt_tokens} | خروجی={total_completion_tokens} | مجموع={total_prompt_tokens+total_completion_tokens}")

        if timer_message and not stop_flag:
            try:
                await context.bot.delete_message(chat_id=timer_message.chat_id, message_id=timer_message.message_id)
            except:
                pass

    except Exception as e:
        await update.message.reply_text(f"❌ خطا: {str(e)}")
    finally:
        if timer_task and not timer_task.done():
            timer_task.cancel()
        if job_id in active_jobs:
            del active_jobs[job_id]
        user_request_lock[user_id] = False

# -------------------- لغو job --------------------
async def cancel_job(query, job_id: str):
    job_info = active_jobs.get(job_id)
    if not job_info:
        await query.answer("این درخواست قبلاً تمام شده یا لغو شده است.")
        return
    job_info["stop_flag"] = True
    if job_info.get("timer_task"):
        job_info["timer_task"].cancel()
    for chat_id, msg_id in job_info.get("sent_messages", []):
        try:
            await query.bot.delete_message(chat_id=chat_id, message_id=msg_id)
        except:
            pass
    if job_info.get("timer_message_id"):
        try:
            await query.bot.delete_message(chat_id=job_info["chat_id"], message_id=job_info["timer_message_id"])
        except:
            pass
    user_id = job_info["user_id"]
    if job_id in active_jobs:
        del active_jobs[job_id]
    user_request_lock[user_id] = False
    await query.edit_message_text("✅ تولید پاسخ لغو شد. تمام پیام‌های مربوطه حذف گردید.")

# -------------------- پردازنده صف --------------------
async def process_queue(app: Application):
    global queue_processing
    while True:
        if not queue_processing and request_queue:
            queue_processing = True
            update, context, user_id, chat_id, prompt = request_queue.popleft()
            if user_request_lock.get(user_id, False):
                await update.message.reply_text("شما در حال حاضر یک درخواست فعال دارید. لطفاً پس از اتمام آن، درخواست جدید بدهید.")
                queue_processing = False
                continue
            user_request_lock[user_id] = True
            try:
                await send_response(update, context, user_id, chat_id, prompt)
            except Exception as e:
                print(f"Error: {e}")
            finally:
                user_request_lock[user_id] = False
                queue_processing = False
        await asyncio.sleep(0.5)

# -------------------- هندلرهای تلگرام --------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if current_status != STATUS_READY:
        if current_status == STATUS_DOWNLOADING:
            await update.message.reply_text(f"📥 مدل در حال دانلود... {download_progress:.1f}%")
        else:
            await update.message.reply_text(f"❌ خطا: {download_error_msg}")
        return
    await show_main_menu(update, context, as_edit=False)

async def show_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, as_edit: bool = False):
    keyboard = [
        [InlineKeyboardButton("🎛️ تنظیمات", callback_data="settings")],
        [InlineKeyboardButton("💬 چت‌های من", callback_data="list_chats")],
        [InlineKeyboardButton("➕ چت جدید", callback_data="new_chat")],
        [InlineKeyboardButton("🔄 ریست اکانت", callback_data="reset_account")],
        [InlineKeyboardButton("ℹ️ راهنما", callback_data="help")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    text = "✅ مدل آماده است. منوی اصلی:"
    if as_edit and update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=reply_markup)
    else:
        await update.message.reply_text(text, reply_markup=reply_markup)

async def safe_edit(query, text, reply_markup=None, parse_mode=None):
    if (query.message.text == text and 
        query.message.reply_markup == reply_markup and 
        parse_mode is None):
        return
    try:
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=parse_mode)
    except Exception as e:
        if "Message is not modified" not in str(e):
            raise

async def refresh_settings_menu(query, user_id):
    settings = load_user_settings(user_id)
    text = (
        f"📏 max_tokens: {settings['max_tokens']}\n"
        f"🌡️ temperature: {settings['temperature']}\n"
        f"🎯 top_p: {settings['top_p']}\n"
        f"📖 n_ctx: {settings['n_ctx']}\n"
        f"⚡ استریمینگ: {'فعال' if settings['streaming'] else 'غیرفعال'}\n"
        f"🔁 ریپلای: {'فعال' if settings['reply'] else 'غیرفعال'}\n"
        f"👨‍💻 حالت برنامه‌نویس: {'فعال' if settings.get('developer_mode', False) else 'غیرفعال'}\n"
        f"🧠 نمایش تایمر: {'فعال' if settings.get('show_thinking_timer', True) else 'غیرفعال'}\n"
        f"🧠 Think: {'فعال' if settings.get('enable_thinking', True) else 'غیرفعال'}\n"
        f"✏️ سیستم پرامپت: {'فعال' if settings.get('system_prompt_enabled', False) else 'خاموش'}\n"
    )
    if settings.get('system_prompt_enabled', False):
        text += f"📝 متن پرامپت: {settings.get('system_prompt', DEFAULT_SETTINGS['system_prompt'])[:50]}..."
    keyboard = [
        [InlineKeyboardButton("ویرایش max_tokens", callback_data="edit_max_tokens")],
        [InlineKeyboardButton("ویرایش temperature", callback_data="edit_temp")],
        [InlineKeyboardButton("ویرایش top_p", callback_data="edit_top_p")],
        [InlineKeyboardButton("ویرایش n_ctx", callback_data="edit_n_ctx")],
        [InlineKeyboardButton("تغییر استریمینگ", callback_data="toggle_streaming")],
        [InlineKeyboardButton("تغییر ریپلای", callback_data="toggle_reply")],
        [InlineKeyboardButton("تغییر حالت برنامه‌نویس", callback_data="toggle_dev_mode")],
        [InlineKeyboardButton("تغییر نمایش تایمر", callback_data="toggle_timer")],
        [InlineKeyboardButton("فعال/غیرفعال کردن Think", callback_data="toggle_thinking")],
        [InlineKeyboardButton("فعال/غیرفعال کردن سیستم پرامپت", callback_data="toggle_system_prompt")],
        [InlineKeyboardButton("✏️ ویرایش متن سیستم پرامپت", callback_data="edit_system_prompt_text")],
        [InlineKeyboardButton("🔙 بازگشت", callback_data="main_menu")]
    ]
    await safe_edit(query, text, InlineKeyboardMarkup(keyboard))

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    user_id = query.from_user.id

    if data.startswith("cancel_"):
        job_id = data.replace("cancel_", "")
        await cancel_job(query, job_id)
        return

    if data == "settings":
        await refresh_settings_menu(query, user_id)
        return

    elif data.startswith("edit_"):
        param = data.replace("edit_", "")
        context.user_data["waiting_for"] = param
        if param == "system_prompt_text":
            await safe_edit(query, "لطفاً متن سیستم پرامپت جدید را بفرستید (این متن در ابتدای هر چت جدید در صورت فعال بودن سیستم پرامپت قرار می‌گیرد):")
        else:
            await safe_edit(query, f"لطفاً مقدار جدید {param} را بفرستید:")
        return

    elif data in ["toggle_streaming", "toggle_reply", "toggle_dev_mode", "toggle_timer", "toggle_system_prompt", "toggle_thinking"]:
        settings = load_user_settings(user_id)
        if data == "toggle_streaming":
            settings["streaming"] = not settings["streaming"]
        elif data == "toggle_reply":
            settings["reply"] = not settings["reply"]
        elif data == "toggle_dev_mode":
            settings["developer_mode"] = not settings.get("developer_mode", False)
        elif data == "toggle_timer":
            settings["show_thinking_timer"] = not settings.get("show_thinking_timer", True)
        elif data == "toggle_system_prompt":
            settings["system_prompt_enabled"] = not settings.get("system_prompt_enabled", False)
        elif data == "toggle_thinking":
            settings["enable_thinking"] = not settings.get("enable_thinking", True)
        save_user_settings(user_id, settings)
        await refresh_settings_menu(query, user_id)
        return

    elif data == "list_chats":
        chats = list_user_chats(user_id)
        if not chats:
            await safe_edit(query, "هیچ چتی ندارید. با گزینه «چت جدید» شروع کنید.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 بازگشت", callback_data="main_menu")]]))
        else:
            keyboard = [[InlineKeyboardButton(f"{name} ({cid[:6]})", callback_data=f"chat_{cid}")] for cid, name in chats]
            keyboard.append([InlineKeyboardButton("🔙 بازگشت", callback_data="main_menu")])
            await safe_edit(query, "چت‌های شما:", reply_markup=InlineKeyboardMarkup(keyboard))
        return

    elif data.startswith("chat_"):
        chat_id = data.replace("chat_", "")
        context.user_data["active_chat"] = chat_id
        await safe_edit(query, f"اکنون در چت {get_chat_name(user_id, chat_id)} هستید. پیام خود را بفرستید.")
        return

    elif data == "new_chat":
        new_id = str(int(time.time()))
        context.user_data["active_chat"] = new_id
        set_chat_name(user_id, new_id, new_id)
        await safe_edit(query, f"چت جدید با شناسه {new_id} ساخته شد. اکنون می‌توانید پیام بفرستید.")
        return

    elif data == "reset_account":
        for f in os.listdir(SETTINGS_DIR):
            if f.startswith(f"{user_id}"):
                os.remove(os.path.join(SETTINGS_DIR, f))
        for f in os.listdir(CHAT_SESSIONS_DIR):
            if f.startswith(f"{user_id}_"):
                os.remove(os.path.join(CHAT_SESSIONS_DIR, f))
        save_user_settings(user_id, DEFAULT_SETTINGS.copy())
        await safe_edit(query, "اکانت شما با موفقیت ریست شد. تمام داده‌ها پاک گردید.")
        await asyncio.sleep(1)
        await show_main_menu(update, context, as_edit=False)
        return

    elif data == "help":
        help_text = (
            "📖 *راهنما:*\n"
            "- `max_tokens`: حداکثر طول پاسخ (توکن). بیشتر = پاسخ بلندتر.\n"
            "- `temperature`: خلاقیت مدل (0 = خشک، 1 = خلاق، 2 = بسیار خلاق).\n"
            "- `top_p`: تنوع کلمات (0.9 مقدار خوب).\n"
            "- `n_ctx`: حافظه مکالمه (بیشتر = خاطره بیشتر، رم بیشتر).\n"
            "- استریمینگ: نمایش زنده پاسخ (ادیت هر ۷ ثانیه).\n"
            "- ریپلای: پاسخ به پیام شما به صورت ریپلای.\n"
            "- حالت برنامه‌نویس: خروجی فقط در فایل txt ارسال می‌شود (بدون نمایش در چت).\n"
            "- نمایش تایمر: هنگام تولید پاسخ، یک پیام با تایمر نشان می‌دهد و می‌توانید لغو کنید.\n"
            "- Think: فعال/غیرفعال کردن فرآیند استدلال داخلی مدل (غیرفعال = سریع‌تر).\n"
            "- سیستم پرامپت: می‌توانید یک دستورالعمل سیستمی برای مدل تنظیم کنید (پیش‌فرض خاموش).\n"
            "- ریست اکانت: تمام داده‌های شما را پاک می‌کند.\n\n"
            "برای تغییر هر گزینه، به بخش تنظیمات بروید."
        )
        await safe_edit(query, help_text, parse_mode=ParseMode.MARKDOWN)
        return

    elif data == "main_menu":
        await show_main_menu(update, context, as_edit=True)
        return

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if current_status != STATUS_READY:
        await update.message.reply_text("مدل هنوز آماده نیست. لطفاً چند دقیقه دیگر تلاش کنید.")
        return

    user_id = update.effective_user.id
    text = update.message.text

    waiting = context.user_data.get("waiting_for")
    if waiting:
        try:
            settings = load_user_settings(user_id)
            if waiting == "max_tokens":
                settings["max_tokens"] = int(text)
            elif waiting == "temp":
                settings["temperature"] = float(text)
            elif waiting == "top_p":
                settings["top_p"] = float(text)
            elif waiting == "n_ctx":
                settings["n_ctx"] = int(text)
            elif waiting == "system_prompt_text":
                settings["system_prompt"] = text.strip()
            save_user_settings(user_id, settings)
            await update.message.reply_text(f"{waiting} به {text} تغییر یافت.")
        except Exception as e:
            await update.message.reply_text(f"مقدار نامعتبر: {e}")
        context.user_data["waiting_for"] = None
        await show_main_menu(update, context, as_edit=False)
        return

    active_chat = context.user_data.get("active_chat")
    if not active_chat:
        await update.message.reply_text("لطفاً ابتدا از منو یک چت انتخاب کنید یا چت جدید بسازید.")
        return

    if user_request_lock.get(user_id, False):
        await update.message.reply_text("شما در حال حاضر یک درخواست فعال دارید. لطفاً پس از اتمام آن، درخواست جدید بدهید.")
        return

    request_queue.append((update, context, user_id, active_chat, text))
    await update.message.reply_text("درخواست شما در صف قرار گرفت. لطفاً صبر کنید...")

# -------------------- خاموشی خودکار --------------------
def shutdown_bot():
    print("⏰ زمان اجرا (۵:۵۰ ساعت) به پایان رسید. خاموش کردن ربات...")
    os._exit(0)

async def post_init(app: Application):
    asyncio.create_task(process_queue(app))

def main():
    ensure_dirs()
    threading.Thread(target=download_model, daemon=True).start()
    timer = threading.Timer(350 * 60, shutdown_bot)
    timer.start()

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    loop = asyncio.get_event_loop()
    loop.create_task(post_init(app))

    print("ربات شروع به کار کرد. مدل در حال دانلود در پس‌زمینه...")
    app.run_polling()

if __name__ == "__main__":
    main()
