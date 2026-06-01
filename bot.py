import asyncio
import json
import os
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple
import re

import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, filters, ContextTypes
)
from llama_cpp import Llama

# -------------------- تنظیمات اولیه --------------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
MODEL_URL = "https://huggingface.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive/resolve/main/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf?download=true"
MODEL_PATH = "model.gguf"
SETTINGS_DIR = "user_settings"
CHAT_SESSIONS_DIR = "chat_sessions"
AGENT_SESSIONS_DIR = "agent_sessions"  # برای آپدیت دوم
LOGS_DIR = "logs"  # برای ذخیره فایل‌های حالت برنامه‌نویس

# وضعیت‌های ربات
STATUS_DOWNLOADING = "downloading"
STATUS_READY = "ready"
STATUS_ERROR = "error"

# متغیرهای سراسری
current_status = STATUS_DOWNLOADING
download_progress = 0
download_error_msg = ""
llm: Optional[Llama] = None

# صف درخواست‌ها (هر کاربر فقط یک درخواست فعال)
user_request_lock = {}  # user_id -> bool
request_queue = deque()
queue_processing = False

# تنظیمات پیش‌فرض
DEFAULT_SETTINGS = {
    "max_tokens": 500,
    "temperature": 0.7,
    "top_p": 0.95,
    "n_ctx": 2048,
    "streaming": True,      # حالت استریمینگ
    "reply": True,          # ریپلای
    "developer_mode": False # حالت برنامه‌نویس
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
            return json.load(f)
    return DEFAULT_SETTINGS.copy()

def save_user_settings(user_id: int, settings: dict):
    path = os.path.join(SETTINGS_DIR, f"{user_id}.json")
    with open(path, "w") as f:
        json.dump(settings, f)

def load_chat_history(user_id: int, chat_id: str) -> List[dict]:
    filename = os.path.join(CHAT_SESSIONS_DIR, f"{user_id}_{chat_id}.json")
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return [{"role": "system", "content": "تو یک دستیار مفید و بی‌سازشکاری."}]

def save_chat_history(user_id: int, chat_id: str, history: List[dict]):
    filename = os.path.join(CHAT_SESSIONS_DIR, f"{user_id}_{chat_id}.json")
    with open(filename, "w") as f:
        json.dump(history, f)

def list_user_chats(user_id: int) -> List[Tuple[str, str]]:
    """بازگرداندن لیست (chat_id, chat_name) برای کاربر"""
    chats = []
    for f in os.listdir(CHAT_SESSIONS_DIR):
        if f.startswith(f"{user_id}_") and f.endswith(".json"):
            chat_id = f.replace(f"{user_id}_", "").replace(".json", "")
            # نام چت را از یک فایل جداگانه یا از همان فایل تاریخچه بخوان
            name_file = os.path.join(CHAT_SESSIONS_DIR, f"{user_id}_{chat_id}_name.txt")
            if os.path.exists(name_file):
                with open(name_file, "r") as nf:
                    chat_name = nf.read().strip()
            else:
                chat_name = chat_id  # fallback
            chats.append((chat_id, chat_name))
    return chats

def set_chat_name(user_id: int, chat_id: str, name: str):
    name_file = os.path.join(CHAT_SESSIONS_DIR, f"{user_id}_{chat_id}_name.txt")
    with open(name_file, "w") as nf:
        nf.write(name)

def generate_chat_name_from_history(user_id: int, chat_id: str):
    """با استفاده از مدل، خلاصه ۳ کلمه‌ای از هدف چت تولید کن"""
    history = load_chat_history(user_id, chat_id)
    if len(history) < 4:  # هنوز ۳ پیام رد و بدل نشده
        return None
    # گرفتن پیام‌های کاربر (بدون سیستم و دستیار)
    user_messages = [msg["content"] for msg in history if msg["role"] == "user"]
    if len(user_messages) < 3:
        return None
    # خلاصه‌سازی با خود مدل (درخواست کوتاه)
    prompt = f"Based on the following user messages, generate a short 3-word title (in Persian) that summarizes the main topic:\n" + "\n".join(user_messages[:3])
    try:
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.3
        )
        summary = response['choices'][0]['message']['content'].strip()
        # محدود به ۳ کلمه
        words = summary.split()[:3]
        return " ".join(words)
    except:
        return None

def update_chat_name_if_needed(user_id: int, chat_id: str):
    """بعد از هر ۳ پیام کاربر، بررسی کن و اسم را به‌روز کن"""
    history = load_chat_history(user_id, chat_id)
    user_msg_count = sum(1 for msg in history if msg["role"] == "user")
    if user_msg_count % 3 == 0 and user_msg_count > 0:
        name_file = os.path.join(CHAT_SESSIONS_DIR, f"{user_id}_{chat_id}_name.txt")
        if not os.path.exists(name_file) or os.path.getsize(name_file) == 0:
            new_name = generate_chat_name_from_history(user_id, chat_id)
            if new_name:
                set_chat_name(user_id, chat_id, new_name)
                return new_name
    return None

# -------------------- دانلود مدل --------------------
def download_model():
    global download_progress, current_status, download_error_msg
    try:
        response = requests.get(MODEL_URL, stream=True, timeout=60)
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
            n_threads=4,
            chat_format="qwen",
            verbose=False
        )
        current_status = STATUS_READY
        download_progress = 100
    except Exception as e:
        current_status = STATUS_ERROR
        download_error_msg = str(e)

# -------------------- تولید پاسخ (با استریم یا بدون) --------------------
def generate_response(user_id: int, chat_id: str, prompt: str):
    """Generator که توکن‌ها را یک‌به‌یک برمی‌گرداند و آمار مصرف توکن را نیز بازمی‌گرداند"""
    settings = load_user_settings(user_id)
    history = load_chat_history(user_id, chat_id)
    history.append({"role": "user", "content": prompt})

    # اگر حالت برنامه‌نویس فعال باشد، خروجی را در فایل ذخیره می‌کنیم
    dev_mode = settings.get("developer_mode", False)

    stream = llm.create_chat_completion(
        messages=history,
        max_tokens=settings["max_tokens"],
        temperature=settings["temperature"],
        top_p=settings["top_p"],
        stream=True
    )
    full_response = ""
    total_tokens = 0
    for chunk in stream:
        if "choices" in chunk and len(chunk["choices"]) > 0:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                full_response += content
                yield content, None  # yield توکن
        # در برخی نسخه‌های llama-cpp-python، توکن مصرفی در آخرین chunk می‌آید
        if "usage" in chunk:
            total_tokens = chunk["usage"].get("total_tokens", 0)

    # ذخیره تاریخچه
    history.append({"role": "assistant", "content": full_response})
    save_chat_history(user_id, chat_id, history)

    # اگر حالت برنامه‌نویس فعال باشد، خروجی خام را در فایل ذخیره کن
    if dev_mode:
        log_file = os.path.join(LOGS_DIR, f"{user_id}_{chat_id}_{int(time.time())}.txt")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"User: {prompt}\n\nAssistant:\n{full_response}\n\n---\nTokens: {total_tokens}")
        # ارسال فایل به کاربر (اختیاری) - اینجا فقط ذخیره می‌کنیم

    yield None, total_tokens  # آخرین yield آمار

# -------------------- ارسال پاسخ با مدیریت ادیت و تقسیم خروجی --------------------
async def send_response(update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int, chat_id: str, prompt: str):
    settings = load_user_settings(user_id)
    streaming = settings.get("streaming", True)

    if not streaming or settings.get("developer_mode", False):
        # حالت غیراستریم یا برنامه‌نویس: یکجا تولید کن و تقسیم به بخش‌های 3500 کاراکتری
        full_response = ""
        total_tokens = 0
        for token, _ in generate_response(user_id, chat_id, prompt):
            if token:
                full_response += token
            else:
                # آخرین yield: آمار
                total_tokens = _ or 0
        # تقسیم به بخش‌های 3500
        parts = [full_response[i:i+3500] for i in range(0, len(full_response), 3500)]
        for idx, part in enumerate(parts):
            prefix = f"(ادامه {idx+1}/{len(parts)})\n" if idx > 0 else ""
            text = prefix + part
            await update.message.reply_text(text, reply_to_message_id=update.message.message_id if settings.get("reply", True) else None)
        # ارسال آمار
        await update.message.reply_text(f"📊 آمار: {total_tokens} توکن مصرف شد.")
        return

    # حالت استریمینگ با ادیت هر ۷ ثانیه و مدیریت خروجی بلند
    generator = generate_response(user_id, chat_id, prompt)
    first_chunk = True
    message = None
    current_text = ""
    last_edit_time = 0
    total_tokens = 0
    part_counter = 1
    message_parts = []  # لیست پیام‌های ارسالی (برای مدیریت ادیت)

    try:
        for token, stats in generator:
            if token:
                current_text += token
                now = time.time()
                # هر ۷ ثانیه ادیت کن (و اگر طول از 3500 گذشت، پیام جدید بفرست)
                if now - last_edit_time >= 7:
                    if len(current_text) > 3500:
                        # بخش فعلی را نهایی کن (تا 3500 کاراکتر)
                        part_text = current_text[:3500]
                        remainder = current_text[3500:]
                        if message:
                            await context.bot.edit_message_text(
                                chat_id=message.chat_id,
                                message_id=message.message_id,
                                text=part_text
                            )
                        # ارسال پیام جدید برای ادامه
                        new_msg = await update.message.reply_text(
                            f"(ادامه {part_counter+1})\n⏳ در حال تولید...",
                            reply_to_message_id=update.message.message_id if settings.get("reply", True) else None
                        )
                        message_parts.append((new_msg.chat_id, new_msg.message_id, part_counter+1))
                        message = new_msg
                        current_text = remainder
                        part_counter += 1
                    else:
                        # ادیت پیام جاری
                        if not first_chunk:
                            try:
                                await context.bot.edit_message_text(
                                    chat_id=message.chat_id,
                                    message_id=message.message_id,
                                    text=current_text
                                )
                            except Exception:
                                pass
                        else:
                            # اولین بار: ارسال پیام اول
                            sent = await update.message.reply_text(
                                "⏳ در حال تولید...",
                                reply_to_message_id=update.message.message_id if settings.get("reply", True) else None
                            )
                            message = sent
                            message_parts.append((sent.chat_id, sent.message_id, 1))
                            first_chunk = False
                    last_edit_time = now
            else:
                # پایان استریم، آمار توکن
                total_tokens = stats or 0
                break

        # پس از اتمام، ادیت نهایی روی آخرین پیام
        if message and current_text:
            await context.bot.edit_message_text(
                chat_id=message.chat_id,
                message_id=message.message_id,
                text=current_text
            )
        # ارسال آمار به عنوان یک پیام جداگانه
        await update.message.reply_text(f"📊 آمار: {total_tokens} توکن مصرف شد.")
    except Exception as e:
        error_text = f"❌ خطا در تولید پاسخ: {str(e)}"
        if message:
            try:
                await context.bot.edit_message_text(
                    chat_id=message.chat_id,
                    message_id=message.message_id,
                    text=error_text
                )
            except:
                await update.message.reply_text(error_text)
        else:
            await update.message.reply_text(error_text)
        raise

# -------------------- پردازنده صف (با قفل کاربری) --------------------
async def process_queue(app: Application):
    global queue_processing
    while True:
        if not queue_processing and request_queue:
            queue_processing = True
            update, context, user_id, chat_id, prompt = request_queue.popleft()
            # قفل کاربری: بررسی می‌کنیم آیا این کاربر هم اکنون درخواست فعال دارد؟
            if user_request_lock.get(user_id, False):
                await update.message.reply_text("شما در حال حاضر یک درخواست فعال دارید. لطفاً پس از اتمام آن، درخواست جدید بدهید.")
                queue_processing = False
                continue
            # قفل را بگذار
            user_request_lock[user_id] = True
            try:
                await send_response(update, context, user_id, chat_id, prompt)
            except Exception as e:
                print(f"Error processing user {user_id}: {e}")
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
    await show_main_menu(update, context)

async def show_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🎛️ تنظیمات", callback_data="settings")],
        [InlineKeyboardButton("💬 چت‌های من", callback_data="list_chats")],
        [InlineKeyboardButton("➕ چت جدید", callback_data="new_chat")],
        [InlineKeyboardButton("🔄 ریست اکانت", callback_data="reset_account")],
        [InlineKeyboardButton("ℹ️ راهنما", callback_data="help")]
    ]
    await update.message.reply_text("✅ مدل آماده است. منوی اصلی:", reply_markup=InlineKeyboardMarkup(keyboard))

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    user_id = query.from_user.id

    # جلوگیری از ادیت تکراری: بررسی می‌کنیم اگر محتوای جدید با قدیم یکی بود، ادیت نکنیم
    async def safe_edit(text, reply_markup=None):
        if query.message.text == text and reply_markup == query.message.reply_markup:
            return
        await query.edit_message_text(text, reply_markup=reply_markup)

    if data == "settings":
        settings = load_user_settings(user_id)
        text = (
            f"📏 max_tokens: {settings['max_tokens']}\n"
            f"🌡️ temperature: {settings['temperature']}\n"
            f"🎯 top_p: {settings['top_p']}\n"
            f"📖 n_ctx: {settings['n_ctx']}\n"
            f"⚡ استریمینگ: {'فعال' if settings['streaming'] else 'غیرفعال'}\n"
            f"🔁 ریپلای: {'فعال' if settings['reply'] else 'غیرفعال'}\n"
            f"👨‍💻 حالت برنامه‌نویس: {'فعال' if settings.get('developer_mode', False) else 'غیرفعال'}"
        )
        keyboard = [
            [InlineKeyboardButton("ویرایش max_tokens", callback_data="edit_max_tokens")],
            [InlineKeyboardButton("ویرایش temperature", callback_data="edit_temp")],
            [InlineKeyboardButton("ویرایش top_p", callback_data="edit_top_p")],
            [InlineKeyboardButton("ویرایش n_ctx", callback_data="edit_n_ctx")],
            [InlineKeyboardButton("تغییر استریمینگ", callback_data="toggle_streaming")],
            [InlineKeyboardButton("تغییر ریپلای", callback_data="toggle_reply")],
            [InlineKeyboardButton("تغییر حالت برنامه‌نویس", callback_data="toggle_dev_mode")],
            [InlineKeyboardButton("🔙 بازگشت", callback_data="main_menu")]
        ]
        await safe_edit(text, InlineKeyboardMarkup(keyboard))

    elif data.startswith("edit_"):
        param = data.replace("edit_", "")
        context.user_data["waiting_for"] = param
        await safe_edit(f"لطفاً مقدار جدید {param} را بفرستید:")

    elif data == "toggle_streaming":
        settings = load_user_settings(user_id)
        settings["streaming"] = not settings["streaming"]
        save_user_settings(user_id, settings)
        await safe_edit(f"حالت استریمینگ {'فعال' if settings['streaming'] else 'غیرفعال'} شد.")
        await asyncio.sleep(1)
        # برگرد به منوی تنظیمات
        await button_callback(update, context)  # Recursive call to refresh settings menu

    elif data == "toggle_reply":
        settings = load_user_settings(user_id)
        settings["reply"] = not settings["reply"]
        save_user_settings(user_id, settings)
        await safe_edit(f"حالت ریپلای {'فعال' if settings['reply'] else 'غیرفعال'} شد.")
        await asyncio.sleep(1)
        await button_callback(update, context)

    elif data == "toggle_dev_mode":
        settings = load_user_settings(user_id)
        settings["developer_mode"] = not settings.get("developer_mode", False)
        save_user_settings(user_id, settings)
        await safe_edit(f"حالت برنامه‌نویس {'فعال' if settings['developer_mode'] else 'غیرفعال'} شد.")
        await asyncio.sleep(1)
        await button_callback(update, context)

    elif data == "list_chats":
        chats = list_user_chats(user_id)
        if not chats:
            await safe_edit("هیچ چتی ندارید. با گزینه «چت جدید» شروع کنید.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 بازگشت", callback_data="main_menu")]]))
        else:
            keyboard = [[InlineKeyboardButton(f"{name} ({cid[:6]})", callback_data=f"chat_{cid}")] for cid, name in chats]
            keyboard.append([InlineKeyboardButton("🔙 بازگشت", callback_data="main_menu")])
            await safe_edit("چت‌های شما:", reply_markup=InlineKeyboardMarkup(keyboard))

    elif data.startswith("chat_"):
        chat_id = data.replace("chat_", "")
        context.user_data["active_chat"] = chat_id
        await safe_edit(f"اکنون در چت {chat_id} هستید. پیام خود را بفرستید.")

    elif data == "new_chat":
        new_id = str(int(time.time()))
        context.user_data["active_chat"] = new_id
        # ذخیره یک نام موقت رندم (همان new_id)
        set_chat_name(user_id, new_id, new_id)
        await safe_edit(f"چت جدید با شناسه {new_id} ساخته شد. اکنون می‌توانید پیام بفرستید.")

    elif data == "reset_account":
        # پاک کردن تمام فایل‌های کاربر
        for f in os.listdir(SETTINGS_DIR):
            if f.startswith(f"{user_id}"):
                os.remove(os.path.join(SETTINGS_DIR, f))
        for f in os.listdir(CHAT_SESSIONS_DIR):
            if f.startswith(f"{user_id}_"):
                os.remove(os.path.join(CHAT_SESSIONS_DIR, f))
        # ریست تنظیمات به پیش‌فرض
        save_user_settings(user_id, DEFAULT_SETTINGS.copy())
        await safe_edit("اکانت شما با موفقیت ریست شد. تمام داده‌ها پاک گردید.")
        await asyncio.sleep(1)
        await show_main_menu(update, context)

    elif data == "help":
        help_text = (
            "📖 *راهنما:*\n"
            "- `max_tokens`: حداکثر طول پاسخ (توکن). بیشتر = پاسخ بلندتر.\n"
            "- `temperature`: خلاقیت مدل (0 = خشک، 1 = خلاق، 2 = بسیار خلاق).\n"
            "- `top_p`: تنوع کلمات (0.9 مقدار خوب).\n"
            "- `n_ctx`: حافظه مکالمه (بیشتر = خاطره بیشتر، رم بیشتر).\n"
            "- استریمینگ: نمایش زنده پاسخ (ادیت هر ۷ ثانیه).\n"
            "- ریپلای: پاسخ به پیام شما به صورت ریپلای.\n"
            "- حالت برنامه‌نویس: ذخیره خروجی خام در فایل متنی.\n"
            "- ریست اکانت: تمام داده‌های شما را پاک می‌کند.\n\n"
            "برای تغییر هر گزینه، به بخش تنظیمات بروید."
        )
        await safe_edit(help_text, parse_mode=ParseMode.MARKDOWN)

    elif data == "main_menu":
        await show_main_menu(update, context)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if current_status != STATUS_READY:
        await update.message.reply_text("مدل هنوز آماده نیست. لطفاً چند دقیقه دیگر تلاش کنید.")
        return

    user_id = update.effective_user.id
    text = update.message.text

    # اگر در حال انتظار برای ورودی تنظیمات هستیم
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
            save_user_settings(user_id, settings)
            await update.message.reply_text(f"{waiting} به {text} تغییر یافت.")
        except Exception as e:
            await update.message.reply_text(f"مقدار نامعتبر: {e}")
        context.user_data["waiting_for"] = None
        # پس از ثبت، به منوی تنظیمات برگرد
        await show_main_menu(update, context)
        return

    active_chat = context.user_data.get("active_chat")
    if not active_chat:
        await update.message.reply_text("لطفاً ابتدا از منو یک چت انتخاب کنید یا چت جدید بسازید.")
        return

    # اضافه کردن به صف (با رعایت قفل کاربری)
    if user_request_lock.get(user_id, False):
        await update.message.reply_text("شما در حال حاضر یک درخواست فعال دارید. لطفاً پس از اتمام آن، درخواست جدید بدهید.")
        return
    request_queue.append((update, context, user_id, active_chat, text))
    await update.message.reply_text("درخواست شما در صف قرار گرفت. لطفاً صبر کنید...")

    # پس از پایان پاسخ (در send_response خودکار)، بررسی کنیم که آیا نیاز به به‌روزرسانی نام چت است
    # اما چون پردازش غیرهمزمان است، می‌توانیم یک تسک جداگانه بعد از اتمام پاسخ اجرا کنیم
    # برای سادگی، در اینجا پس از اضافه شدن به صف، منتظر می‌مانیم و در خود send_response بعد از ذخیره تاریخچه، نام را به‌روز می‌کنیم.
    # در send_response پس از save_chat_history می‌توانیم update_chat_name_if_needed را صدا بزنیم.
    # برای این کار باید تابع send_response را اصلاح کنیم (در داخل generator نمی‌توانیم به راحتی، ولی بعد از پایان می‌توانیم).
    # در کد فعلی، پس از اتمام generator در send_response، می‌توانیم نام را به‌روز کنیم. اصلاح خواهیم کرد.

# -------------------- خاموشی خودکار --------------------
def shutdown_bot():
    print("⏰ زمان اجرا (۵:۵۰ ساعت) به پایان رسید. خاموش کردن ربات...")
    os._exit(0)

# -------------------- اجرای اصلی --------------------
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
