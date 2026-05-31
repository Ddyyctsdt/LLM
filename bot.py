import asyncio
import json
import os
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

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

# وضعیت‌های ربات
STATUS_DOWNLOADING = "downloading"
STATUS_READY = "ready"
STATUS_ERROR = "error"

# متغیرهای سراسری
current_status = STATUS_DOWNLOADING
download_progress = 0
download_error_msg = ""
llm: Optional[Llama] = None

# صف درخواست‌ها: هر عنصر (user_id, chat_id, message_id, text)
request_queue = deque()
queue_lock = threading.Lock()
processing = False

# تنظیمات پیش‌فرض
DEFAULT_SETTINGS = {
    "max_tokens": 500,
    "temperature": 0.7,
    "top_p": 0.95,
    "n_ctx": 2048,      # می‌توانید تا 8192 افزایش دهید (رم بیشتر)
    "streaming": True,   # فعال/غیرفعال کردن استریمینگ
    "reply": True
}

# -------------------- توابع کمکی --------------------
def ensure_dirs():
    os.makedirs(SETTINGS_DIR, exist_ok=True)
    os.makedirs(CHAT_SESSIONS_DIR, exist_ok=True)

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

def list_user_chats(user_id: int) -> List[str]:
    chats = []
    for f in os.listdir(CHAT_SESSIONS_DIR):
        if f.startswith(f"{user_id}_") and f.endswith(".json"):
            chat_id = f.replace(f"{user_id}_", "").replace(".json", "")
            chats.append(chat_id)
    return chats

# -------------------- دانلود مدل (در ترد جداگانه) --------------------
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
        # بارگذاری مدل
        global llm
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=DEFAULT_SETTINGS["n_ctx"],  # مقدار اولیه، بعداً per-user نمی‌شود تغییر داد بدون reload
            n_threads=4,
            chat_format="qwen",
            verbose=False
        )
        current_status = STATUS_READY
        download_progress = 100
    except Exception as e:
        current_status = STATUS_ERROR
        download_error_msg = str(e)

# -------------------- تولید پاسخ با استریم (generator) --------------------
def generate_response(user_id: int, chat_id: str, prompt: str):
    """
    Generator که توکن‌های پاسخ را یک‌به‌یک تولید می‌کند.
    همچنین تاریخچه را مدیریت می‌کند.
    """
    settings = load_user_settings(user_id)
    history = load_chat_history(user_id, chat_id)
    history.append({"role": "user", "content": prompt})

    # فراخوانی استریم
    stream = llm.create_chat_completion(
        messages=history,
        max_tokens=settings["max_tokens"],
        temperature=settings["temperature"],
        top_p=settings["top_p"],
        stream=True
    )

    full_response = ""
    for chunk in stream:
        if "choices" in chunk and len(chunk["choices"]) > 0:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                full_response += content
                yield content  # تحویل لحظه‌ای توکن‌ها

    # پس از اتمام، تاریخچه را ذخیره کن
    history.append({"role": "assistant", "content": full_response})
    save_chat_history(user_id, chat_id, history)

# -------------------- ارسال پاسخ با مدیریت ادیت و تقسیم به پیام‌های متعدد --------------------
async def send_streaming_response(update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int, chat_id: str, prompt: str):
    """
    پاسخ را با استریم و ادیت پیام ارسال می‌کند.
    اگر خروجی از 4096 کاراکتر گذشت، پیام جدیدی می‌فرستد و آن را هم ادیت می‌کند.
    """
    settings = load_user_settings(user_id)
    if not settings.get("streaming", True):
        # حالت غیر استریم: یکباره تولید و ارسال
        full = ""
        for token in generate_response(user_id, chat_id, prompt):
            full += token
        # تقسیم به بخش‌های 4096 کاراکتری
        for i in range(0, len(full), 4096):
            part = full[i:i+4096]
            await update.message.reply_text(part, reply_to_message_id=update.message.message_id)
        return

    # حالت استریم
    generator = generate_response(user_id, chat_id, prompt)
    first_chunk = True
    message = None
    current_text = ""
    message_index = 0
    messages_list = []  # list of (message_id, chat_id)

    try:
        for token in generator:
            current_text += token
            # اگر پیامی وجود نداشته باشد، اولین پیام را بفرست
            if first_chunk:
                # ابتدا یک پیام خالی بفرستیم (یا یک نقطه)
                sent = await update.message.reply_text("⏳ در حال تولید...", reply_to_message_id=update.message.message_id)
                message = sent
                messages_list.append((message.message_id, message.chat_id))
                first_chunk = False
                continue

            # اگر متن فعلی بیشتر از 4000 کاراکتر شد (برای جلوگیری از ارسال ناگهانی پیام جدید)
            if len(current_text) > 4000:
                # پیام فعلی را با متن کامل نهایی کن (ادیت نهایی)
                await context.bot.edit_message_text(
                    chat_id=message.chat_id,
                    message_id=message.message_id,
                    text=current_text[:4096]
                )
                # باقیمانده را برای پیام جدید ذخیره کن
                remaining = current_text[4096:]
                # ارسال پیام جدید
                new_msg = await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="⏳ ادامه...",
                    reply_to_message_id=update.message.message_id
                )
                messages_list.append((new_msg.message_id, new_msg.chat_id))
                message = new_msg
                current_text = remaining
            else:
                # ادیت پیام جاری
                try:
                    await context.bot.edit_message_text(
                        chat_id=message.chat_id,
                        message_id=message.message_id,
                        text=current_text
                    )
                except Exception:
                    # ممکن است پیام قبلی حذف شده باشد یا خطای نرخ رخ دهد – نادیده بگیر
                    pass
            # یک تأخیر بسیار کوتاه برای جلوگیری از flood (0.2 ثانیه)
            await asyncio.sleep(0.2)
    except Exception as e:
        # در صورت خطا، پیام خطا را به همان پیام جاری یا پیام جدید ارسال کن
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
        # درخواست خطادار از صف حذف می‌شود (در پردازنده صف مدیریت می‌شود)
        raise
    else:
        # پس از اتمام، آخرین ادیت را انجام بده (در صورت نیاز)
        if message and current_text:
            try:
                await context.bot.edit_message_text(
                    chat_id=message.chat_id,
                    message_id=message.message_id,
                    text=current_text
                )
            except:
                pass

# -------------------- پردازنده صف --------------------
async def process_queue(app: Application):
    global processing
    while True:
        if not processing and request_queue:
            with queue_lock:
                item = request_queue.popleft()
                processing = True
            try:
                update, context, user_id, chat_id, prompt = item
                await send_streaming_response(update, context, user_id, chat_id, prompt)
            except Exception as e:
                # خطا قبلاً در send_streaming_response لاگ شده، فقط ادامه بده
                print(f"Error processing queue item: {e}")
            finally:
                processing = False
        await asyncio.sleep(0.5)

# -------------------- هندلرهای تلگرام --------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if current_status == STATUS_DOWNLOADING:
        await update.message.reply_text(f"📥 مدل در حال دانلود... {download_progress:.1f}%")
    elif current_status == STATUS_ERROR:
        await update.message.reply_text(f"❌ خطا: {download_error_msg}")
    else:
        await show_main_menu(update, context)

async def show_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🎛️ تنظیمات", callback_data="settings")],
        [InlineKeyboardButton("💬 چت‌های من", callback_data="list_chats")],
        [InlineKeyboardButton("➕ چت جدید", callback_data="new_chat")],
        [InlineKeyboardButton("ℹ️ راهنما", callback_data="help")]
    ]
    await update.message.reply_text("✅ مدل آماده است. منوی اصلی:", reply_markup=InlineKeyboardMarkup(keyboard))

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    user_id = query.from_user.id

    if data == "settings":
        settings = load_user_settings(user_id)
        text = (
            f"تنظیمات شما:\n"
            f"📏 max_tokens: {settings['max_tokens']}\n"
            f"🌡️ temperature: {settings['temperature']}\n"
            f"🎯 top_p: {settings['top_p']}\n"
            f"📖 n_ctx: {settings['n_ctx']}\n"
            f"⚡ استریمینگ: {'فعال' if settings['streaming'] else 'غیرفعال'}\n"
            f"🔁 ریپلای: {'فعال' if settings['reply'] else 'غیرفعال'}"
        )
        keyboard = [
            [InlineKeyboardButton("ویرایش max_tokens", callback_data="edit_max_tokens")],
            [InlineKeyboardButton("ویرایش temperature", callback_data="edit_temp")],
            [InlineKeyboardButton("ویرایش top_p", callback_data="edit_top_p")],
            [InlineKeyboardButton("ویرایش n_ctx", callback_data="edit_n_ctx")],
            [InlineKeyboardButton("تغییر استریمینگ", callback_data="toggle_streaming")],
            [InlineKeyboardButton("تغییر ریپلای", callback_data="toggle_reply")],
            [InlineKeyboardButton("🔙 بازگشت", callback_data="main_menu")]
        ]
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

    elif data == "edit_max_tokens":
        context.user_data["waiting_for"] = "max_tokens"
        await query.edit_message_text("لطفاً مقدار جدید max_tokens را به صورت عدد بفرستید (مثال: 800):")
    elif data == "edit_temp":
        context.user_data["waiting_for"] = "temp"
        await query.edit_message_text("لطفاً مقدار جدید temperature (0 تا 2) را بفرستید:")
    elif data == "edit_top_p":
        context.user_data["waiting_for"] = "top_p"
        await query.edit_message_text("لطفاً مقدار جدید top_p (0 تا 1) را بفرستید:")
    elif data == "edit_n_ctx":
        context.user_data["waiting_for"] = "n_ctx"
        await query.edit_message_text("لطفاً مقدار جدید n_ctx را بفرستید (مثال: 4096):")
    elif data == "toggle_streaming":
        settings = load_user_settings(user_id)
        settings["streaming"] = not settings["streaming"]
        save_user_settings(user_id, settings)
        await query.edit_message_text(f"حالت استریمینگ {'فعال' if settings['streaming'] else 'غیرفعال'} شد.")
        await asyncio.sleep(1)
        await query.edit_message_text("تنظیمات به‌روز شد.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 بازگشت", callback_data="settings")]]))
    elif data == "toggle_reply":
        settings = load_user_settings(user_id)
        settings["reply"] = not settings["reply"]
        save_user_settings(user_id, settings)
        await query.edit_message_text(f"حالت ریپلای {'فعال' if settings['reply'] else 'غیرفعال'} شد.")
        await asyncio.sleep(1)
        await query.edit_message_text("تنظیمات به‌روز شد.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 بازگشت", callback_data="settings")]]))
    elif data == "list_chats":
        chats = list_user_chats(user_id)
        if not chats:
            await query.edit_message_text("هیچ چتی ندارید. با گزینه «چت جدید» شروع کنید.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 بازگشت", callback_data="main_menu")]]))
        else:
            keyboard = [[InlineKeyboardButton(f"چت {c}", callback_data=f"chat_{c}")] for c in chats]
            keyboard.append([InlineKeyboardButton("🔙 بازگشت", callback_data="main_menu")])
            await query.edit_message_text("چت‌های شما:", reply_markup=InlineKeyboardMarkup(keyboard))
    elif data.startswith("chat_"):
        chat_id = data.replace("chat_", "")
        context.user_data["active_chat"] = chat_id
        await query.edit_message_text(f"اکنون در چت {chat_id} هستید. پیام خود را بفرستید.")
    elif data == "new_chat":
        new_id = str(int(time.time()))
        context.user_data["active_chat"] = new_id
        await query.edit_message_text(f"چت جدید با شناسه {new_id} ساخته شد. اکنون می‌توانید پیام بفرستید.")
    elif data == "help":
        help_text = (
            "📖 *راهنما:*\n"
            "- `max_tokens`: حداکثر طول پاسخ (توکن). بیشتر = پاسخ بلندتر.\n"
            "- `temperature`: خلاقیت مدل (0 = خشک، 1 = خلاق، 2 = بسیار خلاق).\n"
            "- `top_p`: تنوع کلمات (0.9 مقدار خوب).\n"
            "- `n_ctx`: حافظه مکالمه (بیشتر = خاطره بیشتر، رم بیشتر).\n"
            "- استریمینگ: نمایش زنده پاسخ.\n"
            "- ریپلای: پاسخ به پیام شما به صورت ریپلای.\n\n"
            "برای تغییر هر گزینه، به بخش تنظیمات بروید."
        )
        await query.edit_message_text(help_text, parse_mode=ParseMode.MARKDOWN)
    elif data == "main_menu":
        keyboard = [
            [InlineKeyboardButton("🎛️ تنظیمات", callback_data="settings")],
            [InlineKeyboardButton("💬 چت‌های من", callback_data="list_chats")],
            [InlineKeyboardButton("➕ چت جدید", callback_data="new_chat")],
            [InlineKeyboardButton("ℹ️ راهنما", callback_data="help")]
        ]
        await query.edit_message_text("منوی اصلی:", reply_markup=InlineKeyboardMarkup(keyboard))

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if current_status != STATUS_READY:
        await update.message.reply_text("مدل هنوز آماده نیست. لطفاً چند دقیقه دیگر تلاش کنید.")
        return

    user_id = update.effective_user.id
    text = update.message.text

    # اگر در حالت تنظیمات هستیم
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
        return

    # اگر چت فعال وجود ندارد
    active_chat = context.user_data.get("active_chat")
    if not active_chat:
        await update.message.reply_text("لطفاً ابتدا از منو یک چت انتخاب کنید یا چت جدید بسازید.")
        return

    # اضافه کردن به صف
    with queue_lock:
        request_queue.append((update, context, user_id, active_chat, text))
    await update.message.reply_text("درخواست شما در صف قرار گرفت. لطفاً صبر کنید...")

# -------------------- تابع خاموشی خودکار --------------------
def shutdown_bot():
    print("⏰ زمان اجرا (۵:۵۰ ساعت) به پایان رسید. خاموش کردن ربات...")
    os._exit(0)

# -------------------- اجرای اصلی --------------------
async def post_init(app: Application):
    # شروع پردازنده صف
    asyncio.create_task(process_queue(app))

def main():
    ensure_dirs()
    # شروع دانلود در ترد جداگانه
    threading.Thread(target=download_model, daemon=True).start()

    # تایمر خاموشی (350 دقیقه)
    timer = threading.Timer(350 * 60, shutdown_bot)
    timer.start()

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # پس از آماده شدن اپلیکیشن، post_init اجرا می‌شود
    loop = asyncio.get_event_loop()
    loop.create_task(post_init(app))

    print("ربات شروع به کار کرد. مدل در حال دانلود در پس‌زمینه...")
    app.run_polling()

if __name__ == "__main__":
    main()
