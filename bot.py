import asyncio
import json
import os
import threading
import time
from pathlib import Path
import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
from llama_cpp import Llama

# ---------- تنظیمات ثابت ----------
BOT_TOKEN = os.getenv("BOT_TOKEN")
MODEL_URL = "https://huggingface.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive/resolve/main/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf?download=true"
MODEL_PATH = "model.gguf"
SETTINGS_FILE = "settings.json"
CHAT_SESSIONS_DIR = "chat_sessions"

# وضعیت‌های ربات
STATUS_DOWNLOADING = "downloading"
STATUS_READY = "ready"
STATUS_ERROR = "error"

# متغیرهای سراسری
current_status = STATUS_DOWNLOADING
download_progress = 0
download_error_msg = ""
llm = None
settings = {}  # {"max_tokens": 300, "temperature": 0.7, "top_p": 0.95, "n_ctx": 2048}
# --------------------------------

def load_settings():
    global settings
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            settings = json.load(f)
    else:
        settings = {
            "max_tokens": 300,
            "temperature": 0.7,
            "top_p": 0.95,
            "n_ctx": 2048,
        }
        save_settings()

def save_settings():
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f)

def load_chat_history(user_id: int, chat_id: str) -> list:
    os.makedirs(CHAT_SESSIONS_DIR, exist_ok=True)
    filename = f"{CHAT_SESSIONS_DIR}/{user_id}_{chat_id}.json"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return [{"role": "system", "content": "تو یک دستیار مفید و بی‌سازشکاری."}]

def save_chat_history(user_id: int, chat_id: str, history: list):
    filename = f"{CHAT_SESSIONS_DIR}/{user_id}_{chat_id}.json"
    with open(filename, "w") as f:
        json.dump(history, f)

def list_chat_sessions(user_id: int) -> list:
    os.makedirs(CHAT_SESSIONS_DIR, exist_ok=True)
    sessions = []
    for f in os.listdir(CHAT_SESSIONS_DIR):
        if f.startswith(f"{user_id}_") and f.endswith(".json"):
            chat_id = f.replace(f"{user_id}_", "").replace(".json", "")
            sessions.append(chat_id)
    return sessions

# ---------- دانلود مدل در پس‌زمینه (غیرمسدودکننده) ----------
def download_model_background():
    global download_progress, current_status, llm, download_error_msg
    try:
        response = requests.get(MODEL_URL, stream=True, timeout=60)
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    download_progress = (downloaded / total_size) * 100
        # بارگذاری مدل با تنظیمات ذخیره شده
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=settings.get("n_ctx", 2048),
            n_threads=4,
            chat_format="qwen",
            verbose=False
        )
        current_status = STATUS_READY
        download_progress = 100
    except Exception as e:
        current_status = STATUS_ERROR
        download_error_msg = str(e)
        print(f"Error: {e}")

# ---------- توابع هوش مصنوعی (تنها زمانی که مدل آماده است) ----------
async def ask_model(user_id: int, chat_id: str, message: str) -> str:
    if current_status != STATUS_READY or llm is None:
        return "مدل هنوز آماده نیست. لطفاً بعداً تلاش کنید."

    history = load_chat_history(user_id, chat_id)
    history.append({"role": "user", "content": message})

    response = llm.create_chat_completion(
        messages=history,
        max_tokens=settings.get("max_tokens", 300),
        temperature=settings.get("temperature", 0.7),
        top_p=settings.get("top_p", 0.95)
    )
    reply = response['choices'][0]['message']['content']
    history.append({"role": "assistant", "content": reply})
    save_chat_history(user_id, chat_id, history)
    return reply

# ---------- منوی اصلی (فقط زمانی نمایش داده می‌شود که مدل آماده باشد) ----------
async def get_main_menu():
    keyboard = [
        [InlineKeyboardButton("🎛️ تنظیمات", callback_data="settings")],
        [InlineKeyboardButton("💬 چت‌های من", callback_data="list_chats")],
        [InlineKeyboardButton("➕ چت جدید", callback_data="new_chat")],
    ]
    return InlineKeyboardMarkup(keyboard)

# ---------- هندلرهای تلگرام ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """پاسخ به /start - بسته به وضعیت مدل"""
    if current_status == STATUS_DOWNLOADING:
        await update.message.reply_text(
            f"📥 مدل در حال دانلود است... {download_progress:.1f}% تکمیل شده.\n"
            "لطفاً چند دقیقه دیگر /start را بزنید."
        )
    elif current_status == STATUS_ERROR:
        await update.message.reply_text(
            f"❌ خطا در دانلود یا بارگذاری مدل:\n{download_error_msg}\n"
            "لطفاً بعداً تلاش کنید."
        )
    else:  # STATUS_READY
        await update.message.reply_text(
            "✅ مدل آماده است. از منوی زیر استفاده کنید:",
            reply_markup=await get_main_menu()
        )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """مدیریت کلیک روی دکمه‌های منو (فقط وقتی مدل آماده است فراخوانی می‌شود)"""
    query = update.callback_query
    await query.answer()
    data = query.data
    user_id = query.from_user.id

    if data == "settings":
        keyboard = [
            [InlineKeyboardButton(f"📏 max_tokens ({settings['max_tokens']})", callback_data="edit_max_tokens")],
            [InlineKeyboardButton(f"🌡️ temperature ({settings['temperature']})", callback_data="edit_temp")],
            [InlineKeyboardButton(f"📖 n_ctx ({settings['n_ctx']})", callback_data="edit_ctx")],
            [InlineKeyboardButton("🔙 بازگشت", callback_data="main_menu")],
        ]
        await query.edit_message_text("تنظیمات فعلی:", reply_markup=InlineKeyboardMarkup(keyboard))

    elif data == "list_chats":
        sessions = list_chat_sessions(user_id)
        if not sessions:
            text = "هیچ چتی ندارید. با «چت جدید» شروع کنید."
            await query.edit_message_text(text, reply_markup=await get_main_menu())
        else:
            keyboard = [[InlineKeyboardButton(f"چت {s}", callback_data=f"chat_{s}")] for s in sessions]
            keyboard.append([InlineKeyboardButton("🔙 بازگشت", callback_data="main_menu")])
            await query.edit_message_text("چت‌های شما:", reply_markup=InlineKeyboardMarkup(keyboard))

    elif data.startswith("chat_"):
        chat_id = data.replace("chat_", "")
        context.user_data['active_chat'] = chat_id
        await query.edit_message_text(
            f"در حال چت با شناسه {chat_id}. پیام خود را بفرستید.\nبرای بازگشت به منو، /start را بزنید."
        )

    elif data == "new_chat":
        new_id = str(int(time.time()))
        context.user_data['active_chat'] = new_id
        await query.edit_message_text(
            f"چت جدید با شناسه {new_id} ساخته شد. اکنون می‌توانید پیام بفرستید."
        )

    elif data == "main_menu":
        await query.edit_message_text("منوی اصلی:", reply_markup=await get_main_menu())

    # ویرایش تنظیمات (نمونه فقط برای max_tokens)
    elif data == "edit_max_tokens":
        await query.edit_message_text("مقدار جدید max_tokens را به صورت عدد بفرستید (مثال: 500):")
        context.user_data['waiting_for'] = 'max_tokens'

    elif data == "edit_temp":
        await query.edit_message_text("مقدار جدید temperature را بفرستید (مثال: 0.8):")
        context.user_data['waiting_for'] = 'temp'

    elif data == "edit_ctx":
        await query.edit_message_text("مقدار جدید n_ctx را بفرستید (مثال: 4096):")
        context.user_data['waiting_for'] = 'ctx'

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """پاسخ به پیام‌های متنی کاربر"""
    user_id = update.effective_user.id
    text = update.message.text

    # اگر در حالت تنظیمات هستیم
    waiting = context.user_data.get('waiting_for')
    if waiting:
        try:
            if waiting == 'max_tokens':
                val = int(text)
                settings['max_tokens'] = val
                await update.message.reply_text(f"max_tokens به {val} تغییر کرد.")
            elif waiting == 'temp':
                val = float(text)
                settings['temperature'] = val
                await update.message.reply_text(f"temperature به {val} تغییر کرد.")
            elif waiting == 'ctx':
                val = int(text)
                settings['n_ctx'] = val
                await update.message.reply_text(f"n_ctx به {val} تغییر کرد.")
            save_settings()
            # اگر مدل در حال اجراست، باید با تنظیمات جدید دوباره بارگذاری شود؟ (نیاز به ریستارت دارد)
            # برای سادگی می‌گوییم بعد از ریست بعدی اعمال می‌شود.
        except Exception as e:
            await update.message.reply_text(f"ورودی نامعتبر: {e}")
        context.user_data['waiting_for'] = None
        return

    # اگر مدل آماده نباشد، فقط وضعیت را گزارش بده
    if current_status != STATUS_READY:
        if current_status == STATUS_DOWNLOADING:
            await update.message.reply_text(f"📥 مدل در حال دانلود... {download_progress:.1f}%")
        else:
            await update.message.reply_text("❌ مدل در دسترس نیست. لطفاً بعداً تلاش کنید.")
        return

    # در غیر این صورت، چت معمولی
    active_chat = context.user_data.get('active_chat')
    if not active_chat:
        await update.message.reply_text("لطفاً ابتدا از منو یک چت انتخاب کنید یا چت جدید بسازید.")
        return

    await update.message.chat.send_action(action="typing")
    reply = await ask_model(user_id, active_chat, text)
    await update.message.reply_text(reply)

# ---------- خاموشی خودکار بعد از ۵ ساعت و ۵۰ دقیقه ----------
def shutdown_bot():
    print("⏰ زمان اجرا به پایان رسید. خاموش کردن ربات...")
    os._exit(0)

def main():
    load_settings()
    # شروع دانلود در ترد جداگانه
    download_thread = threading.Thread(target=download_model_background)
    download_thread.start()

    # تایمر خاموشی
    timer = threading.Timer(350 * 60, shutdown_bot)  # 350 دقیقه = 5:50 ساعت
    timer.start()

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("ربات شروع به کار کرد. مدل در حال دانلود در پس‌زمینه...")
    app.run_polling()

if __name__ == "__main__":
    main()
