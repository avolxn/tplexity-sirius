import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from telethon import TelegramClient

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

api_id = os.getenv("API_ID")
api_hash = os.getenv("API_HASH")
phone = os.getenv("PHONE")
session_name = os.getenv("SESSION_NAME", "my_session")

if not api_id or not api_hash:
    print("❌ Ошибка: Укажите API_ID и API_HASH в .env файле")
    print("Получить можно здесь: https://my.telegram.org")
    print("\n📖 См. подробную инструкцию в файле SETUP_GUIDE.md")
    exit(1)


project_root = Path(__file__).parent.parent.parent.parent
session_path = project_root / f"{session_name}.session"

print("=" * 60)
print("🔐 Авторизация Telegram клиента")
print("=" * 60)
print(f"📁 Файл сессии будет сохранен: {session_path}")
print()


async def main():
    client = TelegramClient(str(session_path), int(api_id), api_hash)

    try:
        await client.connect()
        print("✅ Соединение с Telegram установлено")
        print()

        if not await client.is_user_authorized():
            phone_number = phone or input("📱 Введите номер телефона (с кодом страны, например +79991234567): ").strip()
            print(f"📱 Отправка кода на номер: {phone_number}")
            print()

            await client.send_code_request(phone_number)

            code = input("✉️ Введите код из Telegram/SMS: ").strip()

            try:
                await client.sign_in(phone_number, code)
            except Exception as e:
                if "password" in str(e).lower() or "2FA" in str(e) or "two" in str(e).lower():
                    print()
                    print("🔒 Требуется пароль двухфакторной аутентификации (2FA)")
                    password = input("🔒 Введите пароль 2FA: ").strip()
                    await client.sign_in(password=password)
                else:
                    raise

        print()
        print("=" * 60)
        print("✅ Авторизация успешна!")
        print("=" * 60)
        print()

        me = await client.get_me()
        print(f"👤 Авторизован как: {me.first_name} {me.last_name or ''} (@{me.username or 'без username'})")
        print()
        print(f"📁 Файл сессии сохранен: {session_path}")
        print()
        print("✅ Теперь можно запускать сервисы!")
        print()

    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ Ошибка авторизации: {e}")
        print("=" * 60)
        import traceback

        traceback.print_exc()
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
