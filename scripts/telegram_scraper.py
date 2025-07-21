import os
import pandas as pd
from telethon.sync import TelegramClient
from telethon.tl.types import MessageMediaPhoto
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")

client = TelegramClient('ethio_ecommerce_session', API_ID, API_HASH)
client.start()

channels = [
    'https://t.me/shageronlinestore',
    'https://t.me/AddisShopping', 
    'https://t.me/EthioShop'
]

os.makedirs('data/raw/images', exist_ok=True)

def fetch_messages(channel_url, limit=200):
    print(f"Scraping channel: {channel_url}")
    channel = client.get_entity(channel_url)
    messages = client.iter_messages(channel, limit=limit)

    rows = []
    for i, msg in enumerate(messages):
        row = {
            "channel": channel_url,
            "message": msg.text if msg.text else "",
            "views": msg.views,
            "timestamp": msg.date.strftime("%Y-%m-%d %H:%M:%S") if msg.date else "",
            "sender_id": msg.sender_id,
            "image_path": ""
        }

        # Download image if present
        if msg.media and isinstance(msg.media, MessageMediaPhoto):
            image_path = f"data/raw/images/{channel.username}_{i}.jpg"
            try:
                client.download_media(msg, file=image_path)
                row["image_path"] = image_path
            except Exception as e:
                print(f"Failed to download image: {e}")
        
        if row["message"] or row["image_path"]:
            rows.append(row)

    return rows

def scrape_all_channels():
    all_data = []
    for ch in channels:
        data = fetch_messages(ch)
        all_data.extend(data)
    return pd.DataFrame(all_data)

if __name__ == "__main__":
    df = scrape_all_channels()
    df.drop_duplicates(subset=["message", "image_path"], inplace=True)
    df.to_csv("data/raw/scraped_messages.csv", index=False, encoding='utf-8-sig')
    print(f"Saved {len(df)} messages to data/raw/scraped_messages.csv")
