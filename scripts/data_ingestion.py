import os
import pandas as pd
from telethon.sync import TelegramClient
from telethon.tl.types import MessageMediaPhoto
import pytesseract
from PIL import Image
import logging
import sys
import asyncio
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configure channels
CHANNELS = [
    '@Shageronlinestore',
    '@ethio_electronics',
    'AddisShopping',
    'EthioShop',
    '@EthiopianMarketplace',
    '@addisbazaar'
]

# Create data directories
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/raw/images', exist_ok=True)

async def fetch_messages(client, channel_url, limit=50):
    """Fetch messages from a Telegram channel with OCR for images"""
    logger.info(f"📡 Starting scrape for: {channel_url}")
    
    try:
        channel = await client.get_entity(channel_url)
    except Exception as e:
        logger.error(f"❌ Failed to get entity for {channel_url}: {e}")
        return []
    
    messages_data = []
    async for message in client.iter_messages(channel, limit=limit):
        try:
            # Extract vendor name without special characters
            vendor = ''.join(c for c in channel.title if c.isalnum() or c in " _-")
            
            row = {
                "vendor": vendor,  # Cleaned vendor name
                "message": message.text or "",
                "views": message.views or 0,
                "timestamp": message.date.isoformat() if message.date else "",
                "image_path": "",
                "ocr_text": ""
            }

            # Process images with OCR
            if isinstance(message.media, MessageMediaPhoto):
                try:
                    image_name = f"{vendor}_{message.id}.jpg"
                    image_path = f"data/raw/images/{image_name}"
                    await client.download_media(message.media, file=image_path)
                    row["image_path"] = image_path
                    
                    # Perform OCR
                    image = Image.open(image_path)
                    text = pytesseract.image_to_string(image, lang='amh')
                    row["ocr_text"] = text
                    logger.info(f"📸 Processed image OCR for message {message.id}")
                except Exception as e:
                    logger.error(f"❌ Image processing failed: {e}")
            
            messages_data.append(row)
            
        except Exception as e:
            logger.error(f"❌ Error processing message {message.id}: {e}")
    
    logger.info(f"✅ Scraped {len(messages_data)} messages from {channel_url}")
    return messages_data

async def main():
    """Main function to scrape all channels"""
    # Initialize client with your credentials
    client = TelegramClient(
        session='ethio_ecommerce_session',
        api_id=21894791,  # Replace with your API ID
        api_hash='d4a1d4f8e5c94e2b1a4b6b8c4f7e8a9b'  # Replace with your API HASH
    )
    
    await client.start(
        phone='+251927135189',  # Your phone number
        password='YOUR_PASSWORD'  # Your password if 2FA enabled
    )
    
    me = await client.get_me()
    logger.info(f"🔓 Signed in successfully as {me.first_name}")
    
    # Scrape all channels
    all_data = []
    for channel in CHANNELS:
        channel_data = await fetch_messages(client, channel)
        all_data.extend(channel_data)
    
    # Save data to single file without timestamp
    if all_data:
        df = pd.DataFrame(all_data)
        df.drop_duplicates(subset=["message", "image_path"], inplace=True)
        
        output_file = "data/raw/scraped_messages.csv"
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"💾 Saved {len(df)} messages to {output_file}")
    else:
        logger.warning("⚠️ No data scraped. Check channel URLs and permissions.")
    
    await client.disconnect()

if __name__ == "__main__":
    # Set event loop policy for Windows compatibility
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())