"""
download_missing_products.py — Download missing product images from Zara CDN
Handles anti-bot measures with User-Agent, retries, and rate limiting.
"""
import os
import time
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "product_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "images", "products")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    "Referer": "https://www.zara.com/",
}

MAX_WORKERS = 12
MAX_RETRIES = 3
TIMEOUT = 15


def download_image(pid, url, output_dir):
    out_path = os.path.join(output_dir, f"{pid}.jpg")
    if os.path.exists(out_path):
        return pid, True, "exists"

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            if resp.status_code == 200 and len(resp.content) > 1000:
                with open(out_path, 'wb') as f:
                    f.write(resp.content)
                return pid, True, "downloaded"
            elif resp.status_code == 429:
                time.sleep(2 ** attempt)
            else:
                return pid, False, f"status={resp.status_code}"
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
            else:
                return pid, False, str(e)[:50]

    return pid, False, "max_retries"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    existing = {f.replace('.jpg', '') for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg')}
    missing = df[~df['product_asset_id'].isin(existing)]

    print(f"📦 Total products: {len(df)}")
    print(f"✅ Already downloaded: {len(existing)}")
    print(f"⬇️  To download: {len(missing)}")

    if len(missing) == 0:
        print("Nothing to download!")
        return

    downloaded = 0
    failed = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for _, row in missing.iterrows():
            pid = row['product_asset_id']
            url = row['product_image_url']
            if pd.notna(url):
                futures[executor.submit(download_image, pid, url, OUTPUT_DIR)] = pid

        for i, future in enumerate(as_completed(futures)):
            pid, success, msg = future.result()
            if success:
                downloaded += 1
            else:
                failed += 1

            if (i + 1) % 500 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                print(f"  [{i+1}/{len(futures)}] ✅ {downloaded} ❌ {failed} | {rate:.0f} img/s")

    elapsed = time.time() - start
    final_count = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg')])
    print(f"\n🏆 Done in {elapsed:.0f}s")
    print(f"   Downloaded: {downloaded} | Failed: {failed}")
    print(f"   Total images now: {final_count}/{len(df)}")


if __name__ == "__main__":
    main()
