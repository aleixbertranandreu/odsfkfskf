import os
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import Counter

# --- 1. CONFIGURAZIONE PERCORSI E PARAMETRI ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))

CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "bundles_dataset.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "images", "bundles")

MAX_WORKERS = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"CSV non trovato in: {CSV_PATH}\n"
        f"Directory corrente: {os.getcwd()}\n"
        f"Assicurati che 'bundle_dataset.csv' sia in '{os.path.join(PROJECT_ROOT, 'data', 'raw')}'"
    )

# --- 2. HEADERS CHE SIMULANO UN BROWSER REALE ---
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

# --- 3. FUNZIONE DI DOWNLOAD SINGOLO ---
def download_image(row):
    asset_id = row['bundle_asset_id']
    url = row['bundle_image_url']
    
    save_path = os.path.join(OUTPUT_DIR, f"{asset_id}.jpg")
    
    if os.path.exists(save_path):
        return True, None
        
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True, None
    except Exception as e:
        return False, str(e)

# --- 4. ESECUZIONE MULTITHREADING ---
def main():
    print(f"📂 Project root: {PROJECT_ROOT}")
    print(f"Caricamento dataset da {CSV_PATH}...")
    
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=['bundle_image_url'])
    print(f"Trovate {len(df)} immagini da scaricare.")
    
    success_count = 0
    errors = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_image, row): row for _, row in df.iterrows()}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Download in corso"):
            success, error = future.result()
            if success:
                success_count += 1
            elif error:
                errors.append(error)
    
    print(f"\n🚀 Download completato! {success_count}/{len(df)} immagini salvate in: {OUTPUT_DIR}")
    
    # --- 5. RIEPILOGO ERRORI ---
    if errors:
        error_summary = Counter()
        for e in errors:
            if "404" in e:               error_summary["404 Not Found"] += 1
            elif "403" in e:             error_summary["403 Forbidden"] += 1
            elif "429" in e:             error_summary["429 Rate Limited"] += 1
            elif "timed out" in e.lower() or "timeout" in e.lower():
                                         error_summary["Timeout"] += 1
            elif "ConnectionError" in e: error_summary["Connection Error"] += 1
            else:                        error_summary["Altro"] += 1

        print(f"\n📊 Riepilogo errori ({len(errors)} totali):")
        for err_type, count in error_summary.most_common():
            print(f"  {err_type}: {count}")
    else:
        print("\n✅ Nessun errore!")

if __name__ == "__main__":
    main()