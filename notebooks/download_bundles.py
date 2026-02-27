import os
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- 1. CONFIGURAZIONE PERCORSI E PARAMETRI ---

# Percorso assoluto alla root del progetto (due livelli su rispetto al notebook)
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))

CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "bundles_dataset.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "images", "bundles")

MAX_WORKERS = 50

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.zara.com/",
}

# Crea la cartella se non esiste
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Verifica che il CSV esista prima di procedere
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"CSV non trovato in: {CSV_PATH}\n"
        f"Directory corrente: {os.getcwd()}\n"
        f"Assicurati che 'product_dataset.csv' sia in '{os.path.join(PROJECT_ROOT, 'data', 'raw')}'"
    )

# --- 2. FUNZIONE DI DOWNLOAD SINGOLO ---
def download_image(row):
    asset_id = row.get('bundle_asset_id')
    url = row.get('bundle_image_url')

    if not isinstance(url, str) or not url.startswith("http"):
        return False

    save_path = os.path.join(OUTPUT_DIR, f"{asset_id}.jpg")

    if os.path.exists(save_path):
        return True

    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True

    except Exception as e:
        print(f"Errore con {url}: {e}")
        return False

# --- 3. ESECUZIONE MULTITHREADING ---
def main():
    print(f"📂 Project root: {PROJECT_ROOT}")
    print(f"Caricamento dataset da {CSV_PATH}...")
    
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=['bundle_image_url'])
    print(f"Trovate {len(df)} immagini da scaricare.")
    
    success_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        rows = df.to_dict("records")

        futures = {executor.submit(download_image, row): row for row in rows}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Download in corso"):
            if future.result():
                success_count += 1
                
    print(f"\n🚀 Download completato! {success_count}/{len(df)} immagini salvate in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()