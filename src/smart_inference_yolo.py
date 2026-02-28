# Copia de smart_inference.py pero adaptado para usar el YOLO del usuario
import os
import torch
import numpy as np
import pickle
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO

# --- 1. CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_CARPETA_TEST = os.path.join(BASE_DIR, "data", "images", "bundles")
RUTA_PKL = os.path.join(BASE_DIR, "data", "embeddings", "catalog_embeddings.pkl")
RUTA_PRODUCTOS = os.path.join(BASE_DIR, "data", "raw", "product_dataset.csv") 
RUTA_OUTPUT_CSV = os.path.join(BASE_DIR, "submission_yolo_filtrado.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Iniciando Pipeline YOLO-CLIP en dispositivo: {DEVICE}")

# --- 2. INICIALIZACIÓN DE MODELOS ---
print("⏳ Cargando YOLOv8 (Detector de Ropa Especializado)...")
yolo_model = YOLO('best.pt')

print("⏳ Cargando CLIP (Extracción de Features)...")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_model.eval()

# --- 3. CARGA DE BASE DE DATOS VECTORIAL ---
print("📚 Cargando base de datos vectorial del catálogo...")
with open(RUTA_PKL, 'rb') as f:
    catalogo_db = pickle.load(f)

ids_catalogo = list(catalogo_db.keys())
matriz_catalogo = np.vstack(list(catalogo_db.values()))

# Normalizamos
normas = np.linalg.norm(matriz_catalogo, axis=1, keepdims=True)
matriz_catalogo = matriz_catalogo / np.where(normas == 0, 1e-10, normas)
print(f"✅ Catálogo listo: {len(ids_catalogo)} prendas cargadas. Forma: {matriz_catalogo.shape}")

# --- 3.5. CARGA DEL FILTRO DE CATEGORÍAS ---
print("🗂️ Cargando el Filtro de Categorías...")
df_productos = pd.read_csv(RUTA_PRODUCTOS)
diccionario_categorias = dict(zip(df_productos['product_asset_id'], df_productos['product_description']))

categorias_catalogo = np.array([diccionario_categorias.get(id_prod, "UNKNOWN") for id_prod in ids_catalogo])

# Mapeo: YOLO a Inditex
MAPEO_CATEGORIAS_YOLO = {
    # Las categorías detectadas por YOLO -> Las etiquetas que buscamos en el catálogo manual de inditex
    "top": ['T-SHIRT', 'SHIRT', 'SWEATER', 'JACKET', 'COAT', 'BLOUSE', 'TOP', 'WAISTCOAT', 'CARDIGAN', 'POLO SHIRT', 'BODYSUIT'],
    "trousers": ['TROUSERS', 'JEANS', 'SHORTS'],
    "skirt": ['SKIRT'],
    "dress": ['DRESS', 'JUMPSUIT'],
    "outwear": ['JACKET', 'COAT', 'BLAZER', 'TRENCH']
}

# --- 4. FUNCIÓN INFERENCIA ---
def analizar_bundle_yolo(ruta_imagen):
    try:
        imagen = Image.open(ruta_imagen).convert("RGB")
        w, h = imagen.size
        
        # -- FASE 1: YOLO Detección --
        # Conf=0.15 para no perdernos detalles por ser muy estrictos
        results = yolo_model(imagen, conf=0.15, verbose=False)[0]
        
        recortes = []
        etiquetas_recortes = []
        
        for caja in results.boxes:
            clase_id = int(caja.cls[0])
            nombre_clase = yolo_model.names[clase_id]
            
            # Solo analizamos cosas que son ropa de verdad, según este modelo YOLO específico
            if nombre_clase in ["top", "trousers", "skirt", "dress", "outwear"]:
                x1, y1, x2, y2 = map(int, caja.xyxy[0])
                recortes.append(imagen.crop((x1, y1, x2, y2)))
                etiquetas_recortes.append(nombre_clase)
        
        # -- ESTRATEGIA DE FALLBACK --
        if len(recortes) == 0:
            recortes.append(imagen.crop((0, 0, w, h // 2))) 
            etiquetas_recortes.append("top") 
            
            recortes.append(imagen.crop((0, h // 2, w, h)))   
            etiquetas_recortes.append("trousers")
            
        # -- FASE 2: Inferencia CLIP --
        with torch.no_grad():
            inputs_clip = clip_processor(images=recortes, return_tensors="pt").to(DEVICE)
            vision_outputs = clip_model.vision_model(pixel_values=inputs_clip['pixel_values'].to(DEVICE))
            embeddings_recortes = clip_model.visual_projection(vision_outputs.pooler_output)
            
            embeddings_recortes = embeddings_recortes / embeddings_recortes.norm(dim=-1, keepdim=True)
            embeddings_recortes = embeddings_recortes.cpu().numpy() 
            
        # -- FASE 3: Búsqueda Vectorial Eficiente --
        sim_matrix = np.dot(embeddings_recortes, matriz_catalogo.T) 
        
        # 🔥 FASE 3.5: EL FILTRO MÁGICO ADAPTADO A YOLO 🔥
        for i, etiqueta in enumerate(etiquetas_recortes):
            if etiqueta in MAPEO_CATEGORIAS_YOLO:
                categorias_permitidas = MAPEO_CATEGORIAS_YOLO[etiqueta]
                mascara_penalizar = ~np.isin(categorias_catalogo, categorias_permitidas)
                sim_matrix[i][mascara_penalizar] = -1.0 
        
        # -- FASE 4: Fusión de Resultados --
        mejores_similitudes_por_prenda = np.max(sim_matrix, axis=0) 
        
        top_15_indices = np.argsort(mejores_similitudes_por_prenda)[-15:][::-1]
        top_15_ids = [ids_catalogo[i] for i in top_15_indices]
        
        return top_15_ids
    
    except Exception as e:
        print(f"❌ Error procesando {ruta_imagen}: {e}")
        return ids_catalogo[:15]

# --- 5. BUCLE PRINCIPAL ---
archivos_test = [f for f in os.listdir(RUTA_CARPETA_TEST) if f.endswith(".jpg")]
resultados_submission = []

# IMPORTANTE: No lo lanzamos aquí para que el script main no se vuelva loco
# Lo importaremos en test_yolo o lo ejecutarás directamente.
if __name__ == "__main__":
    print(f"🚀 Iniciando análisis de {len(archivos_test)} bundles con YOLOv8...")

    for archivo in tqdm(archivos_test, desc="Generando Submission YOLO"):
        bundle_id = archivo.replace(".jpg", "")
        ruta_foto = os.path.join(RUTA_CARPETA_TEST, archivo)
        
        top_15 = analizar_bundle_yolo(ruta_foto)
        
        for product_id in top_15:
            resultados_submission.append({
                'bundle_asset_id': bundle_id,
                'product_asset_id': product_id
            })

    # Guardar a CSV
    df = pd.DataFrame(resultados_submission)
    df.to_csv(RUTA_OUTPUT_CSV, index=False)
    print("\n" + "="*50)
    print(f"🏆 ¡PROCESO TERMINADO! Se ha generado '{RUTA_OUTPUT_CSV}'.")
    print("="*50)
