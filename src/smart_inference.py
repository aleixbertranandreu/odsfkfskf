import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import os
import pickle

# --- 1. CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_CARPETA_TEST = os.path.join(BASE_DIR, "data", "images", "bundles")
RUTA_PKL = os.path.join(BASE_DIR, "data", "embeddings", "catalog_embeddings.pkl")
RUTA_PRODUCTOS = os.path.join(BASE_DIR, "data", "raw", "product_dataset.csv") # 🔥 NUEVA RUTA AL CSV DE PRODUCTOS
RUTA_OUTPUT_CSV = os.path.join(BASE_DIR, "submission_owlvit_filtrado.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Iniciando Pipeline en dispositivo: {DEVICE}")

# --- 2. CARGA DE MODELOS ---
print("⏳ Cargando OwlViT (Segmentación Zero-Shot)...")
owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(DEVICE)

print("⏳ Cargando CLIP (Extracción de Features)...")
# FIX SEGURIDAD PYTORCH 2.6
import torchvision
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([torchvision.transforms.Compose])

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)

# --- 3. CARGA DEL CATÁLOGO ---
print("📚 Cargando base de datos vectorial del catálogo...")
with open(RUTA_PKL, "rb") as f:
    catalogo_db = pickle.load(f)

ids_catalogo = list(catalogo_db.keys())
matriz_catalogo = np.vstack(list(catalogo_db.values()))

# Normalizamos
normas = np.linalg.norm(matriz_catalogo, axis=1, keepdims=True)
matriz_catalogo = matriz_catalogo / np.where(normas == 0, 1e-10, normas)
print(f"✅ Catálogo listo: {len(ids_catalogo)} prendas cargadas. Forma: {matriz_catalogo.shape}")

# --- 3.5. CARGA DEL FILTRO DE CATEGORÍAS 🔥 ---
print("🗂️ Cargando el Filtro Mágico de Categorías...")
df_productos = pd.read_csv(RUTA_PRODUCTOS)
diccionario_categorias = dict(zip(df_productos['product_asset_id'], df_productos['product_description']))

# Alineamos las categorías con el orden exacto de la matriz
categorias_catalogo = np.array([diccionario_categorias.get(id_prod, "UNKNOWN") for id_prod in ids_catalogo])

# Mapeo: Qué categorías de Inditex permitimos según lo que vea OWL-ViT
MAPEO_CATEGORIAS = {
    "upper clothing": ['T-SHIRT', 'SHIRT', 'SWEATER', 'JACKET', 'COAT', 'BLOUSE', 'TOP', 'WAISTCOAT', 'CARDIGAN', 'POLO SHIRT', 'BODYSUIT'],
    "pants": ['TROUSERS', 'JEANS', 'SHORTS'],
    "skirt": ['SKIRT'],
    "dress": ['DRESS', 'JUMPSUIT']
}

# --- 4. FUNCIÓN BÚSQUEDA DEL OUTFIT ---
TEXT_QUERIES = [["upper clothing", "pants", "skirt", "dress"]]

def analizar_bundle_optimizado(ruta_imagen):
    try:
        imagen = Image.open(ruta_imagen).convert("RGB")
        w, h = imagen.size
        
        # -- FASE 1: OwlViT Detección --
        inputs_owl = owl_processor(text=TEXT_QUERIES, images=imagen, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs_owl = owl_model(**inputs_owl)
        
        target_sizes = torch.tensor([imagen.size[::-1]])
        results = owl_processor.post_process_grounded_object_detection(outputs=outputs_owl, target_sizes=target_sizes, text_labels=TEXT_QUERIES, threshold=0.1)
        
        cajas = results[0]["boxes"].cpu().numpy()
        scores = results[0]["scores"].cpu().numpy()
        labels_idx = results[0]["labels"].cpu().numpy() # 🔥 Guardamos qué ha visto exactamente
        
        recortes = []
        etiquetas_recortes = []
        
        for caja, score, idx in zip(cajas, scores, labels_idx):
            if score > 0.1:
                x1, y1, x2, y2 = caja
                recortes.append(imagen.crop((x1, y1, x2, y2)))
                etiquetas_recortes.append(TEXT_QUERIES[0][idx]) # Guardamos si es "pants", "upper clothing", etc.
        
        # -- ESTRATEGIA DE FALLBACK --
        if len(recortes) == 0:
            recortes.append(imagen.crop((0, 0, w, h // 2))) 
            etiquetas_recortes.append("upper clothing") # Asumimos que la mitad de arriba es camiseta
            
            recortes.append(imagen.crop((0, h // 2, w, h)))   
            etiquetas_recortes.append("pants") # Asumimos que la mitad de abajo es pantalón
            
        # -- FASE 2: Inferencia CLIP --
        with torch.no_grad():
            inputs_clip = clip_processor(images=recortes, return_tensors="pt").to(DEVICE)
            vision_outputs = clip_model.vision_model(pixel_values=inputs_clip['pixel_values'])
            embeddings_recortes = clip_model.visual_projection(vision_outputs.pooler_output)
            
            embeddings_recortes = embeddings_recortes / embeddings_recortes.norm(dim=-1, keepdim=True)
            embeddings_recortes = embeddings_recortes.cpu().numpy() 
            
        # -- FASE 3: Búsqueda Vectorial Eficiente --
        sim_matrix = np.dot(embeddings_recortes, matriz_catalogo.T) 
        
        # 🔥 FASE 3.5: EL FILTRO MÁGICO 🔥
        for i, etiqueta in enumerate(etiquetas_recortes):
            if etiqueta in MAPEO_CATEGORIAS:
                categorias_permitidas = MAPEO_CATEGORIAS[etiqueta]
                # Castigamos con -1.0 todo lo que NO sea de esta categoría
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
archivos_test = [f for f in os.listdir(RUTA_CARPETA_TEST) if f.endswith('.jpg')]
resultados_submission = []

print(f"🚀 Iniciando análisis de {len(archivos_test)} bundles con Filtro de Categorías...")

for archivo in tqdm(archivos_test, desc="Generando Submission"):
    bundle_id = archivo.replace(".jpg", "")
    ruta_foto = os.path.join(RUTA_CARPETA_TEST, archivo)
    
    top_15 = analizar_bundle_optimizado(ruta_foto)
    
    for product_id in top_15:
        resultados_submission.append({
            'bundle_asset_id': bundle_id,
            'product_asset_id': product_id
        })

# --- 6. GUARDADO ---
df = pd.DataFrame(resultados_submission)
df.to_csv(RUTA_OUTPUT_CSV, index=False)

print("\n" + "="*50)
print(f"🏆 ¡PROCESO TERMINADO! Se ha generado '{RUTA_OUTPUT_CSV}'.")
print("="*50)