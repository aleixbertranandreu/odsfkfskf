from PIL import Image, ImageDraw
import torch
import matplotlib.pyplot as plt
from transformers import OwlViTProcessor, OwlViTForObjectDetection

print("⏳ Cargando procesador visual y modelo OwlViT...")
# Configuramos el dispositivo (tarjeta gráfica si está, sino procesador normal)
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)

# ⚠️ FOTO DE PRUEBA: Pon aquí la ruta de una foto chula donde se vea bien la ropa
ruta_foto = "../data/images/bundles/B_568aa697563e.jpg" 
imagen_original = Image.open(ruta_foto).convert("RGB")

# ¿Qué cosas quieres que la IA busque y separe en la imagen?
etiquetas_a_buscar = [["upper clothing", "pants", "skirt", "dress"]]
print(f"🔍 Buscando específicamente: {etiquetas_a_buscar[0]}")

# 1. Alimentamos a la bestia (imagen + texto)
inputs = processor(text=etiquetas_a_buscar, images=imagen_original, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

# 2. Obtenemos las cajas de resultados adaptadas al tamaño real de tu foto
target_sizes = torch.tensor([imagen_original.size[::-1]])
# Umbral bajo (0.1) para que sea un poco optimista
results = processor.post_process_grounded_object_detection(
    outputs=outputs, 
    target_sizes=target_sizes, 
    text_labels=etiquetas_a_buscar, 
    threshold=0.1
)[0] # Cogemos el resultado de la primera (y única) imagen

# Pintar cajas en la foto original para ver cómo piensa
imagen_anotada = imagen_original.copy()
draw = ImageDraw.Draw(imagen_anotada)

recortes = []
titulos = []

# 3. Analizar lo que ha encontrado
for caja, score, label_idx in zip(results["boxes"], results["scores"], results["labels"]):
    # Extraer coordenadas
    x1, y1, x2, y2 = map(int, caja.tolist())
    
    # ¿Qué etiqueta corresponde a este índice?
    nombre_prenda = etiquetas_a_buscar[0][label_idx.item()]
    confianza = score.item()
    
    # Dibujamos rectángulo rojo en la original
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    draw.text((x1, y1-10), f"{nombre_prenda} ({confianza:.2f})", fill="red")
    
    print(f"✅ ¡Encontrado! {nombre_prenda.upper()} con {confianza*100:.1f}% de seguridad.")
    
    # Recortamos la prenda y la guardamos para enseñarla luego
    recorte = imagen_original.crop((x1, y1, x2, y2))
    recortes.append(recorte)
    titulos.append(f"{nombre_prenda}\nScore: {confianza:.2f}")

# 4. Mostrar el show visual
if len(recortes) > 0:
    print("Encontrado elementos")
else:
    print("🤷‍♂️ OwlViT no ha logrado ver ninguna de esas prendas en la foto con >10% seguridad.")
