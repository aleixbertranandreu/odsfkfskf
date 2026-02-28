import json

with open("notebooks/03_yolo_cropping.ipynb", "r") as f:
    nb = json.load(f)

new_code = """from ultralytics import YOLO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

print("⏳ Cargando el modelo especialista de YOLOv8...")
# Fix Seguridad PyTorch 2.6
import torchvision
import ultralytics
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([torchvision.transforms.Compose, ultralytics.nn.tasks.DetectionModel])
# Cargar el modelo YOLO que hemos descargado (o puesto en la carpeta)
model = YOLO('best.pt') 

# ⚠️ FOTO DE PRUEBA EXISTENTE EN TU REPOSITORIO
ruta_foto = "../data/images/bundles/B_00efaa88351d.jpg" 
imagen_original = Image.open(ruta_foto).convert("RGB")

print("🔍 Analizando el outfit...")
# Realizar la inferencia con YOLO
# conf=0.15 asegura que pille prendas aunque esté poco seguro
results = model(imagen_original, conf=0.15)[0] 

# Pintar cajas en la foto original para ver cómo piensa
imagen_anotada = imagen_original.copy()
draw = ImageDraw.Draw(imagen_anotada)

recortes_ropa = []
titulos = []

# Analizar las cajas detectadas
for caja in results.boxes: 
    clase_id = int(caja.cls[0])
    nombre_clase = model.names[clase_id]
    confianza = float(caja.conf[0])
    
    # Extraer coordenadas
    x1, y1, x2, y2 = map(int, caja.xyxy[0])
    
    # Dibujamos en la foto original
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    draw.text((x1, y1-10), f"{nombre_clase} ({confianza:.2f})", fill="red")
    print(f"👀 Detectado: {nombre_clase} ({confianza:.2f})")
    
    # 🔥 LA MAGIA: Las categorías válidas de este modelo para el catálogo
    # Solo guardamos cosas que sean ropa, ignoramos fondos o ruido
    if nombre_clase in ["top", "trousers", "skirt", "dress", "outwear"]:
        recorte = imagen_original.crop((x1, y1, x2, y2))
        recortes_ropa.append(recorte)
        titulos.append(f"{nombre_clase}\\nScore: {confianza:.2f}")

# Mostrar los resultados limpios
if len(recortes_ropa) > 0:
    # Creamos una fila de imágenes
    fig, axs = plt.subplots(1, len(recortes_ropa) + 1, figsize=(15, 6))
    
    axs[0].imshow(imagen_anotada)
    axs[0].set_title("1. Visión completa YOLO")
    axs[0].axis('off')
    
    for i, (recorte, titulo) in enumerate(zip(recortes_ropa, titulos)):
        axs[i+1].imshow(recorte)
        axs[i+1].set_title(f"2. Recorte Limpio ({titulo})")
        axs[i+1].axis('off')
        
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ YOLO no detectó ninguna prenda válida de ropa.")"""

# Prepare the lines with newlines
source_lines = [line + "\\n" for line in new_code.split("\\n")]
source_lines[-1] = source_lines[-1][:-1] # Remove last newline

nb["cells"][0]["source"] = source_lines
nb["cells"][0]["outputs"] = []

with open("notebooks/03_yolo_cropping.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
