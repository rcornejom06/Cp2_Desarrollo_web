"""
Verificación detallada del dataset
"""
import os
from PIL import Image

dataset_path = 'datasets/BananaLSD/AugmentedSet/'

print("=" * 70)
print("🔍 VERIFICACIÓN DETALLADA DEL DATASET")
print("=" * 70)

if not os.path.exists(dataset_path):
    print(f"\n❌ ERROR: No existe {dataset_path}")
    print("\n¿Tu dataset está en otra ubicación?")
    print("Busca la carpeta que contiene: Cordana, Healthy, Pestalotiopsis, Sigatoka")
    exit(1)

print(f"\n📁 Ruta del dataset: {dataset_path}")

# Listar contenido
items = os.listdir(dataset_path)
print(f"\n📂 Contenido de la carpeta:")
for item in items:
    print(f"   - {item}")

# Verificar clases esperadas
expected_classes = ['Cordana', 'Healthy', 'Pestalotiopsis', 'Sigatoka']
found_classes = []

print(f"\n📊 Análisis por clase:")
print("=" * 70)

total_images = 0
all_ok = True

for class_name in expected_classes:
    class_path = os.path.join(dataset_path, class_name)

    if not os.path.exists(class_path):
        print(f"\n❌ {class_name}")
        print(f"   Carpeta NO encontrada: {class_path}")
        all_ok = False
        continue

    if not os.path.isdir(class_path):
        print(f"\n❌ {class_name}")
        print(f"   {class_path} NO es una carpeta")
        all_ok = False
        continue

    # Contar imágenes
    files = os.listdir(class_path)
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    if len(image_files) == 0:
        print(f"\n❌ {class_name}")
        print(f"   Carpeta vacía: {class_path}")
        print(f"   Archivos encontrados: {files[:5]}")
        all_ok = False
        continue

    found_classes.append(class_name)
    total_images += len(image_files)

    print(f"\n✅ {class_name}")
    print(f"   Imágenes: {len(image_files)}")
    print(f"   Ruta: {class_path}")

    # Verificar que al menos una imagen se puede abrir
    try:
        test_image = os.path.join(class_path, image_files[0])
        img = Image.open(test_image)
        print(f"   Muestra: {image_files[0]} ({img.size}, {img.mode})")
    except Exception as e:
        print(f"   ⚠️  Error abriendo imagen: {e}")
        all_ok = False

print("\n" + "=" * 70)
print("📊 RESUMEN")
print("=" * 70)

print(f"\nClases esperadas: {len(expected_classes)}")
print(f"Clases encontradas: {len(found_classes)}")
print(f"Total de imágenes: {total_images}")

if all_ok and len(found_classes) == 4 and total_images >= 100:
    print("\n✅ DATASET CORRECTO - Listo para entrenar")
else:
    print("\n❌ HAY PROBLEMAS CON EL DATASET")

    if len(found_classes) < 4:
        print(f"\n🔧 Problema: Faltan clases")
        missing = set(expected_classes) - set(found_classes)
        print(f"   Clases faltantes: {missing}")

    if total_images < 100:
        print(f"\n🔧 Problema: Pocas imágenes ({total_images})")
        print(f"   Se necesitan al menos 100 imágenes en total")

    print("\n📋 Solución:")
    print("1. Verifica que el dataset esté descomprimido correctamente")
    print("2. La estructura debe ser:")
    print("   datasets/BananaLSD/AugmentedSet/")
    print("   ├── Cordana/")
    print("   ├── Healthy/")
    print("   ├── Pestalotiopsis/")
    print("   └── Sigatoka/")

print("=" * 70)