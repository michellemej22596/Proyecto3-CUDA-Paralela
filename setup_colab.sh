#!/bin/bash
# Script para configurar el proyecto en Google Colab

echo "=== Configurando proyecto Hough Transform ==="

# Crear estructura de directorios
mkdir -p common
mkdir -p images

# Crear imagen de prueba simple en formato PGM
cat > images/test.pgm << 'EOF'
P5
256 256
255
EOF

# Generar datos de imagen (línea horizontal en el medio)
python3 << 'PYTHON'
import struct

width = 256
height = 256
data = bytearray(width * height)

# Línea horizontal en y=128
for x in range(width):
    data[128 * width + x] = 255

# Línea vertical en x=128
for y in range(height):
    data[y * width + 128] = 255

# Línea diagonal
for i in range(min(width, height)):
    data[i * width + i] = 255

with open('images/test.pgm', 'ab') as f:
    f.write(data)

print(f"Imagen de prueba creada: {sum(1 for x in data if x > 0)} pixels blancos")
PYTHON

echo "=== Configuración completa ==="
echo "Archivos creados:"
echo "  - common/pgm.h"
echo "  - images/test.pgm"
echo ""
echo "Ahora puedes compilar con:"
echo "  nvcc hough.cu -o hough -I."
echo "  ./hough"
