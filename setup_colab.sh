#!/bin/bash
# ============================================================
# Configuración automática para Google Colab
# ============================================================

echo "=== Configurando entorno CUDA para Transformada de Hough ==="

# Verificar si CUDA está disponible
if ! command -v nvcc &> /dev/null; then
    echo "Instalando CUDA toolkit..."
    apt-get update -y && apt-get install -y cuda-toolkit-12-1
else
    echo "CUDA toolkit ya instalado."
fi

# Crear estructura de carpetas
mkdir -p src build images

# Crear imagen de prueba sintética
echo "Generando imagen de prueba (images/test.pgm)..."
cat > images/test.pgm << 'EOF'
P5
256 256
255
EOF

python3 << 'PYTHON'
import struct
width, height = 256, 256
data = bytearray(width * height)
for x in range(width):
    data[128 * width + x] = 255
for y in range(height):
    data[y * width + 128] = 255
for i in range(min(width, height)):
    data[i * width + i] = 255
with open("images/test.pgm", "ab") as f:
    f.write(data)
print("Imagen sintética creada: images/test.pgm")
PYTHON

# Compilar el proyecto 
echo "Compilando con NVCC en Colab..."
nvcc -arch=sm_75 -O2 -diag-suppress=1650 \
    -Xcompiler="-Wno-unused-result -Wno-unused-variable" \
    src/main.cu -o build/hough

if [ $? -ne 0 ]; then
    echo "Error en la compilación."
    exit 1
fi

echo "Compilación exitosa."

# Ejecutar los tres modos
echo ""
echo "=== Ejecutando los 3 modos de Transformada de Hough ==="
for mode in 0 1 2; do
    echo "------------------------------------------------------------"
    echo "Modo ${mode}"
    ./build/hough ${mode}
done

echo ""
echo "Ejecución completa. Archivos generados:"
echo "  - build/output_lines.ppm"
echo "  - build/input_synthetic.pgm"
echo "  - timings_global.txt"
echo "  - timings_constant.txt"
echo "  - timings_const_shared.txt"
echo "============================================================"
