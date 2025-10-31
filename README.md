# Transformada de Hough en CUDA

Proyecto de Computación Paralela y Distribuida - Universidad del Valle de Guatemala

## Descripción

Implementación paralela de la Transformada Lineal de Hough usando CUDA para la detección de líneas rectas en imágenes binarias (blanco y negro). El proyecto explora el uso de diferentes tipos de memoria GPU (Global, Constante y Compartida) para optimizar el rendimiento.

## Equipo

- **Integrante 1**: Silvia Illescas - 22376
- **Integrante 2**: Isabella Miralles - 22293
- **Integrante 3**: Michelle Mejía - 22596

**Docente**: Marlon Fuentes
**Semestre**: 2, 2025
**Fecha de Entrega**: Semana del 12-14 de noviembre, 2025

## Objetivos

- Implementar la Transformada de Hough en CUDA
- Explorar el uso de memoria Constante para valores trigonométricos precalculados
- Utilizar memoria Compartida para reducir accesos a memoria Global
- Comparar el rendimiento de diferentes estrategias de memoria
- Visualizar las líneas detectadas en imágenes

## Estructura del Proyecto

Proyecto3-CUDA-Paralela/

├── README.md

├── Makefile

├── .gitignore

├── src/

│   ├── hough.cu              # Implementación principal

│   ├── hough.h               # Headers y definiciones

│   └── image_utils.cpp       # Utilidades para manejo de imágenes

├── images/

│   ├── input/                # Imágenes de entrada

│   └── output/               # Imágenes con líneas detectadas

├── results/

│   ├── measurements.csv      # Mediciones de tiempo

│   └── analysis.xlsx         # Análisis de resultados

├── docs/

│   ├── informe.pdf           # Informe final

│   ├── presentacion.pptx     # Presentación ejecutiva

│   └── bitacora.md           # Bitácora de desarrollo

└── scripts/

│   └── run_tests.sh          # Script para ejecutar pruebas


## Compilación y Ejecución

### Requisitos

- CUDA Toolkit (versión 11.0 o superior)
- GCC/G++ compatible con CUDA
- GPU NVIDIA con compute capability 3.0+
- OpenCV (opcional, para visualización)

### Compilar

\`\`\`bash
make clean
make
\`\`\`

### Ejecutar

\`\`\`bash
# Versión básica
./hough images/input/test.pgm

# Con parámetros personalizados
./hough images/input/test.pgm --threshold 50 --output images/output/result.png
\`\`\`

## Versiones Implementadas

### Versión 1: Memoria Global
- [x] Cálculo correcto de `gloID`
- [x] Kernel básico funcional
- [x] Medición de tiempos con CUDA Events
- [x] Liberación de memoria

**Tiempo promedio**: ___ ms

### Versión 2: Memoria Global + Constante
- [x] Declaración de `d_Cos` y `d_Sin` en memoria constante
- [x] Uso de `cudaMemcpyToSymbol`
- [x] Eliminación de parámetros del kernel
- [x] Medición de tiempos

**Tiempo promedio**: ___ ms  
**Mejora**: ___% respecto a versión 1

### Versión 3: Global + Constante + Compartida
- [x] Acumulador local en memoria compartida
- [x] Uso de barreras (`__syncthreads()`)
- [x] Operaciones atómicas para sincronización
- [x] Reducción de accesos a memoria global

**Tiempo promedio**: ___ ms  
**Mejora**: ___% respecto a versión 2

## Algoritmo: Transformada de Hough

### Concepto

La Transformada de Hough convierte puntos en el espacio de imagen (x, y) al espacio de parámetros (θ, r), donde:

- **θ**: Ángulo perpendicular a la línea (0° a 180°)
- **r**: Distancia del origen a la línea

### Fórmula

\`\`\`
r(θ) = x·cos(θ) + y·sin(θ)
\`\`\`

### Proceso

1. Para cada pixel "blanco" en la imagen
2. Iterar sobre todos los ángulos θ posibles
3. Calcular r(θ) usando la fórmula
4. Incrementar el acumulador en la posición (θ, r)
5. Las celdas con más votos representan líneas en la imagen

## Resultados

### Tabla Comparativa de Tiempos

| Versión | Tiempo Promedio (ms) | Desv. Estándar | Mejora (%) |
|---------|---------------------|----------------|------------|
| Global  | -                   | -              | -          |
| + Constante | -               | -              | -          |
| + Compartida | -              | -              | -          |

### Imágenes de Resultados

Pendiente


## Pruebas y Mediciones

### Metodología

- **Número de mediciones**: 10 por versión
- **Imagen de prueba**: [Especificar dimensiones y características]
- **Configuración del grid**: [Especificar bloques y threads]
- **Hardware**: [Especificar GPU utilizada]

### Ejecutar Pruebas

\`\`\`bash
./scripts/run_tests.sh
\`\`\`

## Optimizaciones Implementadas

### Memoria Constante
- **Ventaja**: Broadcast eficiente de valores trigonométricos a todos los threads
- **Uso**: Almacenar cos(θ) y sin(θ) precalculados
- **Impacto**: Reducción de cálculos trigonométricos costosos

### Memoria Compartida
- **Ventaja**: Baja latencia, acceso rápido entre threads del mismo bloque
- **Uso**: Acumulador local por bloque
- **Impacto**: Reducción de accesos a memoria global y contención atómica

### Operaciones Atómicas
- **Uso**: `atomicAdd()` para actualizar acumuladores sin race conditions
- **Ubicación**: Tanto en memoria compartida como global

## Debugging

Para habilitar mensajes de debug:

\`\`\`cpp
#define DEBUG 1
\`\`\`

## Referencias

1. NVIDIA CUDA C Programming Guide
2. "Digital Image Processing" - Gonzalez & Woods
3. [CUDA Performance Metrics](https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/)
4. [Hough Transform - Wikipedia](https://en.wikipedia.org/wiki/Hough_transform)

## 📄 Licencia

Proyecto académico - Universidad del Valle de Guatemala
