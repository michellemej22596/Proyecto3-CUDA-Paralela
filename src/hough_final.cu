#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <cassert>

#define CUDA_CHECK(call) do {                                 \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error %s:%d: %s\n",             \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

const int degreeBins = 90; // como en el enunciado (0..178 step 2 -> 90 bins)
const int rBins = 100;
const int DEGREE_STEP = 2; // grados incrementales
const int TIMED_RUNS = 10; // bitácora: 10 mediciones por modo

// Memoria constante unificada (nombres: d_Cos, d_Sin)
__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

// kernel: version global-only (usa punteros a cos/sin en global o en const dependiendo del modo)
__global__
void hough_kernel_global(const unsigned char* img, int W, int H,
                         unsigned int* acc, int degreeBins_local, int rBins_local,
                         float rMax, float rScale,
                         const float* cosArr, const float* sinArr) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = W * H;
    if (gid >= total) return;

    int j = gid % W; // columna (x index)
    int i = gid / W; // fila (y index)
    unsigned char pix = img[gid];
    if (pix == 0) return; // solo pixels "blancos" (non-zero)
    // convert indices to coordinates with origin at center
    float xCoord = (float)j - (float)W / 2.0f;
    float yCoord = (float)i - (float)H / 2.0f;

    for (int t = 0; t < degreeBins_local; ++t) {
        float r = xCoord * cosArr[t] + yCoord * sinArr[t];
        int rIdx = (int)roundf((r + rMax) * rScale);
        if (rIdx >= 0 && rIdx < rBins_local) {
            int idx = t * rBins_local + rIdx;
            atomicAdd(&acc[idx], 1u);
        }
    }
}

// kernel: version that uses memoria compartida localAcc (degreeBins_local * rBins_local entries)
// localAcc stored in dynamic shared memory (int)
__global__
void hough_kernel_shared(const unsigned char* img, int W, int H,
                         unsigned int* acc, int degreeBins_local, int rBins_local,
                         float rMax, float rScale,
                         const float* cosArr, const float* sinArr) {
    extern __shared__ unsigned int locAcc[]; // size must be degreeBins_local * rBins_local (in bytes)
    int locSize = degreeBins_local * rBins_local;
    int locID = threadIdx.x;
    int blockThreads = blockDim.x;

    // Inicializar localAcc en bloques con múltiples hilos
    for (int idx = locID; idx < locSize; idx += blockThreads) {
        locAcc[idx] = 0u;
    }
    __syncthreads();

    // Cada hilo procesa varios pixeles: stride through global grid
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = W * H;
    for (int id = gid; id < total; id += gridDim.x * blockDim.x) {
        unsigned char pix = img[id];
        if (pix == 0) continue;
        int j = id % W;
        int i = id / W;
        float xCoord = (float)j - (float)W / 2.0f;
        float yCoord = (float)i - (float)H / 2.0f;

        for (int t = 0; t < degreeBins_local; ++t) {
            float r = xCoord * cosArr[t] + yCoord * sinArr[t];
            int rIdx = (int)roundf((r + rMax) * rScale);
            if (rIdx >= 0 && rIdx < rBins_local) {
                int idx = t * rBins_local + rIdx;
                atomicAdd(&locAcc[idx], 1u);
            }
        }
    }
    __syncthreads();

    // Reducir localAcc -> acc global usando atomicAdd
    for (int idx = locID; idx < locSize; idx += blockThreads) {
        unsigned int val = locAcc[idx];
        if (val != 0u) {
            atomicAdd(&acc[idx], val);
        }
    }
}

// util: escribe PGM (entrada sintetica) - formato P5 (bytes)
bool writePGM(const char* fname, const unsigned char* data, int W, int H) {
    FILE* f = fopen(fname, "wb");
    if (!f) return false;
    fprintf(f, "P5\n%d %d\n255\n", W, H);
    size_t written = fwrite(data, 1, (size_t)W*H, f);
    fclose(f);
    return written == (size_t)W*H;
}

// util: escribe PPM color con líneas (RGB)
bool writePPM(const char* fname, const unsigned char* rgb, int W, int H) {
    FILE* f = fopen(fname, "wb");
    if (!f) return false;
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    size_t written = fwrite(rgb, 1, (size_t)W*H*3, f);
    fclose(f);
    return written == (size_t)W*H*3;
}

// Dibuja líneas sobre la imagen en RGB usando los parámetros del acumulador
void draw_lines_on_image(const unsigned char* inputPGM, int W, int H,
                         const unsigned int* accHost, int degreeBins_local, int rBins_local,
                         float rMax, float rScale, unsigned char* outRGB, unsigned int threshold) {
    // Inicializa RGB como grayscale de entrada
    for (int i = 0; i < W*H; ++i) {
        unsigned char v = inputPGM[i];
        outRGB[3*i + 0] = v;
        outRGB[3*i + 1] = v;
        outRGB[3*i + 2] = v;
    }
    // Para cada bin (t, r) con votos > threshold, dibuja la línea (r, theta)
    for (int t = 0; t < degreeBins_local; ++t) {
        float theta = (float)(t * DEGREE_STEP) * (M_PI/180.0f);
        float cosT = cosf(theta);
        float sinT = sinf(theta);
        for (int rIdx = 0; rIdx < rBins_local; ++rIdx) {
            int idx = t * rBins_local + rIdx;
            unsigned int votes = accHost[idx];
            if (votes <= threshold) continue;
            // transformar rIdx a r real
            float r = ((float)rIdx) / rScale - rMax;
            // dibujar la línea: para cada x calcula y y marca pixel (alternativa: parametric)
            // Usaremos iteración sobre x
            for (int x = 0; x < W; ++x) {
                float xp = (float)x - (float)W/2.0f;
                // y = (r - x*cos)/sin  (cuidado con sin ~ 0)
                if (fabsf(sinT) < 1e-4f) continue;
                float yfloat = (r - xp * cosT) / sinT;
                int y = (int)roundf(yfloat + (float)H/2.0f);
                if (y >= 0 && y < H) {
                    int id = y * W + x;
                    // marcar en rojo
                    outRGB[3*id + 0] = 255;
                    outRGB[3*id + 1] = 0;
                    outRGB[3*id + 2] = 0;
                }
            }
        }
    }
}

// Genera imagen sintética simple (un par de lineas) en PGM (0/255)
unsigned char* generate_synthetic(int W, int H) {
    unsigned char* img = (unsigned char*)malloc(W*H);
    if (!img) return nullptr;
    // inicializar a negro (0)
    memset(img, 0, W*H);
    // Dibujar unas líneas (blanco=255)
    for (int x = 10; x < W-10; ++x) {
        int y = H/3;
        img[y*W + x] = 255;
    }
    for (int x = 20; x < W-20; ++x) {
        int y = (2*H)/3;
        img[y*W + x] = 255;
    }
    // Diagonal
    for (int k = 0; k < std::min(W,H); ++k) {
        img[k*W + k] = 255;
    }
    return img;
}

int main(int argc, char** argv) {
    int mode = 2; // por defecto: constant + shared
    if (argc >= 2) mode = atoi(argv[1]);
    if (mode < 0 || mode > 2) {
        printf("Uso: %s <mode>\nmode: 0=global, 1=constant, 2=constant+shared\n", argv[0]);
        return 1;
    }
    printf("Modo: %d\n", mode);

    // Parametros de imagen
    const int W = 512;
    const int H = 512;
    const int totalPix = W * H;

    // parámetros Hough
    const int degBins = degreeBins;
    const int rBinsLocal = rBins;
    float rMax = sqrtf((W/2.0f)*(W/2.0f) + (H/2.0f)*(H/2.0f));
    float rScale = (float)rBinsLocal / (2.0f * rMax); // mapea [-rMax, rMax] a [0, rBins)

    // preparar arreglos trig en host
    float hostCos[degreeBins];
    float hostSin[degreeBins];
    for (int t = 0; t < degBins; ++t) {
        float theta = (float)(t * DEGREE_STEP) * (M_PI / 180.0f);
        hostCos[t] = cosf(theta);
        hostSin[t] = sinf(theta);
    }

    // Copiar a memoria constante (si aplica) - pero hacemos cudaMemcpyToSymbol siempre para que los kernels
    // puedan leer desde la memoria constante global unificada d_Cos/d_Sin.
    CUDA_CHECK(cudaMemcpyToSymbol(d_Cos, hostCos, sizeof(float) * degBins));
    CUDA_CHECK(cudaMemcpyToSymbol(d_Sin, hostSin, sizeof(float) * degBins));

    // Además, si el modo global necesita punteros en device (no constante), crearemos arrays device
    float *d_cos_global = nullptr, *d_sin_global = nullptr;
    if (mode == 0) {
        CUDA_CHECK(cudaMalloc(&d_cos_global, sizeof(float) * degBins));
        CUDA_CHECK(cudaMalloc(&d_sin_global, sizeof(float) * degBins));
        CUDA_CHECK(cudaMemcpy(d_cos_global, hostCos, sizeof(float) * degBins, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sin_global, hostSin, sizeof(float) * degBins, cudaMemcpyHostToDevice));
    }

    // Imagen de entrada (generada)
    unsigned char* hostImg = generate_synthetic(W, H);
    if (!hostImg) { fprintf(stderr, "No memory for host image\n"); return 1; }
    writePGM("input_synthetic.pgm", hostImg, W, H);

    // device image
    unsigned char* d_img = nullptr;
    CUDA_CHECK(cudaMalloc(&d_img, (size_t)totalPix));
    CUDA_CHECK(cudaMemcpy(d_img, hostImg, (size_t)totalPix, cudaMemcpyHostToDevice));

    // acumulador global en device
    const int accSize = degBins * rBinsLocal;
    unsigned int* d_acc = nullptr;
    CUDA_CHECK(cudaMalloc(&d_acc, sizeof(unsigned int) * accSize));
    CUDA_CHECK(cudaMemset(d_acc, 0, sizeof(unsigned int) * accSize));

    // host accumulator para lectura final
    unsigned int* hostAcc = (unsigned int*)malloc(sizeof(unsigned int) * accSize);
    memset(hostAcc, 0, sizeof(unsigned int) * accSize);

    // configuración de kernel
    int threads = 256;
    int blocks = (totalPix + threads - 1) / threads;
    if (blocks > 1024*64) blocks = 1024*64; // limit safety
    size_t sharedBytes = 0;
    if (mode == 2) {
        // shared mem size: degBins * rBinsLocal * sizeof(unsigned int)
        sharedBytes = (size_t)degBins * (size_t)rBinsLocal * sizeof(unsigned int);
        // verificar que no exceda limite típico (por bloque). si excede, reducimos blockDim.x para contener.
        // (en GPUs actuales, shared mem per block suele ser >= 48KB, y nuestro ejemplo usa ~36KB para 90*100)
        // Si sobra, continue; si excede, fallback a mode 1 behavior abajo.
        size_t maxSharedPerBlock = 48 * 1024; // asunción conservadora
        if (sharedBytes > maxSharedPerBlock) {
            printf("Advertencia: sharedBytes (%zu) > %zu; reduciendo a modo constante (1)\n", sharedBytes, maxSharedPerBlock);
            mode = 1;
            sharedBytes = 0;
        }
    }

    // preparar archivo de tiempos
    const char* timingFile = (mode==0) ? "timings_global.txt" : (mode==1) ? "timings_constant.txt" : "timings_const_shared.txt";
    FILE* tf = fopen(timingFile, "w");
    if (!tf) { fprintf(stderr, "No se pudo abrir %s\n", timingFile); }

    // Eventos CUDA para medir tiempo
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int run = 0; run < TIMED_RUNS; ++run) {
        // limpiar acumulador device
        CUDA_CHECK(cudaMemset(d_acc, 0, sizeof(unsigned int) * accSize));

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start, 0));

        if (mode == 0) {
            // global: pasar d_cos_global/d_sin_global como punteros device
            hough_kernel_global<<<blocks, threads>>>(
                d_img, W, H, d_acc, degBins, rBinsLocal, rMax, rScale,
                d_cos_global, d_sin_global);
        } else if (mode == 1) {
            float* tmp_cos = nullptr;
            float* tmp_sin = nullptr;
            CUDA_CHECK(cudaMalloc(&tmp_cos, sizeof(float) * degBins));
            CUDA_CHECK(cudaMalloc(&tmp_sin, sizeof(float) * degBins));
            CUDA_CHECK(cudaMemcpyFromSymbol(tmp_cos, d_Cos, sizeof(float) * degBins));
            CUDA_CHECK(cudaMemcpyFromSymbol(tmp_sin, d_Sin, sizeof(float) * degBins));
            hough_kernel_global<<<blocks, threads>>>(
                d_img, W, H, d_acc, degBins, rBinsLocal, rMax, rScale,
                tmp_cos, tmp_sin);
            CUDA_CHECK(cudaFree(tmp_cos));
            CUDA_CHECK(cudaFree(tmp_sin));
        } else { // mode == 2 -> constant + shared
            // similar approach: pass pointers from constant via temporary device arrays, and use shared kernel
            float* tmp_cos = nullptr;
            float* tmp_sin = nullptr;
            CUDA_CHECK(cudaMalloc(&tmp_cos, sizeof(float) * degBins));
            CUDA_CHECK(cudaMalloc(&tmp_sin, sizeof(float) * degBins));
            CUDA_CHECK(cudaMemcpyFromSymbol(tmp_cos, d_Cos, sizeof(float) * degBins));
            CUDA_CHECK(cudaMemcpyFromSymbol(tmp_sin, d_Sin, sizeof(float) * degBins));
            // shared kernel expects dynamic shared mem
            hough_kernel_shared<<<blocks, threads, sharedBytes>>>(
                d_img, W, H, d_acc, degBins, rBinsLocal, rMax, rScale,
                tmp_cos, tmp_sin);
            CUDA_CHECK(cudaFree(tmp_cos));
            CUDA_CHECK(cudaFree(tmp_sin));
        }

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (tf) fprintf(tf, "%f\n", ms);
        printf("Run %d: %f ms\n", run, ms);
    }

    if (tf) fclose(tf);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // copiar acumulador a host
    CUDA_CHECK(cudaMemcpy(hostAcc, d_acc, sizeof(unsigned int) * accSize, cudaMemcpyDeviceToHost));

    // calcular threshold (ejemplo: promedio + 2*std)
    double sum = 0.0;
    double sum2 = 0.0;
    for (int i = 0; i < accSize; ++i) {
        sum += hostAcc[i];
        sum2 += (double)hostAcc[i] * (double)hostAcc[i];
    }
    double mean = sum / accSize;
    double var = sum2 / accSize - mean*mean;
    double stddev = (var > 0.0) ? sqrt(var) : 0.0;
    unsigned int threshold = (unsigned int)floor(mean + 2.0 * stddev);
    printf("mean=%.2f stddev=%.2f threshold=%u\n", mean, stddev, threshold);

    // leer imagen de entrada en host (ya la tenemos hostImg)
    unsigned char* outRGB = (unsigned char*)malloc((size_t)W * (size_t)H * 3);
    draw_lines_on_image(hostImg, W, H, hostAcc, degBins, rBinsLocal, rMax, rScale, outRGB, threshold);
    writePPM("output_lines.ppm", outRGB, W, H);
    printf("Escrita output_lines.ppm\n");

    // liberar memoria
    if (d_cos_global) CUDA_CHECK(cudaFree(d_cos_global));
    if (d_sin_global) CUDA_CHECK(cudaFree(d_sin_global));
    if (d_img) CUDA_CHECK(cudaFree(d_img));
    if (d_acc) CUDA_CHECK(cudaFree(d_acc));
    free(hostAcc);
    free(hostImg);
    free(outRGB);

    CUDA_CHECK(cudaDeviceReset());
    printf("Proceso terminado.\n");
    return 0;
}
