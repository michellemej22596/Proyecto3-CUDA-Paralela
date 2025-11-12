#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

#include "pgm.h"

#define degreeBins 180
#define rBins 100
#define radInc (M_PI / degreeBins)
#define TILE_THETA 32
#define TIMED_RUNS 10

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error en %s (%s:%d): %s\n", \
                #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// ======== MEMORIA CONSTANTE UNIFICADA ========
__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

// ======== KERNELS (lectura desde puntero) ========
__global__ void GPU_HoughTran_global_ptr(const unsigned char *pic, int w, int h, int *acc,
                                         float rMax, float rScale,
                                         const float *dCos, const float *dSin) {
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h) return;
    int xCent = w/2, yCent = h/2;
    int xCoord = gloID % w - xCent;
    int yCoord = yCent - (gloID / w);
    if (pic[gloID] > 0) {
        for (int t = 0; t < degreeBins; ++t) {
            float r = xCoord * dCos[t] + yCoord * dSin[t];
            int rIdx = (int)((r + rMax) / rScale);
            if (rIdx >= 0 && rIdx < rBins) atomicAdd(&acc[t * rBins + rIdx], 1);
        }
    }
}

// ======== KERNELS (lectura desde memoria constante __constant__) ========
__global__ void GPU_HoughTran_const(const unsigned char *pic, int w, int h, int *acc,
                                    float rMax, float rScale) {
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h) return;
    int xCent = w/2, yCent = h/2;
    int xCoord = gloID % w - xCent;
    int yCoord = yCent - (gloID / w);
    if (pic[gloID] > 0) {
        for (int t = 0; t < degreeBins; ++t) {
            float r = xCoord * d_Cos[t] + yCoord * d_Sin[t];
            int rIdx = (int)((r + rMax) / rScale);
            if (rIdx >= 0 && rIdx < rBins) atomicAdd(&acc[t * rBins + rIdx], 1);
        }
    }
}

// ======== KERNEL tiled (shared) - versión que usa punteros ========
extern __shared__ int s_local[]; // for ptr version we won't use __constant__
__global__ void GPU_HoughTran_tiled_ptr(const unsigned char *pic, int w, int h, int *acc,
                                        float rMax, float rScale,
                                        const float *dCos, const float *dSin) {
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h) return;
    int xCent = w/2, yCent = h/2;
    int xCoord = gloID % w - xCent;
    int yCoord = yCent - (gloID / w);

    int tileSizeLimit = TILE_THETA;
    // use dynamic shared memory provided via extern __shared__
    int tileSize = min(tileSizeLimit, degreeBins);
    // We'll process tStart loops manually similar to original
    for (int tStart = 0; tStart < degreeBins; tStart += TILE_THETA) {
        int actualTile = degreeBins - tStart;
        if (actualTile > TILE_THETA) actualTile = TILE_THETA;
        int locSize = actualTile * rBins;
        // map s_local to this tile region
        int *localAcc = s_local; // same pointer
        // init local
        for (int i = threadIdx.x; i < locSize; i += blockDim.x) localAcc[i] = 0;
        __syncthreads();

        if (pic[gloID] > 0) {
            for (int t = 0; t < actualTile; ++t) {
                int tIdx = tStart + t;
                float r = xCoord * dCos[tIdx] + yCoord * dSin[tIdx];
                int rIdx = (int)((r + rMax) / rScale);
                if (rIdx >= 0 && rIdx < rBins) atomicAdd(&localAcc[t * rBins + rIdx], 1);
            }
        }
        __syncthreads();

        for (int i = threadIdx.x; i < locSize; i += blockDim.x) {
            int local_t = i / rBins;
            int local_r = i % rBins;
            int global_t = tStart + local_t;
            atomicAdd(&acc[global_t * rBins + local_r], localAcc[i]);
        }
        __syncthreads();
    }
}

// ======== KERNEL tiled (shared) - version that reads from __constant__ arrays ========
__global__ void GPU_HoughTran_tiled_const(const unsigned char *pic, int w, int h, int *acc,
                                         float rMax, float rScale) {
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h) return;
    int xCent = w/2, yCent = h/2;
    int xCoord = gloID % w - xCent;
    int yCoord = yCent - (gloID / w);

    __shared__ int localAcc[TILE_THETA * rBins];
    for (int tStart = 0; tStart < degreeBins; tStart += TILE_THETA) {
        int actualTile = degreeBins - tStart;
        if (actualTile > TILE_THETA) actualTile = TILE_THETA;
        int locSize = actualTile * rBins;
        for (int i = threadIdx.x; i < locSize; i += blockDim.x) localAcc[i] = 0;
        __syncthreads();

        if (pic[gloID] > 0) {
            for (int t = 0; t < actualTile; ++t) {
                int tIdx = tStart + t;
                float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
                int rIdx = (int)((r + rMax) / rScale);
                if (rIdx >= 0 && rIdx < rBins) atomicAdd(&localAcc[t * rBins + rIdx], 1);
            }
        }
        __syncthreads();

        for (int i = threadIdx.x; i < locSize; i += blockDim.x) {
            int local_t = i / rBins;
            int local_r = i % rBins;
            int global_t = tStart + local_t;
            atomicAdd(&acc[global_t * rBins + local_r], localAcc[i]);
        }
        __syncthreads();
    }
}

// ======== UTIL: write PPM ========
void writePPM(const char *filename, const unsigned char *imgRGB, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) { fprintf(stderr, "Error: no se pudo crear %s\n", filename); return; }
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    fwrite(imgRGB, 1, width * height * 3, fp);
    fclose(fp);
}

int main(int argc, char **argv) {
    int mode = 2;
    if (argc >= 2) mode = atoi(argv[1]);
    if (mode < 0 || mode > 2) {
        printf("Uso: %s [mode]\n mode=0 global\n mode=1 constant\n mode=2 constant+shared\n", argv[0]);
        return 1;
    }
    printf("Modo: %d\n", mode);

    // ensure build dir exists for outputs
    system("mkdir -p build");

    const int w = 100, h = 100;
    const int totalPixels = w * h;

    unsigned char *h_pic = (unsigned char*)malloc((size_t)w * h);
    if (!h_pic) { fprintf(stderr, "No memory for host image\n"); return 1; }
    memset(h_pic, 0, w*h);
    for (int x = 0; x < w; ++x) h_pic[(h/2)*w + x] = 255;
    for (int y = 0; y < h; ++y) h_pic[y*w + (w/2)] = 255;
    for (int i = 0; i < (w < h ? w : h); ++i) h_pic[i*w + i] = 255;

    // trig tables host
    float *hCos = (float*)malloc(sizeof(float) * degreeBins);
    float *hSin = (float*)malloc(sizeof(float) * degreeBins);
    if (!hCos || !hSin) { fprintf(stderr,"No memory for trig tables\n"); return 1; }
    float rad = 0.0f;
    for (int i = 0; i < degreeBins; ++i) {
        hCos[i] = cosf(rad);
        hSin[i] = sinf(rad);
        rad += radInc;
    }

    // device allocations
    unsigned char *d_pic = NULL;
    int *d_acc = NULL;
    float *dCos_global = NULL, *dSin_global = NULL;
    float *dCos_fallback = NULL, *dSin_fallback = NULL;
    bool use_symbol = true;
    bool used_fallback_arrays = false;

    CUDA_CHECK(cudaMalloc(&d_pic, (size_t)totalPixels));
    CUDA_CHECK(cudaMemcpy(d_pic, h_pic, (size_t)totalPixels, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_acc, sizeof(int) * degreeBins * rBins));
    CUDA_CHECK(cudaMemset(d_acc, 0, sizeof(int) * degreeBins * rBins));

    if (mode == 0) {
        // global-only: copy to device arrays and use pointer kernel
        CUDA_CHECK(cudaMalloc(&dCos_global, sizeof(float) * degreeBins));
        CUDA_CHECK(cudaMalloc(&dSin_global, sizeof(float) * degreeBins));
        CUDA_CHECK(cudaMemcpy(dCos_global, hCos, sizeof(float) * degreeBins, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dSin_global, hSin, sizeof(float) * degreeBins, cudaMemcpyHostToDevice));
        printf("Modo 0: usando tablas cos/sin en memoria global.\n");
    } else {
        // attempt to copy to constant symbol
        cudaError_t err1 = cudaMemcpyToSymbol(d_Cos, hCos, sizeof(float) * degreeBins);
        cudaError_t err2 = cudaMemcpyToSymbol(d_Sin, hSin, sizeof(float) * degreeBins);
        if (err1 != cudaSuccess || err2 != cudaSuccess) {
            // fallback: create device arrays and use pointer-based kernels
            fprintf(stderr, "Advertencia: cudaMemcpyToSymbol falló (%s / %s). Usando fallback con arrays device.\n",
                    cudaGetErrorString(err1), cudaGetErrorString(err2));
            use_symbol = false;
            CUDA_CHECK(cudaMalloc(&dCos_fallback, sizeof(float) * degreeBins));
            CUDA_CHECK(cudaMalloc(&dSin_fallback, sizeof(float) * degreeBins));
            CUDA_CHECK(cudaMemcpy(dCos_fallback, hCos, sizeof(float) * degreeBins, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dSin_fallback, hSin, sizeof(float) * degreeBins, cudaMemcpyHostToDevice));
            used_fallback_arrays = true;
        } else {
            printf("Modo %d: tablas cos/sin copiadas a memoria constante.\n", mode);
        }
    }

    float rMax = sqrtf((float)(w*w + h*h)) / 2.0f;
    float rScale = (2.0f * rMax) / (rBins - 1);

    int threads = 256;
    int blocks = (totalPixels + threads - 1) / threads;
    if (blocks < 1) blocks = 1;

    // prepare timing file name
    char timings_file[128];
    if (mode == 0) sprintf(timings_file, "build/timings_global.txt");
    else if (mode == 1) sprintf(timings_file, "build/timings_constant.txt");
    else sprintf(timings_file, "build/timings_const_shared.txt");

    FILE *ftime = fopen(timings_file, "w");
    if (!ftime) { fprintf(stderr, "No se pudo abrir %s para escribir\n", timings_file); }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.0f;

    size_t sharedBytes = 0;
    if (mode == 2) {
        // estimate dynamic shared needed for tiled_ptr (max TILE_THETA * rBins * sizeof(int))
        sharedBytes = (size_t)TILE_THETA * (size_t)rBins * sizeof(int);
    }

    for (int run = 0; run < TIMED_RUNS; ++run) {
        CUDA_CHECK(cudaMemset(d_acc, 0, sizeof(int) * degreeBins * rBins));
        CUDA_CHECK(cudaEventRecord(start, 0));

        if (mode == 0) {
            // use pointer-based kernel (global arrays)
            GPU_HoughTran_global_ptr<<<blocks, threads>>>(d_pic, w, h, d_acc, rMax, rScale, dCos_global, dSin_global);
        } else if (mode == 1) {
            if (use_symbol) {
                GPU_HoughTran_const<<<blocks, threads>>>(d_pic, w, h, d_acc, rMax, rScale);
            } else {
                // fallback pointer kernel
                GPU_HoughTran_global_ptr<<<blocks, threads>>>(d_pic, w, h, d_acc, rMax, rScale, dCos_fallback, dSin_fallback);
            }
        } else { // mode == 2
            if (use_symbol) {
                // use tiled kernel reading from constant memory (uses fixed shared array)
                GPU_HoughTran_tiled_const<<<blocks, threads>>>(d_pic, w, h, d_acc, rMax, rScale);
            } else {
                // use tiled ptr kernel with dynamic shared memory
                GPU_HoughTran_tiled_ptr<<<blocks, threads, sharedBytes>>>(d_pic, w, h, d_acc, rMax, rScale, dCos_fallback, dSin_fallback);
            }
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (ftime) fprintf(ftime, "Run %d: %.3f ms\n", run+1, ms);
        printf("Run %d: %.3f ms\n", run+1, ms);
    }

    if (ftime) fclose(ftime);

    // copy accumulator back
    int *h_acc = (int*)malloc(sizeof(int) * degreeBins * rBins);
    if (!h_acc) { fprintf(stderr, "No memory for host acc\n"); return 1; }
    CUDA_CHECK(cudaMemcpy(h_acc, d_acc, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost));

    // compute threshold mean + 2*std
    double sum = 0.0, sum2 = 0.0;
    int totalCells = degreeBins * rBins;
    for (int i = 0; i < totalCells; ++i) { sum += h_acc[i]; sum2 += (double)h_acc[i] * h_acc[i]; }
    double mean = sum / totalCells;
    double var = sum2 / totalCells - mean * mean;
    double stddev = var > 0.0 ? sqrt(var) : 0.0;
    unsigned int threshold = (unsigned int)floor(mean + 2.0 * stddev);
    printf("mean=%.3f stddev=%.3f threshold=%u\n", mean, stddev, threshold);

    // prepare output image rgb
    unsigned char *outRGB = (unsigned char*)malloc((size_t)w * h * 3);
    if (!outRGB) { fprintf(stderr, "No memory outRGB\n"); return 1; }
    for (int i = 0; i < w*h; ++i) {
        unsigned char v = h_pic[i];
        outRGB[3*i+0] = v; outRGB[3*i+1] = v; outRGB[3*i+2] = v;
    }

    // draw lines using Bresenham
    int linesDrawn = 0;
    for (int t = 0; t < degreeBins; ++t) {
        for (int r = 0; r < rBins; ++r) {
            int votes = h_acc[t * rBins + r];
            if (votes > (int)threshold) {
                linesDrawn++;
                float theta = t * radInc;
                float rVal = r * rScale - rMax;
                float a = cosf(theta), b = sinf(theta);
                float x0 = a * rVal + w / 2.0f;
                float y0 = h / 2.0f - b * rVal;
                int x1 = (int)roundf(x0 + 1000.0f * (-b));
                int y1 = (int)roundf(y0 + 1000.0f * (a));
                int x2 = (int)roundf(x0 - 1000.0f * (-b));
                int y2 = (int)roundf(y0 - 1000.0f * (a));
                int dx = abs(x2 - x1), sx = x1 < x2 ? 1 : -1;
                int dy = -abs(y2 - y1), sy = y1 < y2 ? 1 : -1;
                int err = dx + dy, e2;
                while (true) {
                    if (x1 >= 0 && x1 < w && y1 >= 0 && y1 < h) {
                        int idx = y1 * w + x1;
                        outRGB[3*idx+0] = 255; outRGB[3*idx+1] = 0; outRGB[3*idx+2] = 0;
                    }
                    if (x1 == x2 && y1 == y2) break;
                    e2 = 2 * err;
                    if (e2 >= dy) { err += dy; x1 += sx; }
                    if (e2 <= dx) { err += dx; y1 += sy; }
                }
            }
        }
    }
    printf("Líneas dibujadas: %d\n", linesDrawn);

    // write outputs in build/
    writePPM("build/output_lines.ppm", outRGB, w, h);
    printf("Escrito build/output_lines.ppm\n");
    PGMImage outPGM;
    outPGM.width = w; outPGM.height = h; outPGM.max_gray = 255;
    outPGM.data = h_pic;
    writePGM("build/input_synthetic.pgm", &outPGM);

    // cleanup
    CUDA_CHECK(cudaFree(d_pic));
    CUDA_CHECK(cudaFree(d_acc));
    if (dCos_global) CUDA_CHECK(cudaFree(dCos_global));
    if (dSin_global) CUDA_CHECK(cudaFree(dSin_global));
    if (dCos_fallback) CUDA_CHECK(cudaFree(dCos_fallback));
    if (dSin_fallback) CUDA_CHECK(cudaFree(dSin_fallback));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaDeviceReset());

    free(h_pic);
    free(hCos);
    free(hSin);
    free(h_acc);
    free(outRGB);

    printf("Finalizado correctamente.\n");
    return 0;
}
