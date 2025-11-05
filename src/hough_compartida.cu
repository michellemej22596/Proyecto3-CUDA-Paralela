#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>

#define degreeBins 90
#define rBins 100
#define radInc (M_PI / degreeBins)

__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale) {
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h) return;

    int locID = threadIdx.x;
    
    __shared__ int localAcc[degreeBins * rBins];
    
    for (int i = locID; i < degreeBins * rBins; i += blockDim.x) {
        localAcc[i] = 0;
    }
    __syncthreads();

    if (pic[gloID] > 0) {
        int xCoord = gloID % w;
        int yCoord = gloID / w;
        int xCent = w / 2;
        int yCent = h / 2;

        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            float r = ((xCoord - xCent) * d_Cos[tIdx]) + ((yCoord - yCent) * d_Sin[tIdx]);
            int rIdx = (int)((r + rMax) / rScale);
            
            if (rIdx >= 0 && rIdx < rBins) {
                atomicAdd(&localAcc[tIdx * rBins + rIdx], 1);
            }
        }
    }
    
    __syncthreads();
    
    for (int i = locID; i < degreeBins * rBins; i += blockDim.x) {
        if (localAcc[i] > 0) {
            atomicAdd(&acc[i], localAcc[i]);
        }
    }
}

int main() {
    printf("=== Transformada de Hough - 3 OPTIMIZACIONES ===\n");
    printf("(Global + Constante + Compartida)\n\n");
    
    int w = 256;
    int h = 256;
    
    // Generar imagen de prueba con línea horizontal
    unsigned char *pic = (unsigned char *)malloc(w * h);
    memset(pic, 0, w * h);
    
    int y_line = h / 2;
    for (int x = 50; x < w - 50; x++) {
        pic[y_line * w + x] = 255;
    }
    
    printf("Imagen: %dx%d\n", w, h);
    printf("Bins: %d angulos x %d distancias\n\n", degreeBins, rBins);
    
    float *pcCos = (float *)malloc(sizeof(float) * degreeBins);
    float *pcSin = (float *)malloc(sizeof(float) * degreeBins);
    float rad = 0;
    for (int i = 0; i < degreeBins; i++) {
        pcCos[i] = cos(rad);
        pcSin[i] = sin(rad);
        rad += radInc;
    }
    
    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = (2 * rMax) / rBins;
    
    printf("rMax: %.2f, rScale: %.2f\n\n", rMax, rScale);
    
    // Copiar imagen al device
    unsigned char *d_pic;
    cudaMalloc((void **)&d_pic, w * h);
    cudaMemcpy(d_pic, pic, w * h, cudaMemcpyHostToDevice);
    
    cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(float) * degreeBins);
    cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(float) * degreeBins);
    
    // Crear acumulador en device
    int *d_hough;
    int houghSize = degreeBins * rBins;
    cudaMalloc((void **)&d_hough, sizeof(int) * houghSize);
    cudaMemset(d_hough, 0, sizeof(int) * houghSize);
    
    // Configuración del kernel
    int blockNum = 256;
    int threadNum = 256;
    
    printf("Lanzando kernel: %d bloques x %d threads\n", blockNum, threadNum);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    GPU_HoughTran<<<blockNum, threadNum>>>(d_pic, w, h, d_hough, rMax, rScale);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Tiempo GPU: %.3f ms\n\n", milliseconds);
    
    // Copiar resultados al host
    int *h_hough = (int *)malloc(sizeof(int) * houghSize);
    cudaMemcpy(h_hough, d_hough, sizeof(int) * houghSize, cudaMemcpyDeviceToHost);
    
    // Encontrar máximo
    int max = 0;
    for (int i = 0; i < houghSize; i++) {
        if (h_hough[i] > max) max = h_hough[i];
    }
    
    printf("=== RESULTADOS ===\n");
    printf("Maximo valor acumulado: %d\n\n", max);
    
    if (max > 0) {
        printf("EXITO: Se detectaron lineas!\n");
    } else {
        printf("ADVERTENCIA: No se detectaron lineas\n");
    }
    
    cudaFree(d_pic);
    cudaFree(d_hough);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(pic);
    free(pcCos);
    free(pcSin);
    free(h_hough);
    
    return 0;
}
