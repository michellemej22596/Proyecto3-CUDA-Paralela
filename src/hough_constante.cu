#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define degreeBins 90
#define rBins 100
#define degreeInc 2

__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale) {
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h) return;

    int xCent = w / 2;
    int yCent = h / 2;

    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    if (pic[gloID] > 0) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (int)((r + rMax) / rScale);
            
            if (rIdx >= 0 && rIdx < rBins) {
                atomicAdd(&acc[rIdx * degreeBins + tIdx], 1);
            }
        }
    }
}

void CPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float *pcCos, float *pcSin) {
    int xCent = w / 2;
    int yCent = h / 2;
    
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            int idx = i * w + j;
            if (pic[idx] > 0) {
                int xCoord = j - xCent;
                int yCoord = yCent - i;
                
                for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
                    float r = xCoord * pcCos[tIdx] + yCoord * pcSin[tIdx];
                    int rIdx = (int)((r + rMax) / rScale);
                    
                    if (rIdx >= 0 && rIdx < rBins) {
                        acc[rIdx * degreeBins + tIdx]++;
                    }
                }
            }
        }
    }
}

int main() {
    int w = 256;
    int h = 256;
    
    printf("\n=== Transformada de Hough con MEMORIA CONSTANTE ===\n\n");
    printf("Imagen: %dx%d\n", w, h);
    printf("Bins: %d angulos x %d distancias\n\n", degreeBins, rBins);
    
    float *pcCos = (float *)malloc(sizeof(float) * degreeBins);
    float *pcSin = (float *)malloc(sizeof(float) * degreeBins);
    
    float rad;
    for (int i = 0; i < degreeBins; i++) {
        rad = (float)(i * degreeInc) * M_PI / 180.0;
        pcCos[i] = cos(rad);
        pcSin[i] = sin(rad);
    }
    
    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2.0;
    float rScale = (2 * rMax) / (rBins - 1);
    
    printf("rMax: %.2f, rScale: %.2f\n\n", rMax, rScale);
    
    unsigned char *pic = (unsigned char *)malloc(w * h);
    memset(pic, 0, w * h);
    
    int y = h / 2;
    for (int x = 0; x < w; x++) {
        pic[y * w + x] = 255;
    }
    
    printf("Imagen creada con linea horizontal\n\n");
    
    int *acc = (int *)malloc(sizeof(int) * degreeBins * rBins);
    memset(acc, 0, sizeof(int) * degreeBins * rBins);
    
    unsigned char *d_in;
    int *d_hough;
    
    cudaMalloc((void **)&d_in, sizeof(unsigned char) * w * h);
    cudaMalloc((void **)&d_hough, sizeof(int) * degreeBins * rBins);
    
    cudaMemcpy(d_in, pic, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);
    
    cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(float) * degreeBins);
    cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(float) * degreeBins);
    
    int blockNum = 256;
    int threadNum = 256;
    
    printf("Lanzando kernel: %d bloques x %d threads\n", blockNum, threadNum);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    GPU_HoughTran<<<blockNum, threadNum>>>(d_in, w, h, d_hough, rMax, rScale);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Tiempo GPU (con memoria constante): %.3f ms\n\n", milliseconds);
    
    cudaMemcpy(acc, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);
    
    int maxVal = 0;
    for (int i = 0; i < degreeBins * rBins; i++) {
        if (acc[i] > maxVal) maxVal = acc[i];
    }
    
    printf("=== RESULTADOS ===\n");
    printf("Maximo valor acumulado: %d\n\n", maxVal);
    
    if (maxVal > 0) {
        printf("EXITO: Se detectaron lineas!\n");
        printf("La memoria CONSTANTE mejora el acceso a valores trigonometricos.\n\n");
    } else {
        printf("ADVERTENCIA: No se detectaron lineas\n\n");
    }
    
    cudaFree(d_in);
    cudaFree(d_hough);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    free(pic);
    free(acc);
    free(pcCos);
    free(pcSin);
    
    return 0;
}
