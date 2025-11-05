#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

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

    if (pic[gloID] > 0) {
        int xCoord = gloID % w - xCent;
        int yCoord = yCent - gloID / w;

        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (int)((r + rMax) / rScale);
            
            if (rIdx >= 0 && rIdx < rBins) {
                atomicAdd(acc + (rIdx * degreeBins + tIdx), 1);
            }
        }
    }
}

int main() {
    int w = 256;
    int h = 256;
    
    printf("=== Transformada de Hough BASICA ===\n\n");
    printf("Imagen: %dx%d\n", w, h);
    printf("Bins: %d angulos x %d distancias\n\n", degreeBins, rBins);

    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = 2 * rMax / rBins;

    float *pcCos = (float *)malloc(sizeof(float) * degreeBins);
    float *pcSin = (float *)malloc(sizeof(float) * degreeBins);
    float rad = 0;
    for (int i = 0; i < degreeBins; i++) {
        pcCos[i] = cos(rad);
        pcSin[i] = sin(rad);
        rad += degreeInc * M_PI / 180;
    }

    cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(float) * degreeBins);
    cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(float) * degreeBins);

    unsigned char *h_pic = (unsigned char *)malloc(w * h);
    memset(h_pic, 0, w * h);
    
    // Línea horizontal en el centro
    for (int x = 50; x < 200; x++) {
        h_pic[128 * w + x] = 255;
    }
    
    printf("Imagen creada con linea horizontal\n\n");

    unsigned char *d_pic;
    int *d_hough;
    cudaMalloc((void **)&d_pic, sizeof(unsigned char) * w * h);
    cudaMalloc((void **)&d_hough, sizeof(int) * degreeBins * rBins);

    cudaMemcpy(d_pic, h_pic, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

    int blockNum = ceil(w * h / 256.0);
    printf("Lanzando kernel: %d bloques x 256 threads\n", blockNum);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    GPU_HoughTran<<<blockNum, 256>>>(d_pic, w, h, d_hough, rMax, rScale);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR CUDA: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tiempo GPU: %.3f ms\n\n", milliseconds);

    int *h_hough = (int *)malloc(degreeBins * rBins * sizeof(int));
    cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

    int max = 0;
    for (int i = 0; i < degreeBins * rBins; i++) {
        if (h_hough[i] > max) max = h_hough[i];
    }

    printf("=== RESULTADOS ===\n");
    printf("Maximo valor acumulado: %d\n", max);
    
    if (max > 0) {
        printf("\nEXITO: Se detectaron lineas!\n");
    } else {
        printf("\nADVERTENCIA: No se detectaron lineas\n");
    }

    cudaFree(d_pic);
    cudaFree(d_hough);
    free(h_pic);
    free(h_hough);
    free(pcCos);
    free(pcSin);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
