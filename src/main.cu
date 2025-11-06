#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <stdlib.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define degreeBins 180
#define rBins 100
#define radInc (M_PI / degreeBins)

__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale)
{
    int locID = threadIdx.x;
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h) return;

    int xCent = w / 2;
    int yCent = h / 2;

    __shared__ int localAcc[degreeBins * rBins];
    for (int i = locID; i < degreeBins * rBins; i += blockDim.x)
        localAcc[i] = 0;
    __syncthreads();

    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    if (pic[gloID] > 0)
    {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
            float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (int)((r + rMax) / rScale);
            if (rIdx >= 0 && rIdx < rBins)
                atomicAdd(&localAcc[rIdx * degreeBins + tIdx], 1);
        }
    }

    __syncthreads();

    for (int i = locID; i < degreeBins * rBins; i += blockDim.x)
        atomicAdd(&acc[i], localAcc[i]);
}

int main()
{
    int w = 100, h = 100;

    float *pcCos = (float *)malloc(sizeof(float) * degreeBins);
    float *pcSin = (float *)malloc(sizeof(float) * degreeBins);
    float rad = 0;
    for (int i = 0; i < degreeBins; i++)
    {
        pcCos[i] = cos(rad);
        pcSin[i] = sin(rad);
        rad += radInc;
    }

    cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(float) * degreeBins);
    cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(float) * degreeBins);

    unsigned char *h_pic = (unsigned char *)malloc(w * h);
    for (int i = 0; i < w * h; i++) h_pic[i] = 0;

    // Dibujar líneas sintéticas (horizontal, vertical, diagonal)
    for (int x = 0; x < w; x++) h_pic[h/2 * w + x] = 255;
    for (int y = 0; y < h; y++) h_pic[y * w + w/2] = 255;
    for (int i = 0; i < (w < h ? w : h); i++) h_pic[i * w + i] = 255;

    unsigned char *d_pic;
    cudaMalloc((void **)&d_pic, w * h);
    cudaMemcpy(d_pic, h_pic, w * h, cudaMemcpyHostToDevice);

    float rMax = sqrt(1.0f * w * w + 1.0f * h * h) / 2;
    float rScale = (2.0f * rMax) / (rBins - 1);

    int *d_acc;
    cudaMalloc((void **)&d_acc, sizeof(int) * degreeBins * rBins);
    cudaMemset(d_acc, 0, sizeof(int) * degreeBins * rBins);

    int blockNum = (w * h + 255) / 256;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    GPU_HoughTran<<<blockNum, 256>>>(d_pic, w, h, d_acc, rMax, rScale);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tiempo GPU: %.3f ms\n", milliseconds);

    int *h_acc = (int *)malloc(sizeof(int) * degreeBins * rBins);
    cudaMemcpy(h_acc, d_acc, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

    // ==== Calcular threshold (mean + 2 std) ====
    double sum = 0, sum2 = 0;
    for (int i = 0; i < degreeBins * rBins; i++) {
        sum += h_acc[i];
        sum2 += (double)h_acc[i] * h_acc[i];
    }
    double mean = sum / (degreeBins * rBins);
    double std = sqrt((sum2 / (degreeBins * rBins)) - (mean * mean));
    double threshold = mean + 2 * std;

    printf("Threshold usado = %.2f (mean=%.2f, std=%.2f)\n", threshold, mean, std);

    // ==== Crear imagen RGB para dibujar ====
    unsigned char *rgb = (unsigned char *)malloc(w * h * 3);
    for(int i = 0; i < w*h; i++){
        rgb[3*i+0] = rgb[3*i+1] = rgb[3*i+2] = h_pic[i];
    }

    // ==== Dibujar líneas detectadas ====
    for(int r = 0; r < rBins; r++){
        for(int t = 0; t < degreeBins; t++){
            int votes = h_acc[r * degreeBins + t];
            if(votes > threshold){
                float theta = t * radInc;
                float rVal = r * rScale - rMax;

                float a = cos(theta), b = sin(theta);
                float x0 = a * rVal + w/2, y0 = b * rVal + h/2;

                int x1 = (int)(x0 + 1000*(-b));
                int y1 = (int)(y0 + 1000*(a));
                int x2 = (int)(x0 - 1000*(-b));
                int y2 = (int)(y0 - 1000*(a));

                int dx = abs(x2-x1), sx = x1<x2?1:-1;
                int dy = -abs(y2-y1), sy = y1<y2?1:-1;
                int err = dx + dy, e2;

                while(true){
                    if(x1>=0 && x1<w && y1>=0 && y1<h){
                        int idx = (y1 * w + x1) * 3;
                        rgb[idx] = 255; rgb[idx+1] = 0; rgb[idx+2] = 0;
                    }
                    if(x1==x2 && y1==y2) break;
                    e2 = 2*err;
                    if(e2 >= dy){ err += dy; x1 += sx; }
                    if(e2 <= dx){ err += dx; y1 += sy; }
                }
            }
        }
    }

    stbi_write_png("output_hough.png", w, h, 3, rgb, w*3);
    printf("Imagen con líneas guardada como output_hough.png\n");

    // ==== Liberar memoria ====
    cudaFree(d_pic);
    cudaFree(d_acc);
    free(h_pic);
    free(h_acc);
    free(pcCos);
    free(pcSin);
    free(rgb);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
