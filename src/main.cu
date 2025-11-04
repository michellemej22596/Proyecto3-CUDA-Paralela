#include <stdio.h>
#include <cuda.h>
#include <math.h>

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
    
    // Línea horizontal en el medio
    for (int x = 0; x < w; x++) {
        h_pic[h/2 * w + x] = 255;
    }
    
    // Línea vertical en el medio
    for (int y = 0; y < h; y++) {
        h_pic[y * w + w/2] = 255;
    }
    
    // Diagonal principal
    for (int i = 0; i < (w < h ? w : h); i++) {
        h_pic[i * w + i] = 255;
    }
    
    int whitePixels = 0;
    for (int i = 0; i < w * h; i++) {
        if (h_pic[i] > 0) whitePixels++;
    }
    printf("Pixels blancos en la imagen: %d\n", whitePixels);

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

    int *h_acc = (int *)malloc(sizeof(int) * degreeBins * rBins);
    cudaMemcpy(h_acc, d_acc, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

    printf("Tiempo GPU: %.3f ms\n", milliseconds);
    
    int maxVal = 0;
    int totalVotes = 0;
    int nonZeroBins = 0;
    for (int i = 0; i < degreeBins * rBins; i++) {
        totalVotes += h_acc[i];
        if (h_acc[i] > 0) nonZeroBins++;
        if (h_acc[i] > maxVal) maxVal = h_acc[i];
    }
    printf("Máximo valor acumulado: %d\n", maxVal);
    printf("Total de votos: %d\n", totalVotes);
    printf("Bins con votos: %d / %d\n", nonZeroBins, degreeBins * rBins);
    printf("rMax: %.2f, rScale: %.4f\n", rMax, rScale);

    cudaFree(d_pic);
    cudaFree(d_acc);
    free(h_pic);
    free(h_acc);
    free(pcCos);
    free(pcSin);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
