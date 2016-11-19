#include <stdio.h>
#include <time.h>

void onCPU(float* A, float* B, float* xNow, float* xNext, int Ni) {
    int i,j;
    float sum;

    for (i=0; i<Ni; i++) {
        sum = 0.0;
        for (j=0; j<Ni; j++) {
            if (i != j) {
                sum += A[i*Ni + j] * xNow[j];
            }
        }
        xNext[i] = (B[i] - sum) / A[i*Ni + i];
    }
}

__global__ void onGPU(float* A, float* B,  float* xNow, float* xNext, int Ni) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < Ni) {
        float sum = 0.0;
        int idx_Ai = idx*Ni;   
        for (int j=0; j<Ni; j++) {
            if (idx != j) {
                sum += A[idx_Ai + j] * xNow[j];
            }
        }
        xNext[idx] = (B[idx] - sum) / A[idx_Ai + idx];
    }
}

int main( int argc, char *argv[] ) { 
    time_t start_h, end_h, start_d, end_d;
    float timeCPU, timeGPU;
    float *xNow, *xNext, *A, *B, *xCPU, *xGPU;
    float *xNowDevice, *xNextDevice, *deviceA, *deviceB;
    int N, Ni, iter, tileSize, i , k;   

    Ni=2048 , iter=10;
    N = Ni * Ni;

    if( argc == 2 ) {
        tileSize = atoi(argv[1]);
    }
    else if( argc > 2 ) {
        printf("Too many arguments supplied.\n");
        exit(0);
    }
    else {
        printf("Usage: ./jacobi argument(int).\n");
        exit(0);
    }
    
    printf("Jacobi method:\n\n");
    printf("N = %d, Ni = %d, ", N, Ni);
    printf("thread = %d\n", tileSize);

    A = (float *) malloc(N*sizeof(float));
    B = (float *) malloc(Ni*sizeof(float));
    xNext = (float *) malloc(Ni*sizeof(float));
    xNow = (float *) malloc(Ni*sizeof(float));
    xCPU = (float *) malloc(Ni*sizeof(float));
    xGPU = (float *) malloc(Ni*sizeof(float));

    for (i=0; i<Ni; i++) {
        xNow[i] = 0;
        xNext[i] = 0;
    }

    for (i = 0; i < N; i ++){
        A[i] = rand()/(float)RAND_MAX;
    }
    for (i = 0; i < Ni; i++){
        B[i] = rand()/(float)RAND_MAX;
    } 

    /*
    A[0] = 4;
    A[1] = 0.24;
    A[2] = -0.08;
    A[3] = 0.09;
    A[4] = 3;
    A[5] = -0.15;
    A[6] = 0.04;
    A[7] = -0.08;
    A[8] = 4; 

    B[0] = 8;
    B[1] = 9;
    B[2] = 20;
    */

    //============================================= CPU =============================================//
    start_h = clock();

    for (k=0; k<iter; k++) {
        if (k%2) {
            onCPU( A, B, xNow, xNext, Ni);
        }
        else {
            onCPU( A, B, xNext, xNow, Ni);
        }
    }
    
    end_h = clock();

    if (iter%2 != 0) {
        for (i=0; i<Ni; i++) {
            xCPU[i] = xNext[i];
        }
    }
    else {
        for (i=0; i<Ni; i++) {
            xCPU[i] = xNow[i];
        }
    }

    for (i=0; i<Ni; i++) {
        xNow[i] = 0;
        xNext[i] = 0;
    }

    cudaMalloc((void **) &deviceA, N*sizeof(float));
    cudaMalloc((void **) &deviceB, Ni*sizeof(float));
    cudaMalloc((void **) &xNowDevice, Ni*sizeof(float));
    cudaMalloc((void **) &xNextDevice, Ni*sizeof(float));

    cudaMemcpy(deviceA, A, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, B, sizeof(float)*Ni, cudaMemcpyHostToDevice);
    cudaMemcpy(xNowDevice, xNow, sizeof(float)*Ni, cudaMemcpyHostToDevice);
    cudaMemcpy(xNextDevice, xNext, sizeof(float)*Ni, cudaMemcpyHostToDevice);
   
    int nTiles = Ni/tileSize + (Ni%tileSize == 0?0:1);
    int gridHeight = Ni/tileSize + (Ni%tileSize == 0?0:1);
    int gridWidth = Ni/tileSize + (Ni%tileSize == 0?0:1);
    printf("w=%d, h=%d\n",gridWidth,gridHeight);
    dim3 dGrid(gridHeight, gridWidth), dBlock(tileSize, tileSize);

    //============================================= GPU =============================================//
    start_d = clock();
     
    for (k=0; k<iter; k++) {
        if (k%2) {
            onGPU <<< nTiles, tileSize >>> (deviceA, deviceB, xNowDevice, xNextDevice, Ni);
        }
        else {
            onGPU <<< nTiles, tileSize >>> (deviceA, deviceB, xNextDevice, xNowDevice, Ni);
        }
    }
        
    end_d = clock();

    if (iter%2 != 0 ){
        cudaMemcpy(xGPU, xNextDevice, sizeof(float)*Ni, cudaMemcpyDeviceToHost);
    }
    else {
        cudaMemcpy(xGPU, xNowDevice, sizeof(float)*Ni, cudaMemcpyDeviceToHost);
    }

    free(xNext); free(A); free(xNow); free(B);
    cudaFree(xNextDevice); cudaFree(deviceA); cudaFree(xNowDevice); cudaFree(deviceB);

    /*
    for (i =0 ; i < Ni; i ++) {
        //printf("xCPU[%d]=%f\n",i,xCPU[i]);
        //printf("xGPU[%d]=%f\n",i,xGPU[i]);
    }
    */
    
    timeCPU = ((float)end_h - (float)start_h) / CLOCKS_PER_SEC;
    timeGPU = ((float)end_d - (float)start_d) / CLOCKS_PER_SEC;
    printf("\nTiming:\nCPU: %f\nGPU: %f\n\n", timeCPU, timeGPU);

    return 0;
}