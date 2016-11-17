#include <stdio.h>
#include <time.h>

void jacobiOnHost(float* x_next, float* A, float* x_now, float* b, int Ni, int Nj) {
    int i,j;
    float sigma;

    for (i=0; i<Ni; i++) {
        sigma = 0.0;
        for (j=0; j<Nj; j++) {
            if (i != j) {
                sigma += A[i*Nj + j] * x_now[j];
            }
        }
        x_next[i] = (b[i] - sigma) / A[i*Nj + i];
    }
}

__global__ void jacobiOnDevice(float* x_next, float* A, float* x_now, float* b, int Ni, int Nj) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < Ni) {
        float sigma = 0.0;
        int idx_Ai = idx*Nj;   
        for (int j=0; j<Nj; j++) {
            if (idx != j) {
                sigma += A[idx_Ai + j] * x_now[j];
            }
        }
        x_next[idx] = (b[idx] - sigma) / A[idx_Ai + idx];
    }
}

int main() { 
    time_t start_h, end_h, start_d, end_d;
    float t_host, t_dev;
    float *x_now, *x_next, *A, *b, *x_h, *x_d;
    float *x_now_d, *x_next_d, *A_d, *b_d;
    int N, Ni, Nj, iter, tileSize, i , k;   

    Ni= 2048, Nj=2048, iter=10, tileSize=16;
    N = Ni * Nj;

    printf("\nRunning Jacobi method:\n");
    printf("======================\n\n");
    printf("Parameters:\n");
    printf("N=%d, Ni=%d, Nj=%d, ", N, Ni, Nj);
    printf("tilesize=%d\n", tileSize);

    x_next = (float *) malloc(Ni*sizeof(float));
    A = (float *) malloc(N*sizeof(float));
    x_now = (float *) malloc(Ni*sizeof(float));
    b = (float *) malloc(Ni*sizeof(float));
    x_h = (float *) malloc(Ni*sizeof(float));
    x_d = (float *) malloc(Ni*sizeof(float));

    for (i=0; i<Ni; i++) {
        x_now[i] = 0;
        x_next[i] = 0;
    }

    for (i = 0; i < N; i ++){
        A[i] = rand()/(float)RAND_MAX;
    }
    for (i = 0; i < Ni; i++){
        b[i] = rand()/(float)RAND_MAX;
    }

    start_h = clock();
    // HOST
    for (k=0; k<iter; k++) {
        if (k%2) {
            jacobiOnHost(x_now, A, x_next, b, Ni, Nj);
        }
        else {
            jacobiOnHost(x_next, A, x_now, b, Ni, Nj);
        }
    }
    
    end_h = clock();

    for (i=0; i<Nj; i++) {
        x_h[i] = x_next[i];
    }

    for (i=0; i<Ni; i++) {
        x_now[i] = 0;
        x_next[i] = 0;
    }

    cudaMalloc((void **) &x_next_d, Ni*sizeof(float));
    cudaMalloc((void **) &A_d, N*sizeof(float));
    cudaMalloc((void **) &x_now_d, Ni*sizeof(float));
    cudaMalloc((void **) &b_d, Ni*sizeof(float));

    cudaMemcpy(x_next_d, x_next, sizeof(float)*Ni, cudaMemcpyHostToDevice);
    cudaMemcpy(A_d, A, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(x_now_d, x_now, sizeof(float)*Ni, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, sizeof(float)*Ni, cudaMemcpyHostToDevice);

    int nTiles = Ni/tileSize + (Ni%tileSize == 0?0:1);
    int gridHeight = Nj/tileSize + (Nj%tileSize == 0?0:1);
    int gridWidth = Ni/tileSize + (Ni%tileSize == 0?0:1);
    printf("w=%d, h=%d\n",gridWidth,gridHeight);
    dim3 dGrid(gridHeight, gridWidth), dBlock(tileSize, tileSize);

    start_d = clock();
     
    //  DEVICE
    for (k=0; k<iter; k++) {
        if (k%2) {
            jacobiOnDevice <<< nTiles, tileSize >>> (x_now_d, A_d, x_next_d, b_d, Ni, Nj);
        }
        else {
            jacobiOnDevice <<< nTiles, tileSize >>> (x_next_d, A_d, x_now_d, b_d, Ni, Nj);
        }
    }
        
    end_d = clock();

    cudaMemcpy(x_d, x_next_d, sizeof(float)*Ni, cudaMemcpyDeviceToHost);

    free(x_next); free(A); free(x_now); free(b);
    cudaFree(x_next_d); cudaFree(A_d); cudaFree(x_now_d); cudaFree(b_d);

    /*
    for (i =0 ; i < Ni; i ++) {
        printf("x_h[%d]=%f\n",i,x_h[i]);
        printf("x_d[%d]=%f\n",i,x_d[i]);
    }
    */
    t_host = ((float)end_h - (float)start_h) / CLOCKS_PER_SEC;
    t_dev = ((float)end_d - (float)start_d) / CLOCKS_PER_SEC;
    printf("\nTiming:\nHost: %f\nDevice: %f\n\n", t_host, t_dev);

    printf("successfully.\n");

    return 0;
}
