/*
 * CUDA convolutional neural net
 */

#include <iostream>
#include <iomanip>
#include "ee155_utils.hxx"
#include "matrix.hxx"
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

using namespace std;

template<int size> __device__ void print_array (float arr [size][size]);

const int BS=32;		// The blocks are BS x BS.
const int FILT_SIZE_MAX = 12;	// The filter size (needs not be a power of 2)

///////////////////////////////
// This is the CUDA kernel function for you to write.
//////////////////////////////
__global__ void CNN (float *d_inp, float *d_f, float *d_out, int N_inp, int N_f, int N_out, int pitch_f, int pitch_out) {
    int rB=blockIdx.y, cB=blockIdx.x;
    int rI=threadIdx.y, cI=threadIdx.x;

    __shared__ float shared_inp[BS][BS], shared_f[FILT_SIZE_MAX][FILT_SIZE_MAX];

    int N_out_per_b = BS - N_f + 1;

    int inp_r = (N_out_per_b * rB) + rI;
    int inp_c = (N_out_per_b * cB) + cI;

    if (inp_r < N_inp && inp_c < N_inp) {
        shared_inp[rI][cI] = d_inp[(inp_r * N_inp) + inp_c];
    }

    if (rI < N_f && cI < N_f) {
        shared_f[rI][cI] = d_f[(rI * pitch_f) + cI];
    }

    __syncthreads();

    if (rI < N_out_per_b && cI < N_out_per_b)
    if (inp_r < N_out && inp_c < N_out) {

        float result = 0.0;

        for (int f_r = 0; f_r < N_f; f_r++) 
        for (int f_c = 0; f_c < N_f; f_c++) {
            result += shared_inp[rI + f_r][cI + f_c] * shared_f[f_r][f_c];
        }

        d_out[(inp_r * pitch_out) + inp_c] = result;
    }

}


///////////////////////////////
// This is the host function for you to write.
// It allocates memory and moves data between CPU<->GPU
//
void Matrix::CNN2 (const Matrix &inp, const Matrix &f, int dummy) {
    auto start1 = start_time();

    // Allocate input matrix in device memory. It's a nice 2^N size, so don't
    // bother with cudaMallocPitch().
    assert (1<<inp._log2NColsAlc == inp._nCols);
    int numElem=inp.data.size(), sizeBytes = numElem*4;
    float *d_inp = NULL;
    cudaError_t err = cudaMalloc((void **)&d_inp, sizeBytes);
    ERR_CHK (err, "Failed to allocate device matrix 'inp'");
    //LOG ("\nInput matrix: rows="<<inp._nRows<<", cols="<<inp._nCols);

    // Copy inp from host memory to device memory.
    err = cudaMemcpy (d_inp, inp.data.data(), sizeBytes,cudaMemcpyHostToDevice);
    ERR_CHK (err, "Failed to copy matrix inp from host to device");

    // Allocate device memory for filter. Again, don't bother with
    // cudaMallocPitch(); the filter is small, and Matrix has already picked 
    // a power of 2 columns
    float *d_f = NULL;
    sizeBytes = static_cast<int> (f.data.size()) * 4;
    err = cudaMalloc((void **)&d_f, sizeBytes);
    ERR_CHK (err, "Failed to allocate device matrix for the filter f");

    // Copy f from host memory to device memory.
    err = cudaMemcpy (d_f, f.data.data(), sizeBytes, cudaMemcpyHostToDevice);
    ERR_CHK (err, "Failed to copy matrix f from host to device");

    
    // We're assuming the filter and input matrix are of equal width and length
    int N_inp = inp.N();
    int N_f   = f.N();
    int N_out = this->N();

    int out_bytes = N_out * 4;

    int N_out_per_b = BS - N_f + 1;

    // Allocate device memory for the output matrix. In fact, allocate the
    // entire thing (with padding).
    size_t pitch_out_bytes = 0;
    float *d_out = NULL;

    err = cudaMallocPitch((void**)&d_out, &pitch_out_bytes, out_bytes, N_out);
    ERR_CHK (err, "Failed to allocate device matrix 'out'");

    size_t pitch_out_elems = pitch_out_bytes / 4;

    cudaDeviceSynchronize(); long int time1 = delta_usec (start1);
    auto start2 = start_time();

    int N_thread_blocks = N_out % N_out_per_b == 0 ? N_out / N_out_per_b : (N_out / N_out_per_b) + 1;

    dim3 grid(N_thread_blocks, N_thread_blocks), block(BS, BS); 

    size_t pitch_f_elems = 1<<f._log2NColsAlc; 

    // Launch the CUDA Kernel
    CNN <<<grid, block>>> (d_inp, d_f, d_out, N_inp, N_f, N_out, pitch_f_elems, pitch_out_elems);
    err = cudaGetLastError();
    ERR_CHK (err, "Failed to launch or finish CNN_kernel");

    cudaDeviceSynchronize(); long int time2 = delta_usec (start2);
    auto start3 = start_time();

    // Copy the result from device memory to host memory.
    err = cudaMemcpy2D (this->data.data(), (1<<this->_log2NColsAlc) * 4, d_out, pitch_out_bytes, out_bytes, N_out, cudaMemcpyDeviceToHost);
    ERR_CHK (err, "Failed to copy result from device to host");
    cudaDeviceSynchronize(); long int time3 = delta_usec (start3);

    err = cudaFree(d_inp);
    ERR_CHK (err, "Failed to free CUDA matrix inp");

    err = cudaFree(d_f);
    ERR_CHK (err, "Failed to free CUDA matrix f");

    err = cudaFree(d_out);
    ERR_CHK (err, "Failed to free CUDA matrix out");

    cout << setprecision(3) << fixed;
    LOG ("\tCUDA " <<inp.nRows()<<"x"<<inp.nRows()
	 << " CNN with "<<f.nRows()<<"x"<<f.nRows()<<" filter took "
	 <<(time1+time2+time3)/1000000.0<<" sec; "<<(time1/1000000.0)<<"s copy to, "
	 << (time2/1000000.0)<<"s for computation, "<< (time3/1000000.0)<<"s copy back ");
}