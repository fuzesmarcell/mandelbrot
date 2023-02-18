#include "mandelbrot.h"

#include <cuda_runtime.h>

__global__ void mandelbrot_kernel(float h, float v, int* iter_buffer, int width, int height) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i >= width || j >= height) { return; }

	float x0 = (h * (float)i) - 2.00f;
	float y0 = (v * (float)j) - 1.12f;

	float x = 0.f;
	float y = 0.f;

	int iteration = 0;
	while (x*x + y*y <= 2.f*2.f && iteration < MANDELBROT_MAX_ITERATIONS) {
		float x_temp = x*x - y*y + x0;
		y = 2.f*x*y + y0;
		x = x_temp;
		iteration++;
	}

	iter_buffer[j*width + i] = iteration;
}

void mandelbrot_gpu(int* iter_buffer, int width, int height) {
	cudaError_t err;

	size_t buffer_size = width * height * sizeof(int);
	int* iter_buffer_d;
	err = cudaMalloc((void**)&iter_buffer_d, buffer_size);
	if (err != cudaSuccess) { return; }

	float h = ((0.47f + 2.00f) / (float)(width));
	float v = ((1.12f + 1.12f) / (float)(height));

	dim3 block(32, 32);
	dim3 grid((width + block.x-1) / block.x, (height + block.y - 1) / block.y);
	mandelbrot_kernel<<<grid, block>>>(h, v, iter_buffer_d, width, height);

	err = cudaGetLastError();
	if (err != cudaSuccess) { goto error; }
	
	err = cudaMemcpy(iter_buffer, iter_buffer_d, buffer_size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) { goto error; }

error:
	// Does not really make sense to check the error here at the end if we do not propagate the error :)
	err = cudaFree(iter_buffer_d);
	if (err != cudaSuccess) { return; }
}