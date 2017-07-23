#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */
__global__ void array_set_kernel(float *arr_data, float value, int size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) arr_data[i] = value;
}

__global__ void broadcast_to_kernel(const float *input_data, float *output_data, int size_input, int size_output){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size_output) output_data[i] = input_data[i % size_input];
}

__global__ void reduce_sum_axis_zero_kernel(const float *input_data, float *output_data, int num, int size_output){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size_output){
		float sum = 0;
		for (int j = 0; j < num; ++i) sum += input_data[i * num + j];
		output_data[i] = sum; 
	}
}

__global__ void matrix_elementwise_add_kernel(const float *A_data, const float * B_data, float *output_data, int size_A, int size_B, int size_output){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size_output) output_data[i] = A_data[i % size_A] + B_data[i % size_B];
}

__global__ void matrix_elementwise_add_by_const_kernel(const float *input_data, const float value, float *output_data, int size_output){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size_output) output_data[i] = input_data[i] + value;
}

__global__ void matrix_elementwise_multiply_kernel(const float *A_data, const float * B_data, float *output_data, int size_A, int size_B, int size_output){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size_output) output_data[i] = A_data[i % size_A] * B_data[i % size_B];
}

__global__ void matrix_elementwise_multiply_by_const_kernel(const float *input_data, const float value, float *output_data, int size_output){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size_output) output_data[i] = input_data[i] * value;
}

__global__ void matrix_multiply_kernel(const float *A_data, int row_A, int col_A, bool TA, const float *B_data, int row_B, int col_B, bool TB, float *C_data, int size_C, int row_C){
	int xx = blockIdx.x * blockDim.x + threadIdx.x;
	int i = xx / row_C, j = xx % row_C;
	if (i < size_C){
		int sum = 0;
		if (TA && TB) {
			for (int p = 0; p < col_A; ++p) sum += A_data[i * col_A + p] * B_data[p * col_B + j]; C_data[xx] = sum;
		}
		else if (!TA && TB) {
			for (int p = 0; p < row_A; ++p) sum += A_data[p * col_A + i] * B_data[p * col_B + j]; C_data[xx] = sum;
		}
		else if (TA && !TB) {
			for (int p = 0; p < col_A; ++p) sum += A_data[i * col_A + p] * B_data[j * row_B + p]; C_data[xx] = sum;
		}
		else if (!TA && !TB) {
			for (int p = 0; p < row_A; ++p) sum += A_data[p * col_A + i] * B_data[j * row_B + p]; C_data[xx] = sum;
		}
	}
}

__global__ void relu_kernel(const float *input_data, float *output_data, int size_input, int size_output){
	int i= blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size_output) output_data[i] = max(0, input_data[i]);
}

__global__ void relu_gradient_kernel(const float *input_data, float *output_data, int size_input, int size_output){
	int i= blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size_output){
		if (input_data[i] < 0) output_data[i] = 0;
		if (input_data[i] == 0) output_data[i] = 0.5;
		if (input_data[i] > 0) output_data[i] = 1;
	}
}

__global__ void softmax_kernel(const int row, const int col, const float *input_data, float *output_data, int size_input, int size_output){
	int i= blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size_output) return;
	int xx = i / col * col;
	float maxval = *input_data;
	for (int x = 0; x < col; ++x) maxval = max(maxval, input_data[x + xx]);
	float sum = 0;
	for (int x = 0; x < col; ++x) sum += exp(input_data[x + xx] - maxval);
	output_data[i] = exp(input_data[i] - maxval) / sum;
}

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol, const float *input_a, const float *input_b, float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float maxval = *input_a;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_a[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_a[x] - maxval);
  }
  // Compute per-row loss.
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}

int DLGpuArraySet(DLArrayHandle arr, float value) {
	float *arr_data = (float*)arr->data;

	int size = 1;
	for (int i = 0; i < arr->ndim; ++i) size = size * arr->shape[i];

	dim3 blocks, treads;
	if (size <= 1024) { blocks.x = 1; treads.x = size; }
	else { blocks.x = (size + 1023) / 1024; treads.x = 1024; }

	array_set_kernel<<<blocks, treads>>>(arr_data, value, size);
	return 0;
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
	const float *input_data = (const float*)input->data;
	float *output_data = (float*)output->data;

	int size_input = 1;
	for (int i = 0; i < input->ndim; ++i) size_input = size_input * input->shape[i];
	int size_output = 1;
	for (int i = 0; i < output->ndim; ++i) size_output = size_output * output->shape[i];
	assert(size_output % size_input == 0);

	dim3 blocks, treads;
	if (size_output <= 1024) { blocks.x = 1; treads.x = size_output; }
	else { blocks.x = (size_output + 1023) / 1024; treads.x = 1024; }

	broadcast_to_kernel<<<blocks, treads>>>(input_data, output_data, size_input, size_output);
	return 0;
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
	const float *input_data = (const float*)input->data;
	float *output_data = (float*)output->data;

	int size_input = 1;
	for (int i = 0; i < input->ndim; ++i) size_input = size_input * input->shape[i];
	int size_output = 1;
	for (int i = 0; i < output->ndim; ++i) size_output = size_output * output->shape[i];
	assert(size_input % size_output == 0);
	int num = size_input / size_output;

	dim3 blocks, treads;
	if (size_output <= 1024) { blocks.x = 1; treads.x = size_output; }
	else { blocks.x = (size_output + 1023) / 1024; treads.x = 1024; }

	reduce_sum_axis_zero_kernel<<<blocks, treads>>>(input_data, output_data, num, size_output);
	return 0;
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA, const DLArrayHandle matB, DLArrayHandle output) {
	const float *A_data = (const float*)matA->data;
	const float *B_data = (const float*)matB->data;
	float *output_data = (float*)output->data;

	int size_A = 1;
	for (int i = 0; i < matA->ndim; ++i) size_A = size_A * matA->shape[i];
	int size_B = 1;
	for (int i = 0; i < matB->ndim; ++i) size_B = size_B * matB->shape[i];
	int size_output = 1;
	for (int i = 0; i < output->ndim; ++i) size_output = size_output * output->shape[i];

	dim3 blocks, treads;
	if (size_output <= 1024) { blocks.x = 1; treads.x = size_output; }
	else { blocks.x = (size_output + 1023) / 1024; treads.x = 1024; }

	matrix_elementwise_add_kernel<<<blocks, treads>>>(A_data, B_data, output_data, size_A, size_B, size_output);
	return 0;
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val, DLArrayHandle output) {
  	const float *input_data = (const float*)input->data;
	float *output_data = (float*)output->data;

	int size_input = 1;
	for (int i = 0; i < input->ndim; ++i) size_input = size_input * input->shape[i];
	int size_output = 1;
	for (int i = 0; i < output->ndim; ++i) size_output = size_output * output->shape[i];
	assert(size_input == size_output);

	dim3 blocks, treads;
	if (size_output <= 1024) { blocks.x = 1; treads.x = size_output; }
	else { blocks.x = (size_output + 1023) / 1024; treads.x = 1024; }

	matrix_elementwise_add_by_const_kernel<<<blocks, treads>>>(input_data, val, output_data, size_output);
	return 0;
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA, const DLArrayHandle matB, DLArrayHandle output) {
	const float *A_data = (const float*)matA->data;
	const float *B_data = (const float*)matB->data;
	float *output_data = (float*)output->data;

	int size_A = 1;
	for (int i = 0; i < matA->ndim; ++i) size_A = size_A * matA->shape[i];
	int size_B = 1;
	for (int i = 0; i < matB->ndim; ++i) size_B = size_B * matB->shape[i];
	int size_output = 1;
	for (int i = 0; i < output->ndim; ++i) size_output = size_output * output->shape[i];

	dim3 blocks, treads;
	if (size_output <= 1024) { blocks.x = 1; treads.x = size_output; }
	else { blocks.x = (size_output + 1023) / 1024; treads.x = 1024; }

	matrix_elementwise_multiply_kernel<<<blocks, treads>>>(A_data, B_data, output_data, size_A, size_B, size_output);
	return 0;
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val, DLArrayHandle output) {
	const float *input_data = (const float*)input->data;
	float *output_data = (float*)output->data;

	int size_input = 1;
	for (int i = 0; i < input->ndim; ++i) size_input = size_input * input->shape[i];
	int size_output = 1;
	for (int i = 0; i < output->ndim; ++i) size_output = size_output * output->shape[i];
	assert(size_input == size_output);

	dim3 blocks, treads;
	if (size_output <= 1024) { blocks.x = 1; treads.x = size_output; }
	else { blocks.x = (size_output + 1023) / 1024; treads.x = 1024; }

	matrix_elementwise_multiply_by_const_kernel<<<blocks, treads>>>(input_data, val, output_data, size_output);
	return 0;
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA, const DLArrayHandle matB, bool transposeB, DLArrayHandle matC) {
 	const float *A_data = (const float*)matA->data;
	const float *B_data = (const float*)matB->data;
	float *C_data = (float*)matC->data;

	int row_A = matA->shape[0], col_A = matA->shape[1]; 
	int row_B = matB->shape[0], col_B = matB->shape[1]; 
	int size_C = matC->shape[0] * matC->shape[1];

	dim3 blocks, treads;
	if (size_C <= 1024) { blocks.x = 1; treads.x = size_C; }
	else { blocks.x = (size_C  + 1023) / 1024; treads.x = 1024; }

	matrix_multiply_kernel<<<blocks, treads>>>(A_data, row_A, col_A, transposeA, B_data, row_B, col_B, transposeB, C_data, size_C, matC->shape[0]);
  	return 0;
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
  	const float *input_data = (const float*)input->data;
  	float *output_data = (float*)output->data;

  	int size_input = 1;
  	for (int i = 0; i < input->ndim; ++i) size_input *= input->shape[i];
  	int size_output = 1;
  	for (int i = 0; i < output->ndim; ++i) size_output *= output->shape[i];

  	dim3 blocks, treads;
	if (size_output <= 1024) { blocks.x = 1; treads.x = size_output; }
	else { blocks.x = (size_output + 1023) / 1024; treads.x = 1024; }

	relu_kernel<<<blocks, treads>>>(input_data, output_data, size_input, size_output);
	return 0;
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad, DLArrayHandle output) {
   	const float *input_data = (const float*)input->data;
  	float *output_data = (float*)output->data;

  	int size_input = 1;
  	for (int i = 0; i < input->ndim; ++i) size_input *= input->shape[i];
  	int size_output = 1;
  	for (int i = 0; i < output->ndim; ++i) size_output *= output->shape[i];

  	dim3 blocks, treads;
	if (size_output <= 1024) { blocks.x = 1; treads.x = size_output; }
	else { blocks.x = (size_output + 1023) / 1024; treads.x = 1024; }

	relu_gradient_kernel<<<blocks, treads>>>(input_data, output_data, size_intput, size_output);
	return 0;
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
    const float *input_data = (const float*)input->data;
  	float *output_data = (float*)output->data;

  	int size_input = 1;
  	for (int i = 0; i < input->ndim; ++i) size_input *= input->shape[i];
  	int size_output = 1;
  	for (int i = 0; i < output->ndim; ++i) size_output *= output->shape[i];

  	dim3 blocks, treads;
	if (size_output <= 1024) { blocks.x = 1; treads.x = size_output; }
	else { blocks.x = (size_output + 1023) / 1024; treads.x = 1024; }

	int x=input->shape[0], y=input->shape[1];

	softmax_kernel<<<blocks, treads>>>(x, y ,input_data, output_data, size_intput, size_output);
	return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a, const DLArrayHandle input_b, DLArrayHandle output) {
  assert(input_a->ndim == 2);
  assert(input_b->ndim == 2);
  assert(output->ndim == 1);
  assert(input_a->shape[0] == input_b->shape[0] &&
         input_a->shape[1] == input_b->shape[1]);
  int nrow = input_a->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input_a->shape[1];
  const float *input_data_a = (const float *)input_a->data;
  const float *input_data_b = (const float *)input_b->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
      nrow, ncol, input_data_a, input_data_b, output_data);
  return 0;
}
