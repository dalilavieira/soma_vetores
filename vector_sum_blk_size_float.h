#include<stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>

using namespace std;

float elapsed_time;
cudaEvent_t start, stop;                             //# Declara dois eventos

void time_start(){
  cudaEventCreate(&start);                          //# Irá marcar o inicio da execucao
  cudaEventCreate(&stop);                           //# Irá  marcar o final da execucao
  cudaEventRecord(start, 0);                        //# insere na fila 
}

void time_end() {
  cudaEventRecord(stop, 0);                          //# insere na fila
  cudaEventSynchronize(stop);                        //# espera terminar
  cudaEventElapsedTime(&elapsed_time, start, stop);  //# calcula
}

void host_add(float *a, float *b, float *c) {
	for(int idx=0;idx<N;idx++)
		c[idx] = a[idx] + b[idx];
}

__global__ void device_add(float *a, float *b, float *c) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;
        c[index] = a[index] + b[index];
}


//basically just fills the array with index.
void fill_array(float *data) {
	for(int idx=0;idx<N;idx++)
		data[idx] = idx;
}


void verify_output(float *a, float *b, float*c) {
  int ok = 1;
	for(int idx=0;idx<N;idx++)
    if ( a[idx] + b[idx] != c[idx]) ok = 0;

	if ( ok) printf("Soma de Vetores está correta !\n");
}


int main(void) {
	float *a, *b, *c;
  float *d_a, *d_b, *d_c; // device copies of a, b, c
	int threads_per_block=0, no_of_blocks=0;

	int size = N * sizeof(float);
	
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	printf("device %d: %s \n", 0, deviceProp.name);
	cudaSetDevice(0);

	ofstream myfile;
	myfile.open ("results_matrix.csv");
	myfile << "size, kernell, all\n";

  ofstream gpufile;
	gpufile.open ("gpu.txt");
  gpufile << deviceProp.name;
	
	for (int i = 0; i < data.size(); ++i) {

		// Alloc space for host copies of a, b, c and setup input values
		a = (float *)malloc(size); fill_array(a);
		b = (float *)malloc(size); fill_array(b);
		c = (float *)malloc(size);

		// Alloc space for device copies of a, b, c
		cudaMalloc((void **)&d_a, size);
		cudaMalloc((void **)&d_b, size);
		cudaMalloc((void **)&d_c, size);

	       // Copy inputs to device
		cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

		//threads_per_block = 512;
		threads_per_block = data[i];
		myfile << threads_per_block << ",";
		
		time_start();
		no_of_blocks = N/threads_per_block;	
		device_add<<<no_of_blocks,threads_per_block>>>(d_a,d_b,d_c);
		time_end();

		printf("Time GPU naive: %7.2lf ms\n", elapsed_time);
		//myfile << (elapsed_time * threads_per_block)/(no_of_blocks*pow(10,9)) << ",";
    myfile << (N * pow(10,3))/elapsed_time << ",";

    float before = elapsed_time;

    time_start();
		// Copy result back to host
		cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
		time_end();
		//myfile << ((before+elapsed_time) * threads_per_block)/(no_of_blocks*pow(10,9)) << "\n";
    myfile << (N * pow(10,3))/(elapsed_time+before) << "\n";
		
		verify_output(a,b,c);

		free(a); free(b); free(c);
        	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

	}
  myfile.close();

	return 0;
}

