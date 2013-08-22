//kernelPBO.cu (Rob Farber)
 

#include "fluids.h"


using namespace glm;

class Ray {
public:
  vec3 p, d;
  __device__ Ray(const vec3 & P, const vec3 & D);
  __device__ vec3 solve(const float &t) const;
};

class Camera{
public:
  vec3 center, lookat, up, right, cof;
  float c, d, e, f;

  Camera();
  Camera(vec3 eye, vec3 lookat, vec3 up, int width, int height);
  __device__ Ray compute_ray(float pixel_x, float pixel_y);
  void transform(glm::mat4 t);

};

struct interval
{
	float tmin, tmax;
};

class AABB{
public:
	vec3 bounds[2], center;
	__device__ struct interval intersect(const Ray ray, float t0, float t1) const;

	AABB(){
		bounds[0] = vec3(0);
		bounds[1] = vec3(0);
		center = vec3(0);
	}

	AABB(vec3 min, vec3 max) {
		bounds[0] = min;
		bounds[1] = max;
		center = bounds[0] + (bounds[1] - bounds[0]) / 2.f;
	}
	
};

static void HandleError( cudaError_t err,
 const char *file,
    int line ) {
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
        file, line );
    exit( EXIT_FAILURE );
  }
}
void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__host__ __device__ void printVec3(vec3 v){
	printf("<%f, %f, %f>\n", v.x, v.y, v.z);
}


struct bufferPointers{
	float *u, *v, *w, *u0, *v0, *w0, *dens, *dens0, *sources, *u_s, *v_s, *w_s;
};

struct cudaDims{
	dim3 dimBlockFluid, dimGridFluid; 
	dim3 dimBlockBounds, dimGridBounds;
	dim3 dimBlockDraw, dimGridDraw;

};


struct bufferPointers device_pointers;
struct cudaDims dims;
bool draw_dens_flag = true;
Camera cam;
AABB fluidBounds;

__device__ vec4 marchRay(Ray r, struct bufferPointers p, bool draw_dens_flag, AABB fluidBounds){
	
	
	vec4 ray_color(0);
	float multiplier = 1;
	float opacity = 0.1 * RAY_STEP;
	
	struct interval interval = fluidBounds.intersect(r, -10000, 10000);
	float t = interval.tmin;
	vec3 pos = r.solve(t);

	int pixel_x= blockIdx.x * blockDim.x + threadIdx.x;
	int pixel_y= blockIdx.y * blockDim.y + threadIdx.y;

	bool flag =  false && pixel_x == 512 && pixel_y == 512;

	

	if(flag) printVec3(r.p);
	if(flag) printVec3(r.d);
	
	while(t < interval.tmax && multiplier > 0){
		int i = (int) pos.x;
		int j = (int) pos.y;
		int k = (int) pos.z;
		



		
		if(i > 0 && i <= N && j > 0 && j <= N && k > 0 && k <=N){
			int index = IX(i, j, k);
			if(flag) printVec3(pos);
			if(flag) printf("%d %d %d\n", i, j, k);

			if(flag) printVec3(vec3(p.dens[index]));


			
			if(draw_dens_flag){
		 		ray_color += multiplier * opacity * vec4(p.dens[index]);
		 		multiplier *= (1 - opacity * p.dens[index]);

		 	}else{
		 		float weight =1;
		 		ray_color.x += opacity * abs(weight * p.u[index]);
		 		ray_color.y += opacity * abs(weight * p.v[index]);
		 		ray_color.z += opacity * abs(weight * p.w[index]);
		 	}
		}
		
		t += RAY_STEP;
		pos = r.solve(t);
	}

	return ray_color;
}

__global__ void draw_dens_kernel(struct bufferPointers p, bool draw_dens_flag, AABB fluidBounds, float dt, Camera cam, uchar4* pixels){
	
	//printf("kernel\n");
	int pixel_x= blockIdx.x * blockDim.x + threadIdx.x;

	int pixel_y= blockIdx.y * blockDim.y + threadIdx.y;

	int index = pixel_x+pixel_y*blockDim.x*gridDim.x;


	vec4 color = marchRay(cam.compute_ray(pixel_x, pixel_y), p, draw_dens_flag, fluidBounds);
	
 	// vec4 color(0);
 	// if(draw_dens_flag){
 	// 	color = vec4(p.dens[index]);
 	// }else{
 	// 	color.x = abs(1000 * p.u[index]);
 	// 	color.y = abs(1000 * p.v[index]);
 	// }
 	

	// Each thread writes one pixel location in the texture (textel) 	
	pixels[index].x = (unsigned char)(min(1.0, color.x) * 255.9999f);
	pixels[index].y = (unsigned char)(min(1.0, color.y) * 255.9999f);
	pixels[index].z = (unsigned char)(min(1.0, color.z) * 255.9999f);
	pixels[index].w = (unsigned char)(min(1.0, color.w) * 255.9999f);

	
}

__global__ void set_bnd_kernel ( int b, float * x )
{
	int pixel_x= blockIdx.x * blockDim.x + threadIdx.x;

	int pixel_y= blockIdx.y * blockDim.y + threadIdx.y;

	int i = pixel_x;
	int j = pixel_y;//IX(pixel_x, pixel_y);

	//if(i > 0 && i <= N && j > 0 && j <= N){
		switch(b){
			case 0:
				// x[IX(1 ,i)] +=  x[IX(0,i)];
				// x[IX(N,i)] += x[IX(N+1,i)];
				// x[IX(i,1 )] +=  x[IX(i,0)];
				// x[IX(i,N)] += x[IX(i,N+1)];

				// x[IX(1 ,i, j)] +=  x[IX(0 ,i, j)];
				// x[IX(N,i, j)] += x[IX(N+1,i, j)];
				// x[IX(i,1, j)] +=  x[IX(i,0, j)];
				// x[IX(i,N, j)] += x[IX(i,N+1, j)];
				// x[IX(i, j, 1)] +=  x[IX(i, j, 0)];
				// x[IX(i, j, N)] += x[IX(i, j, N+1)];

				x[IX(0 ,i, j)] =  0;
				x[IX(N+1,i, j)] = 0;
				x[IX(i,0, j)] =  0;
				x[IX(i,N+1, j)] = 0;
				x[IX(i, j, 0)] =  0;
				x[IX(i, j, N+1)] = 0;


				break;
			case 1:
				if(x[IX(0 ,i, j)] < 0) x[IX(0 ,i, j)] =  -x[IX(0,i, j)];
				if(x[IX(N+1,i, j)] > 0) x[IX(N+1,i, j)] = -x[IX(N+1,i, j)];
				break;
			 case 2:

				if(x[IX(i,0 , j)] < 0) x[IX(i,0, j )] = -x[IX(i,0, j)];
				if(x[IX(i,N+1, j)] >0) x[IX(i,N+1, j)] = -x[IX(i,N+1, j)];
				break;

			case 3:
				if(x[IX(i, j, 0)] < 0) x[IX(i, j, 0)] = -x[IX(i,j, 0)];
				if(x[IX(i, j, N+1)] >0) x[IX(i, j, N+1)] = -x[IX(i, j, N+1)];

				break;
			default:
				break;
				// x[IX(0 ,i)] = b == 1 ? -x[IX(1,i)] : x[IX(1,i)];
				// x[IX(N+1,i)] = b == 1 ? -x[IX(N,i)] : x[IX(N,i)];
				// x[IX(i,0 )] = b==2 ? -x[IX(i,1)] : x[IX(i,1)];
				// x[IX(i,N+1)] = b==2 ? -x[IX(i,N)] : x[IX(i,N)];
	
		}
	//}

	// if(i < 32){
	// 	x[IX(0 ,0 )] = 0.5*(x[IX(1,0 )]+x[IX(0 ,1)]);
	// 	x[IX(0 ,N+1)] = 0.5*(x[IX(1,N+1)]+x[IX(0 ,N )]);
	// 	x[IX(N+1,0 )] = 0.5*(x[IX(N,0 )]+x[IX(N+1,1)]);
	// 	x[IX(N+1,N+1)] = 0.5*(x[IX(N,N+1)]+x[IX(N+1,N )]);
	// }
	

}

__host__ void set_bnd(int b, float * x){
	set_bnd_kernel<<<dims.dimGridBounds, dims.dimBlockBounds>>>(b, x);
	checkCUDAError("kernel failed!");
	cudaThreadSynchronize();
}


__global__ void add_source_kernel(float * x, float * s, float dt )
{
	int i= blockIdx.x * blockDim.x + threadIdx.x;

	int j= blockIdx.y * blockDim.y + threadIdx.y;

	int k = blockIdx.z * blockDim.z + threadIdx.z;

	int index = IX(i, j, k);

	x[index] += dt * s[index];
}

__host__ void add_source(struct cudaDims dims, float * x, float * s, float dt ){
	add_source_kernel<<<dims.dimGridFluid, dims.dimBlockFluid>>>(x, s, dt );
	checkCUDAError("kernel failed!"); 
	cudaThreadSynchronize();
}




__global__ void diffuse_kernel(float * x, float * x0, float diff, float dt){
	int i, j, k;
	i = blockIdx.x * blockDim.x + threadIdx.x;

	j = blockIdx.y * blockDim.y + threadIdx.y;

	k = blockIdx.z * blockDim.z + threadIdx.z;

	float a = dt * diff * N * N * N ;
	
	if(i > 0 && i <= N && j > 0 && j <= N && k > 0 && k <=N){
		x[IX(i, j, k)] = (x0[IX(i , j, k)] + a * (
			x[IX(i - 1, j, k)] + 
			x[IX(i + 1, j, k)] +
			x[IX(i, j - 1, k)] + 
			x[IX(i, j + 1, k)] +
			x[IX(i, j, k - 1)] + 
			x[IX(i, j, k + 1)]
			)) / (1 + 6 * a);
	}


	
}

__host__  void diffuse(struct cudaDims dims, int b, float *x, float*x0, float diff, float dt){
	for(int k = 0; k < K; k++){
		diffuse_kernel<<<dims.dimGridFluid, dims.dimBlockFluid>>>(x, x0, diff, dt);
		checkCUDAError("kernel failed!");
		cudaThreadSynchronize();
		set_bnd(b, x);
	}
}




//set bounds after call
__global__ void advect_kernel(float * d, float * d0, float * u, float * v,float * w,  float dt )
{
	int i, j, k, ia[2], ja[2], ka[2];
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	k = blockIdx.z * blockDim.z + threadIdx.z;

	float x, y, z, sa[2], ta[2], ua[2], dt0;
	dt0 = dt * N;
	if(i > 0 && i <= N && j > 0 && j <= N && k > 0 && k <=N){
		x = i - dt0 * u[IX(i, j, k)];
		y = j - dt0 * v[IX(i, j, k)];
		z = k - dt0 * w[IX(i, j, k)];

		if (x < 0.5) x = 0.5;
		if (x > N + 0.5) x = N + 0.5;

		ia[0] = (int) x;
		ia[1] = ia[0] + 1;

		if (y < 0.5) y = 0.5;
		if (y > N + 0.5) y = N + 0.5; 

		ja[0] = (int) y;
		ja[1] = ja[0] + 1;

		if (z < 0.5) z = 0.5;
		if (z > N + 0.5) z = N + 0.5; 

		ka[0] = (int) z;
		ka[1] = ka[0] + 1;

		sa[1] = x - ia[0];
		sa[0] = 1 - sa[1];

		ta[1] = y - ja[0];
		ta[0] = 1 - ta[1];

		ua[1] = z - ka[0];
		ua[0] = 1 - ua[1];

		// d[IX(i,j, k)] = ua[0] * (sa[0] * (ta[0] * d0[IX(ia[0],ja[0], ka[0])] + ta[1] * d0[IX(ia[0], ja[1], ka[0])]) + 
		// 				   sa[1] * (ta[0] * d0[IX(ia[1], ja[0], ka[0])] + ta[1] * d0[IX(ia[1], ja[1], ka[0])])) +
		// 			 ua[1] * (sa[0] * (ta[0] * d0[IX(ia[0],ja[0], ka[1])] + ta[1] * d0[IX(ia[0], ja[1], ka[1])]) + 
		// 				   sa[1] * (ta[0] * d0[IX(ia[1], ja[0], ka[1])] + ta[1] * d0[IX(ia[1], ja[1], ka[1])]));

		d[IX(i,j, k)] = 0;
		for(int a = 0; a < 2; a++){
			for (int b = 0; b < 2; b++){
				for (int c = 0; c < 2; c++){
					d[IX(i,j, k)] += sa[a] * ta[b] * ua[c] * d0[IX(ia[a], ja[b], ka[c])];
				}
			}
		}
	
	}
	//set_bnd ( N, b, d );
}

__host__ void advect(struct cudaDims dims, int b, float * d, float * d0, float * u, float * v, float * w, float dt ){
	advect_kernel<<<dims.dimGridFluid, dims.dimBlockFluid>>>(d, d0, u, v, w, dt);
	checkCUDAError("kernel failed!");
	cudaThreadSynchronize();
	set_bnd(b, d);
}

__global__ void project_kernel_1(float * u, float * v, float *w, float * momentum, float * divergence){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	float h = 1.0/N;
	if(i > 0 && i <= N && j > 0 && j <= N && k > 0 && k <=N){
		divergence[IX(i,j, k)] = -0.5*h*(u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)]+
										 v[IX(i, j + 1, k)] - v[IX(i, j - 1, k)]+
										 v[IX(i, j, k + 1)] - v[IX(i, j, k - 1)]);
		momentum[IX(i,j, k)] = 0;
	}

}

__global__ void project_kernel_2(float * u, float * v, float *w, float * momentum, float * divergence){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	float h = 1.0/N;
	if(i > 0 && i <= N && j > 0 && j <= N && k > 0 && k <=N){
		momentum[IX(i,j, k)] = (divergence[IX(i, j, k)] +
								momentum[IX(i - 1, j, k)] + momentum[IX(i + 1, j, k)]+
								momentum[IX(i, j - 1, k)] + momentum[IX(i, j + 1, k)]+
								momentum[IX(i, j, k-1)] + momentum[IX(i, j, k + 1)])/6;
	}

}

__global__ void project_kernel_3(float * u, float * v, float *w, float * momentum, float * divergence){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	float h = 1.0/N;
	if(i > 0 && i <= N && j > 0 && j <= N && k > 0 && k <=N){
		u[IX(i,j,k)] -= 0.5*(momentum[IX(i+1, j, k)]-momentum[IX(i-1, j, k)])/h;
		v[IX(i,j,k)] -= 0.5*(momentum[IX(i, j+1, k)]-momentum[IX(i, j-1, k)])/h;
		w[IX(i,j,k)] -= 0.5*(momentum[IX(i, j, k+1)]-momentum[IX(i, j, k-1)])/h;
	}

}

__host__ void project (cudaDims dims, float * u, float * v, float *w, float * momentum, float * divergence)
{
	int k;


	project_kernel_1<<<dims.dimGridFluid, dims.dimBlockFluid>>>(u, v, w, momentum, divergence);
	checkCUDAError("kernel failed!");
	cudaThreadSynchronize();

	set_bnd (0, momentum );
	set_bnd (0, divergence ); 
	for ( k=0 ; k<K ; k++ ) {
		project_kernel_2<<<dims.dimGridFluid, dims.dimBlockFluid>>>(u, v, w, momentum, divergence);
		checkCUDAError("kernel failed!");
		cudaThreadSynchronize();
		
		set_bnd (0, momentum ); 
	
	}

	project_kernel_3<<<dims.dimGridFluid, dims.dimBlockFluid>>>(u, v, w, momentum, divergence);
	checkCUDAError("kernel failed!");
	cudaThreadSynchronize();

	set_bnd (1, u ); 
	set_bnd (2, v );
	set_bnd (3, w );
}


void vel_step (struct cudaDims dims, float * u, float * v, float *w, float* u0, float * v0, float * w0, float visc, float dt )
{
	// add_source(dims, u, u0, dt ); 
	// add_source(dims, v, v0, dt ); 
	// add_source(dims, w, w0, dt );

	SWAP (u0, u ); 
	SWAP (v0, v );
	SWAP (w0, w );

	diffuse(dims, 1, u, u0, visc, dt);
	diffuse(dims, 2, v, v0, visc, dt);
	diffuse(dims, 3, w, w0, visc, dt);

	project (dims, u, v, w, u0, v0);

	SWAP ( u0, u ); 
	SWAP ( v0, v );
	SWAP ( w0, w );

	advect (dims, 1, u, u0, u0, v0, w0, dt ); 
	advect (dims, 2, v, v0, u0, v0, w0, dt );
	advect (dims, 3, w, w0, u0, v0, w0, dt );

	project (dims, u, v, w, u0, v0);
}



__host__ void dens_step(struct cudaDims dims, float * x, float * x0, float * u, float * v, float * w, float diff, float dt)
{
	//add_source(dims, x, x0, dt);

	SWAP(x0, x);

	diffuse(dims, 0, x, x0, diff, dt);

	SWAP(x0, x);

	advect(dims, 0, x, x0, u, v, w, dt);

}





// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_kernel(uchar4* pixels, unsigned int image_width, unsigned int image_height, float dt)
{
 



	struct bufferPointers p = device_pointers;

	add_source(dims, p.u, p.u_s, dt ); 
	add_source(dims, p.v, p.v_s, dt ); 
	add_source(dims, p.w, p.w_s, dt );
	add_source(dims, p.dens, p.sources, dt );

	vel_step (dims, p.u, p.v, p.w, p.u0, p.v0, p.w0, VISC, dt );


 	dens_step(dims, p.dens, p.dens0, p.u, p.v, p.w,  DIFF, dt);

 	draw_dens_kernel<<<dims.dimGridDraw, dims.dimBlockDraw>>>(device_pointers, draw_dens_flag, fluidBounds, dt, cam, pixels);
 	checkCUDAError("kernel failed!");
 
 	size_t memsize = SIZE * sizeof(float);

 	// HANDLE_ERROR(cudaMemset((device_pointers.u0), 0, memsize));
 	// HANDLE_ERROR(cudaMemset((device_pointers.v0), 0, memsize));
 	// HANDLE_ERROR(cudaMemset((device_pointers.w0), 0, memsize));
 	//HANDLE_ERROR(cudaMemset((device_pointers.dens0), 0, memsize));
 	// SWAP(device_pointers.v0, device_pointers.v);
 	// SWAP(device_pointers.u0, device_pointers.u);
 	// SWAP(device_pointers.w0, device_pointers.w);
 	//   SWAP(device_pointers.dens0, device_pointers.dens);
}



extern "C" void setup_scene(unsigned int image_width, unsigned int image_height){

	printf("setting up scene\n");

	fluidBounds = AABB(vec3(1), vec3(N+1));

	printf("creating camera\n");

	cam = Camera(vec3(N, -N,  -N), vec3(N/2, N/2, N/2), vec3(0, 1, 0), image_width, image_height);

	printf("computing dimensions\n");
	int block_width=32, block_height=32;

	dims.dimGridFluid = dim3((N + 2) / block_width, (N + 2) / block_height, N + 2);
	dims.dimBlockFluid = dim3(block_width, block_height, 1);

	int block_width_draw=32, block_height_draw=16;
	dims.dimGridDraw = dim3(image_width / block_width_draw, image_height / block_height_draw);
	dims.dimBlockDraw = dim3(block_width_draw, block_height_draw);

	dims.dimGridBounds = dim3((N + 2) / block_width, (N + 2) / block_height);
	dims.dimBlockBounds = dim3(block_width, block_height);

	printf("allocating buffers\n");

	size_t memsize = SIZE * sizeof(float);

	HANDLE_ERROR(cudaMalloc(&(device_pointers.u), memsize));
	HANDLE_ERROR(cudaMemset((device_pointers.u), 0, memsize));

	HANDLE_ERROR(cudaMalloc(&(device_pointers.u0), memsize));
	HANDLE_ERROR(cudaMemset((device_pointers.u0), 0, memsize));

	HANDLE_ERROR(cudaMalloc(&(device_pointers.v), memsize));
	HANDLE_ERROR(cudaMemset((device_pointers.v), 0, memsize));

	HANDLE_ERROR(cudaMalloc(&(device_pointers.v0), memsize));
	HANDLE_ERROR(cudaMemset((device_pointers.v0), 0, memsize));

	HANDLE_ERROR(cudaMalloc(&(device_pointers.w), memsize));
	HANDLE_ERROR(cudaMemset((device_pointers.w), 0, memsize));

	HANDLE_ERROR(cudaMalloc(&(device_pointers.w0), memsize));
	HANDLE_ERROR(cudaMemset((device_pointers.w0), 0, memsize));

	HANDLE_ERROR(cudaMalloc(&(device_pointers.dens), memsize));
	HANDLE_ERROR(cudaMemset((device_pointers.dens), 0, memsize));

	HANDLE_ERROR(cudaMalloc(&(device_pointers.dens0), memsize));
	HANDLE_ERROR(cudaMemset((device_pointers.dens0), 0, memsize));

	HANDLE_ERROR(cudaMalloc(&(device_pointers.u_s), memsize));
	HANDLE_ERROR(cudaMemset((device_pointers.u_s), 0, memsize));

	HANDLE_ERROR(cudaMalloc(&(device_pointers.v_s), memsize));
	HANDLE_ERROR(cudaMemset((device_pointers.v_s), 0, memsize));

	HANDLE_ERROR(cudaMalloc(&(device_pointers.w_s), memsize));
	HANDLE_ERROR(cudaMemset((device_pointers.w_s), 0, memsize));


	HANDLE_ERROR(cudaMalloc(&(device_pointers.sources), memsize));
	HANDLE_ERROR(cudaMemset((device_pointers.sources), 0, memsize));

	// //float source = 1.0;
	// //HANDLE_ERROR(cudaMemcpy(&(device_pointers.dens0[IX(N/2, N/2)]), &source, sizeof(float), cudaMemcpyHostToDevice));

	// //float v[SIZE];
	// // for(int i=0; i<SIZE; i++){
	// // 	v[i] = 1000.0; 
	// // }
	// //
	


	// //float source_array[SIZE];

	printf("instantiating arrays\n");

	float velmag =  0.01;
	float v[SIZE], u[SIZE], w[SIZE], dens0[SIZE], sources[SIZE];
	int offset = 16;
	int min_index = N/2 - offset;
	int max_index = N/2 + offset;
	for(int i = 1; i <= N; i++){
		for(int j = 1; j <= N ; j++){
			for(int k = 1; k <= N ; k++){
				int index = IX(i, j, k);

				u[index] = 0;
				v[index] = 0;
				w[index] = 0;

				if(i > min_index && i < max_index && j > N+2 - offset && k > min_index && k < max_index){
					sources[index] =0.1;
				}else{
					sources[index] =0;
				}
				if(((i/offset)%2 ==0 ^ (k/offset)%2 ==0) && j > N+2 - offset){

				//if(((i/offset)%8 ==0 && (k/offset)%8 ==0) && j == 1){
					dens0[index] = 0.0;
					//w[index] = velmag;
				}else{
					dens0[index] = 0.0;
				}
				// u[index] = velmag;
				// v[index] = velmag;
				// w[index] = velmag;
				

				float R = 4.0;
				float x = i - N/2.0;
				float y = j - N/2.0;
				float z = k - N/2.0;
				float r = length(vec2(x, z));

				if( r > 1){
					//dens0[index] = 1.0;
					// u[index] = velmag * z/r;
					// v[index] = velmag * -100/r;
					// w[index] = velmag * -x/r;
					vec3 vel = vec3(z, -10, -x);
					vel /= r;
					//vel *=  1/(abs(r-R)+1);
					
					//vec3 vel =   (r - R)/abs(r - R) * 1/(abs(r-R)+1) *  velmag * vel;
					u[index] =  vel.x;
					v[index] = 	vel.y;
					w[index] = 	vel.z;
				
				}
				

				// vec3 vel = vec3(z, 1, -x);
				// if(length(vel) > 0.1){
				// 	vec3 vel =   /*(r - R)/abs(r - R) * 1/(abs(r-R)+1)*/  velmag * vel;
				// 	u[index] =  vel.x;
				// 	v[index] = 	vel.y;
				// 	w[index] = 	vel.z;
				// }else{
				// 	u[index] =  0;
				// 	v[index] = 	0;
				// 	w[index] = 	0;
				// }
				

				//printf("%f, %f\n", vel.x, vel.y);
	
				//dens0[index] = 0.1 /(abs(r-R)+1);

				if(abs(r-R) < 0.01){

					u[index] =  0;
					v[index] = 	0;
					w[index] = 	0;
				}
			}
		}
	}
	int index = IX(N/2, N/2, N/2);
	u[index] =  0;
	v[index] = 	0;
	w[index] = 	0;


	HANDLE_ERROR(cudaMemcpy(device_pointers.dens, dens0, memsize, cudaMemcpyHostToDevice));
	// HANDLE_ERROR(cudaMemcpy(device_pointers.u, u, memsize, cudaMemcpyHostToDevice));
	// HANDLE_ERROR(cudaMemcpy(device_pointers.v, v, memsize, cudaMemcpyHostToDevice));
	// HANDLE_ERROR(cudaMemcpy(device_pointers.w, w, memsize, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(device_pointers.sources, sources, memsize, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(device_pointers.u_s, u, memsize, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(device_pointers.v_s, v, memsize, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(device_pointers.w_s, w, memsize, cudaMemcpyHostToDevice));




}

extern "C" void destroy_scene(){
	printf("destroying scene\n");
	cudaFree(device_pointers.u);
	cudaFree(device_pointers.v);
	cudaFree(device_pointers.u0);
	cudaFree(device_pointers.v0);
	cudaFree(device_pointers.dens);
	cudaFree(device_pointers.dens0);
	cudaFree(device_pointers.sources);
}






//! Keyboard events handler for GLUT
void keyboard(unsigned char key, int x, int y)
{
	switch(key) {
	case(27) :case('q') :
		exit(0);
		break;
	case('w'):
		
		break;
	case('a'):
	case('s'):
	case('d'):
      
		break;

	case('v'):
		draw_dens_flag = !draw_dens_flag;
		break;
	
   }
   

   // indicate the display must be redrawn
   glutPostRedisplay();
}

Camera::Camera(vec3 eye, vec3 lookat_in, vec3 up_in, int width, int height){

	float near = 1;
	cof = eye;
	up = normalize(up_in);
	lookat = normalize(lookat_in - cof);
	right = normalize(cross(lookat, up))/2.f ;
	center = cof + (normalize(lookat) * near);
	up = normalize(cross(right, lookat))/2.f ;
	

	printVec3(center);
	printVec3(right);
	printVec3(up);
	printVec3(lookat);

	if (width > height) {
		d = (width - 1) / 2.0;
		c = (height * (width - 1.0))/ (2.0 * width);
		f = (height - 1.0) / 2.0;
		e = (height - 1.0) / 2.0;
	} else {
		d = (width - 1.0) / 2.0;
		c = (width - 1.0) / 2.0;
		f = (height - 1.0) / 2.0;
		e = (width  * (height - 1.0))/ (2.0 * height);
	}
	
}
Camera::Camera(){}



__device__ Ray Camera::compute_ray(float pixel_x, float pixel_y){
	
	float x = (pixel_x - d) / c, y = ((pixel_y - f) / e);
	vec3 pixel_point = center + (up * y) + (right * x);
	Ray result(cof, normalize((pixel_point - cof)));
	return result;
}


void Camera::transform(glm::mat4 t){
	center=vec3(t * vec4(center, 1));
	lookat=vec3(t*vec4(lookat, 0));
 	up = vec3(t * vec4(up, 0));
	right=vec3( t* vec4(right, 0));
	cof=vec3(t*vec4(cof, 1));;


}

__device__ Ray::Ray(const vec3 & P, const vec3 & D): p(P), d(D){}


__device__ vec3 Ray::solve(const float &t) const{
	
	return p+(d*t);
}

__device__ struct interval AABB::intersect(const Ray r, float t0, float t1) const {
	float tmin, tmax, tymin, tymax, tzmin, tzmax;
	struct interval result = {-1, -1};
	if (r.d.x >= 0) {
		tmin = (bounds[0].x - r.p.x) / r.d.x;
		tmax = (bounds[1].x - r.p.x) / r.d.x;
	}
	else {
		tmin = (bounds[1].x - r.p.x) / r.d.x;
		tmax = (bounds[0].x - r.p.x) / r.d.x;
	}
	if (r.d.y >= 0) {
		tymin = (bounds[0].y - r.p.y) / r.d.y;
		tymax = (bounds[1].y - r.p.y) / r.d.y;
	}
	else {
		tymin = (bounds[1].y - r.p.y) / r.d.y;
		tymax = (bounds[0].y - r.p.y) / r.d.y;
	}
	if ( (tmin > tymax) || (tymin > tmax) )
		return result;
	if (tymin > tmin)
		tmin = tymin;
	if (tymax < tmax)
		tmax = tymax;
	if (r.d.z >= 0) {
		tzmin = (bounds[0].z - r.p.z) / r.d.z;
		tzmax = (bounds[1].z - r.p.z) / r.d.z;
	}
	else {
		tzmin = (bounds[1].z - r.p.z) / r.d.z;
		tzmax = (bounds[0].z - r.p.z) / r.d.z;
	}
	if ( (tmin > tzmax) || (tzmin > tmax) )
		return result;
	if (tzmin > tmin)
		tmin = tzmin;
	if (tzmax < tmax)
		tmax = tzmax;

	if((tmin < t1) && (tmax > t0)){
		result.tmin = tmin;
		result.tmax = tmax;
		return result;
	}else{
		return result;
	}
}