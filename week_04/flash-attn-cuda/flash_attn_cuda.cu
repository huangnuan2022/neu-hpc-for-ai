
/*
  FlashAttention HW4
  Build: nvcc -O3 -std=c++17 flash_attn_cuda.cu -o flash_attn_cuda
  Run:   ./flash_attn_cuda N d BR BC seed
*/
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#define CHECK_CUDA(x) do{cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static inline int idx2(int r,int c,int ld){ return r*ld + c; }

/* ------------------baseline attention model ------------------ */
static void naive_attention_cpu(const float*Q,const float*K,const float*V,float*O,int N,int d){
  const float scale = 1.f/std::sqrt((float)d);
  std::vector<float>S((size_t)N*N);
  for(int i=0;i<N;++i) for(int j=0;j<N;++j){
    double acc=0.0; for(int k=0;k<d;++k) acc += (double)Q[i*d+k]*K[j*d+k];
    S[i*(size_t)N+j] = (float)(acc*scale);
  }
  for(int i=0;i<N;++i){
    float m=-INFINITY; for(int j=0;j<N;++j) m=std::max(m,S[i*(size_t)N+j]);
    double denom=0.0; for(int j=0;j<N;++j) denom+=std::exp((double)S[i*(size_t)N+j]-m);
    for(int u=0;u<d;++u){
      double acc=0.0;
      for(int j=0;j<N;++j){
        double p = std::exp((double)S[i*(size_t)N+j]-m) / denom;
        acc += p * (double)V[j*d+u];
      }
      O[i*d+u]=(float)acc;
    }
  }
}

/* ----------------- FlashAttention setting ----------------- */

#ifndef DMAX
#define DMAX 128   // max head dim per-thread registers
#endif
#ifndef BR
#define BR 64      // row tile size or threads per block
#endif
#ifndef BC
#define BC 64      // KV tile size
#endif

// Kernel: one block handles BxR rows. Loop over KxV tiles.
__global__ void flash_attn_kernel(const float* __restrict__ Q,
                                  const float* __restrict__ K,
                                  const float* __restrict__ V,
                                  float* __restrict__ O,
                                  float* __restrict__ m_running,
                                  float* __restrict__ l_running,
                                  int N, int d, float scale)
{
  const int t = threadIdx.x;           // 0..BR-1
  const int i0 = blockIdx.y * BR;      // top row of this block
  const int i  = i0 + t;               // global row

  if (t >= BR) return;

  __shared__ float sK[BC][DMAX];
  __shared__ float sV[BC][DMAX];

  // Load query row to registers
  float q[DMAX];
  if (i < N) {
    #pragma unroll
    for (int u=0; u<DMAX; ++u) q[u]=0.f;
    for (int u=0; u<d; ++u) q[u] = Q[i*d + u];
  }

  // Running row state
  float m_i = (i<N) ? m_running[i] : -INFINITY;
  float l_i = (i<N) ? l_running[i] : 0.f;

  for (int j0=0; j0<N; j0+=BC) {
    int bc = min(BC, N - j0);

    // Load K/V tile into shared (coalesced 1-D striding by threads)
    for (int s = t; s < bc*d; s += BR) {
      int r = s / d; int c = s % d;
      sK[r][c] = K[(j0 + r)*d + c];
      sV[r][c] = V[(j0 + r)*d + c];
    }
    __syncthreads();

    // Tile-local accumulators
    float m_t = -INFINITY;
    float l_t = 0.f;
    float T_row[DMAX];
    #pragma unroll
    for (int u=0; u<DMAX; ++u) T_row[u]=0.f;

    if (i < N) {
      // Sweep keys in this tile
      for (int j=0; j<bc; ++j) {
        double acc=0.0;
        #pragma unroll
        for (int u=0; u<DMAX; ++u) {
          if (u<d) acc += (double)q[u] * (double)sK[j][u];
        }
        float s = (float)(acc * (double)scale);

        // Online update within the tile
        float m_new = (m_t > s) ? m_t : s;
        float a = (m_t==-INFINITY)?0.f:expf(m_t - m_new);
        float b = expf(s - m_new);
        l_t = l_t * a + b;
        #pragma unroll
        for (int u=0; u<DMAX; ++u) {
          if (u<d) T_row[u] = T_row[u]*a + b * sV[j][u];
        }
        m_t = m_new;
      }

      // Merge with running state
      float m_new = (m_i > m_t) ? m_i : m_t;
      float alpha = (m_i==-INFINITY)?0.f:expf(m_i - m_new);
      float beta  = (m_t==-INFINITY)?0.f:expf(m_t - m_new);
      float l_new = l_i*alpha + l_t*beta;

      if (l_new > 0.f) {
        float scale_O = (l_i*alpha)/l_new;
        float scale_T = beta/l_new;
        for (int u=0; u<d; ++u) {
          float old = O[i*d + u];
          O[i*d + u] = old*scale_O + T_row[u]*scale_T;
        }
      } else {
        float inv = (l_t>0.f)? (1.f/l_t) : 0.f;
        for (int u=0; u<d; ++u) O[i*d + u] = T_row[u]*inv;
      }
      m_i = m_new; l_i = l_new; m_running[i]=m_i; l_running[i]=l_i;
    }
    __syncthreads();
  }
}

/* ------------------------ Testing ------------------------ */
int main(int argc,char**argv){
  int N   = (argc>1)? std::atoi(argv[1]) : 512;
  int d   = (argc>2)? std::atoi(argv[2]) : 64;
  int BRu = (argc>3)? std::atoi(argv[3]) : BR;
  int BCu = (argc>4)? std::atoi(argv[4]) : BC;
  int seed= (argc>5)? std::atoi(argv[5]) : 42;
  if (d>DMAX || BRu>BR || BCu>BC) {
    fprintf(stderr,"Recompile with larger DMAX/BR/BC (DMAX=%d BR=%d BC=%d)\n",DMAX,BR,BC);
    return 2;
  }
  printf("CUDA FlashAttention: N=%d d=%d BR=%d BC=%d seed=%d\n",N,d,BRu,BCu,seed);
  const float scale = 1.f/std::sqrt((float)d);

  // Host RNG
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.f,1.f);
  std::vector<float> hQ(N*(size_t)d), hK(N*(size_t)d), hV(N*(size_t)d);
  for(auto& x: hQ) x=dist(rng);
  for(auto& x: hK) x=dist(rng);
  for(auto& x: hV) x=dist(rng);

  // Reference
  std::vector<float> hO_ref(N*(size_t)d);
  naive_attention_cpu(hQ.data(),hK.data(),hV.data(),hO_ref.data(),N,d);

  // Device buffers
  float *Q,*K,*V,*O,*m_run,*l_run;
  CHECK_CUDA(cudaMalloc(&Q,sizeof(float)*hQ.size()));
  CHECK_CUDA(cudaMalloc(&K,sizeof(float)*hK.size()));
  CHECK_CUDA(cudaMalloc(&V,sizeof(float)*hV.size()));
  CHECK_CUDA(cudaMalloc(&O,sizeof(float)*N*(size_t)d));
  CHECK_CUDA(cudaMalloc(&m_run,sizeof(float)*N));
  CHECK_CUDA(cudaMalloc(&l_run,sizeof(float)*N));

  CHECK_CUDA(cudaMemcpy(Q,hQ.data(),sizeof(float)*hQ.size(),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(K,hK.data(),sizeof(float)*hK.size(),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(V,hV.data(),sizeof(float)*hV.size(),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(O,0,sizeof(float)*N*(size_t)d));
  std::vector<float> mi(N,-INFINITY), li(N,0.f);
  CHECK_CUDA(cudaMemcpy(m_run,mi.data(),sizeof(float)*N,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(l_run,li.data(),sizeof(float)*N,cudaMemcpyHostToDevice));

  // Launch
  dim3 block(BR);
  dim3 grid(1, (N + BR - 1)/BR);

  // Warmup
  flash_attn_kernel<<<grid,block>>>(Q,K,V,O,m_run,l_run,N,d,scale);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Time
  cudaEvent_t start,stop; CHECK_CUDA(cudaEventCreate(&start)); CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for(int it=0; it<10; ++it)
    flash_attn_kernel<<<grid,block>>>(Q,K,V,O,m_run,l_run,N,d,scale);
  CHECK_CUDA(cudaEventRecord(stop)); CHECK_CUDA(cudaEventSynchronize(stop));
  float ms=0.f; CHECK_CUDA(cudaEventElapsedTime(&ms,start,stop)); ms/=10.f;

  // Copy result
  std::vector<float> hO(N*(size_t)d);
  CHECK_CUDA(cudaMemcpy(hO.data(),O,sizeof(float)*hO.size(),cudaMemcpyDeviceToHost));

  // Compare
  double rmse=0.0; float mad=0.f;
  for(size_t i=0;i<hO.size();++i){
    double diff = (double)hO[i]-hO_ref[i];
    rmse += diff*diff;
    mad = std::max(mad,(float)std::fabs(diff));
  }
  rmse = std::sqrt(rmse / hO.size());
  printf("Avg time: %.3f ms   MaxAbsDiff=%.3e   RMSE=%.3e\n", ms, mad, rmse);

  CHECK_CUDA(cudaFree(Q)); CHECK_CUDA(cudaFree(K)); CHECK_CUDA(cudaFree(V));
  CHECK_CUDA(cudaFree(O)); CHECK_CUDA(cudaFree(m_run)); CHECK_CUDA(cudaFree(l_run));
  CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));
  return 0;
}
