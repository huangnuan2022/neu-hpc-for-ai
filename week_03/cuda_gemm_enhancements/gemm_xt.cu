%%writefile gemm_xt.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <string>
#include <cmath>
#include <iostream>

#define CHECK_CUDA(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

enum Transpose : int { N=0, T=1 };

// row-major index
__host__ __device__ inline int idx(int r, int c, int ld) { return r*ld + c; }

// logical reads for op(A) and op(B) using stored dims
__host__ __device__ inline float read_opA(const float* A, int i, int k,
                                          int mA, int nA, Transpose tA) {
  // stored A is mA x nA (row-major)
  // if tA==N: opA(i,k) = A[i,k]
  // if tA==T: opA(i,k) = A^T[i,k] = A[k,i]
  return (tA==N) ? A[idx(i,k,nA)] : A[idx(k,i,nA)];
}

__host__ __device__ inline float read_opB(const float* B, int k, int j,
                                          int mB, int nB, Transpose tB) {
  // stored B is mB x nB
  // if tB==N: opB(k,j) = B[k,j]
  // if tB==T: opB(k,j) = B^T[k,j] = B[j,k]
  return (tB==N) ? B[idx(k,j,nB)] : B[idx(j,k,nB)];
}

// ---------------------- NAIVE KERNEL ----------------------
__global__ void gemm_naive_inplace(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C, // in-place
                                   int m, int n, int K,
                                   int mA, int nA, int mB, int nB,
                                   Transpose tA, Transpose tB,
                                   float alpha, float beta) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) return;

  float acc = 0.f;
  for (int k=0; k<K; ++k) {
    float a = read_opA(A, row, k, mA, nA, tA);
    float b = read_opB(B,  k, col, mB, nB, tB);
    acc = fmaf(a, b, acc);
  }
  C[idx(row,col,n)] = alpha*acc + beta*C[idx(row,col,n)];
}

// ---------------------- TILED KERNEL ----------------------
template<int BLOCK>
__global__ void gemm_tiled_inplace(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int m, int n, int K,
                                   int mA, int nA, int mB, int nB,
                                   Transpose tA, Transpose tB,
                                   float alpha, float beta) {
  __shared__ float As[BLOCK][BLOCK+1];
  __shared__ float Bs[BLOCK][BLOCK+1];

  int row = blockIdx.y * BLOCK + threadIdx.y;
  int col = blockIdx.x * BLOCK + threadIdx.x;

  float acc = 0.f;
  int tiles = (K + BLOCK - 1) / BLOCK;

  for (int t=0; t<tiles; ++t) {
    int k0 = t*BLOCK;

    // load tile of op(A): rows=row, cols=k0..k0+BLOCK-1
    int aCol = k0 + threadIdx.x;
    As[threadIdx.y][threadIdx.x] =
      (row<m && aCol<K) ? read_opA(A, row, aCol, mA, nA, tA) : 0.f;

    // load tile of op(B): rows=k0..k0+BLOCK-1, cols=col
    int bRow = k0 + threadIdx.y;
    Bs[threadIdx.y][threadIdx.x] =
      (bRow<K && col<n) ? read_opB(B, bRow, col, mB, nB, tB) : 0.f;

    __syncthreads();
    #pragma unroll
    for (int kk=0; kk<BLOCK; ++kk) {
      acc = fmaf(As[threadIdx.y][kk], Bs[kk][threadIdx.x], acc);
    }
    __syncthreads();
  }

  if (row<m && col<n) {
    float c_old = C[idx(row,col,n)];
    C[idx(row,col,n)] = alpha*acc + beta*c_old;
  }
}

// ---------------------- Host driver ----------------------
struct Args {
  int m=1024, n=1024, k=1024;
  Transpose tA=N, tB=N;
  float alpha=1.f, beta=1.f;
  int repeat=20;
  std::string kernel="tiled";
  int block=32;
  int seed=42;
};

Args parse_args(int argc, char** argv){
  Args a;
  for(int i=1;i<argc;++i){
    std::string s(argv[i]); auto next=[&](int&i){return std::string(argv[++i]);};
    if(s=="--m") a.m=std::stoi(next(i));
    else if(s=="--n") a.n=std::stoi(next(i));
    else if(s=="--k") a.k=std::stoi(next(i));
    else if(s=="--alpha") a.alpha=std::stof(next(i));
    else if(s=="--beta") a.beta=std::stof(next(i));
    else if(s=="--repeat") a.repeat=std::stoi(next(i));
    else if(s=="--kernel") a.kernel=next(i);
    else if(s=="--block") a.block=std::stoi(next(i));
    else if(s=="--tA") { char c=next(i)[0]; a.tA=(c=='T'||c=='t')?T:N; }
    else if(s=="--tB") { char c=next(i)[0]; a.tB=(c=='T'||c=='t')?T:N; }
    else if(s=="--seed") a.seed=std::stoi(next(i));
  }
  return a;
}

static double gflops_gemm(long long m,long long n,long long K,double ms){
  double flops = 2.0*m*n*K + 2.0*m*n; // include axpby
  return flops/(ms*1e6);
}

int main(int argc, char** argv){
  Args args = parse_args(argc, argv);
  int m=args.m, n=args.n, K=args.k;

  // stored shapes according to transpose flags
  int mA = (args.tA==N)? m : K;
  int nA = (args.tA==N)? K : m;
  int mB = (args.tB==N)? K : n;
  int nB = (args.tB==N)? n : K;

  // host buffers
  std::mt19937 rng(args.seed);
  std::uniform_real_distribution<float> dist(-1.f,1.f);
  std::vector<float> A(mA*nA), B(mB*nB), C(m*n), C_ref(m*n);
  for (auto& x:A) x=dist(rng);
  for (auto& x:B) x=dist(rng);
  for (auto& x:C) x=dist(rng);
  C_ref = C; // <-- FIX: just clone C

  // CPU reference (in-place)
  auto opA = [&](int i,int kk)->float{
    return (args.tA==N) ? A[idx(i,kk,nA)] : A[idx(kk,i,nA)];
  };
  auto opB = [&](int kk,int j)->float{
    return (args.tB==N) ? B[idx(kk,j,nB)] : B[idx(j,kk,nB)];
  };
  for(int i=0;i<m;++i){
    for(int j=0;j<n;++j){
      double acc=0.0;
      for(int kk=0; kk<K; ++kk) acc += (double)opA(i,kk) * (double)opB(kk,j);
      C_ref[idx(i,j,n)] = (float)(args.alpha*acc + args.beta*C_ref[idx(i,j,n)]);
    }
  }

  // device buffers
  float *dA,*dB,*dC;
  CHECK_CUDA(cudaMalloc(&dA,sizeof(float)*A.size()));
  CHECK_CUDA(cudaMalloc(&dB,sizeof(float)*B.size()));
  CHECK_CUDA(cudaMalloc(&dC,sizeof(float)*C.size()));
  CHECK_CUDA(cudaMemcpy(dA,A.data(),sizeof(float)*A.size(),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB,B.data(),sizeof(float)*B.size(),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dC,C.data(),sizeof(float)*C.size(),cudaMemcpyHostToDevice));

  // launch config
  dim3 block(16,16);
  dim3 grid((n+block.x-1)/block.x, (m+block.y-1)/block.y);

  cudaEvent_t start,stop; CHECK_CUDA(cudaEventCreate(&start)); CHECK_CUDA(cudaEventCreate(&stop));

  auto run_naive = [&](){
    gemm_naive_inplace<<<grid,block>>>(dA,dB,dC,m,n,K,mA,nA,mB,nB,args.tA,args.tB,args.alpha,args.beta);
  };
  auto run_tiled = [&](){
    if(args.block==16){
      dim3 b(16,16), g((n+15)/16,(m+15)/16);
      gemm_tiled_inplace<16><<<g,b>>>(dA,dB,dC,m,n,K,mA,nA,mB,nB,args.tA,args.tB,args.alpha,args.beta);
    } else {
      dim3 b(32,32), g((n+31)/32,(m+31)/32);
      gemm_tiled_inplace<32><<<g,b>>>(dA,dB,dC,m,n,K,mA,nA,mB,nB,args.tA,args.tB,args.alpha,args.beta);
    }
  };

  // warmup
  if(args.kernel=="naive") run_naive(); else run_tiled();
  CHECK_CUDA(cudaDeviceSynchronize());

  // time
  CHECK_CUDA(cudaEventRecord(start));
  for(int r=0;r<args.repeat;++r){
    if(args.kernel=="naive") run_naive(); else run_tiled();
  }
  CHECK_CUDA(cudaEventRecord(stop)); CHECK_CUDA(cudaEventSynchronize(stop));
  float ms=0.f; CHECK_CUDA(cudaEventElapsedTime(&ms,start,stop)); ms/=args.repeat;

  // copy back and check
  CHECK_CUDA(cudaMemcpy(C.data(),dC,sizeof(float)*C.size(),cudaMemcpyDeviceToHost));
  float mad=0.f; for(size_t i=0;i<C.size();++i) mad=fmaxf(mad, fabsf(C[i]-C_ref[i]));

  std::cout<<"Kernel="<<args.kernel<<" block="<<args.block
           <<"  m="<<m<<" n="<<n<<" k="<<K
           <<"  tA="<<(args.tA==N?'N':'T')<<" tB="<<(args.tB==N?'N':'T')
           <<"  alpha="<<args.alpha<<" beta="<<args.beta<<"\n";
  std::cout<<"Avg time "<<ms<<" ms   Throughput "<<gflops_gemm(m,n,K,ms)<<" GFLOP/s\n";
  std::cout<<"Max |C_gpu - C_ref| = "<<mad<<"\n";

  CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dB)); CHECK_CUDA(cudaFree(dC));
  CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));
  return 0;
}

