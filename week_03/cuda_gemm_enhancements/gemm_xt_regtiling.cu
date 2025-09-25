%%writefile gemm_xt_regtiling.cu
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
__host__ __device__ inline int idx(int r,int c,int ld){ return r*ld + c; }

__host__ __device__ inline float read_opA(const float* A,int i,int k,int mA,int nA,Transpose tA){
  return (tA==N) ? A[idx(i,k,nA)] : A[idx(k,i,nA)];
}
__host__ __device__ inline float read_opB(const float* B,int k,int j,int mB,int nB,Transpose tB){
  return (tB==N) ? B[idx(k,j,nB)] : B[idx(j,k,nB)];
}

/* ---------------- Register-tiled kernel ----------------
 Block computes a BLOCK_M×BLOCK_N tile of C.
 K is traversed in chunks of KT.
 Each thread accumulates a TM×TN micro-tile in registers.
*/
template<int BLOCK_M, int BLOCK_N, int KT, int TM, int TN>
__global__ void gemm_regtiling_inplace(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int m, int n, int K,
                                       int mA, int nA, int mB, int nB,
                                       Transpose tA, Transpose tB,
                                       float alpha, float beta){
  // threadblock arrangement
  constexpr int TX = BLOCK_N / TN;  // threads in x
  constexpr int TY = BLOCK_M / TM;  // threads in y
  static_assert(BLOCK_M%TM==0 && BLOCK_N%TN==0, "tile dims must divide block dims");

  __shared__ float As[BLOCK_M][KT+1];  // +1 to reduce bank conflicts
  __shared__ float Bs[KT][BLOCK_N+1];

  const int tidx = threadIdx.x;     // 0..TX-1
  const int tidy = threadIdx.y;     // 0..TY-1
  const int blockRow = blockIdx.y * BLOCK_M;
  const int blockCol = blockIdx.x * BLOCK_N;

  // top-left output this thread is responsible for
  const int row0 = blockRow + tidy*TM;
  const int col0 = blockCol + tidx*TN;

  // micro-tile accumulators in registers
  float acc[TM][TN];
  #pragma unroll
  for(int i=0;i<TM;++i) for(int j=0;j<TN;++j) acc[i][j] = 0.f;

  // number of K tiles
  const int tiles = (K + KT - 1) / KT;

  for(int t=0; t<tiles; ++t){
    const int k0 = t*KT;

    // -------- cooperative load into shared --------
    // map threads to cover full As (BLOCK_M×KT) and Bs (KT×BLOCK_N)
    // use a linearized id and strided loop to keep code simple & correct
    const int TX_ALL = TX*TY;
    const int tlinear = tidy*TX + tidx;

    // load As
    for(int s = tlinear; s < BLOCK_M*KT; s += TX_ALL){
      int r = s / KT;     // 0..BLOCK_M-1
      int p = s % KT;     // 0..KT-1
      int gRow = blockRow + r;
      int gK   = k0 + p;
      float val = 0.f;
      if(gRow < m && gK < K){
        val = read_opA(A, gRow, gK, mA, nA, tA);
      }
      As[r][p] = val;
    }

    // load Bs
    for(int s = tlinear; s < KT*BLOCK_N; s += TX_ALL){
      int p = s / BLOCK_N;  // 0..KT-1
      int c = s % BLOCK_N;  // 0..BLOCK_N-1
      int gK = k0 + p;
      int gCol = blockCol + c;
      float val = 0.f;
      if(gK < K && gCol < n){
        val = read_opB(B, gK, gCol, mB, nB, tB);
      }
      Bs[p][c] = val;
    }

    __syncthreads();

    // -------- compute on this K-tile --------
    #pragma unroll
    for(int p=0; p<KT; ++p){
      // gather the TM elements this thread needs from As and TN from Bs
      float a_reg[TM];
      float b_reg[TN];

      #pragma unroll
      for(int ii=0; ii<TM; ++ii){
        int r = tidy*TM + ii;
        a_reg[ii] = (row0+ii < m && (k0+p) < K && r < BLOCK_M) ? As[r][p] : 0.f;
      }
      #pragma unroll
      for(int jj=0; jj<TN; ++jj){
        int c = tidx*TN + jj;
        b_reg[jj] = (col0+jj < n && (k0+p) < K && c < BLOCK_N) ? Bs[p][c] : 0.f;
      }

      #pragma unroll
      for(int ii=0; ii<TM; ++ii){
        #pragma unroll
        for(int jj=0; jj<TN; ++jj){
          acc[ii][jj] = fmaf(a_reg[ii], b_reg[jj], acc[ii][jj]);
        }
      }
    }
    __syncthreads();
  }

  // -------- in-place epilogue writeback --------
  #pragma unroll
  for(int ii=0; ii<TM; ++ii){
    int r = row0 + ii;
    if (r >= m) continue;
    #pragma unroll
    for(int jj=0; jj<TN; ++jj){
      int c = col0 + jj;
      if (c >= n) continue;
      int o = idx(r,c,n);
      C[o] = alpha*acc[ii][jj] + beta*C[o];
    }
  }
}

/* ---------------- Naive kernel (for comparison) ---------------- */
__global__ void gemm_naive_inplace(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int m,int n,int K, int mA,int nA,int mB,int nB,
                                   Transpose tA, Transpose tB, float alpha,float beta){
  int i = blockIdx.y*blockDim.y + threadIdx.y;
  int j = blockIdx.x*blockDim.x + threadIdx.x;
  if(i>=m || j>=n) return;
  float acc=0.f;
  for(int k=0;k<K;++k){
    acc = fmaf(read_opA(A,i,k,mA,nA,tA), read_opB(B,k,j,mB,nB,tB), acc);
  }
  C[idx(i,j,n)] = alpha*acc + beta*C[idx(i,j,n)];
}

/* ---------------- Host driver ---------------- */
struct Args {
  int m=1024, n=1024, k=1024;
  Transpose tA=N, tB=N;
  float alpha=1.f, beta=1.f;
  int repeat=30;
  std::string kernel="reg"; // "reg" | "naive"
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
    else if(s=="--tA"){ char c=next(i)[0]; a.tA=(c=='T'||c=='t')?T:N; }
    else if(s=="--tB"){ char c=next(i)[0]; a.tB=(c=='T'||c=='t')?T:N; }
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
  const int m=args.m, n=args.n, K=args.k;

  // stored dims based on transpose flags
  const int mA=(args.tA==N)?m:K, nA=(args.tA==N)?K:m;
  const int mB=(args.tB==N)?K:n, nB=(args.tB==N)?n:K;

  // host buffers
  std::mt19937 rng(args.seed);
  std::uniform_real_distribution<float> dist(-1.f,1.f);
  std::vector<float> A(mA*nA), B(mB*nB), C(m*n), C_ref(m*n);
  for(auto& x:A) x=dist(rng);
  for(auto& x:B) x=dist(rng);
  for(auto& x:C) x=dist(rng);
  C_ref=C;

  auto opA = [&](int i,int kk)->float{ return (args.tA==N)?A[idx(i,kk,nA)]:A[idx(kk,i,nA)]; };
  auto opB = [&](int kk,int j)->float{ return (args.tB==N)?B[idx(kk,j,nB)]:B[idx(j,kk,nB)]; };

  for(int i=0;i<m;++i){
    for(int j=0;j<n;++j){
      double acc=0.0;
      for(int kk=0; kk<K; ++kk) acc += (double)opA(i,kk)*(double)opB(kk,j);
      C_ref[idx(i,j,n)] = (float)(args.alpha*acc + args.beta*C_ref[idx(i,j,n)]);
    }
  }

  // device
  float *dA,*dB,*dC;
  CHECK_CUDA(cudaMalloc(&dA,sizeof(float)*A.size()));
  CHECK_CUDA(cudaMalloc(&dB,sizeof(float)*B.size()));
  CHECK_CUDA(cudaMalloc(&dC,sizeof(float)*C.size()));
  CHECK_CUDA(cudaMemcpy(dA,A.data(),sizeof(float)*A.size(),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB,B.data(),sizeof(float)*B.size(),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dC,C.data(),sizeof(float)*C.size(),cudaMemcpyHostToDevice));

  // launches
  cudaEvent_t start,stop; CHECK_CUDA(cudaEventCreate(&start)); CHECK_CUDA(cudaEventCreate(&stop));

  auto run_naive = [&](){
    dim3 b(16,16), g((n+15)/16,(m+15)/16);
    gemm_naive_inplace<<<g,b>>>(dA,dB,dC,m,n,K,mA,nA,mB,nB,args.tA,args.tB,args.alpha,args.beta);
  };

  // register-tiled params: BLOCK_M=N=64, KT=16, TM=2, TN=2 → 32×32=1024 threads/block
  auto run_reg = [&](){
    dim3 b(32,32), g((n+63)/64,(m+63)/64); // threads = (BLOCK_N/TN, BLOCK_M/TM)
    gemm_regtiling_inplace<64,64,16,2,2>
      <<<g,b>>>(dA,dB,dC,m,n,K,mA,nA,mB,nB,args.tA,args.tB,args.alpha,args.beta);
  };

  if(args.kernel=="naive") run_naive(); else run_reg();
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventRecord(start));
  for(int r=0;r<args.repeat;++r){
    if(args.kernel=="naive") run_naive(); else run_reg();
  }
  CHECK_CUDA(cudaEventRecord(stop)); CHECK_CUDA(cudaEventSynchronize(stop));
  float ms=0.f; CHECK_CUDA(cudaEventElapsedTime(&ms,start,stop)); ms/=args.repeat;

  CHECK_CUDA(cudaMemcpy(C.data(),dC,sizeof(float)*C.size(),cudaMemcpyDeviceToHost));
  float mad=0.f; for(size_t i=0;i<C.size();++i) mad=fmaxf(mad, fabsf(C[i]-C_ref[i]));

  std::cout<<"Kernel="<<args.kernel<<"  m="<<m<<" n="<<n<<" k="<<K
           <<"  tA="<<(args.tA==N?'N':'T')<<" tB="<<(args.tB==N?'N':'T')
           <<"  alpha="<<args.alpha<<" beta="<<args.beta<<"\n";
  std::cout<<"Avg time "<<ms<<" ms   Throughput "<<gflops_gemm(m,n,K,ms)<<" GFLOP/s\n";
  std::cout<<"Max |C_gpu - C_ref| = "<<mad<<"\n";

  CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dB)); CHECK_CUDA(cudaFree(dC));
  CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));
  return 0;
}

