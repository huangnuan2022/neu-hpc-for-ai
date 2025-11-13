#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#define CHECK_CUDA(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
#define CHECK_NCCL(x) do{ ncclResult_t r=(x); if(r!=ncclSuccess){ fprintf(stderr,"NCCL %s:%d %s\n",__FILE__,__LINE__,ncclGetErrorString(r)); exit(1);} }while(0)
__device__ __forceinline__ float gelu(float x){ return 0.5f*x*(1.0f + tanhf(0.7978845608f*(x + 0.044715f*x*x*x))); }
__global__ void gating_top1(const float* X,const float* Wg,int B,int d,int E,int* top1){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=B)return; const float* xi=X+(size_t)i*d; int best=-1; float sbest=-1e30f; for(int e=0;e<E;e++){ const float* we=Wg+(size_t)e*d; float s=0.f; for(int t=0;t<d;t++) s+=xi[t]*we[t]; if(s>sbest){sbest=s;best=e;} } top1[i]=best; }
__global__ void gather_rows(const float* X,const int* idx,float* Y,int n,int d){ int k=blockIdx.x*blockDim.x+threadIdx.x; if(k>=n)return; const float* xs=X+(size_t)idx[k]*d; float* yd=Y+(size_t)k*d; for(int t=0;t<d;t++) yd[t]=xs[t]; }
__global__ void scatter_rows(const float* X,const int* dest,float* Y,int n,int d){ int k=blockIdx.x*blockDim.x+threadIdx.x; if(k>=n)return; const float* xs=X+(size_t)k*d; float* yd=Y+(size_t)dest[k]*d; for(int t=0;t<d;t++) yd[t]=xs[t]; }
__global__ void expert_mlp(const float* Tin,const int* ex,const float* W1,const float* W2,float* Tout,int T,int d,int h,int rank,int E_local){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=T)return; int el=ex[i]; const float* x=Tin+(size_t)i*d; float* y=Tout+(size_t)i*d; for(int t=0;t<d;t++) y[t]=0.f;
  const float* w1=W1+(size_t)(rank*E_local+el)*d*h; const float* w2=W2+(size_t)(rank*E_local+el)*h*d;
  for(int p=0;p<h;p++){ float acc=0.f; for(int k=0;k<d;k++) acc+=x[k]*w1[(size_t)k*h+p]; float u=gelu(acc); const float* w2r=w2+(size_t)p*d; for(int t=0;t<d;t++) y[t]+=u*w2r[t]; }
}
static void read_bin(const std::string& p, void* b, size_t n){ FILE* f=fopen(p.c_str(),"rb"); if(!f){ perror(p.c_str()); exit(1);} fread(b,1,n,f); fclose(f); }
static void read_meta(const std::string& dir,int& N,int& B,int& d,int& h,int& E,int& E_local){ std::string p=dir+"/meta.txt"; FILE* f=fopen(p.c_str(),"r"); if(!f){ perror(p.c_str()); exit(1);} char line[256]; while(fgets(line,sizeof(line),f)){ int v; if(sscanf(line,"N=%d",&v)==1)N=v; else if(sscanf(line,"B=%d",&v)==1)B=v; else if(sscanf(line,"d=%d",&v)==1)d=v; else if(sscanf(line,"h=%d",&v)==1)h=v; else if(sscanf(line,"E=%d",&v)==1)E=v; else if(sscanf(line,"E_local=%d",&v)==1)E_local=v; } fclose(f); }
int main(int argc,char**argv){
  MPI_Init(&argc,&argv); int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
  std::string dir="cases/case1"; for(int i=1;i<argc;i++) if(!strcmp(argv[i],"--case")&&i+1<argc) dir=argv[++i];
  int N,B,d,h,E,E_local; read_meta(dir,N,B,d,h,E,E_local); if(N!=size){ if(rank==0) fprintf(stderr,"world %d != N %d\n",size,N); MPI_Abort(MPI_COMM_WORLD,1); }
  int devs=0; CHECK_CUDA(cudaGetDeviceCount(&devs)); CHECK_CUDA(cudaSetDevice(rank%devs));
  ncclUniqueId id; if(rank==0) CHECK_NCCL(ncclGetUniqueId(&id)); MPI_Bcast(&id,sizeof(id),MPI_BYTE,0,MPI_COMM_WORLD);
  ncclComm_t comm; CHECK_NCCL(ncclCommInitRank(&comm,size,id,rank)); cudaStream_t stream; CHECK_CUDA(cudaStreamCreate(&stream));
  size_t szWg=(size_t)E*d, szW1=(size_t)E*d*h, szW2=(size_t)E*h*d, szX=(size_t)B*d;
  std::vector<float> hWg(szWg),hW1(szW1),hW2(szW2),hX(szX),hYref(szX);
  read_bin(dir+"/Wg.bin",hWg.data(),szWg*4); read_bin(dir+"/W1.bin",hW1.data(),szW1*4); read_bin(dir+"/W2.bin",hW2.data(),szW2*4);
  {char p[1024]; sprintf(p,"%s/X_rank%d.bin",dir.c_str(),rank); read_bin(p,hX.data(),szX*4);}
  {char p[1024]; sprintf(p,"%s/Y_rank%d.bin",dir.c_str(),rank); read_bin(p,hYref.data(),szX*4);}
  float *dX,*dY,*dWg,*dW1,*dW2; int *dTop1; CHECK_CUDA(cudaMalloc(&dX,szX*4)); CHECK_CUDA(cudaMalloc(&dY,szX*4)); CHECK_CUDA(cudaMalloc(&dWg,szWg*4)); CHECK_CUDA(cudaMalloc(&dW1,szW1*4)); CHECK_CUDA(cudaMalloc(&dW2,szW2*4)); CHECK_CUDA(cudaMalloc(&dTop1,B*4));
  CHECK_CUDA(cudaMemcpyAsync(dX,hX.data(),szX*4,cudaMemcpyHostToDevice,stream)); CHECK_CUDA(cudaMemcpyAsync(dWg,hWg.data(),szWg*4,cudaMemcpyHostToDevice,stream)); CHECK_CUDA(cudaMemcpyAsync(dW1,hW1.data(),szW1*4,cudaMemcpyHostToDevice,stream)); CHECK_CUDA(cudaMemcpyAsync(dW2,hW2.data(),szW2*4,cudaMemcpyHostToDevice,stream)); CHECK_CUDA(cudaMemsetAsync(dY,0,szX*4,stream)); CHECK_CUDA(cudaStreamSynchronize(stream));
  dim3 blk(256),grd((B+255)/256); gating_top1<<<grd,blk,0,stream>>>(dX,dWg,B,d,E,dTop1); CHECK_CUDA(cudaStreamSynchronize(stream));
  std::vector<int> top1(B),owner(B),exloc(B); CHECK_CUDA(cudaMemcpy(top1.data(),dTop1,B*4,cudaMemcpyDeviceToHost)); for(int i=0;i<B;i++){ owner[i]=top1[i]/E_local; exloc[i]=top1[i]%E_local; }
  std::vector<int> sendc(size,0),recvc(size,0); for(int i=0;i<B;i++) sendc[owner[i]]++; MPI_Alltoall(sendc.data(),1,MPI_INT,recvc.data(),1,MPI_INT,MPI_COMM_WORLD);
  int send_tot=0,recv_tot=0; std::vector<int> sdis(size,0),rdis(size,0); for(int p=0;p<size;p++){ sdis[p]=send_tot; send_tot+=sendc[p]; rdis[p]=recv_tot; recv_tot+=recvc[p]; }
  int *dIdx,*dEx,*dOri; float *dPack; CHECK_CUDA(cudaMalloc(&dIdx,send_tot*4)); CHECK_CUDA(cudaMalloc(&dEx,send_tot*4)); CHECK_CUDA(cudaMalloc(&dOri,send_tot*4)); CHECK_CUDA(cudaMalloc(&dPack,(size_t)send_tot*d*4));
  std::vector<int> hIdx(send_tot),hEx(send_tot),hOri(send_tot),cur(size,0); for(int i=0;i<B;i++){ int dst=owner[i]; int pos=sdis[dst]+cur[dst]++; hIdx[pos]=i; hEx[pos]=exloc[i]; hOri[pos]=i; }
  CHECK_CUDA(cudaMemcpy(dIdx,hIdx.data(),send_tot*4,cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemcpy(dEx,hEx.data(),send_tot*4,cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemcpy(dOri,hOri.data(),send_tot*4,cudaMemcpyHostToDevice));
  gather_rows<<<(send_tot+255)/256,blk,0,stream>>>(dX,dIdx,dPack,send_tot,d); CHECK_CUDA(cudaStreamSynchronize(stream));
  float *dTokRecv,*dOutRecv; int *dExRecv,*dOriRecv; CHECK_CUDA(cudaMalloc(&dTokRecv,(size_t)recv_tot*d*4)); CHECK_CUDA(cudaMalloc(&dExRecv,recv_tot*4)); CHECK_CUDA(cudaMalloc(&dOriRecv,recv_tot*4)); CHECK_CUDA(cudaMalloc(&dOutRecv,(size_t)recv_tot*d*4));
  CHECK_NCCL(ncclGroupStart()); for(int p=0;p<size;p++){ size_t se=(size_t)sendc[p]*d, re=(size_t)recvc[p]*d; CHECK_NCCL(ncclSend(dPack+(size_t)sdis[p]*d,se,ncclFloat,p,comm,stream)); CHECK_NCCL(ncclRecv(dTokRecv+(size_t)rdis[p]*d,re,ncclFloat,p,comm,stream)); } CHECK_NCCL(ncclGroupEnd());
  CHECK_NCCL(ncclGroupStart()); for(int p=0;p<size;p++){ CHECK_NCCL(ncclSend(dEx+sdis[p],sendc[p],ncclInt,p,comm,stream)); CHECK_NCCL(ncclRecv(dExRecv+rdis[p],recvc[p],ncclInt,p,comm,stream)); CHECK_NCCL(ncclSend(dOri+sdis[p],sendc[p],ncclInt,p,comm,stream)); CHECK_NCCL(ncclRecv(dOriRecv+rdis[p],recvc[p],ncclInt,p,comm,stream)); } CHECK_NCCL(ncclGroupEnd()); CHECK_CUDA(cudaStreamSynchronize(stream));
  expert_mlp<<<(recv_tot+255)/256,blk,0,stream>>>(dTokRecv,dExRecv,(const float*)dW1,(const float*)dW2,dOutRecv,recv_tot,d,h,rank,(E/size)); CHECK_CUDA(cudaStreamSynchronize(stream));
  std::vector<int> sendc_b(size),recvc_b(size); for(int p=0;p<size;p++) sendc_b[p]=recvc[p]; MPI_Alltoall(sendc_b.data(),1,MPI_INT,recvc_b.data(),1,MPI_INT,MPI_COMM_WORLD);
  std::vector<int> sdis_b(size,0),rdis_b(size,0); int send_tot_b=0,recv_tot_b=0; for(int p=0;p<size;p++){ sdis_b[p]=send_tot_b; send_tot_b+=sendc_b[p]; rdis_b[p]=recv_tot_b; recv_tot_b+=recvc_b[p]; }
  float* dOutBack; int* dOriBack; CHECK_CUDA(cudaMalloc(&dOutBack,(size_t)recv_tot_b*d*4)); CHECK_CUDA(cudaMalloc(&dOriBack,recv_tot_b*4));
  CHECK_NCCL(ncclGroupStart()); for(int p=0;p<size;p++){ size_t se=(size_t)sendc_b[p]*d,re=(size_t)recvc_b[p]*d; CHECK_NCCL(ncclSend(dOutRecv+(size_t)rdis[p]*d,se,ncclFloat,p,comm,stream)); CHECK_NCCL(ncclRecv(dOutBack+(size_t)rdis_b[p]*d,re,ncclFloat,p,comm,stream)); } CHECK_NCCL(ncclGroupEnd());
  CHECK_NCCL(ncclGroupStart()); for(int p=0;p<size;p++){ CHECK_NCCL(ncclSend(dOriRecv+rdis[p],sendc_b[p],ncclInt,p,comm,stream)); CHECK_NCCL(ncclRecv(dOriBack+rdis_b[p],recvc_b[p],ncclInt,p,comm,stream)); } CHECK_NCCL(ncclGroupEnd()); CHECK_CUDA(cudaStreamSynchronize(stream));
  scatter_rows<<<(recv_tot_b+255)/256,blk,0,stream>>>(dOutBack,dOriBack,dY,recv_tot_b,d); CHECK_CUDA(cudaStreamSynchronize(stream));
  if(size==1){ std::vector<float> hY(szX); CHECK_CUDA(cudaMemcpy(hY.data(),dY,szX*4,cudaMemcpyDeviceToHost)); double mx=0,mae=0; for(size_t i=0;i<szX;i++){ double e=fabs((double)hY[i]-(double)hYref[i]); if(e>mx)mx=e; mae+=e; } mae/= (double)szX; printf("OK CUDA/NCCL  max|Δ|=%.3e  MAE=%.3e\n",mx,mae); }
  ncclCommDestroy(comm); cudaStreamDestroy(stream); return 0;
}
