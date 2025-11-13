#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "common.h"
static inline float gelu(float x){ return 0.5f*x*(1.0f + tanhf(0.7978845608f*(x + 0.044715f*x*x*x))); }
int main(int argc,char**argv){
  MPI_Init(&argc,&argv); int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
  const char* dir="cases/case1"; for(int i=1;i<argc;i++) if(!strcmp(argv[i],"--case")&&i+1<argc) dir=argv[++i];
  int N,B,d,h,E,E_local; if(read_meta(dir,&N,&B,&d,&h,&E,&E_local)!=0){ MPI_Abort(MPI_COMM_WORLD,1); }
  if(size!=N){ if(rank==0) fprintf(stderr,"world %d != N %d\n",size,N); MPI_Abort(MPI_COMM_WORLD,2); }
  size_t szWg=(size_t)E*d, szW1=(size_t)E*d*h, szW2=(size_t)E*h*d, szX=(size_t)B*d;
  float *Wg=malloc(szWg*4),*W1=malloc(szW1*4),*W2=malloc(szW2*4),*X=malloc(szX*4),*Y=malloc(szX*4),*Yref=malloc(szX*4);
  int32_t* top1=malloc((size_t)B*4);
  char p[1024];
  snprintf(p,sizeof(p),"%s/Wg.bin",dir); read_bin_f32(p,Wg,szWg);
  snprintf(p,sizeof(p),"%s/W1.bin",dir); read_bin_f32(p,W1,szW1);
  snprintf(p,sizeof(p),"%s/W2.bin",dir); read_bin_f32(p,W2,szW2);
  snprintf(p,sizeof(p),"%s/X_rank%d.bin",dir,rank); read_bin_f32(p,X,szX);
  snprintf(p,sizeof(p),"%s/Y_rank%d.bin",dir,rank); read_bin_f32(p,Yref,szX);
  snprintf(p,sizeof(p),"%s/top1_rank%d.bin",dir,rank); read_bin_i32(p,top1,(size_t)B);
  int32_t* owner=malloc((size_t)B*4); int32_t* exloc=malloc((size_t)B*4);
  for(int i=0;i<B;i++){
    int best=-1; float sbest=-1e30f;
    for(int e=0;e<E;e++){ const float* we=Wg+(size_t)e*d; float s=0.f; for(int t=0;t<d;t++) s+=X[(size_t)i*d+t]*we[t]; if(s>sbest){sbest=s;best=e;} }
    if(best!=top1[i]){ fprintf(stderr,"rank%d gating mismatch %d\n",rank,i); MPI_Abort(MPI_COMM_WORLD,3); }
    owner[i]=best/ E_local; exloc[i]=best% E_local;
  }
  int* sendc=calloc(size,sizeof(int)),*recvc=calloc(size,sizeof(int));
  for(int i=0;i<B;i++) sendc[owner[i]]++;
  MPI_Alltoall(sendc,1,MPI_INT,recvc,1,MPI_INT,MPI_COMM_WORLD);
  int* sdis=calloc(size,sizeof(int)),*rdis=calloc(size,sizeof(int));
  int send_tot=0,recv_tot=0; for(int p=0;p<size;p++){ sdis[p]=send_tot; send_tot+=sendc[p]; rdis[p]=recv_tot; recv_tot+=recvc[p]; }
  float* t_send=malloc((size_t)send_tot*d*4); int32_t* ex_send=malloc((size_t)send_tot*4); int32_t* ori_send=malloc((size_t)send_tot*4);
  int* cur=calloc(size,sizeof(int));
  for(int i=0;i<B;i++){ int dst=owner[i]; int pos=sdis[dst]+cur[dst]++; memcpy(t_send+(size_t)pos*d, X+(size_t)i*d, (size_t)d*4); ex_send[pos]=exloc[i]; ori_send[pos]=i; }
  free(cur);
  float* t_recv=malloc((size_t)recv_tot*d*4); int32_t* ex_recv=malloc((size_t)recv_tot*4); int32_t* ori_recv=malloc((size_t)recv_tot*4);
  int *sendc_f=malloc(size*4), *recvc_f=malloc(size*4); for(int p=0;p<size;p++){ sendc_f[p]=sendc[p]*d; recvc_f[p]=recvc[p]*d; }
  MPI_Alltoallv(t_send,sendc_f,sdis,MPI_FLOAT,t_recv,recvc_f,rdis,MPI_FLOAT,MPI_COMM_WORLD);
  MPI_Alltoallv(ex_send,sendc,sdis,MPI_INT,ex_recv,recvc,rdis,MPI_INT,MPI_COMM_WORLD);
  MPI_Alltoallv(ori_send,sendc,sdis,MPI_INT,ori_recv,recvc,rdis,MPI_INT,MPI_COMM_WORLD);
  float* out_recv=malloc((size_t)recv_tot*d*4);
  for(int n=0;n<recv_tot;n++){
    int el=ex_recv[n];
    const float* w1=W1+(size_t)(rank*E_local+el)*d*h;
    const float* w2=W2+(size_t)(rank*E_local+el)*h*d;
    float* y=out_recv+(size_t)n*d; for(int t=0;t<d;t++) y[t]=0.f;
    for(int p2=0;p2<h;p2++){ float acc=0.f; for(int k=0;k<d;k++) acc+=t_recv[(size_t)n*d+k]*w1[(size_t)k*h+p2]; float u=gelu(acc); for(int t=0;t<d;t++) y[t]+=u*w2[(size_t)p2*d+t]; }
  }
  int* sendc_b=malloc(size*4),*recvc_b=malloc(size*4); for(int p=0;p<size;p++) sendc_b[p]=recvc[p]; MPI_Alltoall(sendc_b,1,MPI_INT,recvc_b,1,MPI_INT,MPI_COMM_WORLD);
  int* sdis_b=calloc(size,sizeof(int)),*rdis_b=calloc(size,sizeof(int)); int send_tot_b=0,recv_tot_b=0;
  for(int p=0;p<size;p++){ sdis_b[p]=send_tot_b; send_tot_b+=sendc_b[p]; rdis_b[p]=recv_tot_b; recv_tot_b+=recvc_b[p]; }
  float* out_back=malloc((size_t)recv_tot_b*d*4); int32_t* ori_back=malloc((size_t)recv_tot_b*4);
  int *sendc_bf=malloc(size*4), *recvc_bf=malloc(size*4); for(int p=0;p<size;p++){ sendc_bf[p]=sendc_b[p]*d; recvc_bf[p]=recvc_b[p]*d; }
  MPI_Alltoallv(out_recv,recvc_f,rdis,MPI_FLOAT,out_back,recvc_bf,rdis_b,MPI_FLOAT,MPI_COMM_WORLD);
  MPI_Alltoallv(ori_recv,recvc,rdis,MPI_INT,ori_back,recvc_b,rdis_b,MPI_INT,MPI_COMM_WORLD);
  for(int n=0;n<recv_tot_b;n++){ int i=ori_back[n]; memcpy(Y+(size_t)i*d, out_back+(size_t)n*d, (size_t)d*4); }
  double mx=0.0,mae=0.0; for(size_t i=0;i<szX;i++){ double e=fabs((double)Y[i]-(double)Yref[i]); if(e>mx)mx=e; mae+=e; } mae/= (double)szX;
  if(rank==0) printf("OK MPI all-to-all  max|Δ|=%.3e  MAE=%.3e\n",mx,mae);
  MPI_Finalize(); return 0;
}
