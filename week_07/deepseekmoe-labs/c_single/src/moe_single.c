#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "common.h"
static inline float gelu(float x){ return 0.5f*x*(1.0f + tanhf(0.7978845608f*(x + 0.044715f*x*x*x))); }
int main(int argc,char**argv){
  const char* dir="cases/case1"; int rank=0;
  for(int i=1;i<argc;i++){ if(!strcmp(argv[i],"--case")&&i+1<argc) dir=argv[++i]; else if(!strcmp(argv[i],"--rank")&&i+1<argc) rank=atoi(argv[++i]); }
  int N,B,d,h,E,E_local; if(read_meta(dir,&N,&B,&d,&h,&E,&E_local)!=0) return 1;
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
  for(int i=0;i<B;i++){
    int best=-1; float sbest=-1e30f;
    for(int e=0;e<E;e++){ const float* we=Wg+(size_t)e*d; float s=0.f; for(int t=0;t<d;t++) s+=X[(size_t)i*d+t]*we[t]; if(s>sbest){sbest=s;best=e;} }
    if(best!=top1[i]){ fprintf(stderr,"gating mismatch %d\n",i); return 2; }
    const float* w1=W1+(size_t)best*d*h; const float* w2=W2+(size_t)best*h*d; float* y=Y+(size_t)i*d;
    for(int t=0;t<d;t++) y[t]=0.f;
    for(int p=0;p<h;p++){ float acc=0.f; for(int k=0;k<d;k++) acc+=X[(size_t)i*d+k]*w1[(size_t)k*h+p]; float u=gelu(acc); for(int t=0;t<d;t++) y[t]+=u*w2[(size_t)p*d+t]; }
  }
  double mx=0.0,mae=0.0; for(size_t i=0;i<szX;i++){ double e=fabs((double)Y[i]-(double)Yref[i]); if(e>mx)mx=e; mae+=e; } mae/= (double)szX;
  printf("OK single-thread C  max|Δ|=%.3e  MAE=%.3e\n",mx,mae);
  free(Wg);free(W1);free(W2);free(X);free(Y);free(Yref);free(top1); return 0;
}
