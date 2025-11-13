#ifndef DSM_COMMON_H
#define DSM_COMMON_H
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
static inline int read_meta(const char* dir,int* N,int* B,int* d,int* h,int* E,int* E_local){
  char p[1024]; snprintf(p,sizeof(p),"%s/meta.txt",dir); FILE* f=fopen(p,"r"); if(!f) return -1;
  char line[256]; int got=0,v;
  while(fgets(line,sizeof(line),f)){
    if(sscanf(line,"N=%d",&v)==1){*N=v;got++;continue;}
    if(sscanf(line,"B=%d",&v)==1){*B=v;got++;continue;}
    if(sscanf(line,"d=%d",&v)==1){*d=v;got++;continue;}
    if(sscanf(line,"h=%d",&v)==1){*h=v;got++;continue;}
    if(sscanf(line,"E=%d",&v)==1){*E=v;got++;continue;}
    if(sscanf(line,"E_local=%d",&v)==1){*E_local=v;got++;continue;}
  }
  fclose(f); return got>=5?0:-2;
}
static inline int read_bin_f32(const char* path,float* dst,size_t n){
  FILE* f=fopen(path,"rb"); if(!f) return -1; size_t r=fread(dst,sizeof(float),n,f); fclose(f); return r==n?0:-2;
}
static inline int read_bin_i32(const char* path,int32_t* dst,size_t n){
  FILE* f=fopen(path,"rb"); if(!f) return -1; size_t r=fread(dst,sizeof(int32_t),n,f); fclose(f); return r==n?0:-2;
}
#endif
