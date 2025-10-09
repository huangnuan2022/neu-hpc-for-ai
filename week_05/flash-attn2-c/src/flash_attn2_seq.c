/* FlashAttention-2  */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <limits.h>
static inline int idx2(int r,int c,int ld){return r*ld+c;}
static inline float frand(unsigned*s){*s=1664525u*(*s)+1013904223u;return((float)(*s)/(float)UINT_MAX)*2.f-1.f;}
static void rand_mat(float*A,int R,int C,unsigned seed){unsigned s=(seed?seed:1u);for(int i=0;i<R*C;++i)A[i]=frand(&s);}
static void rowmax(const float*X,int R,int C,float*m){for(int i=0;i<R;++i){float mx=-INFINITY;const float*xi=X+i*C;for(int j=0;j<C;++j)if(xi[j]>mx)mx=xi[j];m[i]=mx;}}
static void gemm_q_kT(const float*Qi,const float*Kj,float*S,int br,int bc,int d,float scale){for(int i=0;i<br;++i){const float*qi=Qi+i*d;for(int j=0;j<bc;++j){const float*kj=Kj+j*d;double acc=0.0;for(int k=0;k<d;++k)acc+=(double)qi[k]*kj[k];S[idx2(i,j,bc)]=(float)(acc*(double)scale);}}}
static void gemm_pt_vj(const float*Pt,const float*Vj,float*TV,int br,int bc,int d){for(int i=0;i<br;++i){for(int u=0;u<d;++u){double acc=0.0;for(int j=0;j<bc;++j)acc+=(double)Pt[idx2(i,j,bc)]*(double)Vj[idx2(j,u,d)];TV[idx2(i,u,d)]=(float)acc;}}}
static void gemm_Pt_dO_accum(const float*P,const float*dOi,float*dVj,int br,int bc,int d){for(int j=0;j<bc;++j){for(int u=0;u<d;++u){double acc=0.0;for(int i=0;i<br;++i)acc+=(double)P[idx2(i,j,bc)]*(double)dOi[idx2(i,u,d)];dVj[idx2(j,u,d)]+=(float)acc;}}}
static void gemm_dO_Vt(const float*dOi,const float*Vj,float*dP,int br,int bc,int d){for(int i=0;i<br;++i){for(int j=0;j<bc;++j){double acc=0.0;for(int u=0;u<d;++u)acc+=(double)dOi[idx2(i,u,d)]*(double)Vj[idx2(j,u,d)];dP[idx2(i,j,bc)]=(float)acc;}}}
static void gemm_dS_Kj_accum(const float*dS,const float*Kj,float*dQi,int br,int bc,int d){for(int i=0;i<br;++i){for(int u=0;u<d;++u){double acc=0.0;for(int j=0;j<bc;++j)acc+=(double)dS[idx2(i,j,bc)]*(double)Kj[idx2(j,u,d)];dQi[idx2(i,u,d)]+=(float)acc;}}}
static void gemm_dS_T_Qi_accum(const float*dS,const float*Qi,float*dKj,int br,int bc,int d){for(int j=0;j<bc;++j){for(int u=0;u<d;++u){double acc=0.0;for(int i=0;i<br;++i)acc+=(double)dS[idx2(i,j,bc)]*(double)Qi[idx2(i,u,d)];dKj[idx2(j,u,d)]+=(float)acc;}}}
static void naive_forward(const float*Q,const float*K,const float*V,float*O,int N,int d){const float scale=1.f/sqrtf((float)d);float*S=(float*)malloc(sizeof(float)*N*N),*P=(float*)malloc(sizeof(float)*N*N);
for(int i=0;i<N;++i)for(int j=0;j<N;++j){double acc=0.0;for(int k=0;k<d;++k)acc+=(double)Q[idx2(i,k,d)]*(double)K[idx2(j,k,d)];S[idx2(i,j,N)]=(float)(acc*(double)scale);}
for(int i=0;i<N;++i){float m=-INFINITY;for(int j=0;j<N;++j)if(S[idx2(i,j,N)]>m)m=S[idx2(i,j,N)];double den=0.0;for(int j=0;j<N;++j){double e=exp((double)S[idx2(i,j,N)]-m);P[idx2(i,j,N)]=(float)e;den+=e;}for(int j=0;j<N;++j)P[idx2(i,j,N)]=(float)((double)P[idx2(i,j,N)]/den);}
for(int i=0;i<N;++i){for(int u=0;u<d;++u){double acc=0.0;for(int j=0;j<N;++j)acc+=(double)P[idx2(i,j,N)]*(double)V[idx2(j,u,d)];O[idx2(i,u,d)]=(float)acc;}} free(S);free(P);}
static void naive_backward(const float*Q,const float*K,const float*V,const float*O,const float*dO,float*dQ,float*dK,float*dV,int N,int d){
memset(dQ,0,sizeof(float)*N*d);memset(dK,0,sizeof(float)*N*d);memset(dV,0,sizeof(float)*N*d);
const float scale=1.f/sqrtf((float)d);float*S=(float*)malloc(sizeof(float)*N*N),*P=(float*)malloc(sizeof(float)*N*N),*dP=(float*)malloc(sizeof(float)*N*N),*dS=(float*)malloc(sizeof(float)*N*N),*D=(float*)malloc(sizeof(float)*N);
for(int i=0;i<N;++i)for(int j=0;j<N;++j){double acc=0.0;for(int k=0;k<d;++k)acc+=(double)Q[idx2(i,k,d)]*(double)K[idx2(j,k,d)];S[idx2(i,j,N)]=(float)(acc*(double)scale);}
for(int i=0;i<N;++i){float m=-INFINITY;for(int j=0;j<N;++j)if(S[idx2(i,j,N)]>m)m=S[idx2(i,j,N)];double den=0.0;for(int j=0;j<N;++j){double e=exp((double)S[idx2(i,j,N)]-m);P[idx2(i,j,N)]=(float)e;den+=e;}for(int j=0;j<N;++j)P[idx2(i,j,N)]=(float)((double)P[idx2(i,j,N)]/den);}
for(int i=0;i<N;++i){double acc=0.0;for(int u=0;u<d;++u)acc+=(double)dO[idx2(i,u,d)]*(double)O[idx2(i,u,d)];D[i]=(float)acc;}
for(int j=0;j<N;++j)for(int u=0;u<d;++u){double acc=0.0;for(int i=0;i<N;++i)acc+=(double)P[idx2(i,j,N)]*(double)dO[idx2(i,u,d)];dV[idx2(j,u,d)]=(float)acc;}
for(int i=0;i<N;++i)for(int j=0;j<N;++j){double acc=0.0;for(int u=0;u<d;++u)acc+=(double)dO[idx2(i,u,d)]*(double)V[idx2(j,u,d)];dP[idx2(i,j,N)]=(float)acc;}
for(int i=0;i<N;++i)for(int j=0;j<N;++j)dS[idx2(i,j,N)]=P[idx2(i,j,N)]*(dP[idx2(i,j,N)]-D[i]);
for(int i=0;i<N;++i)for(int u=0;u<d;++u){double acc=0.0;for(int j=0;j<N;++j)acc+=(double)dS[idx2(i,j,N)]*(double)K[idx2(j,u,d)];dQ[idx2(i,u,d)]=(float)acc;}
for(int j=0;j<N;++j)for(int u=0;u<d;++u){double acc=0.0;for(int i=0;i<N;++i)acc+=(double)dS[idx2(i,j,N)]*(double)Q[idx2(i,u,d)];dK[idx2(j,u,d)]=(float)acc;}
free(S);free(P);free(dP);free(dS);free(D);}
static void fa2_forward(const float*Q,const float*K,const float*V,float*O,float*L,int N,int d,int Br,int Bc){
const float scale=1.f/sqrtf((float)d);
for(int i0=0;i0<N;i0+=Br){int br=(i0+Br<=N)?Br:(N-i0);const float*Qi=Q+i0*d;float*Oi=O+i0*d;float*Li=L+i0;
float*Om=(float*)calloc(br*d,sizeof(float)),*ell=(float*)calloc(br,sizeof(float)),*m=(float*)malloc(sizeof(float)*br);for(int i=0;i<br;++i)m[i]=-INFINITY;
float*S=(float*)malloc(sizeof(float)*br*Bc),*mt=(float*)malloc(sizeof(float)*br),*Pt=(float*)malloc(sizeof(float)*br*Bc),*lt=(float*)malloc(sizeof(float)*br),*TV=(float*)malloc(sizeof(float)*br*d);
for(int j0=0;j0<N;j0+=Bc){int bc=(j0+Bc<=N)?Bc:(N-j0);const float*Kj=K+j0*d;const float*Vj=V+j0*d;
gemm_q_kT(Qi,Kj,S,br,bc,d,scale);rowmax(S,br,bc,mt);
for(int i=0;i<br;++i){float m_new=(m[i]>mt[i])?m[i]:mt[i];float a=(m[i]==-INFINITY)?0.f:expf(m[i]-m_new);for(int j=0;j<bc;++j)Pt[idx2(i,j,bc)]=expf(S[idx2(i,j,bc)]-m_new);float sum=0.f;for(int j=0;j<bc;++j)sum+=Pt[idx2(i,j,bc)];lt[i]=a*ell[i]+sum;m[i]=m_new;}
gemm_pt_vj(Pt,Vj,TV,br,bc,d);
for(int i=0;i<br;++i){float sum=0.f;for(int j=0;j<bc;++j)sum+=Pt[idx2(i,j,bc)];float a=(ell[i]>0.f)?((lt[i]-sum)/ell[i]):0.f;for(int u=0;u<d;++u)Om[idx2(i,u,d)]=Om[idx2(i,u,d)]*a+TV[idx2(i,u,d)];ell[i]=lt[i];}}
for(int i=0;i<br;++i){float inv=(ell[i]>0.f)?(1.f/ell[i]):0.f;for(int u=0;u<d;++u)Oi[idx2(i,u,d)]=Om[idx2(i,u,d)]*inv;Li[i]=m[i]+logf((ell[i]>0.f)?ell[i]:1.f);}
free(S);free(mt);free(Pt);free(lt);free(TV);free(Om);free(ell);free(m);}}
static void fa2_backward(const float*Q,const float*K,const float*V,const float*O,const float*dO,const float*L,float*dQ,float*dK,float*dV,int N,int d,int Br,int Bc){
memset(dQ,0,sizeof(float)*N*d);memset(dK,0,sizeof(float)*N*d);memset(dV,0,sizeof(float)*N*d);
float*D=(float*)malloc(sizeof(float)*N);for(int i=0;i<N;++i){double acc=0.0;for(int u=0;u<d;++u)acc+=(double)dO[idx2(i,u,d)]*(double)O[idx2(i,u,d)];D[i]=(float)acc;}
const float scale=1.f/sqrtf((float)d);
for(int j0=0;j0<N;j0+=Bc){int bc=(j0+Bc<=N)?Bc:(N-j0);const float*Kj=K+j0*d;const float*Vj=V+j0*d;float*dKj=(float*)calloc(bc*d,sizeof(float)),*dVj=(float*)calloc(bc*d,sizeof(float));
for(int i0=0;i0<N;i0+=Br){int br=(i0+Br<=N)?Br:(N-i0);const float*Qi=Q+i0*d;const float*dOi=dO+i0*d;float*dQi=dQ+i0*d;const float*Li=L+i0;const float*Di=D+i0;
float*S=(float*)malloc(sizeof(float)*br*bc),*Pij=(float*)malloc(sizeof(float)*br*bc),*dP=(float*)malloc(sizeof(float)*br*bc),*dS=(float*)malloc(sizeof(float)*br*bc);
gemm_q_kT(Qi,Kj,S,br,bc,d,scale);for(int i=0;i<br;++i){float Li_row=Li[i];for(int j=0;j<bc;++j)Pij[idx2(i,j,bc)]=expf(S[idx2(i,j,bc)]-Li_row);}
gemm_Pt_dO_accum(Pij,dOi,dVj,br,bc,d);gemm_dO_Vt(dOi,Vj,dP,br,bc,d);
for(int i=0;i<br;++i){float Di_row=Di[i];for(int j=0;j<bc;++j)dS[idx2(i,j,bc)]=Pij[idx2(i,j,bc)]*(dP[idx2(i,j,bc)]-Di_row);}
gemm_dS_Kj_accum(dS,Kj,dQi,br,bc,d);gemm_dS_T_Qi_accum(dS,Qi,dKj,br,bc,d);
free(S);free(Pij);free(dP);free(dS);}for(int j=0;j<bc;++j)for(int u=0;u<d;++u){dK[idx2(j0+j,u,d)]+=dKj[idx2(j,u,d)];dV[idx2(j0+j,u,d)]+=dVj[idx2(j,u,d)];}
free(dKj);free(dVj);}free(D);}
static double rmse(const float*a,const float*b,size_t n){double s=0.0;for(size_t i=0;i<n;++i){double d=(double)a[i]-b[i];s+=d*d;}return sqrt(s/n);}
static float maxad(const float*a,const float*b,size_t n){float m=0.f;for(size_t i=0;i<n;++i){float d=fabsf(a[i]-b[i]);if(d>m)m=d;}return m;}
int main(int argc,char**argv){int N=(argc>1)?atoi(argv[1]):512,d=(argc>2)?atoi(argv[2]):64,Br=(argc>3)?atoi(argv[3]):128,Bc=(argc>4)?atoi(argv[4]):128,seed=(argc>5)?atoi(argv[5]):42;
printf("FA-2 package  N=%d d=%d Br=%d Bc=%d seed=%d
",N,d,Br,Bc,seed);
float *Q=(float*)malloc(sizeof(float)*N*d),*K=(float*)malloc(sizeof(float)*N*d),*V=(float*)malloc(sizeof(float)*N*d),*O_ref=(float*)malloc(sizeof(float)*N*d),*O=(float*)malloc(sizeof(float)*N*d),*L=(float*)malloc(sizeof(float)*N);
rand_mat(Q,N,d,seed+1);rand_mat(K,N,d,seed+2);rand_mat(V,N,d,seed+3);naive_forward(Q,K,V,O_ref,N,d);fa2_forward(Q,K,V,O,L,N,d,Br,Bc);
printf("[FWD] MAD=%.3e RMSE=%.3e
",maxad(O_ref,O,(size_t)N*d),rmse(O_ref,O,(size_t)N*d));
float *dO=(float*)malloc(sizeof(float)*N*d);rand_mat(dO,N,d,seed+4);
float *dQ_ref=(float*)malloc(sizeof(float)*N*d),*dK_ref=(float*)malloc(sizeof(float)*N*d),*dV_ref=(float*)malloc(sizeof(float)*N*d),*dQ=(float*)malloc(sizeof(float)*N*d),*dK=(float*)malloc(sizeof(float)*N*d),*dV=(float*)malloc(sizeof(float)*N*d);
naive_backward(Q,K,V,O_ref,dO,dQ_ref,dK_ref,dV_ref,N,d);fa2_backward(Q,K,V,O,dO,L,dQ,dK,dV,N,d,Br,Bc);
printf("[BWD] dQ MAD=%.3e RMSE=%.3e | dK MAD=%.3e RMSE=%.3e | dV MAD=%.3e RMSE=%.3e
",maxad(dQ_ref,dQ,(size_t)N*d),rmse(dQ_ref,dQ,(size_t)N*d),maxad(dK_ref,dK,(size_t)N*d),rmse(dK_ref,dK,(size_t)N*d),maxad(dV_ref,dV,(size_t)N*d),rmse(dV_ref,dV,(size_t)N*d));
free(Q);free(K);free(V);free(O_ref);free(O);free(L);free(dO);free(dQ_ref);free(dK_ref);free(dV_ref);free(dQ);free(dK);free(dV);return 0;}
