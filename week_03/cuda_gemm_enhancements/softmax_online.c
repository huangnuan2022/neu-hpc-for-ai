%%writefile softmax_online.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Online normalizer softmax (Algorithm 3, arXiv:1805.02867)
void softmax_online(const float* x, float* y, int V) {
    if (V<=0) return;
    float m = -INFINITY; // running max
    float d = 0.0f;      // running normalizer
    for (int j=0; j<V; ++j) {
        float xj = x[j];
        float mj = fmaxf(m, xj);
        float d_scaled = (m==-INFINITY) ? 0.0f : d * expf(m - mj);
        d = d_scaled + expf(xj - mj);
        m = mj;
    }
    for (int i=0; i<V; ++i) y[i] = expf(x[i] - m) / d;
}

// Two-pass safe softmax for reference
void softmax_ref(const float* x, float* y, int V) {
    float m = -INFINITY; for (int i=0;i<V;++i) if (x[i]>m) m=x[i];
    double s = 0.0; for (int i=0;i<V;++i) s += exp((double)x[i]-m);
    for (int i=0;i<V;++i) y[i] = (float)(exp((double)x[i]-m)/s);
}

static float max_abs_diff(const float* a,const float* b,int n){
    float m=0.f; for(int i=0;i<n;++i){ float d=fabsf(a[i]-b[i]); if(d>m)m=d; } return m;
}

int main(int argc, char** argv){
    int V = (argc>1)? atoi(argv[1]) : 4096;
    int trials = (argc>2)? atoi(argv[2]) : 64;
    unsigned seed = (argc>3)? (unsigned)atoi(argv[3]) : 42u;
    srand(seed);

    float* x = (float*)malloc(sizeof(float)*V);
    float* y1= (float*)malloc(sizeof(float)*V);
    float* y2= (float*)malloc(sizeof(float)*V);
    if(!x||!y1||!y2){ fprintf(stderr,"OOM\n"); return 1; }

    double worst=0.0, avg=0.0;
    for(int t=0;t<trials;++t){
        for(int i=0;i<V;++i){
            float r = (float)rand()/(float)RAND_MAX;    // [0,1]
            x[i] = 40.f*(2.f*r-1.f);                    // [-40, 40] range
        }
        softmax_online(x,y1,V);
        softmax_ref(x,y2,V);
        float mad = max_abs_diff(y1,y2,V);
        if (mad>worst) worst=mad;
        avg += mad;
    }
    avg/=trials;
    double s=0.0; for(int i=0;i<V;++i) s+=y1[i];

    printf("OK  V=%d trials=%d  max_abs_diff=%.3e  avg_abs_diff=%.3e  sum(y)=%.9f\n",
           V, trials, worst, avg, s);

    free(x); free(y1); free(y2);
    return 0;
}

