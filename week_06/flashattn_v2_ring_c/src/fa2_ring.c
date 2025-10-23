// src/fa2_ring.c
// Minimal, pure‑C (CPU) ring‑style FlashAttention‑2 forward pass

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>

#ifdef _OPENMP
  #include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline void* xaligned_alloc(size_t align, size_t bytes){
    void* p=NULL;
#if defined(_MSC_VER)
    p = _aligned_malloc(bytes, align);
    if(!p){ fprintf(stderr, "alloc failed\\n"); exit(1); }
#else
    if(posix_memalign(&p, align, bytes)!=0){
        fprintf(stderr, "posix_memalign failed\\n");
        exit(1);
    }
#endif
    return p;
}

static inline float frand(uint64_t* s){
    // xorshift64*
    uint64_t x = *s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *s = x;
    uint64_t r = x * UINT64_C(2685821657736338717);
    // map to [-1,1]
    return (float)((r >> 40) / (double)(1ULL<<24)) * 2.0f - 1.0f;
}

static inline double tnow(){
#if defined(_OPENMP)
    return omp_get_wtime();
#else
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
#endif
}

// ------------------------------- math helpers --------------------------------

static inline void zero_f32(float* __restrict a, size_t n){
    memset(a, 0, n*sizeof(float));
}

static inline float dot_f32(const float* __restrict x, const float* __restrict y, int d){
    float s=0.f;
    for(int i=0;i<d;i++) s += x[i]*y[i];
    return s;
}

// ------------------------ naive dense attention (oracle) ---------------------
// y = softmax( Q K^T / sqrt(d) ) V
static void naive_attn(const float* Q, const float* K, const float* V,
                       int S, int d, float* __restrict Y)
{
    const float scale = 1.0f / sqrtf((float)d);
    float* scores = (float*)xaligned_alloc(64, (size_t)S * sizeof(float));

    for(int i=0;i<S;i++){
        // compute all logits s_j
        float m = -INFINITY;
        const float* qi = Q + (size_t)i*d;
        for(int j=0;j<S;j++){
            float s = scale * dot_f32(qi, K + (size_t)j*d, d);
            scores[j] = s;
            if(s>m) m=s;
        }
        // softmax row and multiply by V
        float denom=0.f;
        for(int j=0;j<S;j++){
            scores[j] = expf(scores[j]-m);
            denom += scores[j];
        }
        for(int k=0;k<d;k++){
            float acc=0.f;
            for(int j=0;j<S;j++){
                acc += scores[j]*V[(size_t)j*d+k];
            }
            Y[(size_t)i*d+k] = acc/denom;
        }
    }
    free(scores);
}

// ------------------- ring FA2 forward (single head, CPU) --------------------
// We split sequence S into N blocks of equal size Br=S/N.
// For device i: query block Q_i (Br×d), and we iterate over KV blocks
// j=(i-t mod N), applying the online normalizer per query row.
//
// At the end, O_i (numerator) is divided by l_i (denominator) row-wise
// to produce the final output.
//
static void attn_block_update(const float* __restrict Qb, int Br,
                              const float* __restrict Kb,
                              const float* __restrict Vb, int Bc,
                              int d,
                              float* __restrict Ob,
                              float* __restrict m,  // per-row running max
                              float* __restrict l)  // per-row running denom
{
    const float scale = 1.0f / sqrtf((float)d);

    // For each query row r in this block
    for(int r=0;r<Br;r++){
        const float* q = Qb + (size_t)r*d;

        // 1) pass: row-wise block max
        float mb = -INFINITY;
        for(int c=0;c<Bc;c++){
            const float* k = Kb + (size_t)c*d;
            float s = scale * dot_f32(q,k,d);
            if(s>mb) mb=s;
        }

        // 2) pass: accumulate local numerator and denom
        float lb = 0.f;
        // temp accumulator for the V-weighted sum
        // use heap to keep stack small for large d
        float* w = (float*)alloca((size_t)d*sizeof(float));
        for(int k=0;k<d;k++) w[k]=0.f;

        for(int c=0;c<Bc;c++){
            const float* k = Kb + (size_t)c*d;
            const float* v = Vb + (size_t)c*d;
            float s = scale * dot_f32(q,k,d);
            float e = expf(s - mb);
            lb += e;
            for(int k=0;k<d;k++) w[k] += e * v[k];
        }

        // 3) online merge with running (m[r], l[r], O)
        float mprev = m[r];
        float mnew  = (mb > mprev) ? mb : mprev;
        float alpha = expf(mprev - mnew);
        float beta  = expf(mb    - mnew);
        l[r] = alpha * l[r] + beta * lb;

        float* o = Ob + (size_t)r*d;
        for(int k=0;k<d;k++){
            o[k] = alpha * o[k] + beta * w[k];
        }
    }
}

static void ring_fa2_forward(const float* Q, const float* K, const float* V,
                             int S, int d, int N,
                             float* __restrict Y)
{
    if(S % N != 0){
        fprintf(stderr, "S must be divisible by N (got S=%d N=%d)\\n", S, N);
        exit(1);
    }
    const int Br = S / N;   // query block size
    const int Bc = Br;      // KV block size (kept equal here)

    // Per-device buffers: numerator O (Br×d), row max m (Br), row denom l (Br)
    float** O = (float**)xaligned_alloc(64, N*sizeof(float*));
    float** m = (float**)xaligned_alloc(64, N*sizeof(float*));
    float** l = (float**)xaligned_alloc(64, N*sizeof(float*));
    for(int i=0;i<N;i++){
        O[i] = (float*)xaligned_alloc(64, (size_t)Br*d*sizeof(float));
        m[i] = (float*)xaligned_alloc(64, (size_t)Br*sizeof(float));
        l[i] = (float*)xaligned_alloc(64, (size_t)Br*sizeof(float));
        zero_f32(O[i], (size_t)Br*d);
        for(int r=0;r<Br;r++){ m[i][r] = -INFINITY; l[i][r] = 0.f; }
    }

    // Ring: N steps
    for(int t=0;t<N;t++){
        // Parallelize across devices if OpenMP is available.
        #pragma omp parallel for schedule(static) if(N>1)
        for(int i=0;i<N;i++){
            int qidx = i;
            int kvidx = (i - t + N) % N;
            const float* Qb = Q + (size_t)qidx*Br*d;
            const float* Kb = K + (size_t)kvidx*Bc*d;
            const float* Vb = V + (size_t)kvidx*Bc*d;
            attn_block_update(Qb, Br, Kb, Vb, Bc, d, O[i], m[i], l[i]);
        }
    }

    // Finalize: row-wise divide by denominator
    for(int i=0;i<N;i++){
        float* Oi = O[i];
        float* li = l[i];
        for(int r=0;r<Br;r++){
            float inv = 1.0f / (li[r] + 1e-20f);
            for(int k=0;k<d;k++){
                Y[(size_t)(i*Br + r)*d + k] = Oi[(size_t)r*d + k] * inv;
            }
        }
    }

    for(int i=0;i<N;i++){ free(O[i]); free(m[i]); free(l[i]); }
    free(O); free(m); free(l);
}

// ------------------------------ driver / test --------------------------------

int main(int argc, char** argv){
    int S=1024, d=64, N=4;
    uint64_t seed=42;

    if(argc>=2) S    = atoi(argv[1]);
    if(argc>=3) d    = atoi(argv[2]);
    if(argc>=4) N    = atoi(argv[3]);
    if(argc>=5) seed = (uint64_t)strtoull(argv[4], NULL, 10);

    if(N<1 || N>8){
        fprintf(stderr, "N must be in [1,8]\\n");
        return 1;
    }
    if(S % N != 0){
        fprintf(stderr, "ERROR: S (%d) must be divisible by N (%d).\\n", S, N);
        return 1;
    }

    printf("flashattn-v2 ring (pure C, CPU)  S=%d d=%d N=%d seed=%llu\\n",
           S, d, N, (unsigned long long)seed);

    float* Q = (float*)xaligned_alloc(64, (size_t)S*d*sizeof(float));
    float* K = (float*)xaligned_alloc(64, (size_t)S*d*sizeof(float));
    float* V = (float*)xaligned_alloc(64, (size_t)S*d*sizeof(float));

    // Fill with reproducible randoms
    uint64_t s = seed ? seed : 1;
    for(size_t i=0;i<(size_t)S*d;i++){ Q[i]=frand(&s); K[i]=frand(&s); V[i]=frand(&s); }

    float* Y_ring  = (float*)xaligned_alloc(64, (size_t)S*d*sizeof(float));
    float* Y_naive = (float*)xaligned_alloc(64, (size_t)S*d*sizeof(float));

    double t0 = tnow();
    ring_fa2_forward(Q,K,V,S,d,N,Y_ring);
    double t1 = tnow();

    // Small sizes for the oracle are fine. For very large S, this is O(S^2).
    double t2 = tnow();
    naive_attn(Q,K,V,S,d,Y_naive);
    double t3 = tnow();

    // compare
    double max_abs=0.0, mae=0.0;
    for(size_t i=0;i<(size_t)S*d;i++){
        double diff = fabs((double)Y_ring[i] - (double)Y_naive[i]);
        if(diff>max_abs) max_abs=diff;
        mae += diff;
    }
    mae /= (double)(S*d);

    printf("Correctness vs naive: max|Δ| = %.6g, MAE = %.6g\\n", max_abs, mae);
    printf("Timing: ring = %.3f ms, naive = %.3f ms (not comparable for big S; naive is O(S^2))\\n",
           1e3*(t1-t0), 1e3*(t3-t2));

#ifdef _OPENMP
    printf("OpenMP: enabled with %d thread(s)\\n", omp_get_max_threads());
#else
    printf("OpenMP: disabled\\n");
#endif

    free(Q); free(K); free(V); free(Y_ring); free(Y_naive);
    return (max_abs < 2e-5 ? 0 : 2);
}
