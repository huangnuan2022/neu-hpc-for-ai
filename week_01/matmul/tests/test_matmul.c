#include "mat.h"      // Mat type
#include "matmul.h"   // matmul_serial / matmul_threads declarations
#include "util.h"     // now_sec, fill_*, array_almost_equal
#include <stdio.h>    // printf
#include <stdlib.h>   // exit


// --- Helper to run one small case and validate both serial & threaded ---
static int run_small_case(size_t M, size_t K, size_t N, size_t threads) {
    printf("Small test: A=%zux%zu, B=%zux%zu -> C=%zux%zu ... ",
           M, K, K, N, M, N);
    fflush(stdout);

    Mat A = mat_alloc(M, K);
    Mat B = mat_alloc(K, N);
    Mat C1 = mat_alloc(M, N);
    Mat C2 = mat_alloc(M, N);

    if (!A.data || !B.data || !C1.data || !C2.data) {
        fprintf(stderr, "Allocation failed in small test.\n");
        return 0;
    }

    // Deterministic content for easy manual checking
    fill_sequential(A.data, M*K, 0.5f, 0.1f);
    fill_sequential(B.data, K*N, -0.3f, 0.2f);

    // Serial reference
    matmul_serial(&A, &B, &C1);

    // Threaded result
    matmul_threads(&A, &B, &C2, threads);

    int ok = array_almost_equal(C1.data, C2.data, M*N, 1e-5f, 1e-6f);

    printf("%s\n", ok ? "OK" : "FAIL");

    mat_free(&A); mat_free(&B); mat_free(&C1); mat_free(&C2);
    return ok;
}

static void run_small_suite(void) {
    int all = 1;
    // Corner cases
    all &= run_small_case(1,1,1, 4);  // A=1x1, B=1x1
    all &= run_small_case(1,1,5, 4);  // A=1x1, B=1x5
    all &= run_small_case(2,1,3, 4);  // A=2x1, B=1x3
    all &= run_small_case(2,2,2, 4);  // A=2x2, B=2x2
    // And others: non-square / mismatched aspect ratios
    all &= run_small_case(3,4,2, 4);
    all &= run_small_case(4,2,5, 4);

    if (!all) {
        fprintf(stderr, "Some small tests FAILED.\n");
        // Exit with non-zero so CI/graders catch it
        exit(1);
    }
}

// --- Performance benchmark on large matrices ---
static void run_perf(size_t M, size_t K, size_t N) {
    printf("\n=== Performance: A=%zux%zu, B=%zux%zu (C=%zux%zu) ===\n", M, K, K, N, M, N);

    Mat A = mat_alloc(M, K);
    Mat B = mat_alloc(K, N);
    Mat C = mat_alloc(M, N);
    Mat Ref = mat_alloc(M, N);

    if (!A.data || !B.data || !C.data || !Ref.data) {
        fprintf(stderr, "Allocation failed in perf test.\n");
        exit(1);
    }

    // Fixed seed → reproducible performance & correctness
    fill_random(A.data, M*K, 123);
    fill_random(B.data, K*N, 456);

    // Warm-up to stabilize caches/PLT
    matmul_serial(&A, &B, &Ref);

    // Serial baseline
    double t0 = now_sec();
    matmul_serial(&A, &B, &Ref);
    double t1 = now_sec();
    double t_serial = t1 - t0;
    printf("Serial: %.3f s  (baseline)\n", t_serial);

    const size_t thread_list[] = {1, 4, 16, 32, 64, 128};
    printf("%8s %12s %10s\n", "Threads", "Time(s)", "Speedup");

    for (size_t i = 0; i < sizeof(thread_list)/sizeof(thread_list[0]); ++i) {
        size_t T = thread_list[i];

        double s0 = now_sec();
        matmul_threads(&A, &B, &C, T);
        double s1 = now_sec();
        double tt = s1 - s0;

        // Validate threaded result against serial reference
        if (!array_almost_equal(Ref.data, C.data, M*N, 1e-5f, 1e-6f)) {
            fprintf(stderr, "Thread=%zu result mismatch!\n", T);
            exit(1);
        }

        printf("%8zu %12.3f %10.2f\n", T, tt, t_serial / tt);
    }

    mat_free(&A); mat_free(&B); mat_free(&C); mat_free(&Ref);
}

int main(void) {
    printf("== Matrix Multiplication Assignment ==\n");

    // 1) Small corner-case tests (functional correctness)
    run_small_suite();

    // 2) Large matrices for measurable speedup (adjust if needed)
    // Memory estimate: 3 * M * N * 4 bytes. For M=K=N=2048 → ~48 MB total.
    // Increase to 3072 or 4096 on desktops/servers for clearer scaling.
    const size_t M = 2048, K = 2048, N = 2048;
    run_perf(M, K, N);

    printf("\nAll done.\n");
    return 0;
}
