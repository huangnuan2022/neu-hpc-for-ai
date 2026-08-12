#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct XorShift64 {
    uint64_t state;

    explicit XorShift64(uint64_t seed) : state(seed ? seed : 88172645463393265ull) {}

    uint32_t next_u32() {
        uint64_t x = state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        state = x;
        return static_cast<uint32_t>(x >> 32);
    }

    float uniform() {
        return (static_cast<float>(next_u32()) / static_cast<float>(UINT32_MAX)) * 2.0f - 1.0f;
    }
};

struct Timer {
    Clock::time_point start;

    Timer() : start(Clock::now()) {}

    double elapsed_ms() const {
        auto end = Clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

int parse_int(int argc, char** argv, const char* flag, int default_value) {
    for (int i = 0; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], flag) == 0) {
            return std::atoi(argv[i + 1]);
        }
    }
    return default_value;
}

uint64_t parse_seed(int argc, char** argv, uint64_t default_value) {
    for (int i = 0; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--seed") == 0) {
            return static_cast<uint64_t>(std::strtoull(argv[i + 1], nullptr, 10));
        }
    }
    return default_value;
}

std::vector<float> random_vec(size_t n, XorShift64& rng, float scale = 1.0f) {
    std::vector<float> out(n);
    for (float& v : out) {
        v = rng.uniform() * scale;
    }
    return out;
}

float dot_row(const std::vector<float>& a, int a_row,
              const std::vector<float>& b, int b_row, int dim) {
    const size_t ao = static_cast<size_t>(a_row) * dim;
    const size_t bo = static_cast<size_t>(b_row) * dim;
    float acc = 0.0f;
    for (int d = 0; d < dim; ++d) {
        acc += a[ao + d] * b[bo + d];
    }
    return acc;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("max_abs_diff requires equal vector sizes");
    }
    double max_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    }
    return max_diff;
}

void attention_materialized_reference(const std::vector<float>& q,
                                      const std::vector<float>& k,
                                      const std::vector<float>& v,
                                      std::vector<float>& out,
                                      int seq, int dim) {
    std::vector<float> scores(static_cast<size_t>(seq) * seq);
    const float scale = 1.0f / std::sqrt(static_cast<float>(dim));

    for (int i = 0; i < seq; ++i) {
        float row_max = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < seq; ++j) {
            const float score = dot_row(q, i, k, j, dim) * scale;
            scores[static_cast<size_t>(i) * seq + j] = score;
            row_max = std::max(row_max, score);
        }

        double denom = 0.0;
        for (int j = 0; j < seq; ++j) {
            const double e = std::exp(static_cast<double>(scores[static_cast<size_t>(i) * seq + j] - row_max));
            scores[static_cast<size_t>(i) * seq + j] = static_cast<float>(e);
            denom += e;
        }

        for (int d = 0; d < dim; ++d) {
            double acc = 0.0;
            for (int j = 0; j < seq; ++j) {
                acc += static_cast<double>(scores[static_cast<size_t>(i) * seq + j]) *
                       static_cast<double>(v[static_cast<size_t>(j) * dim + d]);
            }
            out[static_cast<size_t>(i) * dim + d] = static_cast<float>(acc / denom);
        }
    }
}

void streaming_update_one_query(const std::vector<float>& q,
                                const std::vector<float>& k,
                                const std::vector<float>& v,
                                std::vector<float>& out,
                                int query, int key_begin, int key_end,
                                int seq, int dim,
                                float& m, float& l,
                                std::vector<float>& acc) {
    (void)seq;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dim));

    for (int key = key_begin; key < key_end; ++key) {
        const float score = dot_row(q, query, k, key, dim) * scale;
        const float m_new = std::max(m, score);
        const float exp_m = std::exp(m - m_new);
        const float exp_s = std::exp(score - m_new);
        const float l_new = l * exp_m + exp_s;

        for (int d = 0; d < dim; ++d) {
            acc[d] = acc[d] * exp_m + v[static_cast<size_t>(key) * dim + d] * exp_s;
        }

        m = m_new;
        l = l_new;
    }

    for (int d = 0; d < dim; ++d) {
        out[static_cast<size_t>(query) * dim + d] = acc[d] / l;
    }
}

void attention_streaming_single(const std::vector<float>& q,
                                const std::vector<float>& k,
                                const std::vector<float>& v,
                                std::vector<float>& out,
                                int seq, int dim) {
    std::vector<float> acc(dim);
    for (int query = 0; query < seq; ++query) {
        std::fill(acc.begin(), acc.end(), 0.0f);
        float m = -std::numeric_limits<float>::infinity();
        float l = 0.0f;
        streaming_update_one_query(q, k, v, out, query, 0, seq, seq, dim, m, l, acc);
    }
}

void attention_ring_sequence_parallel(const std::vector<float>& q,
                                      const std::vector<float>& k,
                                      const std::vector<float>& v,
                                      std::vector<float>& out,
                                      int seq, int dim, int shards) {
    if (seq % shards != 0) {
        throw std::runtime_error("seq must be divisible by shards");
    }

    const int local_seq = seq / shards;
    std::vector<float> acc(dim);

    for (int rank = 0; rank < shards; ++rank) {
        const int q_begin = rank * local_seq;
        const int q_end = q_begin + local_seq;

        for (int query = q_begin; query < q_end; ++query) {
            std::fill(acc.begin(), acc.end(), 0.0f);
            float m = -std::numeric_limits<float>::infinity();
            float l = 0.0f;

            for (int step = 0; step < shards; ++step) {
                const int key_shard = (rank - step + shards) % shards;
                const int key_begin = key_shard * local_seq;
                const int key_end = key_begin + local_seq;
                streaming_update_one_query(q, k, v, out, query, key_begin, key_end,
                                           seq, dim, m, l, acc);
            }
        }
    }
}

struct AttentionRun {
    double naive_ms = 0.0;
    double streaming_ms = 0.0;
    double ring_ms = 0.0;
    double max_diff_streaming = 0.0;
    double max_diff_ring = 0.0;
};

AttentionRun run_attention_once(int seq, int dim, int shards, uint64_t seed) {
    XorShift64 rng(seed);
    auto q = random_vec(static_cast<size_t>(seq) * dim, rng, 0.25f);
    auto k = random_vec(static_cast<size_t>(seq) * dim, rng, 0.25f);
    auto v = random_vec(static_cast<size_t>(seq) * dim, rng, 0.25f);

    std::vector<float> ref(static_cast<size_t>(seq) * dim);
    std::vector<float> streaming(static_cast<size_t>(seq) * dim);
    std::vector<float> ring(static_cast<size_t>(seq) * dim);

    AttentionRun run;
    {
        Timer timer;
        attention_materialized_reference(q, k, v, ref, seq, dim);
        run.naive_ms = timer.elapsed_ms();
    }
    {
        Timer timer;
        attention_streaming_single(q, k, v, streaming, seq, dim);
        run.streaming_ms = timer.elapsed_ms();
    }
    {
        Timer timer;
        attention_ring_sequence_parallel(q, k, v, ring, seq, dim, shards);
        run.ring_ms = timer.elapsed_ms();
    }

    run.max_diff_streaming = max_abs_diff(ref, streaming);
    run.max_diff_ring = max_abs_diff(ref, ring);
    return run;
}

float gelu(float x) {
    return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

int top1_expert(const std::vector<float>& x, const std::vector<float>& wg,
                int token, int dim, int experts) {
    int best = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    for (int e = 0; e < experts; ++e) {
        float score = 0.0f;
        for (int d = 0; d < dim; ++d) {
            score += x[static_cast<size_t>(token) * dim + d] * wg[static_cast<size_t>(e) * dim + d];
        }
        if (score > best_score) {
            best_score = score;
            best = e;
        }
    }
    return best;
}

void expert_forward(const std::vector<float>& x,
                    const std::vector<float>& w1,
                    const std::vector<float>& w2,
                    std::vector<float>& out,
                    int token, int expert,
                    int dim, int hidden) {
    std::vector<float> tmp(hidden, 0.0f);
    for (int h = 0; h < hidden; ++h) {
        float acc = 0.0f;
        for (int d = 0; d < dim; ++d) {
            const size_t w1_idx = (static_cast<size_t>(expert) * dim + d) * hidden + h;
            acc += x[static_cast<size_t>(token) * dim + d] * w1[w1_idx];
        }
        tmp[h] = gelu(acc);
    }

    for (int d = 0; d < dim; ++d) {
        float acc = 0.0f;
        for (int h = 0; h < hidden; ++h) {
            const size_t w2_idx = (static_cast<size_t>(expert) * hidden + h) * dim + d;
            acc += tmp[h] * w2[w2_idx];
        }
        out[static_cast<size_t>(token) * dim + d] = acc;
    }
}

void moe_dense_reference(const std::vector<float>& x,
                         const std::vector<float>& wg,
                         const std::vector<float>& w1,
                         const std::vector<float>& w2,
                         std::vector<float>& out,
                         std::vector<int>& chosen,
                         int tokens, int dim, int hidden, int experts) {
    for (int t = 0; t < tokens; ++t) {
        const int expert = top1_expert(x, wg, t, dim, experts);
        chosen[t] = expert;
        expert_forward(x, w1, w2, out, t, expert, dim, hidden);
    }
}

void moe_routed_simulator(const std::vector<float>& x,
                          const std::vector<float>& wg,
                          const std::vector<float>& w1,
                          const std::vector<float>& w2,
                          std::vector<float>& out,
                          std::vector<int>& chosen,
                          std::vector<int>& route_counts,
                          int tokens, int dim, int hidden, int experts, int shards) {
    if (experts % shards != 0) {
        throw std::runtime_error("experts must be divisible by shards");
    }

    const int local_experts = experts / shards;
    std::vector<std::vector<int>> buckets(shards);

    for (int t = 0; t < tokens; ++t) {
        const int expert = top1_expert(x, wg, t, dim, experts);
        const int owner = expert / local_experts;
        chosen[t] = expert;
        route_counts[owner] += 1;
        buckets[owner].push_back(t);
    }

    for (int owner = 0; owner < shards; ++owner) {
        for (int token : buckets[owner]) {
            expert_forward(x, w1, w2, out, token, chosen[token], dim, hidden);
        }
    }
}

struct MoeRun {
    double dense_ms = 0.0;
    double routed_ms = 0.0;
    double max_diff = 0.0;
    int max_route_count = 0;
    int min_route_count = 0;
};

MoeRun run_moe_once(int tokens, int dim, int hidden, int experts, int shards, uint64_t seed) {
    XorShift64 rng(seed);
    auto x = random_vec(static_cast<size_t>(tokens) * dim, rng, 0.2f);
    auto wg = random_vec(static_cast<size_t>(experts) * dim, rng, 0.2f);
    auto w1 = random_vec(static_cast<size_t>(experts) * dim * hidden, rng, 0.05f);
    auto w2 = random_vec(static_cast<size_t>(experts) * hidden * dim, rng, 0.05f);

    std::vector<float> ref(static_cast<size_t>(tokens) * dim);
    std::vector<float> routed(static_cast<size_t>(tokens) * dim);
    std::vector<int> chosen_ref(tokens);
    std::vector<int> chosen_routed(tokens);
    std::vector<int> route_counts(shards, 0);

    MoeRun run;
    {
        Timer timer;
        moe_dense_reference(x, wg, w1, w2, ref, chosen_ref, tokens, dim, hidden, experts);
        run.dense_ms = timer.elapsed_ms();
    }
    {
        Timer timer;
        moe_routed_simulator(x, wg, w1, w2, routed, chosen_routed, route_counts,
                             tokens, dim, hidden, experts, shards);
        run.routed_ms = timer.elapsed_ms();
    }

    run.max_diff = max_abs_diff(ref, routed);
    run.max_route_count = *std::max_element(route_counts.begin(), route_counts.end());
    run.min_route_count = *std::min_element(route_counts.begin(), route_counts.end());
    return run;
}

double mean(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

uint64_t attention_score_matrix_bytes(int seq) {
    return static_cast<uint64_t>(seq) * static_cast<uint64_t>(seq) * sizeof(float);
}

uint64_t ring_working_bytes_per_shard(int seq, int dim, int shards) {
    const int local_seq = seq / shards;
    const uint64_t state = static_cast<uint64_t>(local_seq) * 2ull * sizeof(float);
    const uint64_t acc = static_cast<uint64_t>(local_seq) * static_cast<uint64_t>(dim) * sizeof(float);
    const uint64_t kv_pingpong = 2ull * static_cast<uint64_t>(local_seq) *
                                 static_cast<uint64_t>(dim) * sizeof(float);
    return state + acc + kv_pingpong;
}

int run_attention_cli(int argc, char** argv) {
    const int seq = parse_int(argc, argv, "--seq", 256);
    const int dim = parse_int(argc, argv, "--dim", 64);
    const int shards = parse_int(argc, argv, "--shards", 4);
    const int iters = parse_int(argc, argv, "--iters", 3);
    const uint64_t seed = parse_seed(argc, argv, 42);

    if (seq <= 0 || dim <= 0 || shards <= 0 || iters <= 0) {
        throw std::runtime_error("seq, dim, shards, and iters must be positive");
    }
    if (seq % shards != 0) {
        throw std::runtime_error("seq must be divisible by shards");
    }

    std::vector<double> naive_ms;
    std::vector<double> streaming_ms;
    std::vector<double> ring_ms;
    double max_diff_streaming = 0.0;
    double max_diff_ring = 0.0;

    for (int i = 0; i < iters; ++i) {
        AttentionRun run = run_attention_once(seq, dim, shards, seed + static_cast<uint64_t>(i));
        naive_ms.push_back(run.naive_ms);
        streaming_ms.push_back(run.streaming_ms);
        ring_ms.push_back(run.ring_ms);
        max_diff_streaming = std::max(max_diff_streaming, run.max_diff_streaming);
        max_diff_ring = std::max(max_diff_ring, run.max_diff_ring);
    }

    const uint64_t naive_bytes = attention_score_matrix_bytes(seq);
    const uint64_t ring_bytes = ring_working_bytes_per_shard(seq, dim, shards);
    const double memory_reduction_pct = 100.0 * (1.0 - static_cast<double>(ring_bytes) /
                                                       static_cast<double>(naive_bytes));

    std::cout << std::fixed << std::setprecision(6)
              << "{"
              << "\"mode\":\"attention\","
              << "\"seq\":" << seq << ","
              << "\"dim\":" << dim << ","
              << "\"shards\":" << shards << ","
              << "\"iters\":" << iters << ","
              << "\"naive_ms\":" << mean(naive_ms) << ","
              << "\"streaming_ms\":" << mean(streaming_ms) << ","
              << "\"ring_ms\":" << mean(ring_ms) << ","
              << "\"streaming_vs_naive_speedup\":" << mean(naive_ms) / mean(streaming_ms) << ","
              << "\"ring_cpu_sim_vs_naive_speedup\":" << mean(naive_ms) / mean(ring_ms) << ","
              << "\"max_diff_streaming\":" << max_diff_streaming << ","
              << "\"max_diff_ring\":" << max_diff_ring << ","
              << "\"naive_score_matrix_bytes\":" << naive_bytes << ","
              << "\"ring_working_bytes_per_shard\":" << ring_bytes << ","
              << "\"estimated_memory_reduction_pct\":" << memory_reduction_pct
              << "}" << std::endl;
    return 0;
}

int run_moe_cli(int argc, char** argv) {
    const int tokens = parse_int(argc, argv, "--tokens", 256);
    const int dim = parse_int(argc, argv, "--dim", 32);
    const int hidden = parse_int(argc, argv, "--hidden", 64);
    const int experts = parse_int(argc, argv, "--experts", 8);
    const int shards = parse_int(argc, argv, "--shards", 4);
    const int iters = parse_int(argc, argv, "--iters", 3);
    const uint64_t seed = parse_seed(argc, argv, 1337);

    if (tokens <= 0 || dim <= 0 || hidden <= 0 || experts <= 0 || shards <= 0 || iters <= 0) {
        throw std::runtime_error("tokens, dim, hidden, experts, shards, and iters must be positive");
    }
    if (experts % shards != 0) {
        throw std::runtime_error("experts must be divisible by shards");
    }

    std::vector<double> dense_ms;
    std::vector<double> routed_ms;
    double max_diff = 0.0;
    int max_route_count = 0;
    int min_route_count = tokens;

    for (int i = 0; i < iters; ++i) {
        MoeRun run = run_moe_once(tokens, dim, hidden, experts, shards,
                                  seed + static_cast<uint64_t>(i));
        dense_ms.push_back(run.dense_ms);
        routed_ms.push_back(run.routed_ms);
        max_diff = std::max(max_diff, run.max_diff);
        max_route_count = std::max(max_route_count, run.max_route_count);
        min_route_count = std::min(min_route_count, run.min_route_count);
    }

    const uint64_t token_payload_bytes = static_cast<uint64_t>(tokens) *
                                         static_cast<uint64_t>(dim) * sizeof(float);
    const double route_imbalance = static_cast<double>(max_route_count) /
                                   std::max(1, min_route_count);

    std::cout << std::fixed << std::setprecision(6)
              << "{"
              << "\"mode\":\"moe\","
              << "\"tokens\":" << tokens << ","
              << "\"dim\":" << dim << ","
              << "\"hidden\":" << hidden << ","
              << "\"experts\":" << experts << ","
              << "\"shards\":" << shards << ","
              << "\"iters\":" << iters << ","
              << "\"dense_ms\":" << mean(dense_ms) << ","
              << "\"routed_ms\":" << mean(routed_ms) << ","
              << "\"routed_cpu_sim_vs_dense_speedup\":" << mean(dense_ms) / mean(routed_ms) << ","
              << "\"max_diff\":" << max_diff << ","
              << "\"token_payload_bytes\":" << token_payload_bytes << ","
              << "\"route_imbalance\":" << route_imbalance << ","
              << "\"max_route_count\":" << max_route_count << ","
              << "\"min_route_count\":" << min_route_count
              << "}" << std::endl;
    return 0;
}

void print_usage(const char* argv0) {
    std::cerr << "Usage:\n"
              << "  " << argv0 << " attention --seq 512 --dim 64 --shards 4 --iters 3\n"
              << "  " << argv0 << " moe --tokens 512 --dim 32 --hidden 64 --experts 8 --shards 4 --iters 3\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            print_usage(argv[0]);
            return 2;
        }

        const std::string mode(argv[1]);
        if (mode == "attention") {
            return run_attention_cli(argc - 1, argv + 1);
        }
        if (mode == "moe") {
            return run_moe_cli(argc - 1, argv + 1);
        }

        print_usage(argv[0]);
        return 2;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << std::endl;
        return 1;
    }
}
