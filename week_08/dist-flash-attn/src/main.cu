#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "flash_attn.h"
#include "utils.h"

namespace {

struct Options {
    int seq = 1024;
    int dim = 64;
    int gpus = 1;
    int warmup = 5;
    int iterations = 20;
    int seed = 42;
    bool overlap = true;
    std::string case_dir;
};

int parse_positive(const char* value, const char* name)
{
    const int parsed = std::atoi(value);
    if (parsed <= 0) {
        throw std::runtime_error(std::string(name) + " must be positive");
    }
    return parsed;
}

Options parse_options(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (argument == "--seq") {
            options.seq = parse_positive(value("--seq"), "seq");
        } else if (argument == "--dim") {
            options.dim = parse_positive(value("--dim"), "dim");
        } else if (argument == "--gpus") {
            options.gpus = parse_positive(value("--gpus"), "gpus");
        } else if (argument == "--warmup") {
            options.warmup = parse_positive(value("--warmup"), "warmup");
        } else if (argument == "--iterations") {
            options.iterations = parse_positive(value("--iterations"), "iterations");
        } else if (argument == "--seed") {
            options.seed = std::atoi(value("--seed"));
        } else if (argument == "--case-dir") {
            options.case_dir = value("--case-dir");
        } else if (argument == "--no-overlap") {
            options.overlap = false;
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: " << argv[0]
                << " [--seq N] [--dim N] [--gpus 1|2|4] [--warmup N]"
                << " [--iterations N] [--seed N] [--case-dir PATH] [--no-overlap]\n";
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    if (options.gpus > 8 || options.seq % options.gpus != 0) {
        throw std::runtime_error("gpus must be <= 8 and divide seq exactly");
    }
    return options;
}

std::vector<float> read_f32(const std::string& path, size_t elements)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("could not open " + path);
    }
    const std::streamsize expected = static_cast<std::streamsize>(elements * sizeof(float));
    if (input.tellg() != expected) {
        throw std::runtime_error("unexpected byte count in " + path);
    }
    input.seekg(0);
    std::vector<float> values(elements);
    if (!input.read(reinterpret_cast<char*>(values.data()), expected)) {
        throw std::runtime_error("could not read " + path);
    }
    return values;
}

void load_metadata(const std::string& directory, int& seq, int& dim)
{
    std::ifstream input(directory + "/meta.txt");
    if (!input) {
        throw std::runtime_error("could not open PyTorch fixture metadata");
    }
    std::string line;
    while (std::getline(input, line)) {
        const size_t separator = line.find('=');
        if (separator == std::string::npos) {
            continue;
        }
        const std::string key = line.substr(0, separator);
        if (key == "seq") {
            seq = std::stoi(line.substr(separator + 1));
        } else if (key == "dim") {
            dim = std::stoi(line.substr(separator + 1));
        }
    }
}

std::vector<float> random_values(size_t elements, int seed)
{
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
    std::vector<float> values(elements);
    for (float& value : values) {
        value = distribution(generator);
    }
    return values;
}

double percentile(std::vector<double> values, double quantile)
{
    std::sort(values.begin(), values.end());
    const double position = (values.size() - 1) * quantile;
    const size_t lower = static_cast<size_t>(std::floor(position));
    const size_t upper = static_cast<size_t>(std::ceil(position));
    if (lower == upper) {
        return values[lower];
    }
    const double fraction = position - lower;
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

struct ErrorMetrics {
    double max_absolute = 0.0;
    double max_relative = 0.0;
};

ErrorMetrics compare(const std::vector<float>& actual, const std::vector<float>& expected)
{
    if (actual.size() != expected.size()) {
        throw std::runtime_error("cannot compare outputs with different sizes");
    }
    ErrorMetrics metrics;
    for (size_t index = 0; index < actual.size(); ++index) {
        const double absolute = std::abs(
            static_cast<double>(actual[index]) - static_cast<double>(expected[index]));
        const double denominator = std::max(std::abs(static_cast<double>(expected[index])), 1.0e-8);
        metrics.max_absolute = std::max(metrics.max_absolute, absolute);
        metrics.max_relative = std::max(metrics.max_relative, absolute / denominator);
    }
    return metrics;
}

struct DeviceBuffers {
    std::vector<float*> q;
    std::vector<float*> k;
    std::vector<float*> v;
    std::vector<float*> out;
};

DeviceBuffers allocate_shards(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    int seq,
    int dim,
    int gpus)
{
    DeviceBuffers buffers;
    buffers.q.resize(gpus);
    buffers.k.resize(gpus);
    buffers.v.resize(gpus);
    buffers.out.resize(gpus);
    const size_t local_elements = static_cast<size_t>(seq / gpus) * dim;
    const size_t local_bytes = local_elements * sizeof(float);
    for (int gpu = 0; gpu < gpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(gpu));
        CHECK_CUDA(cudaMalloc(&buffers.q[gpu], local_bytes));
        CHECK_CUDA(cudaMalloc(&buffers.k[gpu], local_bytes));
        CHECK_CUDA(cudaMalloc(&buffers.v[gpu], local_bytes));
        CHECK_CUDA(cudaMalloc(&buffers.out[gpu], local_bytes));
        const size_t offset = static_cast<size_t>(gpu) * local_elements;
        CHECK_CUDA(cudaMemcpy(
            buffers.q[gpu], q.data() + offset, local_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(
            buffers.k[gpu], k.data() + offset, local_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(
            buffers.v[gpu], v.data() + offset, local_bytes, cudaMemcpyHostToDevice));
    }
    return buffers;
}

void free_shards(DeviceBuffers& buffers)
{
    for (size_t gpu = 0; gpu < buffers.q.size(); ++gpu) {
        CHECK_CUDA(cudaSetDevice(static_cast<int>(gpu)));
        CHECK_CUDA(cudaFree(buffers.q[gpu]));
        CHECK_CUDA(cudaFree(buffers.k[gpu]));
        CHECK_CUDA(cudaFree(buffers.v[gpu]));
        CHECK_CUDA(cudaFree(buffers.out[gpu]));
    }
}

std::vector<float> gather(const DeviceBuffers& buffers, int seq, int dim)
{
    const int gpus = static_cast<int>(buffers.out.size());
    const size_t local_elements = static_cast<size_t>(seq / gpus) * dim;
    const size_t local_bytes = local_elements * sizeof(float);
    std::vector<float> output(static_cast<size_t>(seq) * dim);
    for (int gpu = 0; gpu < gpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(gpu));
        CHECK_CUDA(cudaMemcpy(
            output.data() + static_cast<size_t>(gpu) * local_elements,
            buffers.out[gpu], local_bytes, cudaMemcpyDeviceToHost));
    }
    return output;
}

std::vector<double> benchmark(
    FlashAttnWorkspace* workspace,
    DeviceBuffers& buffers,
    int warmup,
    int iterations)
{
    for (int iteration = 0; iteration < warmup; ++iteration) {
        flash_attn_workspace_prepare_kv(
            workspace, buffers.k.data(), buffers.v.data());
        flash_attn_workspace_forward(workspace, buffers.q.data(), buffers.out.data());
    }
    std::vector<double> samples;
    samples.reserve(iterations);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        flash_attn_workspace_prepare_kv(
            workspace, buffers.k.data(), buffers.v.data());
        samples.push_back(flash_attn_workspace_forward(
            workspace, buffers.q.data(), buffers.out.data()));
    }
    return samples;
}

void print_samples(const std::vector<double>& samples)
{
    std::cout << '[';
    for (size_t index = 0; index < samples.size(); ++index) {
        if (index > 0) {
            std::cout << ',';
        }
        std::cout << samples[index];
    }
    std::cout << ']';
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        Options options = parse_options(argc, argv);
        std::string reference_name = "custom_single_gpu";
        std::vector<float> reference;
        if (!options.case_dir.empty()) {
            load_metadata(options.case_dir, options.seq, options.dim);
            reference_name = "pytorch_sdpa";
        }

        const size_t elements = static_cast<size_t>(options.seq) * options.dim;
        std::vector<float> q;
        std::vector<float> k;
        std::vector<float> v;
        if (!options.case_dir.empty()) {
            q = read_f32(options.case_dir + "/q.f32", elements);
            k = read_f32(options.case_dir + "/k.f32", elements);
            v = read_f32(options.case_dir + "/v.f32", elements);
            reference = read_f32(options.case_dir + "/reference.f32", elements);
        } else {
            q = random_values(elements, options.seed);
            k = random_values(elements, options.seed + 1);
            v = random_values(elements, options.seed + 2);
        }

        FlashAttnConfig single_config{options.seq, options.dim, 1, 0};
        DeviceBuffers single_buffers = allocate_shards(q, k, v, options.seq, options.dim, 1);
        FlashAttnWorkspace* single_workspace = flash_attn_workspace_create(&single_config);
        const std::vector<double> single_samples = benchmark(
            single_workspace, single_buffers, options.warmup, options.iterations);
        const std::vector<float> single_output = gather(single_buffers, options.seq, options.dim);

        FlashAttnConfig ring_config{
            options.seq, options.dim, options.gpus, options.overlap ? 1 : 0};
        DeviceBuffers ring_buffers = allocate_shards(q, k, v, options.seq, options.dim, options.gpus);
        FlashAttnWorkspace* ring_workspace = flash_attn_workspace_create(&ring_config);
        const std::vector<double> ring_samples = benchmark(
            ring_workspace, ring_buffers, options.warmup, options.iterations);
        const std::vector<float> ring_output = gather(ring_buffers, options.seq, options.dim);

        if (reference.empty()) {
            reference = single_output;
        }
        const ErrorMetrics single_error = compare(single_output, reference);
        const ErrorMetrics ring_error = compare(ring_output, reference);
        const double single_median = percentile(single_samples, 0.50);
        const double ring_median = percentile(ring_samples, 0.50);
        const size_t score_bytes = flash_attn_full_score_matrix_bytes(&ring_config);
        const size_t minimal_bytes = flash_attn_minimal_state_bytes_per_gpu(&ring_config);
        const size_t explicit_bytes = flash_attn_workspace_bytes_per_gpu(ring_workspace);
        std::vector<size_t> measured_deltas(options.gpus);
        for (int gpu = 0; gpu < options.gpus; ++gpu) {
            measured_deltas[gpu] = flash_attn_workspace_measured_delta_bytes(
                ring_workspace, gpu);
        }

        std::cout << std::fixed << std::setprecision(6)
                  << "{\"schema_version\":\"1.0\""
                  << ",\"seq\":" << options.seq
                  << ",\"dim\":" << options.dim
                  << ",\"gpus\":" << options.gpus
                  << ",\"warmup\":" << options.warmup
                  << ",\"iterations\":" << options.iterations
                  << ",\"overlap_kv_rotation\":" << (options.overlap ? "true" : "false")
                  << ",\"reference\":\"" << reference_name << "\""
                  << ",\"single_gpu_samples_ms\":";
        print_samples(single_samples);
        std::cout << ",\"ring_samples_ms\":";
        print_samples(ring_samples);
        std::cout << ",\"single_gpu_median_ms\":" << single_median
                  << ",\"single_gpu_p95_ms\":" << percentile(single_samples, 0.95)
                  << ",\"ring_median_ms\":" << ring_median
                  << ",\"ring_p95_ms\":" << percentile(ring_samples, 0.95)
                  << ",\"speedup_vs_single_gpu\":" << single_median / ring_median
                  << ",\"single_max_abs_error\":" << single_error.max_absolute
                  << ",\"single_max_rel_error\":" << single_error.max_relative
                  << ",\"ring_max_abs_error\":" << ring_error.max_absolute
                  << ",\"ring_max_rel_error\":" << ring_error.max_relative
                  << ",\"full_score_matrix_bytes\":" << score_bytes
                  << ",\"minimal_ring_state_bytes_per_gpu\":" << minimal_bytes
                  << ",\"explicit_workspace_bytes_per_gpu\":" << explicit_bytes
                  << ",\"estimated_minimal_state_reduction_pct\":"
                  << 100.0 * (1.0 - static_cast<double>(minimal_bytes) / score_bytes)
                  << ",\"explicit_workspace_reduction_pct\":"
                  << 100.0 * (1.0 - static_cast<double>(explicit_bytes) / score_bytes)
                  << ",\"measured_cuda_allocation_delta_bytes_per_gpu\":[";
        for (size_t gpu = 0; gpu < measured_deltas.size(); ++gpu) {
            if (gpu > 0) {
                std::cout << ',';
            }
            std::cout << measured_deltas[gpu];
        }
        std::cout << "]}\n";

        flash_attn_workspace_destroy(ring_workspace);
        flash_attn_workspace_destroy(single_workspace);
        free_shards(ring_buffers);
        free_shards(single_buffers);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
