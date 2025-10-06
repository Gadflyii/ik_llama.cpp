#include "ggml-cpu-traits.h"
#include "ggml.h"
#include <cstring>

// Virtual destructor implementation
ggml::cpu::tensor_traits::~tensor_traits() = default;
ggml::cpu::extra_buffer_type::~extra_buffer_type() = default;

// Global dispatch for custom compute operations
// Called from ggml.c before standard dispatch
bool ggml_cpu_extra_compute_forward(
    struct ggml_compute_params * params,
    struct ggml_tensor * op) {

    // For most operations, custom traits are stored on the weight tensor (src[0])
    // not the output tensor. Check src[0]->extra for operations with sources.
    struct ggml_tensor * traits_tensor = nullptr;

    if (op->src[0] && op->src[0]->extra) {
        traits_tensor = op->src[0];
    } else if (op->extra) {
        traits_tensor = op;
    }

    // Check if we found custom traits
    if (traits_tensor) {
        auto traits = (ggml::cpu::tensor_traits*)traits_tensor->extra;
        // Let traits handle the operation
        return traits->compute_forward(params, op);
    }

    return false;  // No custom compute, use default dispatch
}

// Global work size calculation for custom operations
bool ggml_cpu_extra_work_size(int n_threads, const struct ggml_tensor * op, size_t * size) {
    // Check for custom traits
    const struct ggml_tensor * traits_tensor = nullptr;

    if (op->src[0] && op->src[0]->extra) {
        traits_tensor = op->src[0];
    } else if (op->extra) {
        traits_tensor = op;
    }

    if (traits_tensor) {
        auto traits = (ggml::cpu::tensor_traits*)traits_tensor->extra;
        return traits->work_size(n_threads, op, *size);
    }

    return false;
}
