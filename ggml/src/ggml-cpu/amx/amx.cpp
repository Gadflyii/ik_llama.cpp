#include "amx.h"
#include "common.h"
#include "mmq.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "../traits.h"

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include <cstdlib>
#include <cstring>
#include <memory>

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)

// AMX type_traits
namespace ggml::cpu::amx {
class tensor_traits : public ggml::cpu::tensor_traits {
    bool work_size(int /* n_threads */, const struct ggml_tensor * op, size_t & size) override {
        size = ggml_backend_amx_desired_wsize(op);
        return true;
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        if (op->op == GGML_OP_MUL_MAT) {
            ggml_backend_amx_mul_mat((const struct ggml_compute_params *)params, op);
            return true;
        }
        return false;
    }
};

static ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, struct ggml_tensor *) {
    static tensor_traits traits;
    return &traits;
}
}  // namespace ggml::cpu::amx

// AMX buffer interface
// Compatibility wrappers for our buffer interface
static const char * GGML_CALL ggml_backend_amx_buffer_get_name(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return "AMX";
}

static void GGML_CALL ggml_backend_amx_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    free(buffer->context);
}

static void * GGML_CALL ggml_backend_amx_buffer_get_base(ggml_backend_buffer_t buffer) {
    return (void *) (buffer->context);
}

// Wrapper: our interface returns void, upstream returns ggml_status
static void GGML_CALL ggml_backend_amx_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    tensor->extra = (void *) ggml::cpu::amx::get_tensor_traits(buffer, tensor);
    GGML_UNUSED(buffer);
    // Note: ignoring ggml_status return value for our interface compatibility
}

static void GGML_CALL ggml_backend_amx_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                                  uint8_t value, size_t offset, size_t size) {
    memset((char *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

// Forward declare mirror buffer structure for NUMA replication
#if defined(__linux__)
#define GGML_NUMA_MAX_NODES 8
#define GGML_MIRROR_BUFFER_MAGIC 0x4D49524E  // "MIRN" in hex
struct ggml_numa_mirror_buffer {
    uint32_t magic;                             // magic number for reliable identification
    uint32_t n_replicas;
    uint32_t active_nodes[GGML_NUMA_MAX_NODES];
    void *   replicas[GGML_NUMA_MAX_NODES];
    size_t   size;
    void *   original_base;
    bool     read_only;
};
#endif

static void GGML_CALL ggml_backend_amx_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                               const void * data, size_t offset, size_t size) {
#if defined(__linux__)
    // Check if this is a mirror buffer using magic number
    if (buffer->context) {
        struct ggml_numa_mirror_buffer * mirror = (struct ggml_numa_mirror_buffer *) buffer->context;
        // Verify magic number to ensure this is actually a mirror buffer
        // Only replicate if this is a read-only buffer (model weights)
        if (mirror->magic == GGML_MIRROR_BUFFER_MAGIC && mirror->n_replicas > 1 && mirror->read_only) {
            // This is a read-only mirror buffer - replicate to all nodes
            // Get the buffer base which is the first replica
            void * buffer_base = mirror->original_base ? mirror->original_base : mirror->replicas[mirror->active_nodes[0]];
            size_t tensor_offset = (char *)tensor->data - (char *)buffer_base;

            for (uint32_t i = 0; i < mirror->n_replicas; i++) {
                uint32_t node = mirror->active_nodes[i];
                // Temporarily update tensor->data to point to this replica
                void * original_data = tensor->data;
                tensor->data = (char *)mirror->replicas[node] + tensor_offset;

                // Convert/copy to this replica
                if (qtype_has_amx_kernels(tensor->type)) {
                    ggml_backend_amx_convert_weight(tensor, data, offset, size);
                } else {
                    memcpy((char *) tensor->data + offset, data, size);
                }

                // Restore original pointer
                tensor->data = original_data;
            }
            return;
        }
    }
#endif

    // Regular (non-mirror) buffer
    if (qtype_has_amx_kernels(tensor->type)) {
        ggml_backend_amx_convert_weight(tensor, data, offset, size);
    } else {
        memcpy((char *) tensor->data + offset, data, size);
    }

    GGML_UNUSED(buffer);
}

/*
// need to figure what we need to do with buffer->extra.
static void ggml_backend_amx_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(!qtype_has_amx_kernels(tensor->type));
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_amx_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        if (qtype_has_amx_kernels(src->type)) {
            ggml_backend_amx_convert_weight(dst, src->data, 0, ggml_nbytes(dst));
        } else {
            memcpy(dst->data, src->data, ggml_nbytes(src));
        }
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}
*/

static void GGML_CALL ggml_backend_amx_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

// Buffer interface struct - reordered to match our ggml_backend_buffer_i definition
static ggml_backend_buffer_i ggml_backend_amx_buffer_interface = {
    /* .get_name        = */ ggml_backend_amx_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_amx_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_amx_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_amx_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_amx_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_amx_buffer_set_tensor,
    /* .get_tensor      = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_amx_buffer_clear,
    /* .reset           = */ nullptr,
};

static const char * ggml_backend_amx_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "AMX";

    GGML_UNUSED(buft);
}

#if defined(__linux__)
// Forward declaration for NUMA mirror support
extern "C" enum ggml_numa_strategy ggml_get_numa_strategy(void);
#endif

static ggml_backend_buffer_t ggml_backend_amx_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
#if defined(__linux__)
    // Check if NUMA mirror mode is active at allocation time
    if (ggml_get_numa_strategy() == GGML_NUMA_STRATEGY_MIRROR) {
        // Need to allocate mirror buffer and wrap it with AMX interface
        // Delegate to CPU buffer type which will create mirror
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);

        if (buffer == nullptr) {
            return nullptr;
        }

        // Override the buffer type to report as AMX
        buffer->buft = buft;

        // CRITICAL: Override init_tensor to set AMX traits so AMX kernels are used
        // Without this, tensors won't have AMX traits and will fall back to regular CPU ops
        buffer->iface.init_tensor = ggml_backend_amx_buffer_init_tensor;
        buffer->iface.set_tensor = ggml_backend_amx_buffer_set_tensor;

        return buffer;
    }
#endif

    void * data = ggml_aligned_malloc(size);
    if (data == NULL) {
        fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_amx_buffer_interface, data, size);
}

static size_t ggml_backend_amx_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

namespace ggml::cpu::amx {
class extra_buffer_type : ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {
        // handle only 2d gemm for now
        auto is_contiguous_2d = [](const struct ggml_tensor * t) {
            return ggml_is_contiguous(t) && t->ne[3] == 1 && t->ne[2] == 1;
        };

        if (op->op == GGML_OP_MUL_MAT && is_contiguous_2d(op->src[0]) &&  // src0 must be contiguous
            is_contiguous_2d(op->src[1]) &&                               // src1 must be contiguous
            op->src[0]->buffer && op->src[0]->buffer->buft == ggml_backend_amx_buffer_type() &&
            op->ne[0] % (TILE_N * 2) == 0 &&                              // out_features is 32x
            (qtype_has_amx_kernels(op->src[0]->type) || (op->src[0]->type == GGML_TYPE_F16))) {
            // src1 must be host buffer
            if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                return false;
            }
            // src1 must be float32
            if (op->src[1]->type == GGML_TYPE_F32) {
                return true;
            }
        }
        return false;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        if (op->op == GGML_OP_MUL_MAT && op->src[0]->buffer &&
            op->src[0]->buffer->buft == ggml_backend_amx_buffer_type()) {
            return (ggml::cpu::tensor_traits *) op->src[0]->extra;
        }

        return nullptr;
    }
};
}  // namespace ggml::cpu::amx

static size_t ggml_backend_amx_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_backend_amx_get_alloc_size(tensor);

    GGML_UNUSED(buft);
}

static bool ggml_backend_amx_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;  // AMX buffers are host-accessible and use CPU backend for compute
}

#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

static bool ggml_amx_init() {
#if defined(__linux__)
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        fprintf(stderr, "AMX is not ready to be used!\n");
        return false;
    }
    return true;
#elif defined(_WIN32)
    return true;
#else
    return false;
#endif
}

ggml_backend_buffer_type_t ggml_backend_amx_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_buffer_type_amx = {
        /* .iface = */ {
                        /* .get_name         = */ ggml_backend_amx_buffer_type_get_name,
                        /* .alloc_buffer     = */ ggml_backend_amx_buffer_type_alloc_buffer,
                        /* .get_alignment    = */ ggml_backend_amx_buffer_type_get_alignment,
                        /* .get_max_size     = */ nullptr,  // defaults to SIZE_MAX
                        /* .get_alloc_size   = */ ggml_backend_amx_buffer_type_get_alloc_size,
                        /* .is_host          = */ ggml_backend_amx_buffer_type_is_host,
                        },
        /* .context = */ new ggml::cpu::amx::extra_buffer_type(),
        // Note: Upstream has .device field, but ik_llama.cpp doesn't
        // We use .is_host = true to associate with CPU backend
    };

    if (!ggml_amx_init()) {
        return nullptr;
    }

    return &ggml_backend_buffer_type_amx;
}

#endif  // defined(__AMX_INT8__) && defined(__AVX512VNNI__)
