#pragma once

#include "ggml-backend.h"

// GGML internal header

#ifdef __cplusplus
extern "C" {
#endif

// Always declare - implementation will return nullptr if AMX not supported
ggml_backend_buffer_type_t ggml_backend_amx_buffer_type(void);

#ifdef __cplusplus
}
#endif
