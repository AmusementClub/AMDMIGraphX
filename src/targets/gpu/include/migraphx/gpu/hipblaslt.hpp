/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_HIPBLASLT_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_HIPBLASLT_HPP
#include <migraphx/manage_ptr.hpp>
#include <migraphx/gpu/config.hpp>
#if MIGRAPHX_USE_HIPBLASLT
#include <hipblaslt/hipblaslt.h>
#define HIPBLASLT_WORKSPACE_SIZE (2 * 128 * 1024 * 1024)
#ifndef CHECK_HIPBLAS_ERROR
#define CHECK_HIPBLAS_ERROR(error)                    \
    if(error != HIPBLAS_STATUS_SUCCESS)               \
    {                                                 \
        fprintf(stderr,                               \
                "hipBLAS error: '%s'(%d) at %s:%d\n", \
                hipblasStatusToString(error),         \
                error,                                \
                __FILE__,                             \
                __LINE__);                            \
        exit(EXIT_FAILURE);                           \
    }
#endif
#endif // MIGRAPHX_USE_HIPBLASLT

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

#if MIGRAPHX_USE_HIPBLASLT
using hipblaslt_handle_ptr     = MIGRAPHX_MANAGE_PTR(hipblasLtHandle_t, hipblasLtDestroy);
using hipblaslt_workspace_ptr  = MIGRAPHX_MANAGE_PTR(void*, hipFree);
using hipblaslt_preference_ptr = MIGRAPHX_MANAGE_PTR(hipblasLtMatmulPreference_t,
                                                     hipblasLtMatmulPreferenceDestroy);

hipblaslt_handle_ptr create_hipblaslt_handle_ptr();
hipblaslt_handle_ptr create_hipblaslt_handle_ptr(hipStream_t s);
hipblaslt_preference_ptr create_hipblaslt_preference_ptr();
hipblaslt_workspace_ptr create_hipblaslt_workspace_ptr();
bool hipblaslt_supported();
#endif // MIGRAPHX_USE_HIPBLASLT

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_MIGRAPHLIB_HIPBLASLT_HPP