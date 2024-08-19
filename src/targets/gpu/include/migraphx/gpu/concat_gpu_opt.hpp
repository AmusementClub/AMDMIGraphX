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
#ifndef MIGRAPHX_GUARD_RTGLIB_CONCAT_GPU_OPT_HPP
#define MIGRAPHX_GUARD_RTGLIB_CONCAT_GPU_OPT_HPP

#include <migraphx/op/concat.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/serialize.hpp>

namespace migraphx {
namespace gpu {

struct concat_gpu_optimization
{
    std::string allocate() const { return "hip::allocate"; }
    optional<migraphx::op::concat> get_concat(const migraphx::operation& op) const
    {
        if(op.name() != "gpu::precompile_op")
            return nullopt;
        auto r = from_value<operation>(op.to_value().at("op"));
        if(r.name() == "concat")
            return any_cast<migraphx::op::concat>(r);
        return nullopt;
    }
};

} // namespace gpu

} // namespace migraphx

#endif
