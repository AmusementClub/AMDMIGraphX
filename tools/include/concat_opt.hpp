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
#ifndef MIGRAPHX_GUARD_CONCAT_OPT_HPP
#define MIGRAPHX_GUARD_CONCAT_OPT_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include <migraphx/operation.hpp>
#include <migraphx/op/concat.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

#ifdef DOXYGEN

/// An interface for target-dependent optimization for the concat instruction
struct concat_optimization
{
    /// A name of the target-dependent allocate operator
    std::string allocate() const;
    /// Return the target-independent concat operator
    optional<op::concat> get_concat(const operation& op) const;
};

#else

<%
    interface(
        'concat_optimization',
        virtual('allocate', returns = 'std::string', const = True),
        virtual(
            'get_concat', returns = 'optional<op::concat>', op = 'const operation&', const = True))
%>

#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
