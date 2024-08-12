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
#ifndef MIGRAPHX_GUARD_OPERATORS_PACK_INT4_HPP
#define MIGRAPHX_GUARD_OPERATORS_PACK_INT4_HPP

#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <migraphx/check_shapes.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/value.hpp>
#include <migraphx/config.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/argument.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
struct pack_int4
{
    int64_t axis = -1;

    std::string name() const { return "pack_int4"; }

    value attributes() const
    {
        value normalize   = value::object{};
        normalize["axis"] = value::array{normalize_attribute::include_min};
        return {{"normalize_axes", normalize}};
    }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"));
    }

    migraphx::shape normalize_compute_shape(std::vector<migraphx::shape> inputs) const
    {
        check_shapes{inputs, *this}.same_dims().has(1);
        auto in_shape = inputs.front();
        if(in_shape.type() != migraphx::shape::int8_type and
           in_shape.type() != migraphx::shape::uint8_type)
        {
            MIGRAPHX_THROW("PACK_INT4: Only Int8 or Uint8 is supported for packing");
        }
        auto new_lens = in_shape.lens();
        if(new_lens[axis] % 2 != 0)
        {
            MIGRAPHX_THROW("PACK_INT4: Can not pack axis that has odd lengths");
        }
        new_lens[axis] /= 2;
        return {in_shape.type(), new_lens};
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        auto input    = args.front();
        auto in_shape = input.get_shape();

        argument result{output_shape};

        visit_all(result, input)([&](auto out, auto inp) {
            par_for(output_shape.elements(), [&](auto i) {
                using type = typename decltype(inp)::value_type;
                if constexpr(std::is_signed<type>{})
                {
                    const int8_t min_4bit  = -8; // clip min value
                    const int8_t max_4bit  = 7;  // clip max value
                    auto data_idx          = output_shape.multi(i);
                    auto in_data_multi_idx = data_idx;
                    in_data_multi_idx[axis] *= 2;
                    int8_t val1 = inp[in_data_multi_idx];
                    val1        = std::min(std::max(val1, min_4bit), max_4bit); // clip

                    in_data_multi_idx[axis] += 1;
                    int8_t val2 = inp[in_data_multi_idx];
                    val2        = std::min(std::max(val2, min_4bit), max_4bit); // clip

                    out[i] = (val2 << 4) | (val1 & 0xf); // pack
                }
                else
                {
                    const uint8_t max_4bit = 15; // clip max value
                    auto data_idx          = output_shape.multi(i);
                    auto in_data_multi_idx = data_idx;
                    in_data_multi_idx[axis] *= 2;
                    uint8_t val1 = inp[in_data_multi_idx];
                    val1         = std::min(val1, max_4bit); // clip

                    in_data_multi_idx[axis] += 1;
                    uint8_t val2 = inp[in_data_multi_idx];
                    val2         = std::min(val2, max_4bit); // clip

                    out[i] = (val2 << 4) | (val1 & 0xf); // pack
                }
            });
        });
        return result;
    }
};
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
