/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>

struct test_conv_add_relu : verify_program<test_conv_add_relu>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto input =
            mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto weights =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto bias_literal = migraphx::literal{migraphx::shape{migraphx::shape::float_type, {4}},
                                              {2.0f, 2.0f, 2.0f, 2.0f}};
        auto bias         = mm->add_literal(bias_literal);
        auto conv         = mm->add_instruction(migraphx::make_op("convolution"), input, weights);
        auto bcast_bias   = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv->get_shape().lens()}}),
            bias);
        auto bias_add = mm->add_instruction(migraphx::make_op("add"), conv, bcast_bias);
        mm->add_instruction(migraphx::make_op("relu"), bias_add);
        return p;
    }
};

struct test_conv_add_relu_conv : verify_program<test_conv_add_relu_conv>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto input =
            mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 4, 3, 3}});
        auto weights =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::float_type, {4, 4, 1, 1}});
        auto bias_literal = migraphx::literal{migraphx::shape{migraphx::shape::float_type, {4}},
                                              {2.0f, 2.0f, 2.0f, 2.0f}};
        auto bias         = mm->add_literal(bias_literal);
        auto conv         = mm->add_instruction(migraphx::make_op("convolution"), input, weights);
        auto bcast_bias   = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv->get_shape().lens()}}),
            bias);
        auto bias_add = mm->add_instruction(migraphx::make_op("add"), conv, bcast_bias);
        auto relu     = mm->add_instruction(migraphx::make_op("relu"), bias_add);
        mm->add_instruction(migraphx::make_op("convolution"), relu, weights);
        return p;
    }
};
