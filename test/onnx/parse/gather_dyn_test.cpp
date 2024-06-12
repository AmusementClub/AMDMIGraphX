/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <onnx_test.hpp>

TEST_CASE(gather_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {{1, 4}, {4, 4}, {5, 5}, {6, 6}}});
    auto l1 = mm->add_parameter(
        "indices", migraphx::shape{migraphx::shape::int32_type, {{1, 4}, {3, 3}, {4, 4}, {5, 5}}});
    auto cont_l0 = mm->add_instruction(migraphx::make_op("contiguous"), l0);
    auto cont_l1 = mm->add_instruction(migraphx::make_op("contiguous"), l1);

    int axis       = 1;
    auto gather_op = migraphx::make_op("gather", {{"axis", axis}});
    auto ret       = mm->add_instruction(gather_op, cont_l0, cont_l1);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 4};
    auto prog                     = read_onnx("gather_dyn_test.onnx", options);

    EXPECT(p == prog);
}
