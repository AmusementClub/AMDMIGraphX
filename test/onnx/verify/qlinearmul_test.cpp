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

#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(qlinearmul_test)
{
    // github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearMul
    migraphx::program p = read_onnx("qlinearmul_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape a{migraphx::shape::uint8_type, {64}};
    std::vector<uint8_t> data_a = {0,   2,   4,   6,   8,   10,  12,  14,  16,  18,  20,  22,  24,
                                   26,  28,  30,  32,  34,  36,  38,  40,  42,  44,  46,  48,  50,
                                   52,  54,  56,  58,  60,  62,  64,  66,  68,  70,  72,  74,  76,
                                   78,  80,  82,  84,  86,  88,  90,  92,  94,  96,  98,  100, 102,
                                   104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126};

    migraphx::shape b{migraphx::shape::uint8_type, {64}};
    std::vector<uint8_t> data_b = {128, 126, 124, 122, 120, 118, 116, 114, 112, 110, 108, 106, 104,
                                   102, 100, 98,  96,  94,  92,  90,  88,  86,  84,  82,  80,  78,
                                   76,  74,  72,  70,  68,  66,  64,  62,  60,  58,  56,  54,  52,
                                   50,  48,  46,  44,  42,  40,  38,  36,  34,  32,  30,  28,  26,
                                   24,  22,  20,  18,  16,  14,  12,  10,  8,   6,   4,   2};

    migraphx::parameter_map pp;
    pp["A"]     = migraphx::argument(a, data_a.data());
    pp["B"]     = migraphx::argument(b, data_b.data());
    auto result = p.eval(pp).back();

    std::vector<uint8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<uint8_t> gold = {100, 111, 122, 132, 142, 151, 160, 169, 177, 185, 192, 199, 206,
                                 212, 218, 223, 228, 233, 237, 241, 244, 247, 250, 252, 254, 255,
                                 255, 255, 255, 255, 255, 255, 254, 252, 250, 247, 244, 241, 237,
                                 233, 228, 223, 218, 212, 206, 199, 192, 185, 177, 169, 160, 151,
                                 142, 132, 122, 111, 100, 89,  77,  65,  52,  39,  26,  12};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
