
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
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/json.hpp>
#include "models.hpp"
namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {
migraphx::program resnet50(unsigned batch) // NOLINT(readability-function-size)
{
    migraphx::program p;
    migraphx::module_ref mmain   = p.get_main_module();
    auto x_main_module_0         = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 0)));
    auto x_main_module_1         = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 1)));
    auto x_input_tensor_module_0 = mmain->add_parameter(
        "input_tensor:0", migraphx::shape{migraphx::shape::float_type, {batch, 3, 224, 224}});
    auto x_main_module_3 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 2));
    auto x_main_module_4 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 512, 3, 3}}, 3));
    auto x_main_module_5 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 4));
    auto x_main_module_6 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {2048, 512, 1, 1}}, 5));
    auto x_main_module_7 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 64, 1, 1}}, 6));
    auto x_main_module_8 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 7));
    auto x_main_module_9 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 8));
    auto x_main_module_10 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 2048, 1, 1}}, 9));
    auto x_main_module_11 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 64, 1, 1}}, 10));
    auto x_main_module_12 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 11));
    auto x_main_module_13 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 12));
    auto x_main_module_14 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 512, 3, 3}}, 13));
    auto x_main_module_15 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 64, 3, 3}}, 14));
    auto x_main_module_16 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 15));
    auto x_main_module_17 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 16));
    auto x_main_module_18 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {2048, 512, 1, 1}}, 17));
    auto x_main_module_19 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 64, 1, 1}}, 18));
    auto x_main_module_20 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 19));
    auto x_main_module_21 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 20));
    auto x_main_module_22 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 2048, 1, 1}}, 21));
    auto x_main_module_23 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 256, 1, 1}}, 22));
    auto x_main_module_24 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 23));
    auto x_main_module_25 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 24));
    auto x_main_module_26 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 512, 3, 3}}, 25));
    auto x_main_module_27 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 64, 3, 3}}, 26));
    auto x_main_module_28 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 27));
    auto x_main_module_29 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 28));
    auto x_main_module_30 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {2048, 512, 1, 1}}, 29));
    auto x_main_module_31 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 64, 1, 1}}, 30));
    auto x_main_module_32 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 31));
    auto x_main_module_33 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 32));
    auto x_main_module_34 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 256, 1, 1}}, 33));
    auto x_main_module_35 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 34));
    auto x_main_module_36 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 64, 3, 3}}, 35));
    auto x_main_module_37 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 36));
    auto x_main_module_38 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 37));
    auto x_main_module_39 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 38)));
    auto x_main_module_40 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 64, 1, 1}}, 39));
    auto x_main_module_41 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 40));
    auto x_main_module_42 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 41));
    auto x_main_module_43 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 42)));
    auto x_main_module_44 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 3, 3}}, 43));
    auto x_main_module_45 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 44));
    auto x_main_module_46 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 128, 1, 1}}, 45));
    auto x_main_module_47 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 46));
    auto x_main_module_48 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 512, 1, 1}}, 47));
    auto x_main_module_49 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 48));
    auto x_main_module_50 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 512, 1, 1}}, 49));
    auto x_main_module_51 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 50));
    auto x_main_module_52 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}}, 51));
    auto x_main_module_53 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 52));
    auto x_main_module_54 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 256, 1, 1}}, 53));
    auto x_main_module_55 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 54));
    auto x_main_module_56 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 1024, 1, 1}}, 55));
    auto x_main_module_57 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 56));
    auto x_main_module_58 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}}, 57));
    auto x_main_module_59 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 58));
    auto x_main_module_60 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 256, 1, 1}}, 59));
    auto x_main_module_61 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 60));
    auto x_main_module_62 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 1024, 1, 1}}, 61));
    auto x_main_module_63 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 62));
    auto x_main_module_64 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}}, 63));
    auto x_main_module_65 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 64));
    auto x_main_module_66 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 256, 1, 1}}, 65));
    auto x_main_module_67 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 66));
    auto x_main_module_68 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 256, 1, 1}}, 67));
    auto x_main_module_69 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 68));
    auto x_main_module_70 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 3, 3}}, 69));
    auto x_main_module_71 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 70));
    auto x_main_module_72 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 128, 1, 1}}, 71));
    auto x_main_module_73 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 72));
    auto x_main_module_74 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 512, 1, 1}}, 73));
    auto x_main_module_75 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 74));
    auto x_main_module_76 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 3, 3}}, 75));
    auto x_main_module_77 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 76));
    auto x_main_module_78 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 128, 1, 1}}, 77));
    auto x_main_module_79 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 78));
    auto x_main_module_80 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 512, 1, 1}}, 79));
    auto x_main_module_81 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 80));
    auto x_main_module_82 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 3, 3}}, 81));
    auto x_main_module_83 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 82));
    auto x_main_module_84 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 128, 1, 1}}, 83));
    auto x_main_module_85 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 84));
    auto x_main_module_86 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 512, 1, 1}}, 85));
    auto x_main_module_87 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 86));
    auto x_main_module_88 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 256, 1, 1}}, 87));
    auto x_main_module_89 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 3, 7, 7}}, 88));
    auto x_main_module_90 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1001}}, 89));
    auto x_main_module_91 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 90));
    auto x_main_module_92 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048, 1001}}, 91));
    auto x_main_module_93 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 1024, 1, 1}}, 92));
    auto x_main_module_94 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 93));
    auto x_main_module_95 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}}, 94));
    auto x_main_module_96 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 95));
    auto x_main_module_97 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 256, 1, 1}}, 96));
    auto x_main_module_98 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 97));
    auto x_main_module_99  = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 1024, 1, 1}}, 98));
    auto x_main_module_100 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 99));
    auto x_main_module_101 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}}, 100));
    auto x_main_module_102 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 101));
    auto x_main_module_103 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 256, 1, 1}}, 102));
    auto x_main_module_104 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 103));
    auto x_main_module_105 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 1024, 1, 1}}, 104));
    auto x_main_module_106 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 105));
    auto x_main_module_107 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}}, 106));
    auto x_main_module_108 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 107));
    auto x_main_module_109 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 256, 1, 1}}, 108));
    auto x_main_module_110 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 109));
    auto x_main_module_111 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {2048, 1024, 1, 1}}, 110));
    auto x_main_module_112 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 111));
    auto x_main_module_113 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 1024, 1, 1}}, 112));
    auto x_main_module_114 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[3,3,3,3],padding_mode:0,stride:[2,"
                               "2],use_dynamic_same_auto_pad:0}"),
        x_input_tensor_module_0,
        x_main_module_89);
    auto x_main_module_115 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_39);
    auto x_main_module_116 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_37);
    auto x_main_module_117 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_41);
    auto x_main_module_118 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_43);
    auto x_main_module_119 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,112,112]}"), x_main_module_117);
    auto x_main_module_120 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_114, x_main_module_119);
    auto x_main_module_121 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_0);
    auto x_main_module_122 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_118, x_main_module_121);
    auto x_main_module_123 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_1);
    auto x_main_module_124 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_122, x_main_module_123);
    auto x_main_module_125 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,112,112]}"), x_main_module_124);
    auto x_main_module_126 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_120, x_main_module_125);
    auto x_main_module_127 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,112,112]}"), x_main_module_115);
    auto x_main_module_128 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_126, x_main_module_127);
    auto x_main_module_129 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,112,112]}"), x_main_module_116);
    auto x_main_module_130 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_128, x_main_module_129);
    auto x_main_module_131 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_130);
    auto x_main_module_132 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:1,padding:[0,0,1,1],stride:[2,2]}"),
        x_main_module_131);
    auto x_main_module_133 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_132,
        x_main_module_11);
    auto x_main_module_134 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,64,56,56]}"), x_main_module_13);
    auto x_main_module_135 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_133, x_main_module_134);
    auto x_main_module_136 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_135);
    auto x_main_module_137 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_136,
        x_main_module_15);
    auto x_main_module_138 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,64,56,56]}"), x_main_module_17);
    auto x_main_module_139 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_137, x_main_module_138);
    auto x_main_module_140 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_139);
    auto x_main_module_141 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_140,
        x_main_module_19);
    auto x_main_module_142 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,256,56,56]}"), x_main_module_21);
    auto x_main_module_143 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_141, x_main_module_142);
    auto x_main_module_144 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_132,
        x_main_module_7);
    auto x_main_module_145 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,256,56,56]}"), x_main_module_9);
    auto x_main_module_146 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_144, x_main_module_145);
    auto x_main_module_147 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_143, x_main_module_146);
    auto x_main_module_148 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_147);
    auto x_main_module_149 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_148,
        x_main_module_23);
    auto x_main_module_150 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,64,56,56]}"), x_main_module_25);
    auto x_main_module_151 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_149, x_main_module_150);
    auto x_main_module_152 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_151);
    auto x_main_module_153 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_152,
        x_main_module_27);
    auto x_main_module_154 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,64,56,56]}"), x_main_module_29);
    auto x_main_module_155 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_153, x_main_module_154);
    auto x_main_module_156 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_155);
    auto x_main_module_157 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_156,
        x_main_module_31);
    auto x_main_module_158 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,256,56,56]}"), x_main_module_33);
    auto x_main_module_159 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_157, x_main_module_158);
    auto x_main_module_160 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_159, x_main_module_148);
    auto x_main_module_161 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_160);
    auto x_main_module_162 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_161,
        x_main_module_34);
    auto x_main_module_163 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,64,56,56]}"), x_main_module_35);
    auto x_main_module_164 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_162, x_main_module_163);
    auto x_main_module_165 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_164);
    auto x_main_module_166 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_165,
        x_main_module_36);
    auto x_main_module_167 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,64,56,56]}"), x_main_module_38);
    auto x_main_module_168 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_166, x_main_module_167);
    auto x_main_module_169 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_168);
    auto x_main_module_170 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_169,
        x_main_module_40);
    auto x_main_module_171 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,256,56,56]}"), x_main_module_42);
    auto x_main_module_172 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_170, x_main_module_171);
    auto x_main_module_173 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_172, x_main_module_161);
    auto x_main_module_174 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_173);
    auto x_main_module_175 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_174,
        x_main_module_68);
    auto x_main_module_176 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,128,56,56]}"), x_main_module_69);
    auto x_main_module_177 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_175, x_main_module_176);
    auto x_main_module_178 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_177);
    auto x_main_module_179 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[2,"
                               "2],use_dynamic_same_auto_pad:0}"),
        x_main_module_178,
        x_main_module_70);
    auto x_main_module_180 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,128,28,28]}"), x_main_module_71);
    auto x_main_module_181 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_179, x_main_module_180);
    auto x_main_module_182 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_181);
    auto x_main_module_183 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_182,
        x_main_module_72);
    auto x_main_module_184 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,512,28,28]}"), x_main_module_73);
    auto x_main_module_185 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_183, x_main_module_184);
    auto x_main_module_186 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[2,"
                               "2],use_dynamic_same_auto_pad:0}"),
        x_main_module_174,
        x_main_module_66);
    auto x_main_module_187 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,512,28,28]}"), x_main_module_67);
    auto x_main_module_188 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_186, x_main_module_187);
    auto x_main_module_189 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_185, x_main_module_188);
    auto x_main_module_190 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_189);
    auto x_main_module_191 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_190,
        x_main_module_74);
    auto x_main_module_192 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,128,28,28]}"), x_main_module_75);
    auto x_main_module_193 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_191, x_main_module_192);
    auto x_main_module_194 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_193);
    auto x_main_module_195 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_194,
        x_main_module_76);
    auto x_main_module_196 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,128,28,28]}"), x_main_module_77);
    auto x_main_module_197 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_195, x_main_module_196);
    auto x_main_module_198 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_197);
    auto x_main_module_199 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_198,
        x_main_module_78);
    auto x_main_module_200 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,512,28,28]}"), x_main_module_79);
    auto x_main_module_201 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_199, x_main_module_200);
    auto x_main_module_202 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_201, x_main_module_190);
    auto x_main_module_203 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_202);
    auto x_main_module_204 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_203,
        x_main_module_80);
    auto x_main_module_205 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,128,28,28]}"), x_main_module_81);
    auto x_main_module_206 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_204, x_main_module_205);
    auto x_main_module_207 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_206);
    auto x_main_module_208 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_207,
        x_main_module_82);
    auto x_main_module_209 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,128,28,28]}"), x_main_module_83);
    auto x_main_module_210 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_208, x_main_module_209);
    auto x_main_module_211 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_210);
    auto x_main_module_212 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_211,
        x_main_module_84);
    auto x_main_module_213 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,512,28,28]}"), x_main_module_85);
    auto x_main_module_214 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_212, x_main_module_213);
    auto x_main_module_215 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_214, x_main_module_203);
    auto x_main_module_216 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_215);
    auto x_main_module_217 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_216,
        x_main_module_86);
    auto x_main_module_218 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,128,28,28]}"), x_main_module_87);
    auto x_main_module_219 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_217, x_main_module_218);
    auto x_main_module_220 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_219);
    auto x_main_module_221 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_220,
        x_main_module_44);
    auto x_main_module_222 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,128,28,28]}"), x_main_module_45);
    auto x_main_module_223 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_221, x_main_module_222);
    auto x_main_module_224 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_223);
    auto x_main_module_225 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_224,
        x_main_module_46);
    auto x_main_module_226 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,512,28,28]}"), x_main_module_47);
    auto x_main_module_227 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_225, x_main_module_226);
    auto x_main_module_228 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_227, x_main_module_216);
    auto x_main_module_229 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_228);
    auto x_main_module_230 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_229,
        x_main_module_50);
    auto x_main_module_231 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,256,28,28]}"), x_main_module_51);
    auto x_main_module_232 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_230, x_main_module_231);
    auto x_main_module_233 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_232);
    auto x_main_module_234 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[2,"
                               "2],use_dynamic_same_auto_pad:0}"),
        x_main_module_233,
        x_main_module_52);
    auto x_main_module_235 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,256,14,14]}"), x_main_module_53);
    auto x_main_module_236 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_234, x_main_module_235);
    auto x_main_module_237 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_236);
    auto x_main_module_238 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_237,
        x_main_module_54);
    auto x_main_module_239 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,1024,14,14]}"), x_main_module_55);
    auto x_main_module_240 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_238, x_main_module_239);
    auto x_main_module_241 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[2,"
                               "2],use_dynamic_same_auto_pad:0}"),
        x_main_module_229,
        x_main_module_48);
    auto x_main_module_242 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,1024,14,14]}"), x_main_module_49);
    auto x_main_module_243 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_241, x_main_module_242);
    auto x_main_module_244 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_240, x_main_module_243);
    auto x_main_module_245 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_244);
    auto x_main_module_246 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_245,
        x_main_module_56);
    auto x_main_module_247 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,256,14,14]}"), x_main_module_57);
    auto x_main_module_248 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_246, x_main_module_247);
    auto x_main_module_249 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_248);
    auto x_main_module_250 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_249,
        x_main_module_58);
    auto x_main_module_251 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,256,14,14]}"), x_main_module_59);
    auto x_main_module_252 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_250, x_main_module_251);
    auto x_main_module_253 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_252);
    auto x_main_module_254 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_253,
        x_main_module_60);
    auto x_main_module_255 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,1024,14,14]}"), x_main_module_61);
    auto x_main_module_256 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_254, x_main_module_255);
    auto x_main_module_257 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_256, x_main_module_245);
    auto x_main_module_258 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_257);
    auto x_main_module_259 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_258,
        x_main_module_62);
    auto x_main_module_260 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,256,14,14]}"), x_main_module_63);
    auto x_main_module_261 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_259, x_main_module_260);
    auto x_main_module_262 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_261);
    auto x_main_module_263 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_262,
        x_main_module_64);
    auto x_main_module_264 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,256,14,14]}"), x_main_module_65);
    auto x_main_module_265 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_263, x_main_module_264);
    auto x_main_module_266 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_265);
    auto x_main_module_267 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_266,
        x_main_module_88);
    auto x_main_module_268 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,1024,14,14]}"), x_main_module_91);
    auto x_main_module_269 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_267, x_main_module_268);
    auto x_main_module_270 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_269, x_main_module_258);
    auto x_main_module_271 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_270);
    auto x_main_module_272 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_271,
        x_main_module_93);
    auto x_main_module_273 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,256,14,14]}"), x_main_module_94);
    auto x_main_module_274 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_272, x_main_module_273);
    auto x_main_module_275 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_274);
    auto x_main_module_276 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_275,
        x_main_module_95);
    auto x_main_module_277 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,256,14,14]}"), x_main_module_96);
    auto x_main_module_278 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_276, x_main_module_277);
    auto x_main_module_279 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_278);
    auto x_main_module_280 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_279,
        x_main_module_97);
    auto x_main_module_281 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,1024,14,14]}"), x_main_module_98);
    auto x_main_module_282 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_280, x_main_module_281);
    auto x_main_module_283 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_282, x_main_module_271);
    auto x_main_module_284 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_283);
    auto x_main_module_285 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_284,
        x_main_module_99);
    auto x_main_module_286 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,256,14,14]}"), x_main_module_100);
    auto x_main_module_287 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_285, x_main_module_286);
    auto x_main_module_288 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_287);
    auto x_main_module_289 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_288,
        x_main_module_101);
    auto x_main_module_290 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,256,14,14]}"), x_main_module_102);
    auto x_main_module_291 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_289, x_main_module_290);
    auto x_main_module_292 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_291);
    auto x_main_module_293 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_292,
        x_main_module_103);
    auto x_main_module_294 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,1024,14,14]}"), x_main_module_104);
    auto x_main_module_295 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_293, x_main_module_294);
    auto x_main_module_296 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_295, x_main_module_284);
    auto x_main_module_297 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_296);
    auto x_main_module_298 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_297,
        x_main_module_105);
    auto x_main_module_299 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,256,14,14]}"), x_main_module_106);
    auto x_main_module_300 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_298, x_main_module_299);
    auto x_main_module_301 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_300);
    auto x_main_module_302 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_301,
        x_main_module_107);
    auto x_main_module_303 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,256,14,14]}"), x_main_module_108);
    auto x_main_module_304 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_302, x_main_module_303);
    auto x_main_module_305 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_304);
    auto x_main_module_306 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_305,
        x_main_module_109);
    auto x_main_module_307 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,1024,14,14]}"), x_main_module_110);
    auto x_main_module_308 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_306, x_main_module_307);
    auto x_main_module_309 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_308, x_main_module_297);
    auto x_main_module_310 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_309);
    auto x_main_module_311 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[2,"
                               "2],use_dynamic_same_auto_pad:0}"),
        x_main_module_310,
        x_main_module_111);
    auto x_main_module_312 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,2048,7,7]}"), x_main_module_112);
    auto x_main_module_313 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_311, x_main_module_312);
    auto x_main_module_314 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_310,
        x_main_module_113);
    auto x_main_module_315 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,512,14,14]}"), x_main_module_3);
    auto x_main_module_316 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_314, x_main_module_315);
    auto x_main_module_317 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_316);
    auto x_main_module_318 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[2,"
                               "2],use_dynamic_same_auto_pad:0}"),
        x_main_module_317,
        x_main_module_4);
    auto x_main_module_319 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,512,7,7]}"), x_main_module_5);
    auto x_main_module_320 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_318, x_main_module_319);
    auto x_main_module_321 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_320);
    auto x_main_module_322 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_321,
        x_main_module_6);
    auto x_main_module_323 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,2048,7,7]}"), x_main_module_8);
    auto x_main_module_324 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_322, x_main_module_323);
    auto x_main_module_325 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_324, x_main_module_313);
    auto x_main_module_326 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_325);
    auto x_main_module_327 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_326,
        x_main_module_10);
    auto x_main_module_328 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,512,7,7]}"), x_main_module_12);
    auto x_main_module_329 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_327, x_main_module_328);
    auto x_main_module_330 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_329);
    auto x_main_module_331 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_330,
        x_main_module_14);
    auto x_main_module_332 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,512,7,7]}"), x_main_module_16);
    auto x_main_module_333 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_331, x_main_module_332);
    auto x_main_module_334 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_333);
    auto x_main_module_335 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_334,
        x_main_module_18);
    auto x_main_module_336 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,2048,7,7]}"), x_main_module_20);
    auto x_main_module_337 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_335, x_main_module_336);
    auto x_main_module_338 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_337, x_main_module_326);
    auto x_main_module_339 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_338);
    auto x_main_module_340 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_339,
        x_main_module_22);
    auto x_main_module_341 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,512,7,7]}"), x_main_module_24);
    auto x_main_module_342 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_340, x_main_module_341);
    auto x_main_module_343 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_342);
    auto x_main_module_344 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_343,
        x_main_module_26);
    auto x_main_module_345 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,512,7,7]}"), x_main_module_28);
    auto x_main_module_346 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_344, x_main_module_345);
    auto x_main_module_347 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_346);
    auto x_main_module_348 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_347,
        x_main_module_30);
    auto x_main_module_349 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,2048,7,7]}"), x_main_module_32);
    auto x_main_module_350 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_348, x_main_module_349);
    auto x_main_module_351 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_350, x_main_module_339);
    auto x_main_module_352 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_351);
    auto x_main_module_353 = mmain->add_instruction(
        migraphx::make_json_op("reduce_mean", "{axes:[2,3]}"), x_main_module_352);
    auto x_main_module_354 = mmain->add_instruction(
        migraphx::make_json_op("reshape", "{dims:[-1,1,1,2048]}"), x_main_module_353);
    auto x_main_module_355 = mmain->add_instruction(
        migraphx::make_json_op("squeeze", "{axes:[1,2]}"), x_main_module_354);
    auto x_main_module_356 =
        mmain->add_instruction(migraphx::make_op("dot"), x_main_module_355, x_main_module_92);
    auto x_main_module_357 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1001]}"), x_main_module_90);
    auto x_main_module_358 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_356, x_main_module_357);
    auto x_main_module_359 =
        mmain->add_instruction(migraphx::make_op("identity"), x_main_module_358);
    auto x_main_module_360 =
        mmain->add_instruction(migraphx::make_json_op("softmax", "{axis:1}"), x_main_module_359);
    auto x_main_module_361 =
        mmain->add_instruction(migraphx::make_op("identity"), x_main_module_360);
    auto x_main_module_362 =
        mmain->add_instruction(migraphx::make_json_op("argmax", "{axis:1}"), x_main_module_359);
    auto x_main_module_363 =
        mmain->add_instruction(migraphx::make_json_op("squeeze", "{axes:[1]}"), x_main_module_362);
    auto x_main_module_364 =
        mmain->add_instruction(migraphx::make_op("identity"), x_main_module_363);
    mmain->add_return({x_main_module_364, x_main_module_361});

    return p;
}
} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx
