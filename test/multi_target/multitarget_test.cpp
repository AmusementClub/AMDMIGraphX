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
#include <iostream>
#include <set>
#include <unordered_set>
#include <vector>
#include <random>
#include <cmath>
#include <migraphx/program.hpp>
#include <migraphx/target.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/module.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/functional.hpp>
#include <basic_ops.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/generate_root_modules.hpp>
#include "migraphx/target_assignments.hpp"
#include "test.hpp"

// check if it is custom_op or run_on_module operator
bool has_target_attr(const migraphx::instruction& ins)
{
    return ins.get_operator().attributes().contains("target");
}

auto nonprefixed_ops()
{
    // ops without prefixes
    static std::unordered_set<std::string> op_map = {
        "select_module", "load", "if", "nonmaxsuppression", "multibroadcast"};
    return op_map;
}

bool is_compiled_gpu_module(const migraphx::module& m)
{
    return std::all_of(m.begin(), m.end(), [](auto ins) {
        auto ins_name = ins.name();
        if(not migraphx::starts_with(ins_name, "@"))
        {
            if(not migraphx::starts_with(ins_name, "gpu::") and
               not migraphx::starts_with(ins_name, "hip::") and
               not migraphx::starts_with(ins_name, "check_context") and
               not migraphx::contains(nonprefixed_ops(), ins_name) and not has_target_attr(ins))
            {
                return false;
            }
        }
        return true;
    });
}

bool is_compiled_fpga_module(const migraphx::module& m)
{
    return std::all_of(m.begin(), m.end(), [](auto ins) {
        auto ins_name = ins.name();
        if(not migraphx::starts_with(ins_name, "@"))
        {
            if(not migraphx::starts_with(ins_name, "fpga::") and
               not migraphx::starts_with(ins_name, "check_context") and
               not migraphx::contains(nonprefixed_ops(), ins_name) and not has_target_attr(ins))
            {
                return false;
            }
        }
        return true;
    });
}

bool is_compiled_cpu_module(const migraphx::module& m)
{
    return std::all_of(m.begin(), m.end(), [](auto ins) {
        auto ins_name = ins.name();
        if(not migraphx::starts_with(ins_name, "@"))
        {
            if(not migraphx::starts_with(ins_name, "cpu::") and
               not migraphx::starts_with(ins_name, "dnnl::") and
               not migraphx::starts_with(ins_name, "check_context") and not has_target_attr(ins) and
               not migraphx::contains(nonprefixed_ops(), ins_name))
            {
                return false;
            }
        }
        return true;
    });
}

bool is_compiled_ref_module(const migraphx::module& m)
{
    return std::all_of(m.begin(), m.end(), [](auto ins) {
        auto ins_name = ins.name();
        if(not migraphx::starts_with(ins_name, "@"))
        {
            if((not migraphx::starts_with(ins_name, "ref::") and
                not migraphx::starts_with(ins_name, "check_context") and
                not has_target_attr(ins)) and
               not migraphx::contains(nonprefixed_ops(), ins_name))
            {
                return false;
            }
        }
        return true;
    });
}

// NOLINT
bool check_compiled_program(const migraphx::program& p,
                            const std::vector<migraphx::target>& targets)
{
    auto mods           = p.get_modules();
    bool check_compiled = true;
    for(const auto* mod : mods)
    {
        for(const auto& ins : *mod)
        {
            if(ins.name() == "run_on_target")
            {
                auto* mod_input = ins.module_inputs().front();
                std::size_t target_id =
                    ins.get_operator().to_value()["target_id"].to<std::size_t>();
                auto target_name = targets.at(target_id).name();
                if(target_name == "gpu")
                    check_compiled &= is_compiled_gpu_module(*mod_input);
                else if(target_name == "cpu")
                    check_compiled &= is_compiled_cpu_module(*mod_input);
                else if(target_name == "fpga")
                    check_compiled &= is_compiled_fpga_module(*mod_input);
                else if(target_name == "ref")
                    check_compiled &= is_compiled_ref_module(*mod_input);
            }
        }
    }
    return check_compiled;
}

TEST_CASE(multitarget_compile_cpu_gpu)
{
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto s       = migraphx::shape{migraphx::shape::float_type, {8}};
    auto x_param = mm->add_parameter("x", s);
    auto y_param = mm->add_parameter("y", s);
    auto z_param = mm->add_parameter("z", s);
    auto cpu_ins = mm->add_instruction(migraphx::make_op("add"), x_param, y_param);
    auto gpu_ins = mm->add_instruction(migraphx::make_op("add"), cpu_ins, z_param);
    mm->add_return({gpu_ins});
    migraphx::target_assignments tass;
    tass.insert(tass.begin(), std::make_pair(cpu_ins, 1));
    tass.insert(tass.begin(), std::make_pair(gpu_ins, 0));
    migraphx::generate_root_modules(p, tass);
    migraphx::compile_options gpu_opts;
    gpu_opts.offload_copy = true;
    p.compile({migraphx::make_target("gpu"), migraphx::make_target("cpu")}, {gpu_opts});
    EXPECT(check_compiled_program(p, {migraphx::make_target("gpu"), migraphx::make_target("cpu")}));
    migraphx::parameter_map params;
    params["x"] = migraphx::fill_argument(s, 1);
    params["y"] = migraphx::fill_argument(s, 2);
    params["z"] = migraphx::fill_argument(s, 3);
    auto result = p.eval(params).back();
    auto gold   = migraphx::fill_argument(s, 6);
    EXPECT(gold == result);
}

TEST_CASE(single_target_multi_compile)
{
    migraphx::program p;
    migraphx::shape boxes_s{migraphx::shape::float_type, {1, 6, 4}};
    auto* mm         = p.get_main_module();
    auto boxes_param = mm->add_parameter("boxes", boxes_s);

    migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 6}};
    std::vector<float> scores_vec = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};
    auto scores_l                 = mm->add_literal(migraphx::literal(scores_s, scores_vec));
    auto max_out_l                = mm->add_literal(int64_t{4});
    auto iou_threshold            = mm->add_literal(0.5f);
    auto score_threshold          = mm->add_literal(0.0f);
    auto r                        = mm->add_instruction(
        migraphx::make_op("nonmaxsuppression",
                                                 {{"center_point_box", true}, {"use_dyn_output", true}}),
        boxes_param,
        scores_l,
        max_out_l,
        iou_threshold,
        score_threshold);
    mm->add_return({r});
    // do target assignments
    migraphx::target_assignments tass;
    tass.insert(tass.begin(), std::make_pair(r, 0));
    migraphx::generate_root_modules(p, tass);
    // compile using multi-target compilation path
    migraphx::compile_options gpu_opts;
    gpu_opts.offload_copy = true;
    // need to add "ref" to avoid ambigious call to "compile()"
    p.compile({migraphx::make_target("gpu"), migraphx::make_target("ref")}, {gpu_opts});
    EXPECT(check_compiled_program(p, {migraphx::make_target("gpu"), migraphx::make_target("ref")}));
    // eval
    migraphx::parameter_map params;
    std::vector<float> boxes_vec  = {0.5, 0.5,  1.0, 1.0, 0.5, 0.6,  1.0, 1.0, 0.5, 0.4,   1.0, 1.0,
                                     0.5, 10.5, 1.0, 1.0, 0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0};
    params["boxes"]               = migraphx::argument(boxes_s, boxes_vec.data());
    auto output                   = p.eval(params).back();
    std::vector<int64_t> gold_vec = {0, 0, 3, 0, 0, 0, 0, 0, 5};
    auto gold =
        migraphx::argument(migraphx::shape{migraphx::shape::int64_type, {3, 3}}, gold_vec.data());
    EXPECT(output == gold);
}

TEST_CASE(multitarget_compile_if_then_else)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape cond_s{migraphx::shape::bool_type};
    auto cond = mm->add_parameter("cond", cond_s);
    migraphx::shape ds{migraphx::shape::float_type, {2, 3}};
    auto x = mm->add_parameter("x", ds);
    auto y = mm->add_parameter("y", ds);

    auto* then_mod = p.create_module("if_gpu_mod");
    std::vector<float> data1(ds.elements(), 1);
    auto l1 = then_mod->add_literal(migraphx::literal(ds, data1));
    // auto gpu_x = then_mod->add_parameter("gpu_x", ds);
    auto a1 = then_mod->add_instruction(migraphx::make_op("add"), x, l1);
    then_mod->add_return({a1});

    auto* else_mod = p.create_module("else_cpu_mod");
    std::vector<float> data2(ds.elements(), 2);
    auto l2 = else_mod->add_literal(migraphx::literal(ds, data2));
    // auto cpu_y = else_mod->add_parameter("cpu_y", ds);
    auto a2 = else_mod->add_instruction(migraphx::make_op("mul"), y, l2);
    else_mod->add_return({a2});

    auto ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
    auto r   = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret);
    mm->add_return({r});
    migraphx::target_assignments tass;
    tass.insert(tass.begin(), std::make_pair(l1, 0));
    tass.insert(tass.begin(), std::make_pair(a1, 0));
    tass.insert(tass.begin(), std::make_pair(l2, 1));
    tass.insert(tass.begin(), std::make_pair(a2, 1));
    migraphx::generate_root_modules(p, tass);
    // compile
    migraphx::compile_options gpu_opts;
    gpu_opts.offload_copy = true;
    p.compile(
        {migraphx::make_target("gpu"), migraphx::make_target("cpu"), migraphx::make_target("ref")},
        {gpu_opts});
    EXPECT(check_compiled_program(p,
                                  {migraphx::make_target("gpu"),
                                   migraphx::make_target("cpu"),
                                   migraphx::make_target("ref")}));
    migraphx::parameter_map params;
    params["x"] = migraphx::fill_argument(ds, 2);
    params["y"] = migraphx::fill_argument(ds, 3);
    for(bool cond_val : {true, false})
    {
        params["cond"] = migraphx::argument(cond_s, &cond_val);
        auto result    = p.eval(params).back();
        auto gold      = migraphx::fill_argument(ds, (cond_val ? 3 : 6));
        EXPECT(gold == result);
    }
}

// TODO : FPGA compilation is broken right now, below test mentions fpga but doesn't compile for it
TEST_CASE(multitarget_compile_nested_if_then_else)
{
    std::unordered_map<std::size_t, std::size_t> counter_map = {{0, 0}, {1, 0}};
    migraphx::shape ds{migraphx::shape::float_type, {2, 3}};
    migraphx::target_assignments tass;
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape cond_s{migraphx::shape::bool_type};
    auto cond_0             = mm->add_parameter("cond_0", cond_s);
    auto cond_1             = mm->add_parameter("cond_1", cond_s);
    auto x                  = mm->add_parameter("x", ds);
    auto y                  = mm->add_parameter("y", ds);
    auto z                  = mm->add_parameter("z", ds);
    auto create_test_module = [&](migraphx::program& prog, std::size_t tid) {
        std::string mod_name =
            "target_" + std::to_string(tid) + "_" + std::to_string(counter_map[tid]++);
        auto* test_mod = prog.create_module(mod_name);
        std::vector<float> data(ds.elements(), -1);
        auto l1               = test_mod->add_literal(migraphx::literal(ds, data));
        auto test_mod_param_0 = test_mod->add_parameter(mod_name + "_param_0", ds);
        auto test_mod_param_1 = test_mod->add_parameter(mod_name + "_param_1", ds);
        auto test_mod_param_2 = test_mod->add_parameter(mod_name + "_param_2", ds);
        auto ins1 = test_mod->add_instruction(migraphx::make_op("add"), test_mod_param_0, l1);
        auto ins2 = test_mod->add_instruction(migraphx::make_op("mul"), ins1, test_mod_param_1);
        auto ins3 = test_mod->add_instruction(migraphx::make_op("sub"), ins2, test_mod_param_2);
        test_mod->add_return({ins3});
        tass.insert(tass.begin(), std::make_pair(ins1, tid));
        tass.insert(tass.begin(), std::make_pair(ins2, tid));
        tass.insert(tass.begin(), std::make_pair(ins3, tid));
        return test_mod;
    };

    auto* then_mod        = p.create_module("then_mod");
    auto then_mod_cond    = then_mod->add_parameter("then_mod_cond", cond_s);
    auto then_mod_param_0 = then_mod->add_parameter("then_mod_param_0", ds);
    auto then_mod_param_1 = then_mod->add_parameter("then_mod_param_1", ds);
    auto then_mod_param_2 = then_mod->add_parameter("then_mod_param_2", ds);
    auto then_mod_ref_ins =
        then_mod->add_instruction(migraphx::make_op("add"), then_mod_param_0, then_mod_param_1);
    tass.insert(tass.begin(), std::make_pair(then_mod_ref_ins, 3));
    auto then_mod_if =
        then_mod->add_instruction(migraphx::make_op("if"),
                                  {then_mod_cond,
                                   then_mod_param_0,
                                   then_mod_param_1,
                                   then_mod_param_2,
                                   then_mod_ref_ins,
                                   then_mod_param_1,
                                   then_mod_param_2},
                                  {create_test_module(p, 1), create_test_module(p, 0)});
    auto then_mod_if_0 =
        then_mod->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), then_mod_if);
    then_mod->add_return({then_mod_if_0});

    // create nested else_mod with multiple targets.
    // else_mod has one instruction that runs a module on "fpga" and another instruction that
    // creates nested modules using "If" that runs on "cpu" and "gpu"
    auto* else_mod        = p.create_module("else_mod");
    auto else_mod_cond    = else_mod->add_parameter("else_mod_cond", cond_s);
    auto else_mod_param_0 = else_mod->add_parameter("else_mod_param_0", ds);
    auto else_mod_param_1 = else_mod->add_parameter("else_mod_param_1", ds);
    auto else_mod_param_2 = else_mod->add_parameter("else_mod_param_2", ds);
    auto else_mod_fpga_ins =
        else_mod->add_instruction(migraphx::make_op("add"), else_mod_param_0, else_mod_param_2);
    tass.insert(tass.begin(), std::make_pair(else_mod_fpga_ins, 2));
    auto else_mod_if =
        else_mod->add_instruction(migraphx::make_op("if"),
                                  {else_mod_cond,
                                   else_mod_fpga_ins,
                                   else_mod_param_0,
                                   else_mod_param_1,
                                   else_mod_param_2,
                                   else_mod_param_1,
                                   else_mod_param_0},
                                  {create_test_module(p, 0), create_test_module(p, 1)});
    auto else_mod_if_0 =
        else_mod->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), else_mod_if);
    else_mod->add_return({else_mod_if_0});

    // Create nested and multi-target main module using "If"
    auto main_if_ins = mm->add_instruction(
        migraphx::make_op("if"), {cond_0, cond_1, x, y, z, cond_1, x, y, z}, {then_mod, else_mod});
    auto r = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), main_if_ins);
    mm->add_return({r});

    // compile
    migraphx::compile_options gpu_opts;
    gpu_opts.offload_copy = true;
    migraphx::generate_root_modules(p, tass);
    p.compile({migraphx::make_target("gpu"),
               migraphx::make_target("cpu"),
               migraphx::make_target("ref"),
               migraphx::make_target("ref")},
              {gpu_opts});
    EXPECT(check_compiled_program(p,
                                  {migraphx::make_target("gpu"),
                                   migraphx::make_target("cpu"),
                                   migraphx::make_target("ref"),
                                   migraphx::make_target("ref")}));
    // do evaluation using different conditions
    migraphx::parameter_map params;
    float x_i   = 2.0;
    float y_i   = 3.0;
    float z_i   = 4.0;
    params["x"] = migraphx::fill_argument(ds, x_i);
    params["y"] = migraphx::fill_argument(ds, y_i);
    params["z"] = migraphx::fill_argument(ds, z_i);
    // cover all paths with different combination of conditions
    std::vector<std::pair<bool, bool>> test_conds = {
        {true, true}, {true, false}, {false, true}, {false, false}};
    for(auto [cond_val_0, cond_val_1] : test_conds)
    {
        params["cond_0"] = migraphx::argument(cond_s, &cond_val_0);
        params["cond_1"] = migraphx::argument(cond_s, &cond_val_1);
        auto result      = p.eval(params).back();
        // main has one instruction that is : if_then_else
        // then mod is doing : {tmp = x+y; (cond) ? (((x-1)*y)-z)  : (((tmp-1)*y)-z);}
        // else mod is doing : {tmp = x+z; (cond) ? (((tmp-1)*x)-y) : (((z-1)*y)-x);}
        float gold_i = -1.0;
        if(cond_val_0)
        {
            float tmp_i = x_i + y_i;
            gold_i      = (cond_val_1) ? (((x_i - 1) * y_i) - z_i) : (((tmp_i - 1) * y_i) - z_i);
        }
        else
        {
            float tmp_i = x_i + z_i;
            gold_i      = (cond_val_1) ? (((tmp_i - 1) * x_i) - y_i) : (((z_i - 1) * y_i) - x_i);
        }
        auto gold = migraphx::fill_argument(ds, gold_i);
        EXPECT(gold == result);
    }
}

// TODO : FPGA compilation is broken right now, below test mentions fpga but doesn't compile for it
TEST_CASE(multitarget_select_module)
{
    migraphx::program p;
    migraphx::target_assignments tass;
    // create batch submodules
    auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
        auto* submod = p.create_module(module_name);
        migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 4}};
        auto sm_input = submod->add_parameter("data", sm_shape);
        migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
        auto literal_ins = submod->add_literal(migraphx::literal{lit_s, {6}});
        auto broadcast_lit =
            submod->add_instruction(migraphx::make_op("multibroadcast"), literal_ins, sm_input);
        auto add_ins0 = submod->add_instruction(migraphx::make_op("add"), sm_input, broadcast_lit);
        auto add_ins1 = submod->add_instruction(migraphx::make_op("add"), add_ins0, broadcast_lit);
        tass.insert(tass.begin(), std::make_pair(broadcast_lit, batch_size - 1));
        tass.insert(tass.begin(), std::make_pair(add_ins0, batch_size - 1));
        tass.insert(tass.begin(), std::make_pair(add_ins1, batch_size - 1));
        submod->add_return({add_ins1});
        return submod;
    };
    auto* batch1 = create_submodule(1, "batch_1");
    auto* batch2 = create_submodule(2, "batch_2");
    auto* batch3 = create_submodule(3, "batch_3");
    auto* batch4 = create_submodule(4, "batch_4");

    auto* mm = p.get_main_module();
    migraphx::shape dyn_s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
    auto input                              = mm->add_parameter("data", dyn_s);
    std::vector<migraphx::shape> sub_shapes = {};
    sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {4, 4}}});
    sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {4, 4}}});
    migraphx::shape out_attr = migraphx::shape{sub_shapes};
    auto sm_ins              = mm->add_instruction(
        migraphx::make_op("select_module", {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
        {input},
        {batch1, batch2, batch3, batch4});
    auto ret0 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
    mm->add_return({ret0});
    // compile
    migraphx::compile_options gpu_opts;
    gpu_opts.offload_copy = true;
    migraphx::generate_root_modules(p, tass);
    p.compile({migraphx::make_target("gpu"),
               migraphx::make_target("cpu"),
               migraphx::make_target("ref"),
               migraphx::make_target("ref")},
              {gpu_opts});
    EXPECT(check_compiled_program(p,
                                  {migraphx::make_target("gpu"),
                                   migraphx::make_target("cpu"),
                                   migraphx::make_target("ref"),
                                   migraphx::make_target("ref")}));
    // program does the 12+x where x has dynamic shape {{1, 4}, {4, 4}}
    for(const size_t bs : {1, 2, 3, 4})
    {
        migraphx::shape arg_shape{migraphx::shape::float_type, {bs, 4}};
        migraphx::parameter_map params;
        params["data"] = migraphx::generate_argument(arg_shape, arg_shape.elements());
        std::vector<float> input_data;
        params["data"].visit([&](const auto& vec) { input_data.assign(vec.begin(), vec.end()); });
        std::transform(input_data.begin(), input_data.end(), input_data.begin(), [](const auto& i) {
            return i + 12.0;
        });
        auto result = p.eval(params).back();
        EXPECT(migraphx::argument(arg_shape, input_data.data()) == result);
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
