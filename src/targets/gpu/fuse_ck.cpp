#include <migraphx/gpu/fuse_ck.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/env.hpp>
namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_CK_GEMM);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_CK_GEMM_FUSION);

struct module;

namespace gpu {

struct ck_gemm
{
    operation op = make_op("dot");

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op, "op"));
    }

    std::string name() const { return "gpu::ck_gemm"; }

    void check_gemm_shape(const shape& s) const
    {
        if(not contains(range(s.strides().rbegin(), s.strides().rbegin() + 3), 1))
            MIGRAPHX_THROW("Invalid shape for ck_gemm");
    }

    shape compute_shape(std::vector<shape> inputs, const std::vector<module_ref>& mods) const
    {
        check_shapes{inputs, *this}.same_ndims();
        // if(mods.size() != 1)
        //     MIGRAPHX_THROW("should have one submodule.");
        if(inputs.size() < 2)
            MIGRAPHX_THROW("should have at least two inputs.");
        auto a = inputs[0];
        auto b = inputs[1];
        for(const auto& input : inputs)
            check_gemm_shape(input);
        auto r = op.compute_shape({a, b});
        if(mods.empty())
            return r;
        return r.with_type(mods.front()->get_output_shapes().front().type());
    }
};
MIGRAPHX_REGISTER_OP(ck_gemm);

struct ck_gemm_int8
{
    operation op = make_op("quant_dot");

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op, "op"));
    }

    std::string name() const { return "gpu::ck_gemm_int8"; }

    void check_gemm_shape(const shape& s) const
    {
        if(not contains(range(s.strides().rbegin(), s.strides().rbegin() + 3), 1))
            MIGRAPHX_THROW("Invalid shape for ck_gemm");
    }

    shape compute_shape(std::vector<shape> inputs, const std::vector<module_ref>& mods) const
    {
        check_shapes{inputs, *this}.same_ndims();
        // if(mods.size() != 1)
        //     MIGRAPHX_THROW("should have one submodule.");
        if(inputs.size() < 2)
            MIGRAPHX_THROW("should have at least two inputs.");
        auto a = inputs[0];
        auto b = inputs[1];
        for(const auto& input : inputs)
            check_gemm_shape(input);
        auto r = op.compute_shape({a, b});
        if(mods.empty())
            return r.with_type(migraphx::shape::int8_type);
        return r.with_type(mods.front()->get_output_shapes().front().type());
    }
};
MIGRAPHX_REGISTER_OP(ck_gemm_int8);

namespace {

MIGRAPHX_PRED_MATCHER(is_ck_gemm, instruction_ref ins)
{
    if(ins->name() != "dot" and ins->name() != "quant_dot")
        return false;
    auto a = ins->inputs().front()->get_shape();
    auto b = ins->inputs().back()->get_shape();
    // if(a.lens().back() > 2048)
    //     return false;
    return true;
}

struct find_ck_gemm_pointwise
{
    // Find a gemm followed by a pointwise operation.
    auto matcher() const
    {
        auto gemm =
            match::skip(match::name("contiguous"))(match::name("dot")(is_ck_gemm().bind("gemm")));
        return match::name("pointwise")(match::any_of[match::inputs()](gemm.bind("x")));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins      = r.result;
        auto gemm_ins = r.instructions["gemm"];
        auto x_ins    = r.instructions["x"]; // input after contiguous
        auto* pm      = ins->module_inputs().front();
        auto names    = pm->get_parameter_names();
        std::sort(names.begin(), names.end());
        auto inputs   = ins->inputs();
        auto gemm_it  = std::find(inputs.begin(), inputs.end(), x_ins);
        auto gemm_idx = gemm_it - inputs.begin();
        assert(gemm_it != inputs.end());
        if(ins->get_shape().type() != shape::half_type)
            return;
        if(gemm_idx != 0)
        {
            auto first_param    = pm->get_parameter(names[0]);
            auto gemm_param     = pm->get_parameter(names[gemm_idx]);
            auto new_gemm_param = pm->add_parameter(names[0] + "_0", gemm_param->get_shape());
            auto new_first_param =
                pm->add_parameter(names[gemm_idx] + "_0", first_param->get_shape());
            pm->replace_instruction(gemm_param, new_gemm_param);
            pm->replace_instruction(first_param, new_first_param);
            pm->remove_instruction(first_param);
            pm->remove_instruction(gemm_param);
        }
        inputs.erase(gemm_it);
        inputs.insert(inputs.begin(), gemm_ins->inputs().begin(), gemm_ins->inputs().end());

        mpm.get_module().replace_instruction(ins, ck_gemm{}, inputs, {pm});
    }
};

struct find_ck_gemm_pointwise_int8
{
    // Find a gemm followed by a pointwise operation.
    auto matcher() const
    {
        auto gemm =
            match::skip(match::name("contiguous"))(match::name("quant_dot")(is_ck_gemm().bind("gemm")));
        return match::name("pointwise")(match::any_of[match::inputs()](gemm.bind("x")));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins      = r.result;
        auto gemm_ins = r.instructions["gemm"];
        auto x_ins    = r.instructions["x"]; // input after contiguous
        auto next_ins = std::next(ins);
        // if (next_ins->name() == "quant_dot")
        // {
        //     std::cout << "\nins: ";
        //     ins->debug_print(); 
        //     std::cout << "\ngemm_ins: ";
        //     gemm_ins->debug_print(); 
        //     std::cout << "\nx_ins: ";
        //     x_ins->debug_print(); 
        //     std::cout << "\nnext: ";
        //     next_ins->debug_print();
        //     mpm.get_module().debug_print();
        // }
        
        
        auto* pm      = ins->module_inputs().front();
        auto names    = pm->get_parameter_names();
        
        std::sort(names.begin(), names.end());
        auto inputs   = ins->inputs();
        auto gemm_it  = std::find(inputs.begin(), inputs.end(), x_ins);
        auto gemm_idx = gemm_it - inputs.begin();
        assert(gemm_it != inputs.end());
        // if(ins->get_shape().type() != shape::half_type)
        //     return;
        // if (next_ins->name() == "reshape")
        // {
        //     std::cout << "PM before: " << std::endl;
        //     pm->debug_print();
        // }
        if(gemm_idx != 0)
        {
            auto first_param    = pm->get_parameter(names[0]);
            auto gemm_param     = pm->get_parameter(names[gemm_idx]);
            auto new_gemm_param = pm->add_parameter(names[0] + "_0", gemm_param->get_shape());
            auto new_first_param =
                pm->add_parameter(names[gemm_idx] + "_0", first_param->get_shape());
            pm->replace_instruction(gemm_param, new_gemm_param);
            pm->replace_instruction(first_param, new_first_param);
            pm->remove_instruction(first_param);
            pm->remove_instruction(gemm_param);
        }
        // if (next_ins->name() == "reshape")
        // {
        //     std::cout << "PM after: " << std::endl;
        //     pm->debug_print();
        // }
        inputs.erase(gemm_it);
        inputs.insert(inputs.begin(), gemm_ins->inputs().begin(), gemm_ins->inputs().end());
        // std::cout << "Next_ins inputs: " << std::endl;
        // for (auto& in : next_ins->inputs())
        // {
        //     in->debug_print();
        // }
        // auto out_shape = compute_shape(ck_gemm_int8{}, inputs, {pm});
        // instruction::replace(ins, ck_gemm_int8{}, out_shape.with_type(migraphx::shape::half_type), inputs, {pm});
        mpm.get_module().replace_instruction(ins, ck_gemm_int8{}, inputs, {pm});
        // std::cout << "Next_ins inputs (post replace): " << std::endl;
        // for (auto& in : std::next(ins)->inputs())
        // {
        //     in->debug_print();
        // }
        // if (next_ins->name() == "softmax" or next_ins->name() == "reshape")
        // {
        //     std::cout << "After replace: " << std::endl;
        //     mpm.get_module().debug_print();
        // }
    }
};

struct find_ck_gemm
{
    auto matcher() const { return match::name("dot")(is_ck_gemm().bind("gemm")); }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins = r.result;
        mpm.get_module().replace_instruction(ins, ck_gemm{ins->get_operator()}, ins->inputs());
    }
};

struct find_ck_gemm_int8
{
    auto matcher() const { return match::name("quant_dot")(is_ck_gemm().bind("gemm")); }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins = r.result;
        mpm.get_module().replace_instruction(ins, ck_gemm_int8{ins->get_operator()}, ins->inputs());
    }
};

} // namespace

void fuse_ck::apply(module_pass_manager& mpm) const
{
    if(not enabled(MIGRAPHX_DISABLE_CK_GEMM_FUSION{}))
    {
        match::find_matches(mpm, find_ck_gemm_pointwise{});
        match::find_matches(mpm, find_ck_gemm_pointwise_int8{});
    }
    if(not enabled(MIGRAPHX_DISABLE_CK_GEMM{}))
    {
        match::find_matches(mpm, find_ck_gemm{});
        match::find_matches(mpm, find_ck_gemm_int8{});
    }
}

} // namespace gpu

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
