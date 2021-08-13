#include <migraphx/layout_nhwc.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/eliminate_contiguous.hpp>
#include <migraphx/dead_code_elimination.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class Predicate>
std::vector<instruction_ref> find_lasts(const module& m, Predicate pred)
{
    std::vector<instruction_ref> result;
    fix([&](auto self, auto ins) {
        if(pred(ins))
        {
            result.push_back(ins);
            return;
        }
        for(auto input : ins->inputs())
            self(input);
    })(std::prev(m.end()));
    return result;
}

void preserve_output_layout(module& m)
{
    std::vector<instruction_ref> outputs =
        find_lasts(m, [](auto ins) { return ins->get_shape().lens().size() == 4; });
    for(auto output : outputs)
    {
        auto permutation = find_permutation(output->get_shape());
        auto layout      = m.insert_instruction(
            std::next(output), make_op("layout", {{"permutation", permutation}}), output);
        m.replace_instruction(output, layout);
    }
}

void transform_convolutions(module& m)
{
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "convolution")
            continue;
        if(ins->get_shape().lens().size() != 4)
            continue;
        auto args = ins->inputs();
        std::transform(args.begin(), args.end(), args.begin(), [&](auto& i) {
            return m.insert_instruction(ins, make_op("layout", {{"permutation", {0, 2, 3, 1}}}), i);
        });
        auto conv = m.insert_instruction(ins, ins->get_operator(), args);
        auto c    = m.insert_instruction(ins, make_op("contiguous"), conv);
        m.replace_instruction(ins, c);
    }
}

void layout_nhwc::apply(module& m) const
{
    preserve_output_layout(m);
    transform_convolutions(m);
    dead_code_elimination{}.apply(m);
    eliminate_contiguous{"contiguous"}.apply(m);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
