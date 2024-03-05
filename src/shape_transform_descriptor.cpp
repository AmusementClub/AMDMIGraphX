#include <migraphx/shape_transform_descriptor.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/common_dims.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/stringutils.hpp>
#include <map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class Range>
static auto elements(const Range& r)
{
    return std::accumulate(r.begin(), r.end(), std::size_t{1}, std::multiplies<>{});
}

template <class Iterator, class Projection>
static auto compute_end_dim(Iterator start, Iterator last, std::size_t dim, Projection proj)
{
    std::size_t x = 1;
    auto it       = std::find_if(start, last, [&](auto d) {
        x *= proj(d);
        return x >= dim;
    });
    if(x != dim)
        return start;
    return it;
}

static void debug_print(const std::vector<shape_transform_descriptor::dimension::sub>& subs)
{
    for(const auto& s : subs)
        std::cout << s.len << ":" << to_string_range(s.axis, "x") << ",";
}
static void debug_print(const std::vector<shape_transform_descriptor::dimension>& dims)
{
    for(const auto& d : dims)
    {
        std::cout << "[";
        debug_print(d.subdimensions);
        std::cout << "],";
    }
    std::cout << std::endl;
}

shape_transform_descriptor::shape_transform_descriptor(const std::vector<std::size_t>& dims)
    : rank(dims.size())
{
    transform(dims,
              range(dims.size()),
              std::back_inserter(dimensions),
              [](std::size_t d, std::size_t a) -> dimension {
                  return {{dimension::sub{d, {a}}}};
              });
}

static std::vector<shape_transform_descriptor::dimension::sub>
get_all_subdimensions(const std::vector<shape_transform_descriptor::dimension>& dimensions)
{
    std::vector<shape_transform_descriptor::dimension::sub> result;
    for(const auto& dim : dimensions)
    {
        result.insert(result.end(), dim.subdimensions.begin(), dim.subdimensions.end());
    }
    return result;
}

static std::vector<std::size_t> compute_dims(const operation& op,
                                             const std::vector<std::size_t>& idims)
{
    shape s{shape::float_type, idims};
    return op.compute_shape({s}).lens();
}

bool shape_transform_descriptor::apply(const std::vector<operation>& ops)
{
    std::vector<std::size_t> dims;
    std::transform(dimensions.begin(),
                   dimensions.end(),
                   std::back_inserter(dims),
                   [](const dimension& d) { return d.len(); });
    for(const auto& op : ops)
    {
        auto v = op.to_value();
        if(contains({"reshape", "squeeze", "unsqueeze", "flatten"}, op.name()))
        {
            dims = compute_dims(op, dims);
            if(not apply_reshape(dims))
                return false;
        }
        else if(op.name() == "transpose")
        {
            dims = compute_dims(op, dims);
            if(not apply_transpose(v["permutation"].to_vector<std::int64_t>()))
                return false;
        }
        else if(op.name() == "multibroadcast")
        {
            dims = compute_dims(op, dims);
            if(not apply_broadcast(dims))
                return false;
        }
        else
        {
            return false;
        }
    }
    return true;
}
bool shape_transform_descriptor::apply_reshape(const std::vector<std::size_t>& rdims)
{
    assert(migraphx::elements(rdims) == this->elements());
    std::vector<dimension> new_dims;
    auto subs     = get_all_subdimensions(dimensions);
    std::size_t i = 0;
    std::size_t r = 0;
    while(i < subs.size() and r < rdims.size())
    {
        const auto& sub = subs[i];
        auto idim       = sub.len;
        auto rdim       = rdims[r];
        if(idim == rdim)
        {
            new_dims.push_back({{sub}});
        }
        // squeeze
        else if(rdim > idim)
        {
            auto start = subs.begin() + i;
            auto it = compute_end_dim(start, subs.end(), rdim, std::mem_fn(&dimension::sub::len));
            if(it == start)
                return false;
            auto n = it - start;
            i += n;
            new_dims.push_back({{start, it + 1}});
        }
        // unsqueeze
        else // if(rdim < idim)
        {
            auto start = rdims.begin() + r;
            auto it    = compute_end_dim(start, rdims.end(), idim, id{});
            if(it == start)
                return false;
            auto n = it - start;
            r += n;
            transform(range(n + 1), std::back_inserter(new_dims), [&](auto j) -> dimension {
                auto new_sub = sub;
                if(not new_sub.axis.empty())
                    new_sub.axis.push_back(j);
                new_sub.len = start[j];
                return {{new_sub}};
            });
        }
        r++;
        i++;
    }

    // Handle trailing 1s
    if(new_dims.size() < rdims.size() and not new_dims.empty())
    {
        for(auto d : range(rdims.begin() + new_dims.size(), rdims.end()))
        {
            if(d != 1)
                return false;
            new_dims.push_back({{dimension::sub{1}}});
        }
    }

    if(rdims.size() != new_dims.size())
        return false;
    dimensions = new_dims;
    return true;
}
bool shape_transform_descriptor::apply_transpose(const std::vector<std::int64_t>& permutation)
{
    if(permutation.size() != dimensions.size())
        return false;
    dimensions = reorder_dims(dimensions, permutation);
    return true;
}

bool shape_transform_descriptor::apply_broadcast(const std::vector<std::size_t>& out_lens,
                                                 optional<std::size_t> axis)
{
    auto offset = out_lens.size() - dimensions.size();
    std::vector<dimension> new_dims;
    std::transform(out_lens.begin(),
                   out_lens.begin() + offset,
                   std::back_inserter(new_dims),
                   [&](auto len) -> dimension {
                       return {{dimension::sub{len, {}}}};
                   });
    std::transform(out_lens.begin() + offset,
                   out_lens.end(),
                   dimensions.begin(),
                   std::back_inserter(new_dims),
                   [&](auto len, const dimension& dim) -> dimension {
                       if(len == dim.len())
                           return dim;
                       if(dim.len() != 1)
                           MIGRAPHX_THROW("Wrong out_lens for broadcast");
                       return {{dimension::sub{len, {}}}};
                   });
    dimensions = new_dims;
    return true;
}

void shape_transform_descriptor::dimension::simplify()
{
    if(subdimensions.size() < 2)
        return;
    // Remove dimensions of 1
    subdimensions.erase(std::remove_if(std::next(subdimensions.begin()),
                                       subdimensions.end(),
                                       [&](const sub& d) { return d.len == 1; }),
                        subdimensions.end());
    // Remove adjacent dimensions
    subdimensions.erase(adjacent_remove_if(subdimensions.begin(),
                                           subdimensions.end(),
                                           [&](const sub& d1, const sub& d2) {
                                               if(d1.axis.size() < 2)
                                                   return false;
                                               if(d2.axis.size() < 2)
                                                   return false;
                                               if(not std::equal(d1.axis.begin(),
                                                                 d1.axis.end() - 1,
                                                                 d2.axis.begin(),
                                                                 d2.axis.end() - 1))
                                                   return false;
                                               auto a1 = d1.axis.back();
                                               auto a2 = d2.axis.back();
                                               return (std::max(a1, a2) - std::min(a1, a2)) == 1;
                                           }),
                        subdimensions.end());
}

template <class Predicate>
static auto find_subdimension(shape_transform_descriptor& td, Predicate p)
{
    for(auto& d : td.dimensions)
    {
        auto it = std::find_if(d.subdimensions.begin(), d.subdimensions.end(), p);
        if(it != d.subdimensions.end())
            return std::make_tuple(std::ref(d.subdimensions), it);
    }
    MIGRAPHX_THROW("Searching for non-existent subdimension");
}

void shape_transform_descriptor::simplify()
{
    for(auto& d : dimensions)
        d.simplify();

    std::map<std::size_t, std::size_t> missing_axes;
    std::vector<std::size_t> last_axis;
    {
        // Group axis
        std::map<std::size_t, std::vector<dimension::sub*>> axes_map;
        for(auto& d : dimensions)
        {
            for(auto& s : d.subdimensions)
            {
                if(s.axis.empty())
                    continue;
                axes_map[s.axis.front()].push_back(&s);
            }
        }
        if(axes_map.empty())
            return;

        // Renumber subaxis
        for(auto&& p : axes_map)
        {
            const auto& axis = p.first;
            auto& subs       = p.second;
            if(subs.size() == 1)
            {
                subs[0]->axis = {axis};
            }
            else
            {
                std::sort(subs.begin(), subs.end(), by(std::less<>{}, [](const dimension::sub* s) {
                              return s->axis;
                          }));
                for(std::size_t i : range(subs.size()))
                    subs[i]->axis = {axis, i};
            }
        }

        // Find last axis
        last_axis = std::prev(axes_map.end())->second.back()->axis;

        // Find missing axes
        for(auto axis : range(rank))
        {
            if(contains(axes_map, axis))
                continue;
            auto it            = axes_map.upper_bound(axis);
            missing_axes[axis] = it == axes_map.end() ? rank : it->first;
        }
    }

    // Reinsert removed axes of 1
    for(auto&& p : missing_axes)
    {
        auto missing_axis = p.first;
        auto next_axis    = p.second;
        if(next_axis == rank)
        {
            auto x = find_subdimension(
                *this, [&](const dimension::sub& s) { return s.axis == last_axis; });
            std::get<0>(x).insert(std::next(std::get<1>(x)), dimension::sub{1, {missing_axis}});
        }
        else
        {
            auto x = find_subdimension(*this, [&](const dimension::sub& s) {
                if(s.axis.empty())
                    return false;
                if(s.axis.front() != next_axis)
                    return false;
                if(s.axis.size() == 1)
                    return true;
                assert(s.axis.size() == 2);
                return s.axis.back() == 0;
            });
            std::get<0>(x).insert(std::get<1>(x), dimension::sub{1, {missing_axis}});
        }
    }
}

static bool is_broadcast_dim(const shape_transform_descriptor::dimension& d)
{
    if(d.subdimensions.empty())
        return true;
    if(d.subdimensions.size() != 1)
        return false;
    const auto& sub = d.subdimensions.front();
    return sub.axis.empty();
}

std::vector<operation> shape_transform_descriptor::generate() const
{
    std::vector<operation> result;
    std::vector<shape_transform_descriptor::dimension> new_dims = dimensions;
    // Need multibroadcast
    if(std::any_of(new_dims.begin(), new_dims.end(), &is_broadcast_dim))
    {
        std::vector<std::size_t> out_lens;
        std::transform(new_dims.begin(),
                       new_dims.end(),
                       std::back_inserter(out_lens),
                       [](const dimension& d) { return d.len(); });
        result.push_back(make_op("multibroadcast", {{"out_lens", out_lens}}));
    }
    // Need squeeze reshape
    if(std::any_of(new_dims.begin(), new_dims.end(), [](const dimension& d) {
           if(d.subdimensions.size() != 1)
               return true;
           return is_broadcast_dim(d);
       }))
    {
        std::vector<std::size_t> dims;
        std::transform(new_dims.begin(),
                       new_dims.end(),
                       std::back_inserter(dims),
                       [](const dimension& d) -> std::size_t {
                           if(is_broadcast_dim(d))
                               return 1;
                           return d.len();
                       });
        result.push_back(make_op("reshape", {{"dims", dims}}));
    }

    // Remove broadcast
    new_dims.erase(std::remove_if(new_dims.begin(), new_dims.end(), &is_broadcast_dim),
                   new_dims.end());

    auto subs = get_all_subdimensions(new_dims);
    // Need multibroadcast
    if(std::any_of(
           subs.begin(), subs.end(), [](const dimension::sub& s) { return s.axis.empty(); }))
    {
        std::vector<std::size_t> out_lens;
        std::transform(subs.begin(),
                       subs.end(),
                       std::back_inserter(out_lens),
                       [](const dimension::sub& s) { return s.len; });
        result.push_back(make_op("multibroadcast", {{"out_lens", out_lens}}));
    }

    auto compare_sub = [](auto f) {
        return by(f, [](const dimension::sub& s) -> const auto& { return s.axis; });
    };
    // Need transpose
    if(not std::is_sorted(subs.begin(), subs.end(), compare_sub(std::less<>{})))
    {
        auto permutation = sort_permutation(subs, compare_sub(std::greater<>{}));
        result.push_back(make_op("transpose", {{"permutation", permutation}}));
        reorder_dims(subs, invert_permutation(permutation));
    }
    // Need reshape unsqeeze
    if(std::any_of(
           subs.begin(), subs.end(), [](const dimension::sub& s) { return s.axis.size() != 1; }))
    {
        std::vector<std::size_t> dims;
        std::transform(subs.begin(),
                       subs.end(),
                       std::back_inserter(dims),
                       [](const dimension::sub& s) -> std::size_t {
                           if(s.axis.empty())
                               return 1;
                           return s.len;
                       });
        result.push_back(make_op("reshape", {{"dims", dims}}));
    }
    std::reverse(result.begin(), result.end());
    return result;
}

std::size_t shape_transform_descriptor::dimension::len() const
{
    return transform_accumulate(subdimensions.begin(),
                                subdimensions.end(),
                                std::size_t{1},
                                std::multiplies<>{},
                                [](const auto& s) { return s.len; });
}

std::size_t shape_transform_descriptor::elements() const
{
    return transform_accumulate(dimensions.begin(),
                                dimensions.end(),
                                std::size_t{1},
                                std::multiplies<>{},
                                [](const auto& s) { return s.len(); });
}

std::vector<operation> optimize_shape_transforms(const std::vector<std::size_t>& dims,
                                                 const std::vector<operation>& ops)
{
    shape_transform_descriptor sd{dims};
    if(not sd.apply(ops))
        return ops;
    sd.simplify();
    return sd.generate();
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx