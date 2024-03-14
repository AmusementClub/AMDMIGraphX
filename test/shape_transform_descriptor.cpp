#include <migraphx/shape_transform_descriptor.hpp>
#include <migraphx/make_op.hpp>
#include <test.hpp>

using migraphx::make_op;
using migraphx::shape_transform_descriptor;
using all_lens   = std::vector<std::vector<std::size_t>>;
using final_lens = std::vector<std::size_t>;
using all_axes   = std::vector<std::vector<std::vector<std::size_t>>>;
using d_axes     = std::vector<std::vector<std::size_t>>;
using ops        = std::vector<migraphx::operation>;

all_lens get_all_lens(const shape_transform_descriptor& d)
{
    all_lens result;
    std::transform(d.dimensions.begin(),
                   d.dimensions.end(),
                   std::back_inserter(result),
                   [](const auto& dimension) {
                       std::vector<std::size_t> sub_lens;
                       std::transform(dimension.subdimensions.begin(),
                                      dimension.subdimensions.end(),
                                      std::back_inserter(sub_lens),
                                      [](const auto& x) { return x.len; });
                       return sub_lens;
                   });
    return result;
}

final_lens get_final_lens(const shape_transform_descriptor& d)
{
    final_lens result;
    std::transform(d.dimensions.begin(),
                   d.dimensions.end(),
                   std::back_inserter(result),
                   [](const auto& x) { return x.len(); });
    return result;
}

all_axes get_all_axes(const shape_transform_descriptor& d)
{
    all_axes result;
    std::transform(d.dimensions.begin(),
                   d.dimensions.end(),
                   std::back_inserter(result),
                   [](const auto& dimension) {
                       std::vector<std::vector<std::size_t>> sub_axis;
                       std::transform(dimension.subdimensions.begin(),
                                      dimension.subdimensions.end(),
                                      std::back_inserter(sub_axis),
                                      [](const auto& x) { return x.axis; });
                       return sub_axis;
                   });
    return result;
}

template <class... Ts>
shape_transform_descriptor make_descriptor(const std::vector<std::size_t>& dims, const Ts&... xs)
{
    auto desc = shape_transform_descriptor{dims};
    CHECK(desc.apply({xs...}));
    return desc;
}

TEST_CASE(record_reshape)
{
    auto desc = make_descriptor({256, 3, 16, 16}, make_op("reshape", {{"dims", {16, 16, 48, 16}}}));
    EXPECT(get_final_lens(desc) == final_lens{16, 16, 48, 16});
    EXPECT(get_all_lens(desc) == all_lens{{16}, {16}, {3, 16}, {16}});
    EXPECT(get_all_axes(desc) ==
           all_axes{d_axes{{0, 0}}, d_axes{{0, 1}}, d_axes{{1}, {2}}, d_axes{{3}}});
}

TEST_CASE(record_reshape_1s)
{
    auto desc = make_descriptor({3, 4, 4}, make_op("reshape", {{"dims", {3, 1, 4, 1, 4}}}));
    EXPECT(get_final_lens(desc) == final_lens{3, 1, 4, 1, 4});
    EXPECT(get_all_lens(desc) == all_lens{{3}, {1}, {4}, {1}, {4}});
    EXPECT(get_all_axes(desc) ==
           all_axes{d_axes{{0}}, d_axes{{1, 0}}, d_axes{{1, 1}}, d_axes{{2, 0}}, d_axes{{2, 1}}});
}

TEST_CASE(record_reshape_trailing_1s)
{
    auto desc = make_descriptor({3, 4, 4}, make_op("reshape", {{"dims", {3, 4, 4, 1, 1}}}));
    EXPECT(get_final_lens(desc) == final_lens{3, 4, 4, 1, 1});
    EXPECT(get_all_lens(desc) == all_lens{{3}, {4}, {4}, {1}, {1}});
    EXPECT(get_all_axes(desc) ==
           all_axes{d_axes{{0}}, d_axes{{1}}, d_axes{{2}}, d_axes{{}}, d_axes{{}}});
}

TEST_CASE(record_squeeze_trailing_1s)
{
    auto desc = make_descriptor({3, 4, 4, 1, 1}, make_op("reshape", {{"dims", {3, 4, 4}}}));
    EXPECT(get_final_lens(desc) == final_lens{3, 4, 4});
    EXPECT(get_all_lens(desc) == all_lens{{3}, {4}, {4}});
    EXPECT(get_all_axes(desc) == all_axes{d_axes{{0}}, d_axes{{1}}, d_axes{{2}}});
}

TEST_CASE(record_reshape_squeeze_trailing_1s)
{
    auto desc = make_descriptor({3, 4, 4},
                                make_op("reshape", {{"dims", {3, 4, 4, 1, 1}}}),
                                make_op("reshape", {{"dims", {3, 4, 4}}}));
    EXPECT(get_final_lens(desc) == final_lens{3, 4, 4});
    EXPECT(get_all_lens(desc) == all_lens{{3}, {4}, {4}});
    EXPECT(get_all_axes(desc) == all_axes{d_axes{{0}}, d_axes{{1}}, d_axes{{2}}});
}

TEST_CASE(record_transpose)
{
    auto desc =
        make_descriptor({256, 3, 16, 16}, make_op("transpose", {{"permutation", {0, 2, 3, 1}}}));
    EXPECT(get_final_lens(desc) == final_lens{256, 16, 16, 3});
    EXPECT(get_all_lens(desc) == all_lens{{256}, {16}, {16}, {3}});
    EXPECT(get_all_axes(desc) == all_axes{d_axes{{0}}, d_axes{{2}}, d_axes{{3}}, d_axes{{1}}});
}

TEST_CASE(record_multibroadcast)
{
    auto desc =
        make_descriptor({1, 3, 1, 1}, make_op("multibroadcast", {{"out_lens", {256, 3, 16, 16}}}));
    EXPECT(get_final_lens(desc) == final_lens{256, 3, 16, 16});
    EXPECT(get_all_lens(desc) == all_lens{{256}, {3}, {16}, {16}});
    EXPECT(get_all_axes(desc) == all_axes{d_axes{{}}, d_axes{{1}}, d_axes{{}}, d_axes{{}}});
}

TEST_CASE(record_broadcast1)
{
    auto desc =
        make_descriptor({3}, make_op("broadcast", {{"axis", 1}, {"out_lens", {256, 3, 16, 16}}}));
    EXPECT(get_final_lens(desc) == final_lens{256, 3, 16, 16});
    EXPECT(get_all_lens(desc) == all_lens{{256}, {3}, {16}, {16}});
    EXPECT(get_all_axes(desc) == all_axes{d_axes{{}}, d_axes{{0}}, d_axes{{}}, d_axes{{}}});
}

TEST_CASE(record_broadcast2)
{
    auto desc = make_descriptor(
        {32, 10}, make_op("broadcast", {{"axis", 1}, {"out_lens", {256, 32, 10, 16, 16}}}));
    EXPECT(get_final_lens(desc) == final_lens{256, 32, 10, 16, 16});
    EXPECT(get_all_lens(desc) == all_lens{{256}, {32}, {10}, {16}, {16}});
    EXPECT(get_all_axes(desc) ==
           all_axes{d_axes{{}}, d_axes{{0}}, d_axes{{1}}, d_axes{{}}, d_axes{{}}});
}

TEST_CASE(optimize_transpose_transpose)
{
    EXPECT(migraphx::optimize_shape_transforms(
               {3, 5, 2},
               {
                   make_op("transpose", {{"permutation", {0, 2, 1}}}),
                   make_op("transpose", {{"permutation", {1, 0, 2}}}),
               }) == ops{
                         make_op("transpose", {{"permutation", {2, 0, 1}}}),
                     });
}

TEST_CASE(optimize_reshape_reshape)
{
    EXPECT(migraphx::optimize_shape_transforms({3, 5, 2},
                                               {
                                                   make_op("reshape", {{"dims", {30}}}),
                                                   make_op("reshape", {{"dims", {3, 10}}}),
                                               }) == ops{
                                                         make_op("reshape", {{"dims", {3, 10}}}),
                                                     });
}

TEST_CASE(optimize_reshape_transpose_reshape_to_none)
{
    EXPECT(migraphx::optimize_shape_transforms(
               {6, 5, 2},
               {
                   make_op("reshape", {{"dims", {6, 5, 2, 1, 1}}}),
                   make_op("transpose", {{"permutation", {0, 1, 2, 4, 3}}}),
                   make_op("reshape", {{"dims", {6, 5, 2}}}),
               }) == ops{});
}

TEST_CASE(optimize_reshape_transpose_reshape_to_transpose)
{
    EXPECT(migraphx::optimize_shape_transforms(
               {6, 5, 2},
               {
                   make_op("reshape", {{"dims", {2, 3, 5, 2}}}),
                   make_op("transpose", {{"permutation", {0, 1, 3, 2}}}),
                   make_op("reshape", {{"dims", {6, 2, 5}}}),
               }) == ops{
                         make_op("transpose", {{"permutation", {0, 2, 1}}}),
                     });
}

TEST_CASE(optimize_reshape_transpose_reshape_to_reshape)
{
    EXPECT(migraphx::optimize_shape_transforms(
               {6, 5, 2},
               {
                   make_op("reshape", {{"dims", {6, 5, 2, 1}}}),
                   make_op("transpose", {{"permutation", {0, 1, 3, 2}}}),
                   make_op("reshape", {{"dims", {6, 10}}}),
               }) == ops{
                         make_op("reshape", {{"dims", {6, 10}}}),
                     });
}

TEST_CASE(optimize_multibroadcast_transpose_reshape)
{
    EXPECT(migraphx::optimize_shape_transforms(
               {1, 5, 2},
               {
                   make_op("multibroadcast", {{"out_lens", {20, 5, 2}}}),
                   make_op("transpose", {{"permutation", {0, 2, 1}}}),
                   make_op("reshape", {{"dims", {20, 10}}}),
               }) == ops{
                         make_op("transpose", {{"permutation", {0, 2, 1}}}),
                         make_op("reshape", {{"dims", {1, 10}}}),
                         make_op("multibroadcast", {{"out_lens", {20, 10}}}),
                     });
}

TEST_CASE(optimize_resize)
{
    EXPECT(migraphx::optimize_shape_transforms(
               {3, 4, 4},
               {
                   make_op("reshape", {{"dims", {3, 1, 4, 1, 4}}}),
                   make_op("multibroadcast", {{"out_lens", {3, 2, 4, 2, 4}}}),
                   make_op("reshape", {{"dims", {3, 8, 8}}}),
               }) == ops{
                         make_op("unsqueeze", {{"axes", {1, 3}}}),
                         make_op("multibroadcast", {{"out_lens", {3, 2, 4, 2, 4}}}),
                         make_op("reshape", {{"dims", {3, 8, 8}}}),
                     });
}

TEST_CASE(optimize_reshape_2_squeeze)
{
    EXPECT(migraphx::optimize_shape_transforms({3, 1, 5, 1, 2, 1, 1},
                                               {
                                                   make_op("reshape", {{"dims", {3, 5, 2}}}),
                                               }) ==
           ops{
               make_op("squeeze", {{"axes", {1, 3, 5, 6}}}),
           });
}

TEST_CASE(optimize_reshape_2_unsqueeze)
{
    EXPECT(migraphx::optimize_shape_transforms(
               {3, 5, 2},
               {
                   make_op("reshape", {{"dims", {3, 1, 5, 1, 2, 1, 1}}}),
               }) == ops{
                         make_op("unsqueeze", {{"axes", {1, 3, 5, 6}}}),
                     });
}

TEST_CASE(optimize_unsqueeze_multibroadcast)
{
    EXPECT(migraphx::optimize_shape_transforms(
               {32, 10},
               {
                   make_op("unsqueeze", {{"axes", {0, 3, 4}}}),
                   make_op("multibroadcast", {{"out_lens", {256, 32, 10, 16, 16}}}),
               }) == ops{
                         make_op("broadcast", {{"axis", 1}, {"out_lens", {256, 32, 10, 16, 16}}}),
                     });
}

TEST_CASE(optimize_multibroadcast_reshape)
{
    EXPECT(migraphx::optimize_shape_transforms(
               {1, 4, 1},
               {
                   make_op("multibroadcast", {{"out_lens", {2, 4, 6}}}),
                   make_op("reshape", {{"dims", {2, 2, 2, 6}}}),
               }) == ops{
                         make_op("reshape", {{"dims", {1, 2, 2, 1}}}),
                         make_op("multibroadcast", {{"out_lens", {2, 2, 2, 6}}}),
                     });
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
