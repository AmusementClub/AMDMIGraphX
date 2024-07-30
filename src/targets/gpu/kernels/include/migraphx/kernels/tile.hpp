#ifndef MIGRAPHX_GUARD_KERNELS_TILE_HPP
#define MIGRAPHX_GUARD_KERNELS_TILE_HPP

#include <migraphx/kernels/prestore.hpp>
#include <migraphx/kernels/preload.hpp>

namespace migraphx {

struct tile
{
    struct load
    {
    };
    struct store
    {
    };

    static constexpr auto outer()
    {
        return [](auto axis, auto a) {
            return transform_i(a, [=](auto i) {
                if constexpr(i <= axis)
                    return a[i];
                else
                    return 1;
            });
        };
    }

    static constexpr auto inner()
    {
        return [](auto axis, auto a) {
            return transform_i(a, [=](auto i) {
                if constexpr(i > axis)
                    return a[i];
                else
                    return 1;
            });
        };
    }

    template <index_int Axis, class Select, class Shape>
    static constexpr auto slice(Select select, Shape)
    {
        constexpr Shape s{};
        return make_shape(select(_c<Axis>, s.lens), select(_c<Axis>, s.strides));
    }

    template <index_int Axis, class T>
    static constexpr auto slice_tensor(index_int i, T x)
    {
        constexpr auto s = get_shape_c<T>{};
        auto offset      = slice(outer(), s).index(i);
        return make_tensor_view(x.data() + offset, slice(inner(), s));
    }

    template <class T, class... Ts>
    static constexpr auto get_size(T, Ts...)
    {
        // TODO: Assert all slices are the same size
        constexpr auto size = slice(outer(), get_shape_c<T>{}).elements();
        return size;
    }

    template <index_int Axis>
    static __device__ auto auto_slice(index idx)
    {
        return make_transform([=](auto f, auto... xs) {
            idx.group_stride(get_size(xs...),
                             [=](auto group) { f(slice_tensor<Axis>(group, xs)...); });
        });
    }
};

template <index_int Axis, class... Mode>
__device__ auto auto_tile()
{
    auto idx = make_index();
    return transform_args(tile::auto_slice<Axis>(idx),
                          auto_prestore<is_same<Mode, tile::store>{}...>(idx),
                          auto_preload<is_same<Mode, tile::load>{}...>(idx));
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_TILE_HPP
