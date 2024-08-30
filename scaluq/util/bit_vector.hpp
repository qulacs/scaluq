#pragma once

#include <algorithm>
#include <bit>
#include <iostream>
#include <vector>

#include "../types.hpp"

namespace scaluq {
namespace internal {
class BitVector {
public:
    constexpr static std::uint64_t BIT_SIZE = sizeof(std::uint64_t) * 8;

    BitVector(std::uint64_t sz = 1) : _data((sz + BIT_SIZE - 1) / BIT_SIZE) {}
    BitVector(const std::vector<bool>& vec) : _data((vec.size() + BIT_SIZE - 1) / BIT_SIZE) {
        for (std::uint64_t i = 0; i < vec.size(); ++i) {
            set(i, vec[i]);
        }
    }

    [[nodiscard]] inline const std::vector<std::uint64_t>& data_raw() const { return _data; }
    [[nodiscard]] inline std::vector<std::uint64_t>& data_raw() { return _data; }

    [[nodiscard]] inline bool get(std::uint64_t idx) const {
        if (idx >= _data.size() * BIT_SIZE) return false;
        return _data[idx / BIT_SIZE] >> (idx % BIT_SIZE) & 1ULL;
    }
    inline void set(std::uint64_t idx, bool b) {
        if (idx >= _data.size() * BIT_SIZE) _data.resize(idx / BIT_SIZE + 1);
        if (b)
            _data[idx / BIT_SIZE] |= 1ULL << (idx % BIT_SIZE);
        else
            _data[idx / BIT_SIZE] &= ~(1ULL << (idx % BIT_SIZE));
    }

    template <bool Const>
    class _Reference {
        friend BitVector;

    public:
        _Reference& operator=(bool b) {
            static_assert(!Const);
            _container.set(_idx, b);
            return *this;
        }
        _Reference& operator&=(bool b) {
            static_assert(!Const);
            _container.set(_idx, b && _container.get(_idx));
            return *this;
        }
        _Reference& operator|=(bool b) {
            static_assert(!Const);
            _container.set(_idx, b || _container.get(_idx));
            return *this;
        }
        _Reference& operator^=(bool b) {
            static_assert(!Const);
            _container.set(_idx, b ^ _container.get(_idx));
            return *this;
        }
        operator bool() const { return _container.get(_idx); }

    private:
        using ContainerReference = std::conditional_t<Const, const BitVector&, BitVector&>;
        ContainerReference _container;
        const int _idx;

        _Reference(ContainerReference container, int idx) : _container(container), _idx(idx) {}
    };

    using ConstReference = _Reference<true>;
    using Reference = _Reference<false>;

    [[nodiscard]] inline ConstReference operator[](int idx) const {
        return ConstReference(*this, idx);
    }
    [[nodiscard]] inline Reference operator[](int idx) { return Reference(*this, idx); }

    inline BitVector& operator&=(const BitVector& rhs) {
        if (rhs._data.size() < _data.size()) {
            _data.resize(rhs._data.size());
        }
        for (std::uint64_t i = 0; i < _data.size(); i++) _data[i] &= rhs._data[i];
        return *this;
    }
    inline BitVector operator&(const BitVector& rhs) const { return BitVector(*this) &= rhs; }
    inline BitVector& operator|=(const BitVector& rhs) {
        if (rhs._data.size() > _data.size()) {
            _data.resize(rhs._data.size());
        }
        for (std::uint64_t i = 0; i < rhs._data.size(); i++) _data[i] |= rhs._data[i];
        return *this;
    }
    inline BitVector operator|(const BitVector& rhs) const { return BitVector(*this) |= rhs; }
    inline BitVector& operator^=(const BitVector& rhs) {
        if (rhs._data.size() > _data.size()) {
            _data.resize(rhs._data.size());
        }
        for (std::uint64_t i = 0; i < rhs._data.size(); i++) _data[i] ^= rhs._data[i];
        return *this;
    }
    inline BitVector operator^(const BitVector& rhs) const { return BitVector(*this) ^= rhs; }
    inline BitVector& operator-=(const BitVector& rhs) {
        for (std::uint64_t i = 0; i < std::min(_data.size(), rhs._data.size()); i++)
            _data[i] &= ~rhs._data[i];
        return *this;
    }
    inline BitVector operator-(const BitVector& rhs) const { return BitVector(*this) -= rhs; }

    inline std::weak_ordering operator<=>(const BitVector& other) const {
        std::uint64_t sz = std::max(_data.size(), other._data.size());
        for (std::uint64_t i = sz; i-- != 0;) {
            std::uint64_t l = i >= _data.size() ? 0ULL : _data[i];
            std::uint64_t r = i >= other._data.size() ? 0ULL : other._data[i];
            if (l != r) return l <=> r;
            if (i == 0) break;
        }
        return std::weak_ordering::equivalent;
    }
    inline bool operator==(const BitVector& other) const {
        std::uint64_t sz = std::max(_data.size(), other._data.size());
        for (std::uint64_t i = sz; i-- != 0;) {
            std::uint64_t l = i >= _data.size() ? 0ULL : _data[i];
            std::uint64_t r = i >= other._data.size() ? 0ULL : other._data[i];
            if (l != r) return false;
        }
        return true;
    }

    operator std::vector<bool>() const {
        std::vector<bool> vec(_data.size() * BIT_SIZE);
        for (std::uint64_t i = 0; i < vec.size(); ++i) {
            vec[i] = get(i);
        }
        return vec;
    }

    inline bool empty() const {
        return std::ranges::all_of(_data, [](std::uint64_t x) { return x == 0; });
    }
    inline std::uint64_t msb() const {
        for (std::uint64_t i = _data.size() - 1; i != std::numeric_limits<std::uint64_t>::max();
             i--) {
            if (_data[i] != 0) return (i + 1) * BIT_SIZE - std::countl_zero(_data[i]) - 1;
        }
        return std::numeric_limits<std::uint64_t>::max();
    }
    inline std::uint64_t countr_zero() const {
        std::uint64_t res = 0;
        for (std::uint64_t i = 0; i < _data.size(); i++) {
            std::uint64_t to_add = std::countr_zero(_data[i]);
            res += to_add;
            if (to_add < BIT_SIZE) break;
        }
        return res;
    }
    inline std::uint64_t countr_one() const {
        std::uint64_t res = 0;
        for (std::uint64_t i = 0; i < _data.size(); i++) {
            std::uint64_t to_add = std::countr_one(_data[i]);
            res += to_add;
            if (to_add < BIT_SIZE) break;
        }
        return res;
    }
    inline std::uint64_t popcount() const {
        std::uint64_t res = 0;
        for (std::uint64_t i = 0; i < _data.size(); i++) res += std::popcount(_data[i]);
        return res;
    }

private:
    std::vector<std::uint64_t> _data;
};
}  // namespace internal
}  // namespace scaluq
