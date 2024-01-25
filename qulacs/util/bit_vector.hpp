#include <algorithm>
#include <bit>
#include <iostream>
#include <vector>

#include "../types.hpp"

namespace qulacs {
class BitVector {
public:
    constexpr static UINT BIT_SIZE = sizeof(UINT) * 8;

    BitVector(UINT sz = 1) : _data((sz + BIT_SIZE - 1) / BIT_SIZE) {}

    [[nodiscard]] inline const std::vector<UINT>& data_raw() const { return _data; }
    [[nodiscard]] inline std::vector<UINT>& data_raw() { return _data; }

    [[nodiscard]] inline bool get(UINT idx) const {
        if (idx >= _data.size() * BIT_SIZE) return false;
        return _data[idx / BIT_SIZE] >> (idx % BIT_SIZE) & 1ULL;
    }
    inline void set(UINT idx, bool b) {
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
        for (UINT i = 0; i < _data.size(); i++) _data[i] &= rhs._data[i];
        return *this;
    }
    inline BitVector operator&(const BitVector& rhs) const { return BitVector(*this) &= rhs; }
    inline BitVector& operator|=(const BitVector& rhs) {
        if (rhs._data.size() > _data.size()) {
            _data.resize(rhs._data.size());
        }
        for (UINT i = 0; i < rhs._data.size(); i++) _data[i] |= rhs._data[i];
        return *this;
    }
    inline BitVector operator|(const BitVector& rhs) const { return BitVector(*this) |= rhs; }
    inline BitVector& operator^=(const BitVector& rhs) {
        if (rhs._data.size() > _data.size()) {
            _data.resize(rhs._data.size());
        }
        for (UINT i = 0; i < rhs._data.size(); i++) _data[i] ^= rhs._data[i];
        return *this;
    }
    inline BitVector operator^(const BitVector& rhs) const { return BitVector(*this) ^= rhs; }
    inline BitVector& operator-=(const BitVector& rhs) {
        for (UINT i = 0; i < std::min(_data.size(), rhs._data.size()); i++)
            _data[i] &= ~rhs._data[i];
        return *this;
    }
    inline BitVector operator-(const BitVector& rhs) const { return BitVector(*this) -= rhs; }

    inline auto operator<=>(const BitVector& other) {
        UINT sz = std::max(_data.size(), other._data.size());
        for (UINT i = sz; i-- != 0;) {
            UINT l = i >= _data.size() ? 0ULL : _data[i];
            UINT r = i >= other._data.size() ? 0ULL : other._data[i];
            if (l < r) return -1;
            if (l > r) return 1;
            if (i == 0) break;
        }
        return 0;
    }
    inline bool operator==(const BitVector& other) {
        UINT sz = std::max(_data.size(), other._data.size());
        for (UINT i = sz; i-- != 0;) {
            UINT l = i >= _data.size() ? 0ULL : _data[i];
            UINT r = i >= other._data.size() ? 0ULL : other._data[i];
            if (l != r) return false;
        }
        return true;
    }

    inline bool empty() const {
        return std::ranges::all_of(_data, [](UINT x) { return x == 0; });
    }
    inline UINT msb() const {
        for (UINT i = _data.size() - 1; i != std::numeric_limits<UINT>::max(); i--) {
            if (_data[i] != 0) return (i + 1) * BIT_SIZE - std::countl_zero(_data[i]) - 1;
        }
        return std::numeric_limits<UINT>::max();
    }
    inline UINT countr_zero() const {
        UINT res = 0;
        for (UINT i = 0; i < _data.size(); i++) {
            UINT to_add = std::countr_zero(_data[i]);
            res += to_add;
            if (to_add < BIT_SIZE) break;
        }
        return res;
    }
    inline UINT countr_one() const {
        UINT res = 0;
        for (UINT i = 0; i < _data.size(); i++) {
            UINT to_add = std::countr_one(_data[i]);
            res += to_add;
            if (to_add < BIT_SIZE) break;
        }
        return res;
    }
    inline UINT popcount() const {
        UINT res = 0;
        for (UINT i = 0; i < _data.size(); i++) res += std::popcount(_data[i]);
        return res;
    }

private:
    std::vector<UINT> _data;
};
}  // namespace qulacs
