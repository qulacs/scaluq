#pragma once

#include "../types.hpp"

namespace scaluq {
namespace internal {
//! Flags for bit property: diagonal in X-basis
#define FLAG_X_COMMUTE ((UINT)(0x01))
//! Flags for bit property: diagonal in Y-basis
#define FLAG_Y_COMMUTE ((UINT)(0x02))
//! Flags for bit property: diagonal in Z-basis
#define FLAG_Z_COMMUTE ((UINT)(0x04))

const UINT invalid_qubit = 9999;

class QubitInfo {
protected:
    UINT _index;

public:
    virtual ~QubitInfo() {}

    UINT index() const { return _index; }
    QubitInfo(UINT index) : _index(index) {}
};

class TargetQubitInfo;
class ControlQubitInfo;

class TargetQubitInfo : public QubitInfo {
private:
    UINT _commutation_property;

public:
    TargetQubitInfo(UINT index) : QubitInfo(index), _commutation_property(0) {}
    TargetQubitInfo(UINT index, UINT commutation_property)
        : QubitInfo(index), _commutation_property(commutation_property) {}

    bool is_commute_X() const { return (_commutation_property & FLAG_X_COMMUTE) != 0; }
    bool is_commute_Y() const { return (_commutation_property & FLAG_Y_COMMUTE) != 0; }
    bool is_commute_Z() const { return (_commutation_property & FLAG_Z_COMMUTE) != 0; }

    virtual bool is_commute_with(const TargetQubitInfo& info) const;
    virtual bool is_commute_with(const ControlQubitInfo& info) const;

    virtual UINT get_merged_property(UINT property) const {
        return _commutation_property & property;
    }

    virtual UINT get_merged_property(const TargetQubitInfo& info) const {
        return _commutation_property & info._commutation_property;
    }

    virtual UINT get_merged_property(const ControlQubitInfo& info) const {
        (void)info;
        return _commutation_property & FLAG_Z_COMMUTE;
    }
};

class ControlQubitInfo : public QubitInfo {
private:
    UINT _control_value;

public:
    UINT control_value() const { return _control_value; }
    ControlQubitInfo(UINT index) : QubitInfo(index), _control_value(1) {}
    ControlQubitInfo(UINT index, UINT control_value)
        : QubitInfo(index), _control_value(control_value) {}

    virtual bool is_commute_with(const TargetQubitInfo& info) const;
    virtual bool is_commute_with(const ControlQubitInfo& info) const;
};
}  // namespace internal
}  // namespace scaluq
