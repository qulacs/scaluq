#include "qubit_info.hpp"

namespace scaluq {
namespace internal {
bool TargetQubitInfo::is_commute_with(const TargetQubitInfo& info) const {
    if (this->index() != info.index())
        return true;
    else if ((_commutation_property & info._commutation_property) != 0)
        return true;
    else
        return false;
}
bool TargetQubitInfo::is_commute_with(const ControlQubitInfo& info) const {
    if (this->index() != info.index())
        return true;
    else if (this->is_commute_Z())
        return true;
    else
        return false;
}
bool ControlQubitInfo::is_commute_with(const TargetQubitInfo& info) const {
    if (this->index() != info.index())
        return true;
    else if (info.is_commute_Z())
        return true;
    else
        return false;
}
bool ControlQubitInfo::is_commute_with(const ControlQubitInfo& info) const {
    (void)info;
    return true;
}

}  // namespace internal
}  // namespace scaluq
