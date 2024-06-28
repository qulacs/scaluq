#include "utility.hpp"

#include <set>

#include "../constant.hpp"
#include "gate/gate_factory.hpp"

namespace scaluq {
static std::map<GateType, std::map<GateType, std::pair<GateType, double>>> single_qubit_merge = {
    {GateType::X,
     {{GateType::X, {GateType::I, 0.}},
      {GateType::Y, {GateType::Z, -PI() / 2}},
      {GateType::Z, {GateType::Y, PI() / 2}},
      {GateType::H, {GateType::SqrtYdag, -PI() / 4}},
      {GateType::SqrtX, {GateType::SqrtXdag, 0.}},
      {GateType::SqrtXdag, {GateType::SqrtX, 0.}},
      {GateType::SqrtYdag, {GateType::H, -PI() / 4}}}},
    {GateType::Y,
     {
         {GateType::X, {GateType::Z, PI() / 2}},
         {GateType::Y, {GateType::I, 0.}},
         {GateType::Z, {GateType::X, -PI() / 2}},
         {GateType::SqrtY, {GateType::SqrtYdag, 0.}},
         {GateType::SqrtYdag, {GateType::SqrtY, 0.}},
     }},
    {GateType::Z,
     {
         {GateType::X, {GateType::Y, -PI() / 2}},
         {GateType::Y, {GateType::X, PI() / 2}},
         {GateType::Z, {GateType::I, 0.}},
         {GateType::H, {GateType::SqrtY, PI() / 4}},
         {GateType::S, {GateType::Sdag, 0.}},
         {GateType::Sdag, {GateType::S, 0.}},
         {GateType::SqrtY, {GateType::H, PI() / 4}},
         {GateType::P0, {GateType::P0, 0.}},
         {GateType::P1, {GateType::P1, PI()}},
     }},
    {GateType::H,
     {
         {GateType::X, {GateType::SqrtY, PI() / 4}},
         {GateType::Z, {GateType::SqrtYdag, -PI() / 4}},
         {GateType::H, {GateType::I, 0.}},
         {GateType::SqrtY, {GateType::X, PI() / 4}},
         {GateType::SqrtYdag, {GateType::Z, -PI() / 4}},
     }},
    {GateType::S,
     {
         {GateType::Z, {GateType::Sdag, 0.}},
         {GateType::S, {GateType::Z, 0.}},
         {GateType::Sdag, {GateType::I, 0.}},
         {GateType::Tdag, {GateType::T, 0.}},
         {GateType::P0, {GateType::P0, 0.}},
         {GateType::P1, {GateType::P1, PI() / 2}},
     }},
    {GateType::Sdag,
     {
         {GateType::Z, {GateType::S, 0.}},
         {GateType::S, {GateType::I, 0.}},
         {GateType::Sdag, {GateType::Z, 0.}},
         {GateType::T, {GateType::Tdag, 0.}},
         {GateType::P0, {GateType::P0, 0.}},
         {GateType::P1, {GateType::P1, -PI() / 2}},
     }},
    {GateType::T,
     {{GateType::Z, {GateType::Tdag, 0.}},
      {GateType::Sdag, {GateType::Tdag, 0.}},
      {GateType::T, {GateType::S, 0.}},
      {GateType::Tdag, {GateType::I, 0.}},
      {GateType::P0, {GateType::P0, 0.}},
      {GateType::P1, {GateType::P1, PI() / 4}}}},
    {GateType::Tdag,
     {
         {GateType::Z, {GateType::T, 0.}},
         {GateType::S, {GateType::T, 0.}},
         {GateType::T, {GateType::I, 0.}},
         {GateType::Tdag, {GateType::Sdag, 0.}},
         {GateType::P0, {GateType::P0, 0.}},
         {GateType::P1, {GateType::P1, -PI() / 4}},
     }},
    {GateType::SqrtX,
     {
         {GateType::X, {GateType::SqrtXdag, 0.}},
         {GateType::SqrtX, {GateType::X, 0.}},
         {GateType::SqrtXdag, {GateType::I, 0.}},
     }},
    {GateType::SqrtXdag,
     {{GateType::X, {GateType::SqrtXdag, 0.}},
      {GateType::SqrtX, {GateType::I, 0.}},
      {GateType::SqrtXdag, {GateType::X, 0.}}}},
    {GateType::SqrtY,
     {{GateType::Y, {GateType::SqrtYdag, 0.}},
      {GateType::H, {GateType::Z, PI() / 4}},
      {GateType::SqrtY, {GateType::Y, 0.}},
      {GateType::SqrtYdag, {GateType::I, 0.}}}},
    {GateType::SqrtYdag,
     {{GateType::Y, {GateType::SqrtY, 0.}},
      {GateType::H, {GateType::X, -PI() / 4}},
      {GateType::SqrtY, {GateType::I, 0.}},
      {GateType::SqrtYdag, {GateType::Y, 0.}}}},
    {GateType::P0,
     {
         {GateType::Z, {GateType::P0, 0.}},
         {GateType::S, {GateType::P0, 0.}},
         {GateType::Sdag, {GateType::P0, 0.}},
         {GateType::T, {GateType::P0, 0.}},
         {GateType::Tdag, {GateType::P0, 0.}},
         {GateType::P0, {GateType::P0, 0.}},
     }},
    {GateType::P1,
     {
         {GateType::Z, {GateType::P1, PI()}},
         {GateType::S, {GateType::P1, PI() / 2}},
         {GateType::Sdag, {GateType::P1, -PI() / 2}},
         {GateType::T, {GateType::P1, PI() / 4}},
         {GateType::Tdag, {GateType::P1, -PI() / 4}},
         {GateType::P1, {GateType::P1, 0.}},
     }}};
Gate create_single_qubit_gate_from_type(const GateType& gate_type, UINT target) {
    switch (gate_type) {
        case GateType::I:
            return gate::I();
        case GateType::X:
            return gate::X(target);
        case GateType::Y:
            return gate::Y(target);
        case GateType::Z:
            return gate::Z(target);
        case GateType::H:
            return gate::H(target);
        case GateType::S:
            return gate::S(target);
        case GateType::Sdag:
            return gate::Sdag(target);
        case GateType::T:
            return gate::T(target);
        case GateType::Tdag:
            return gate::Tdag(target);
        case GateType::SqrtX:
            return gate::SqrtX(target);
        case GateType::SqrtXdag:
            return gate::SqrtXdag(target);
        case GateType::SqrtY:
            return gate::SqrtY(target);
        case GateType::SqrtYdag:
            return gate::SqrtYdag(target);
        case GateType::P0:
            return gate::P0(target);
        case GateType::P1:
            return gate::P1(target);
        default:
            assert(0);
    }
}
std::pair<Gate, double> merge_gate(const Gate& gate1, const Gate& gate2) {
    if (gate1.gate_type() == GateType::I) return {gate2->copy(), 0.};
    if (gate2.gate_type() == GateType::I) return {gate1->copy(), 0.};
    if (gate1.gate_type() == GateType::GlobalPhase)
        return {gate2->copy(), GlobalPhaseGate(gate1)->phase()};
    if (gate2.gate_type() == GateType::GlobalPhase)
        return {gate1->copy(), GlobalPhaseGate(gate2)->phase()};
    auto gate1_targets = gate1->get_target_qubit_list();
    std::ranges::copy(gate1->get_control_qubit_list(), std::back_inserter(gate1_targets));
    auto gate2_targets = gate2->get_target_qubit_list();
    std::ranges::copy(gate2->get_control_qubit_list(), std::back_inserter(gate2_targets));
    if (gate1_targets.size() == 1) {
        if (gate2_targets.size() == 1) {
            if (gate1_targets[0] == gate2_targets[0]) {
                UINT target = gate1_targets[0];
                auto it1 = single_qubit_merge.find(gate1.gate_type());
                if (it1 != single_qubit_merge.end()) {
                    auto it2 = it1->second.find(gate2.gate_type());
                    if (it2 != it1->second.end())
                        return {create_single_qubit_gate_from_type(it2->second.first, target),
                                it2->second.second};
                }
            }
            auto matrix1 = gate1->get_matrix().value();
            auto matrix2 = gate2->get_matrix().value();
            auto matrix = matrix2 * matrix1;
            // TODO DenseMatrixGate
        }
    }
}
}  // namespace scaluq
