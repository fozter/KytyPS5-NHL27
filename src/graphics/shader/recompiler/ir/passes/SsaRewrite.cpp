#include "graphics/shader/recompiler/ir/passes/SsaRewrite.h"

#include <algorithm>
#include <bit>
#include <unordered_map>
#include <variant>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

struct SccTag {
	auto operator<=>(const SccTag&) const = default;
};
struct ThreadBitScalarReg {
	ScalarReg reg {};
	auto      operator<=>(const ThreadBitScalarReg&) const = default;
};
struct ScalarMaskTag {
	ScalarReg reg {};
	auto      operator<=>(const ScalarMaskTag&) const = default;
};
struct ExecTag {
	auto operator<=>(const ExecTag&) const = default;
};
struct ExecLoTag {
	auto operator<=>(const ExecLoTag&) const = default;
};
struct ExecHiTag {
	auto operator<=>(const ExecHiTag&) const = default;
};
struct VccTag {
	auto operator<=>(const VccTag&) const = default;
};
struct VccLoTag {
	auto operator<=>(const VccLoTag&) const = default;
};
struct VccHiTag {
	auto operator<=>(const VccHiTag&) const = default;
};
struct M0Tag {
	auto operator<=>(const M0Tag&) const = default;
};
struct GotoVariable {
	uint32_t index                                  = 0;
	auto     operator<=>(const GotoVariable&) const = default;
};

using Variable =
    std::variant<ScalarReg, ThreadBitScalarReg, ScalarMaskTag, VectorReg, GotoVariable, SccTag,
                 ExecTag, ExecLoTag, ExecHiTag, VccTag, VccLoTag, VccHiTag, M0Tag>;
// Goto-variable definitions keyed by (block, variable) in one open-addressing table (perf,
// uncommitted): a node map per variable and block was ~25% of the SSA rewrite on NHL 27.
class GotoDefinitions {
public:
	[[nodiscard]] const Value& Get(Block* block, uint32_t variable) const {
		static const Value empty {};
		if (slots.empty()) {
			return empty;
		}
		for (size_t index = Hash(block, variable);; index = (index + 1) & mask) {
			const auto& slot = slots[index];
			if (slot.block == nullptr) {
				return empty;
			}
			if (slot.block == block && slot.variable == variable) {
				return slot.value;
			}
		}
	}
	void Set(Block* block, uint32_t variable, Value value) {
		if ((count + 1) * 2 > slots.size()) {
			Grow();
		}
		for (size_t index = Hash(block, variable);; index = (index + 1) & mask) {
			auto& slot = slots[index];
			if (slot.block == nullptr) {
				slot = {block, variable, value};
				count++;
				return;
			}
			if (slot.block == block && slot.variable == variable) {
				slot.value = value;
				return;
			}
		}
	}

private:
	struct Slot {
		Block*   block    = nullptr;
		uint32_t variable = 0;
		Value    value;
	};
	[[nodiscard]] size_t Hash(Block* block, uint32_t variable) const {
		const auto key =
		    reinterpret_cast<uintptr_t>(block) ^ (static_cast<uint64_t>(variable) * 0x9E3779B97F4A7C15ull);
		return static_cast<size_t>((key * 0xD6E8FEB86659FD93ull) >> shift);
	}
	void Grow() {
		const auto old      = std::move(slots);
		const auto capacity = old.empty() ? size_t {64} : old.size() * 2;
		slots.assign(capacity, Slot {});
		mask  = capacity - 1;
		shift = 64 - std::countr_zero(capacity);
		count = 0;
		for (const auto& slot: old) {
			if (slot.block != nullptr) {
				Set(slot.block, slot.variable, slot.value);
			}
		}
	}
	std::vector<Slot> slots;
	size_t            mask  = 0;
	int               shift = 64;
	size_t            count = 0;
};

struct DefTable {
	const Value& Get(Block* block, ScalarReg reg) {
		if (RegIndex(reg) >= NumScalarRegs) {
			EXIT("SSA scalar read index is out of range: %u", RegIndex(reg));
		}
		return block->ssa_sreg_values[RegIndex(reg)];
	}
	void Set(Block* block, ScalarReg reg, Value value) {
		if (RegIndex(reg) >= NumScalarRegs) {
			EXIT("SSA scalar write index is out of range: %u", RegIndex(reg));
		}
		block->ssa_sreg_values[RegIndex(reg)] = value;
	}
	const Value& Get(Block* block, ThreadBitScalarReg value) {
		EXIT_IF(RegIndex(value.reg) >= NumScalarRegs);
		return block->ssa_thread_bit_sreg_values[RegIndex(value.reg)];
	}
	void Set(Block* block, ThreadBitScalarReg value, Value definition) {
		EXIT_IF(RegIndex(value.reg) >= NumScalarRegs);
		block->ssa_thread_bit_sreg_values[RegIndex(value.reg)] = definition;
	}
	const Value& Get(Block* block, ScalarMaskTag value) {
		EXIT_IF(RegIndex(value.reg) >= NumScalarRegs);
		return block->ssa_sreg_mask_tags[RegIndex(value.reg)];
	}
	void Set(Block* block, ScalarMaskTag value, Value definition) {
		EXIT_IF(RegIndex(value.reg) >= NumScalarRegs);
		block->ssa_sreg_mask_tags[RegIndex(value.reg)] = definition;
	}

	const Value& Get(Block* block, VectorReg reg) {
		EXIT_IF(RegIndex(reg) >= NumVectorRegs);
		return block->ssa_vreg_values[RegIndex(reg)];
	}
	void Set(Block* block, VectorReg reg, Value value) {
		EXIT_IF(RegIndex(reg) >= NumVectorRegs);
		block->ssa_vreg_values[RegIndex(reg)] = value;
	}
	const Value& Get(Block* block, GotoVariable value) {
		return goto_variables.Get(block, value.index);
	}
	void Set(Block* block, GotoVariable value, Value definition) {
		goto_variables.Set(block, value.index, definition);
	}

	const Value& Get(Block* block, SccTag) { return block->ssa_flag_values[0]; }
	void         Set(Block* block, SccTag, Value value) { block->ssa_flag_values[0] = value; }
	const Value& Get(Block* block, ExecTag) { return block->ssa_flag_values[1]; }
	void         Set(Block* block, ExecTag, Value value) { block->ssa_flag_values[1] = value; }
	const Value& Get(Block* block, ExecLoTag) { return block->ssa_flag_values[2]; }
	void         Set(Block* block, ExecLoTag, Value value) { block->ssa_flag_values[2] = value; }
	const Value& Get(Block* block, ExecHiTag) { return block->ssa_flag_values[3]; }
	void         Set(Block* block, ExecHiTag, Value value) { block->ssa_flag_values[3] = value; }
	const Value& Get(Block* block, VccTag) { return block->ssa_flag_values[4]; }
	void         Set(Block* block, VccTag, Value value) { block->ssa_flag_values[4] = value; }
	const Value& Get(Block* block, VccLoTag) { return block->ssa_flag_values[5]; }
	void         Set(Block* block, VccLoTag, Value value) { block->ssa_flag_values[5] = value; }
	const Value& Get(Block* block, VccHiTag) { return block->ssa_flag_values[6]; }
	void         Set(Block* block, VccHiTag, Value value) { block->ssa_flag_values[6] = value; }
	const Value& Get(Block* block, M0Tag) { return block->ssa_flag_values[7]; }
	void         Set(Block* block, M0Tag, Value value) { block->ssa_flag_values[7] = value; }

	GotoDefinitions goto_variables;
};

Value InitialValue(ScalarReg) {
	return Value(0u);
}
Value InitialValue(ThreadBitScalarReg) {
	return Value(false);
}
Value InitialValue(ScalarMaskTag) {
	return Value(false);
}
Value InitialValue(VectorReg) {
	return Value(0u);
}
Value InitialValue(GotoVariable) {
	return Value(false);
}
Value InitialValue(SccTag) {
	return Value(false);
}
Value InitialValue(ExecTag) {
	return Value(false);
}
Value InitialValue(ExecLoTag) {
	return Value(0u);
}
Value InitialValue(ExecHiTag) {
	return Value(0u);
}
Value InitialValue(VccTag) {
	return Value(false);
}
Value InitialValue(VccLoTag) {
	return Value(0u);
}
Value InitialValue(VccHiTag) {
	return Value(0u);
}
Value InitialValue(M0Tag) {
	return Value(0u);
}

enum class ReadStep { Start, SetValue, PushPhiArgument };

struct ReadState {
	Block*   block = nullptr;
	Value    result;
	Inst*    phi  = nullptr;
	size_t   pred = 0;
	ReadStep step = ReadStep::Start;
};

class Pass {
public:
	template <typename T>
	void Write(T variable, Block* block, Value value) {
		definitions.Set(block, variable, value);
	}

	template <typename T>
	Value Read(T variable, Block* root) {
		stack.assign({{}, {.block = root}});
		const auto prepare_phi = [&]() {
			auto&      state        = stack.back();
			const auto predecessors = state.block->ImmPredecessors();
			if (state.pred == predecessors.size()) {
				const auto result = TryRemoveTrivialPhi(*state.phi, InitialValue(variable));
				const auto block  = state.block;
				stack.pop_back();
				stack.back().result = result;
				Write(variable, block, result);
				return;
			}
			state.step = ReadStep::PushPhiArgument;
			stack.push_back({.block = predecessors[state.pred]});
		};

		do {
			auto& state = stack.back();
			auto* block = state.block;
			switch (state.step) {
				case ReadStep::Start: {
					if (const auto& def = definitions.Get(block, variable); !def.IsEmpty()) {
						state.result = def;
					} else if (!block->IsSsaSealed()) {
						auto& phi = *block->PrependNewInst(block->begin(), ValueOpcode::Phi);
						phi.SetFlags(InitialValue(variable).GetType());
						incomplete_phis[block].emplace_back(Variable(variable), &phi);
						state.result                               = Value(&phi);
					} else if (const auto predecessors = block->ImmPredecessors();
					           predecessors.size() == 1) {
						state.step = ReadStep::SetValue;
						stack.push_back({.block = predecessors.front()});
						break;
					} else {
						auto& phi = *block->PrependNewInst(block->begin(), ValueOpcode::Phi);
						phi.SetFlags(InitialValue(variable).GetType());
						Write(variable, block, Value(&phi));
						state.phi = &phi;
						prepare_phi();
						break;
					}
					[[fallthrough]];
				}
				case ReadStep::SetValue: {
					const auto result = state.result;
					Write(variable, block, result);
					stack.pop_back();
					stack.back().result = result;
					break;
				}
				case ReadStep::PushPhiArgument: {
					const auto predecessors = block->ImmPredecessors();
					state.phi->AddPhiOperand(predecessors[state.pred], state.result);
					state.pred++;
					prepare_phi();
					break;
				}
			}
		} while (stack.size() > 1);
		return stack.back().result;
	}

	void Seal(Block* block) {
		if (const auto found = incomplete_phis.find(block); found != incomplete_phis.end()) {
			// Same order and last-write-wins as the former std::map<Variable, Inst*> (perf,
			// uncommitted: one node per Phi before).
			auto phis = std::move(found->second);
			found->second.clear();
			std::stable_sort(phis.begin(), phis.end(),
			                 [](const auto& a, const auto& b) { return a.first < b.first; });
			for (size_t index = 0; index < phis.size(); index++) {
				if (index + 1 < phis.size() && !(phis[index].first < phis[index + 1].first)) {
					continue;
				}
				auto& [variable, phi] = phis[index];
				std::visit([&](auto value) { AddPhiOperands(value, *phi, block); }, variable);
			}
			// Reading a variable never adds an incomplete Phi to the block being sealed.
			EXIT_IF(!incomplete_phis[block].empty());
		}
		block->SsaSeal();
	}

private:
	template <typename T>
	Value AddPhiOperands(T variable, Inst& phi, Block* block) {
		for (auto* predecessor: block->ImmPredecessors()) {
			phi.AddPhiOperand(predecessor, Read(variable, predecessor));
		}
		return TryRemoveTrivialPhi(phi, InitialValue(variable));
	}

	Value TryRemoveTrivialPhi(Inst& phi, Value initial_value) {
		Value same;
		for (size_t index = 0; index < phi.NumArgs(); index++) {
			const auto operand = phi.Arg(index).Resolve();
			if (operand == same.Resolve() || operand == Value(&phi)) {
				continue;
			}
			if (!same.IsEmpty()) {
				return Value(&phi);
			}
			same = operand;
		}
		if (same.IsEmpty()) {
			same = initial_value;
		}
		const auto users = phi.Uses();
		phi.ReplaceUsesWith(same);
		for (const auto& use: users) {
			if (use.user->GetOpcode() == ValueOpcode::Phi) {
				TryRemoveTrivialPhi(*use.user, initial_value);
			}
		}
		return same;
	}

	std::unordered_map<Block*, std::vector<std::pair<Variable, Inst*>>> incomplete_phis;
	DefTable                                                            definitions;
	std::vector<ReadState>                                              stack;
};

void VisitInstruction(Pass& pass, Block* block, Inst& inst) {
	switch (inst.GetOpcode()) {
		case ValueOpcode::SetScalarRegister:
			pass.Write(inst.Arg(0).ScalarRegister(), block, inst.Arg(1));
			break;
		case ValueOpcode::SetThreadBitScalarRegister:
			pass.Write(ThreadBitScalarReg {inst.Arg(0).ScalarRegister()}, block, inst.Arg(1));
			break;
		case ValueOpcode::SetScalarMaskTag:
			pass.Write(ScalarMaskTag {inst.Arg(0).ScalarRegister()}, block, inst.Arg(1));
			break;
		case ValueOpcode::SetVectorRegister:
			pass.Write(inst.Arg(0).VectorRegister(), block, inst.Arg(1));
			break;
		case ValueOpcode::SetGotoVariable:
			pass.Write(GotoVariable {inst.Arg(0).U32()}, block, inst.Arg(1));
			break;
		case ValueOpcode::SetScc: pass.Write(SccTag {}, block, inst.Arg(0)); break;
		case ValueOpcode::SetExec: pass.Write(ExecTag {}, block, inst.Arg(0)); break;
		case ValueOpcode::SetExecLo: pass.Write(ExecLoTag {}, block, inst.Arg(0)); break;
		case ValueOpcode::SetExecHi: pass.Write(ExecHiTag {}, block, inst.Arg(0)); break;
		case ValueOpcode::SetVcc: pass.Write(VccTag {}, block, inst.Arg(0)); break;
		case ValueOpcode::SetVccLo: pass.Write(VccLoTag {}, block, inst.Arg(0)); break;
		case ValueOpcode::SetVccHi: pass.Write(VccHiTag {}, block, inst.Arg(0)); break;
		case ValueOpcode::SetM0: pass.Write(M0Tag {}, block, inst.Arg(0)); break;
		case ValueOpcode::GetScalarRegister:
			inst.ReplaceUsesWith(pass.Read(inst.Arg(0).ScalarRegister(), block));
			break;
		case ValueOpcode::GetThreadBitScalarRegister:
			inst.ReplaceUsesWith(
			    pass.Read(ThreadBitScalarReg {inst.Arg(0).ScalarRegister()}, block));
			break;
		case ValueOpcode::GetScalarMaskTag:
			inst.ReplaceUsesWith(pass.Read(ScalarMaskTag {inst.Arg(0).ScalarRegister()}, block));
			break;
		case ValueOpcode::GetVectorRegister:
			inst.ReplaceUsesWith(pass.Read(inst.Arg(0).VectorRegister(), block));
			break;
		case ValueOpcode::GetGotoVariable:
			inst.ReplaceUsesWith(pass.Read(GotoVariable {inst.Arg(0).U32()}, block));
			break;
		case ValueOpcode::GetScc: inst.ReplaceUsesWith(pass.Read(SccTag {}, block)); break;
		case ValueOpcode::GetExec: inst.ReplaceUsesWith(pass.Read(ExecTag {}, block)); break;
		case ValueOpcode::GetExecLo: inst.ReplaceUsesWith(pass.Read(ExecLoTag {}, block)); break;
		case ValueOpcode::GetExecHi: inst.ReplaceUsesWith(pass.Read(ExecHiTag {}, block)); break;
		case ValueOpcode::GetVcc: inst.ReplaceUsesWith(pass.Read(VccTag {}, block)); break;
		case ValueOpcode::GetVccLo: inst.ReplaceUsesWith(pass.Read(VccLoTag {}, block)); break;
		case ValueOpcode::GetVccHi: inst.ReplaceUsesWith(pass.Read(VccHiTag {}, block)); break;
		case ValueOpcode::GetM0: inst.ReplaceUsesWith(pass.Read(M0Tag {}, block)); break;
		default: break;
	}
}

} // namespace

void RewriteToSsa(const BlockList& blocks) {
	Pass pass;
	for (auto* block: blocks) {
		for (auto& inst: *block) {
			VisitInstruction(pass, block, inst);
		}
	}
	for (auto* block: blocks) {
		pass.Seal(block);
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
