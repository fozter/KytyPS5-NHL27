#include "common/assert.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/BufferFormat.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <fmt/format.h>
#include <xbyak/xbyak.h>
#include <map>
#include <new>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler::IR {

namespace {

[[noreturn]] void Fail(std::string_view message) {
	EXIT("shader IR validation failed: %s", std::string(message).c_str());
	std::abort();
}

// Open-addressing pointer -> index table for ValidateProgram (perf, uncommitted).
template <typename T>
class PointerIndex {
public:
	explicit PointerIndex(size_t expected) {
		size_t capacity = 16;
		while (capacity < expected * 2) {
			capacity <<= 1;
		}
		slots.assign(capacity, {nullptr, 0});
		mask  = capacity - 1;
		shift = 64 - std::countr_zero(capacity);
	}
	// False if the key is already present.
	bool Insert(const T* key, size_t value) {
		if ((count + 1) * 2 > slots.size()) {
			Grow();
		}
		for (size_t index = Hash(key);; index = (index + 1) & mask) {
			if (slots[index].first == key) {
				return false;
			}
			if (slots[index].first == nullptr) {
				slots[index] = {key, value};
				count++;
				return true;
			}
		}
	}
	[[nodiscard]] const size_t* Find(const T* key) const {
		if (key == nullptr) {
			return nullptr;
		}
		for (size_t index = Hash(key);; index = (index + 1) & mask) {
			if (slots[index].first == key) {
				return &slots[index].second;
			}
			if (slots[index].first == nullptr) {
				return nullptr;
			}
		}
	}
	[[nodiscard]] bool   contains(const T* key) const { return Find(key) != nullptr; }
	[[nodiscard]] size_t at(const T* key) const {
		const auto* found = Find(key);
		EXIT_IF(found == nullptr);
		return *found;
	}

private:
	[[nodiscard]] size_t Hash(const T* key) const {
		return static_cast<size_t>((reinterpret_cast<uintptr_t>(key) * 0x9E3779B97F4A7C15ull) >> shift);
	}
	void Grow() {
		const auto old = std::move(slots);
		slots.assign(old.size() * 2, {nullptr, 0});
		mask  = slots.size() - 1;
		shift = 64 - std::countr_zero(slots.size());
		count = 0;
		for (const auto& [key, value]: old) {
			if (key != nullptr) {
				Insert(key, value);
			}
		}
	}
	std::vector<std::pair<const T*, size_t>> slots;
	size_t                                   mask  = 0;
	int                                      shift = 64;
	size_t                                   count = 0;
};

bool IsRegisterStatePseudo(ValueOpcode opcode) {
	switch (opcode) {
		case ValueOpcode::GetThreadBitScalarRegister:
		case ValueOpcode::SetThreadBitScalarRegister:
		case ValueOpcode::GetScalarMaskTag:
		case ValueOpcode::SetScalarMaskTag:
		case ValueOpcode::GetScalarRegister:
		case ValueOpcode::SetScalarRegister:
		case ValueOpcode::GetVectorRegister:
		case ValueOpcode::SetVectorRegister:
		case ValueOpcode::GetGotoVariable:
		case ValueOpcode::SetGotoVariable:
		case ValueOpcode::GetScc:
		case ValueOpcode::SetScc:
		case ValueOpcode::GetExec:
		case ValueOpcode::SetExec:
		case ValueOpcode::GetExecLo:
		case ValueOpcode::SetExecLo:
		case ValueOpcode::GetExecHi:
		case ValueOpcode::SetExecHi:
		case ValueOpcode::GetVcc:
		case ValueOpcode::SetVcc:
		case ValueOpcode::GetVccLo:
		case ValueOpcode::SetVccLo:
		case ValueOpcode::GetVccHi:
		case ValueOpcode::SetVccHi:
		case ValueOpcode::GetM0:
		case ValueOpcode::SetM0: return true;
		default: return false;
	}
}

bool IsRuntimeRead(ValueOpcode opcode) {
	return opcode == ValueOpcode::ReadConstBuffer ||
	       AddressOpcodeInfoOf(opcode).access == AddressAccess::Read;
}

bool EquivalentValue(const ResourcePlan& program, Value left, Value right,
                     std::vector<std::pair<const Inst*, const Inst*>>& visited) {
	left  = left.Resolve();
	right = right.Resolve();
	if (left == right) {
		return true;
	}
	if (left.IsImmediate() || right.IsImmediate() || left.GetType() != right.GetType()) {
		return false;
	}
	const auto* lhs = left.TryInstruction();
	const auto* rhs = right.TryInstruction();
	if (lhs == nullptr || rhs == nullptr || lhs->GetOpcode() != rhs->GetOpcode() ||
	    lhs->NumArgs() != rhs->NumArgs()) {
		return false;
	}
	if (std::ranges::find(visited, std::pair {lhs, rhs}) != visited.end()) {
		return true;
	}
	visited.emplace_back(lhs, rhs);
	if (IsRuntimeRead(lhs->GetOpcode())) {
		const auto li = lhs->Flags<MemoryFlags>().index;
		const auto ri = rhs->Flags<MemoryFlags>().index;
		if (li >= program.memory_info.size() || ri >= program.memory_info.size() ||
		    program.memory_info[li] != program.memory_info[ri]) {
			return false;
		}
	} else if (lhs->GetOpcode() != ValueOpcode::ReadConst &&
	           lhs->Flags<uint64_t>() != rhs->Flags<uint64_t>()) {
		return false;
	}
	for (size_t index = 0; index < lhs->NumArgs(); index++) {
		if (lhs->GetOpcode() == ValueOpcode::Phi && lhs->PhiBlock(index) != rhs->PhiBlock(index)) {
			return false;
		}
		if (!EquivalentValue(program, lhs->Arg(index), rhs->Arg(index), visited)) {
			return false;
		}
	}
	return true;
}

} // namespace

ResourcePlan::ResourcePlan() = default;
ResourcePlan::ResourcePlan(ResourcePlan&&) noexcept = default;

ResourcePlan::~ResourcePlan() {
	for (auto& inst: value_storage) {
		inst.Invalidate();
	}
}

ResourcePlan& ResourcePlan::operator=(ResourcePlan&& other) noexcept {
	if (this != &other) {
		this->~ResourcePlan();
		new (this) ResourcePlan(std::move(other));
	}
	return *this;
}

Program::~Program() {
	// Planning expressions can refer to block values but outlive block storage in the base class.
	for (auto& inst: value_storage) {
		inst.Invalidate();
	}
	// Values may cross block boundaries. Detach all arguments before any block starts destroying
	// its instruction storage so reverse-use links always point to live definitions.
	for (auto* block: blocks) {
		for (auto& inst: *block) {
			inst.Invalidate();
		}
	}
}

Program& Program::operator=(Program&& other) noexcept {
	if (this != &other) {
		this->~Program();
		new (this) Program(std::move(other));
	}
	return *this;
}

CompiledShaderInfo Program::TakeCompiledInfo() && {
	CompiledShaderInfo result {
	    .stage           = stage,
	    .shader_hash     = shader_hash,
	    .wave_size       = wave_size,
	    .user_data_base  = user_data_base,
	    .user_data_count = user_data_count,
	    .scratch_dwords  = scratch_dwords,
	    .has_address_writes = has_address_writes,
	    .info            = std::move(info),
	    .bindings        = std::move(bindings),
	};
	for (const auto& output: result.info.outputs) {
		if (output.kind == StageOutputKind::Parameter && output.index < 32) {
			result.param_export_mask |= 1u << output.index;
		}
	}
	return result;
}

bool EquivalentValue(const ResourcePlan& program, Value left, Value right) {
	std::vector<std::pair<const Inst*, const Inst*>> visited;
	return EquivalentValue(program, left, right, visited);
}

bool EquivalentSelectorDomain(const ResourcePlan& program,
                              const DescriptorSource::IndirectDescriptor& left,
                              const DescriptorSource::IndirectDescriptor& right) {
	return left.selector == right.selector && left.signed_count == right.signed_count &&
	       left.workgroup_axis == right.workgroup_axis &&
	       EquivalentValue(program, left.key_count, right.key_count) &&
	       EquivalentValue(program, left.selector_first, right.selector_first) &&
	       EquivalentValue(program, left.selector_mask, right.selector_mask);
}

Value ResolveInvariantPhi(const ResourcePlan& program, Value value) {
	value            = value.Resolve();
	const auto* root = value.TryInstruction();
	if (root == nullptr || root->GetOpcode() != ValueOpcode::Phi) {
		return value;
	}
	Value                           invariant;
	std::vector<Value>              pending {value};
	std::unordered_set<const Inst*> visited;
	while (!pending.empty()) {
		const auto current = pending.back().Resolve();
		pending.pop_back();
		const auto* inst = current.TryInstruction();
		if (inst != nullptr && inst->GetOpcode() == ValueOpcode::Phi) {
			if (!visited.insert(inst).second) {
				continue;
			}
			for (size_t index = 0; index < inst->NumArgs(); index++) {
				pending.push_back(inst->Arg(index));
			}
			continue;
		}
		if (invariant.IsEmpty()) {
			invariant = current;
		} else if (!EquivalentValue(program, invariant, current)) {
			return {};
		}
	}
	return invariant;
}

Value ResolveActiveU32(Value value, Value active) {
	for (uint32_t depth = 0; depth <= 32; ++depth) {
		value = value.Resolve();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr || inst->GetOpcode() != ValueOpcode::SelectU32 ||
		    inst->Arg(0).Resolve() != active)
			return value;
		value = inst->Arg(1);
	}
	return {};
}

uint32_t StorageBufferElementBits(const Program& program, const MemoryInfo& memory) {
	if (!memory.formatted) return memory.data_bits;
	const auto format = memory.typed
	                        ? Format::DecodeTBufferFormat(memory.data_format, memory.number_format)
	                        : program.info.buffers[memory.resource].descriptor_format;
	const auto info = Format::GetFormatInfo(format);
	return info.type == Format::ComponentType::Unknown || info.packed_bitfield
	           ? 32u : info.component_bits[0];
}

bool HasShaderMemoryWrites(const Program& program) {
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto op     = inst.GetOpcode();
			const auto buffer = BufferAccessOf(op);
			const auto image  = ImageOpcodeInfoOf(op).access;
			if (buffer == BufferAccess::Write || buffer == BufferAccess::Atomic ||
			    image == ImageAccess::Write || image == ImageAccess::Atomic ||
			    AddressOpcodeInfoOf(op).access == AddressAccess::Write) {
				return true;
			}
		}
	}
	return false;
}

namespace {
void ValidateProgramChecks(const Program& program, bool require_ssa);
} // namespace

void ValidateProgram(const Program& program, bool require_ssa) {
	// Diagnostic switch (uncommitted): KYTY_IR_VALIDATE=0 skips validation, which only ever EXITs
	// on malformed IR. Its time is summed in g_diag_validate_ms ([shphase] cum_val).
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_IR_VALIDATE");
		return value == nullptr || value[0] != '0';
	}();
	if (!enabled) {
		return;
	}
	const auto begin = std::chrono::steady_clock::now();
	ValidateProgramChecks(program, require_ssa);
	g_diag_validate_ms +=
	    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
}

namespace {
void ValidateProgramChecks(const Program& program, bool require_ssa) {
	if (program.blocks.size() != program.block_storage.size()) {
		return Fail("value IR block storage is inconsistent");
	}
	// Flat indices, small vectors, precomputed reverse-use flags and a dominator tree instead of
	// node maps, per-block sets, per-argument use searches and a block x block bit matrix (perf,
	// uncommitted): validation runs twice per shader and was ~20% of NHL 27's translation. Same
	// checks in the same order.
	size_t instruction_count = 0;
	for (const auto& storage: program.block_storage) {
		instruction_count += storage != nullptr ? storage->Instructions().size() : 0;
	}
	PointerIndex<Block>          block_indices(program.blocks.size());
	std::unordered_set<uint32_t> block_ids;
	PointerIndex<Inst>           instruction_ordinals(instruction_count);
	std::vector<const Inst*>     instructions;
	std::vector<size_t>          positions;
	std::vector<size_t>          arg_offsets;
	std::vector<size_t>          block_of;
	instructions.reserve(instruction_count);
	block_of.reserve(instruction_count);
	positions.reserve(instruction_count);
	arg_offsets.reserve(instruction_count);
	size_t total_args = 0;
	for (size_t block_index = 0; block_index < program.blocks.size(); block_index++) {
		const auto* block = program.blocks[block_index];
		if (block == nullptr || program.block_storage[block_index] == nullptr ||
		    block != program.block_storage[block_index].get()) {
			return Fail("value IR block pointer is inconsistent");
		}
		if (!block_indices.Insert(block, block_index)) {
			return Fail("value IR block pointer is duplicated");
		}
		if (block->id == UINT32_MAX) {
			return Fail("value IR block uses the reserved exit id");
		}
		if (!block_ids.insert(block->id).second) {
			return Fail("value IR block id is duplicated");
		}
		size_t position = 0;
		for (const auto& inst: *block) {
			if (!instruction_ordinals.Insert(&inst, instructions.size())) {
				return Fail("value IR instruction is duplicated");
			}
			instructions.push_back(&inst);
			block_of.push_back(block_index);
			positions.push_back(position++);
			arg_offsets.push_back(total_args);
			total_args += inst.NumArgs();
		}
	}
	// has_reverse_use[arg_offsets[user] + operand]: the argument's definition lists that use.
	std::vector<uint8_t> has_reverse_use(total_args, 0);
	// arg_definitions[arg_offsets[user] + operand]: ordinal of the argument's definition.
	std::vector<size_t> arg_definitions(total_args, SIZE_MAX);
	for (const auto* definition: instructions) {
		for (const auto& use: definition->Uses()) {
			const auto* ordinal = instruction_ordinals.Find(use.user);
			if (ordinal != nullptr && use.operand < use.user->NumArgs() &&
			    use.user->Arg(use.operand).TryInstruction() == definition) {
				has_reverse_use[arg_offsets[*ordinal] + use.operand] = 1;
			}
		}
	}
	if (!program.blocks.empty() && !program.blocks.front()->ImmPredecessors().empty()) {
		return Fail("value IR entry block has a predecessor");
	}

	const auto contains = [](const std::vector<const Block*>& list, const Block* block) {
		return std::ranges::find(list, block) != list.end();
	};
	std::vector<const Block*> predecessors;
	std::vector<const Block*> successors;
	std::vector<const Block*> expected_successors;
	std::vector<const Block*> incoming_blocks;
	size_t                    ordinal = 0;
	for (size_t block_index = 0; block_index < program.blocks.size(); block_index++) {
		const auto* block = program.blocks[block_index];
		predecessors.clear();
		for (const auto* predecessor: block->ImmPredecessors()) {
			if (predecessor == nullptr || !block_indices.contains(predecessor)) {
				return Fail("value IR block has a foreign predecessor");
			}
			if (contains(predecessors, predecessor)) {
				return Fail("value IR block predecessor is duplicated");
			}
			predecessors.push_back(predecessor);
			if (std::ranges::find(predecessor->ImmSuccessors(), block) ==
			    predecessor->ImmSuccessors().end()) {
				return Fail("value IR predecessor edge is not reciprocal");
			}
		}

		successors.clear();
		for (const auto* successor: block->ImmSuccessors()) {
			if (successor == nullptr || !block_indices.contains(successor)) {
				return Fail("value IR block has a foreign successor");
			}
			if (contains(successors, successor)) {
				return Fail("value IR block successor is duplicated");
			}
			successors.push_back(successor);
			if (std::ranges::find(successor->ImmPredecessors(), block) ==
			    successor->ImmPredecessors().end()) {
				return Fail("value IR successor edge is not reciprocal");
			}
		}

		expected_successors.clear();
		const auto add_target = [&](const Block* target) {
			if (target == nullptr || !block_indices.contains(target)) {
				return false;
			}
			if (!contains(expected_successors, target)) {
				expected_successors.push_back(target);
			}
			return true;
		};
		const auto& terminator             = block->terminator;
		const auto  validate_control_value = [&](Value value, Type type) {
			if (value.IsEmpty() || value.GetType() != type) {
				return false;
			}
			const auto* definition = value.TryInstruction();
			return definition == nullptr || instruction_ordinals.contains(definition);
		};
		switch (terminator.kind) {
			case CFG::TerminatorKind::Branch:
				if (!add_target(terminator.true_block)) {
					return Fail("value IR branch target is missing");
				}
				break;
			case CFG::TerminatorKind::ConditionalBranch:
				if (!add_target(terminator.true_block) || !add_target(terminator.false_block)) {
					return Fail("value IR conditional branch target is missing");
				}
				if (!validate_control_value(block->condition, Type::U1)) {
					return Fail("value IR conditional branch condition is invalid");
				}
				break;
			case CFG::TerminatorKind::IndirectBranch: {
				if (!validate_control_value(block->indirect_target, Type::U32)) {
					return Fail("value IR indirect branch selector is invalid");
				}
				std::unordered_set<uint32_t> values;
				for (const auto& branch: terminator.cases) {
					if (!values.insert(branch.value).second) {
						return Fail("value IR indirect selector value is duplicated");
					}
					if (!add_target(branch.target)) {
						return Fail("value IR indirect branch target is missing");
					}
				}
				break;
			}
			case CFG::TerminatorKind::Return:
			case CFG::TerminatorKind::Unsupported: break;
		}
		if ((terminator.merge_block != nullptr &&
		     !block_indices.contains(terminator.merge_block)) ||
		    (terminator.continue_block != nullptr &&
		     !block_indices.contains(terminator.continue_block))) {
			return Fail("value IR structured control target is missing");
		}
		if (successors.size() != expected_successors.size() ||
		    !std::ranges::all_of(successors, [&](const Block* successor) {
			    return contains(expected_successors, successor);
		    })) {
			return Fail("value IR terminator and successor graph disagree");
		}

		bool saw_non_phi = false;
		for (const auto& inst: *block) {
			const auto inst_ordinal = ordinal++;
			if (inst.Parent() != block) {
				return Fail("value IR instruction has the wrong parent block");
			}
			if (inst.GetOpcode() == ValueOpcode::Phi) {
				if (saw_non_phi) {
					return Fail("value IR Phi appears after a non-Phi instruction");
				}
				if (inst.NumPhiBlocks() != inst.NumArgs()) {
					return Fail("value IR Phi parent table is inconsistent");
				}
				if (inst.NumArgs() == 0) {
					return Fail("value IR Phi has no incoming values");
				}
				incoming_blocks.clear();
				for (size_t arg_index = 0; arg_index < inst.NumArgs(); arg_index++) {
					const auto* predecessor = inst.PhiBlock(arg_index);
					if (predecessor == nullptr || !block_indices.contains(predecessor) ||
					    !contains(predecessors, predecessor)) {
						return Fail("value IR Phi has a foreign or non-predecessor parent");
					}
					if (contains(incoming_blocks, predecessor)) {
						return Fail("value IR Phi parent is duplicated");
					}
					incoming_blocks.push_back(predecessor);
					if (inst.Arg(arg_index).GetType() != inst.GetType()) {
						return Fail("value IR Phi incoming type does not match its result");
					}
				}
				if (incoming_blocks.size() != predecessors.size()) {
					return Fail("value IR Phi does not cover every predecessor");
				}
			} else {
				saw_non_phi = true;
			}
			if (require_ssa && IsRegisterStatePseudo(inst.GetOpcode())) {
				return Fail(fmt::format("register-state pseudo {} survived SSA rewrite",
				                        ValueOpcodeName(inst.GetOpcode())));
			}
			if (inst.GetOpcode() != ValueOpcode::Phi && inst.GetOpcode() != ValueOpcode::Identity &&
			    inst.GetType() == Type::Opaque) {
				return Fail(fmt::format("untyped opcode {} survived translation",
				                        ValueOpcodeName(inst.GetOpcode())));
			}
			const bool fixed_signature =
			    inst.GetOpcode() != ValueOpcode::Phi && inst.GetOpcode() != ValueOpcode::Identity;
			if (fixed_signature && inst.NumArgs() != NumArgsOf(inst.GetOpcode())) {
				return Fail(fmt::format("{} has {} arguments, expected {}",
				                        ValueOpcodeName(inst.GetOpcode()), inst.NumArgs(),
				                        NumArgsOf(inst.GetOpcode())));
			}
			if (inst.GetOpcode() == ValueOpcode::ReadConstBuffer) {
				const auto memory_index = inst.Flags<MemoryFlags>().index;
				if (memory_index >= program.memory_info.size()) {
					return Fail(fmt::format("{} has an invalid memory-info index",
					                        ValueOpcodeName(inst.GetOpcode())));
				}
				const auto& memory = program.memory_info[memory_index];
				if (memory.kind != ResourceKind::ScalarBuffer &&
				    memory.kind != ResourceKind::IndirectBuffer) {
					return Fail(fmt::format("{} has an invalid scalar-memory resource kind",
					                        ValueOpcodeName(inst.GetOpcode())));
				}
				const bool valid_group_width =
				    memory.component_count == 1u || memory.component_count == 2u ||
				    memory.component_count == 4u || memory.component_count == 8u ||
				    memory.component_count == 16u;
				if (memory.data_bits != 32u || memory.data_dwords != 1u || !valid_group_width ||
				    memory.component_index >= memory.component_count) {
					return Fail(fmt::format("{} has inconsistent scalar-memory metadata",
					                        ValueOpcodeName(inst.GetOpcode())));
				}
			}
			const auto address_info = AddressOpcodeInfoOf(inst.GetOpcode());
			if (address_info.access != AddressAccess::None) {
				const auto memory_index = inst.Flags<MemoryFlags>().index;
				if (memory_index >= program.memory_info.size()) {
					return Fail(fmt::format("{} has an invalid memory-info index",
					                        ValueOpcodeName(inst.GetOpcode())));
				}
				const auto& memory = program.memory_info[memory_index];
				if (!IsAddressResourceKind(memory.kind) ||
				    (memory.kind == ResourceKind::ScalarAddress &&
				     inst.GetOpcode() != ValueOpcode::LoadAddressU32)) {
					return Fail(fmt::format("{} has an invalid address resource kind",
					                        ValueOpcodeName(inst.GetOpcode())));
				}
				const bool scalar_address = memory.kind == ResourceKind::ScalarAddress;
				const bool valid_group_width =
				    scalar_address
				        ? memory.component_count == 1u || memory.component_count == 2u ||
				              memory.component_count == 4u || memory.component_count == 8u ||
				              memory.component_count == 16u
				        : memory.component_count >= 1u && memory.component_count <= 4u;
				if (memory.data_bits != address_info.data_bits || memory.data_dwords != 1u ||
				    !valid_group_width || memory.component_index >= memory.component_count ||
				    memory.sampler != 0u) {
					return Fail(fmt::format("{} has inconsistent address-memory metadata",
					                        ValueOpcodeName(inst.GetOpcode())));
				}
			}
			const auto buffer_components = BufferComponentCount(inst.GetOpcode());
			if (buffer_components != 0u) {
				const auto memory_index = inst.Flags<MemoryFlags>().index;
				if (memory_index >= program.memory_info.size()) {
					return Fail(fmt::format("{} has an invalid memory-info index",
					                        ValueOpcodeName(inst.GetOpcode())));
				}
				const auto& memory = program.memory_info[memory_index];
				const bool  vector_buffer = memory.kind == ResourceKind::Buffer ||
				                            memory.kind == ResourceKind::IndirectBuffer;
				if (!vector_buffer && memory.kind != ResourceKind::ScalarBuffer) {
					return Fail(fmt::format("{} has a non-buffer resource kind",
					                        ValueOpcodeName(inst.GetOpcode())));
				}
				if (memory.kind == ResourceKind::IndirectBuffer &&
				    !memory.SupportsIndirectBufferLoad(inst.GetOpcode())) {
					return Fail("indirect buffer requires a scalar, raw 8/16-bit, raw DWORD x1/x2/x3/x4, "
					            "or formatted X load");
				}
				const auto access = BufferAccessOf(inst.GetOpcode());
				const bool packed_formatted =
				    (access == BufferAccess::Read || access == BufferAccess::Write) && memory.formatted &&
				    memory.data_bits == 16u && (memory.component_count + 1u) / 2u == buffer_components;
				if (buffer_components > 1u &&
				    (!vector_buffer || (memory.data_bits != 32u && !packed_formatted) ||
				     memory.data_dwords != buffer_components || memory.component_index != 0u)) {
					return Fail(fmt::format("{} has inconsistent native-wide metadata",
					                        ValueOpcodeName(inst.GetOpcode())));
				}
				if (buffer_components == 1u &&
				    (inst.GetOpcode() == ValueOpcode::LoadBufferU32 ||
				     inst.GetOpcode() == ValueOpcode::StoreBufferU32) &&
				    memory.data_dwords != 1u) {
					return Fail(fmt::format("{} retains scalar-sibling width metadata",
					                        ValueOpcodeName(inst.GetOpcode())));
				}
			}
			const auto shared_components = SharedComponentCount(inst.GetOpcode());
			if (shared_components != 0u) {
				const auto memory_index = inst.Flags<MemoryFlags>().index;
				if (memory_index >= program.memory_info.size()) {
					return Fail(fmt::format("{} has an invalid memory-info index",
					                        ValueOpcodeName(inst.GetOpcode())));
				}
				const auto& memory = program.memory_info[memory_index];
				if ((memory.kind != ResourceKind::Lds && memory.kind != ResourceKind::Gds) ||
				    memory.resource != 0u || memory.sampler != 0u || memory.component_count == 0u ||
				    memory.component_index >= memory.component_count) {
					return Fail(fmt::format("{} has invalid shared-memory metadata",
					                        ValueOpcodeName(inst.GetOpcode())));
				}
				uint32_t expected_bits = 32u;
				if (inst.GetOpcode() == ValueOpcode::LoadSharedU8 ||
				    inst.GetOpcode() == ValueOpcode::WriteSharedU8) {
					expected_bits = 8u;
				} else if (inst.GetOpcode() == ValueOpcode::LoadSharedU16 ||
				           inst.GetOpcode() == ValueOpcode::WriteSharedU16) {
					expected_bits = 16u;
				}
				if (memory.data_bits != expected_bits || memory.data_dwords != shared_components ||
				    (shared_components > 1u && memory.component_index != 0u)) {
					return Fail(fmt::format("{} has inconsistent shared-memory width",
					                        ValueOpcodeName(inst.GetOpcode())));
				}
			}
			const auto image_info = ImageOpcodeInfoOf(inst.GetOpcode());
			if (image_info.access != ImageAccess::None) {
				const auto memory_index = inst.Flags<MemoryFlags>().index;
				if (memory_index >= program.memory_info.size()) {
					return Fail(fmt::format("{} has an invalid memory-info index",
					                        ValueOpcodeName(inst.GetOpcode())));
				}
				const auto& memory = program.memory_info[memory_index];
				if (memory.kind != ResourceKind::Image ||
				    image_info.resource_class == ImageResourceClass::None) {
					return Fail(fmt::format("{} has invalid image-memory metadata",
					                        ValueOpcodeName(inst.GetOpcode())));
				}
			}
			if (inst.GetOpcode() == ValueOpcode::SetAttribute &&
			    inst.Flags<ExportFlags>().index >= program.export_info.size()) {
				return Fail("SetAttribute has an invalid export-info index");
			}
			uint32_t composite_components = 0u;
			switch (inst.GetOpcode()) {
				case ValueOpcode::CompositeExtractU64: composite_components = 2u; break;
				case ValueOpcode::CompositeExtractU32x2: composite_components = 2u; break;
				case ValueOpcode::CompositeExtractU32x3: composite_components = 3u; break;
				case ValueOpcode::CompositeExtractU32x4: composite_components = 4u; break;
				default: break;
			}
			if (composite_components != 0u &&
			    (!inst.Arg(1).IsImmediate() || inst.Arg(1).GetType() != Type::U32 ||
			     inst.Arg(1).U32() >= composite_components)) {
				return Fail(fmt::format("{} has an invalid component index",
				                        ValueOpcodeName(inst.GetOpcode())));
			}
			for (size_t arg_index = 0; arg_index < inst.NumArgs(); arg_index++) {
				const auto arg = inst.Arg(arg_index);
				if (arg.IsEmpty()) {
					return Fail(
					    fmt::format("{} has an empty argument", ValueOpcodeName(inst.GetOpcode())));
				}
				if (fixed_signature && arg.GetType() != ArgTypeOf(inst.GetOpcode(), arg_index)) {
					return Fail(fmt::format("{} argument {} has type {}, expected {}",
					                        ValueOpcodeName(inst.GetOpcode()), arg_index,
					                        TypeName(arg.GetType()),
					                        TypeName(ArgTypeOf(inst.GetOpcode(), arg_index))));
				}
				const auto* definition = arg.TryInstruction();
				const auto* definition_ordinal =
				    definition != nullptr ? instruction_ordinals.Find(definition) : nullptr;
				if (definition != nullptr && definition_ordinal == nullptr) {
					return Fail("value IR argument has a foreign definition");
				}
				if (definition != nullptr) {
					arg_definitions[arg_offsets[inst_ordinal] + arg_index] = *definition_ordinal;
					if (has_reverse_use[arg_offsets[inst_ordinal] + arg_index] == 0) {
						return Fail(fmt::format("{} argument {} is absent from {} reverse uses",
						                        ValueOpcodeName(inst.GetOpcode()), arg_index,
						                        ValueOpcodeName(definition->GetOpcode())));
					}
				}
			}
		}
	}

	if (program.blocks.empty()) {
		return;
	}

	std::vector<bool>   reachable(program.blocks.size(), false);
	std::vector<size_t> pending {0u};
	while (!pending.empty()) {
		const auto block_index = pending.back();
		pending.pop_back();
		if (reachable[block_index]) {
			continue;
		}
		reachable[block_index] = true;
		for (const auto* successor: program.blocks[block_index]->ImmSuccessors()) {
			pending.push_back(block_indices.at(successor));
		}
	}
	if (!std::ranges::all_of(reachable, [](bool value) { return value; })) {
		return Fail("value IR contains an unreachable block");
	}

	// Immediate dominators (Cooper, Harvey, Kennedy) over a reverse postorder, then dominator-tree
	// DFS intervals. Every block is reachable here, so this is the same (reflexive) relation as
	// the former per-block bit-set fixpoint.
	const auto          block_count = program.blocks.size();
	std::vector<size_t> postorder;
	std::vector<size_t> postorder_number(block_count, 0);
	{
		postorder.reserve(block_count);
		std::vector<bool>                      visited(block_count, false);
		std::vector<std::pair<size_t, size_t>> stack {{0u, 0u}};
		visited.front() = true;
		while (!stack.empty()) {
			const auto index      = stack.back().first;
			const auto successors = program.blocks[index]->ImmSuccessors();
			if (stack.back().second < successors.size()) {
				const auto successor = block_indices.at(successors[stack.back().second++]);
				if (!visited[successor]) {
					visited[successor] = true;
					stack.emplace_back(successor, 0u);
				}
				continue;
			}
			postorder_number[index] = postorder.size();
			postorder.push_back(index);
			stack.pop_back();
		}
	}
	constexpr size_t    Undefined = SIZE_MAX;
	std::vector<size_t> immediate_dominator(block_count, Undefined);
	immediate_dominator.front() = 0;
	const auto intersect        = [&](size_t left, size_t right) {
		while (left != right) {
			while (postorder_number[left] < postorder_number[right]) {
				left = immediate_dominator[left];
			}
			while (postorder_number[right] < postorder_number[left]) {
				right = immediate_dominator[right];
			}
		}
		return left;
	};
	for (bool changed = true; changed;) {
		changed = false;
		for (auto it = postorder.rbegin(); it != postorder.rend(); ++it) {
			const auto index = *it;
			if (index == 0) {
				continue;
			}
			size_t next = Undefined;
			for (const auto* predecessor: program.blocks[index]->ImmPredecessors()) {
				const auto predecessor_index = block_indices.at(predecessor);
				if (immediate_dominator[predecessor_index] == Undefined) {
					continue;
				}
				next = next == Undefined ? predecessor_index : intersect(predecessor_index, next);
			}
			if (next != immediate_dominator[index]) {
				immediate_dominator[index] = next;
				changed                    = true;
			}
		}
	}
	std::vector<size_t> child_start(block_count + 1, 0);
	for (size_t index = 1; index < block_count; index++) {
		child_start[immediate_dominator[index] + 1]++;
	}
	for (size_t index = 0; index < block_count; index++) {
		child_start[index + 1] += child_start[index];
	}
	std::vector<size_t> children(block_count > 0 ? block_count - 1 : 0);
	{
		auto cursor = child_start;
		for (size_t index = 1; index < block_count; index++) {
			children[cursor[immediate_dominator[index]]++] = index;
		}
	}
	std::vector<size_t> enter(block_count, 0);
	std::vector<size_t> leave(block_count, 0);
	{
		size_t                                 clock = 0;
		std::vector<std::pair<size_t, size_t>> stack {{0u, child_start[0]}};
		enter.front() = clock++;
		while (!stack.empty()) {
			const auto index = stack.back().first;
			if (stack.back().second < child_start[index + 1]) {
				const auto child = children[stack.back().second++];
				enter[child]     = clock++;
				stack.emplace_back(child, child_start[child]);
				continue;
			}
			leave[index] = clock++;
			stack.pop_back();
		}
	}

	const auto dominates_index = [&](size_t definition_index, size_t use_index) {
		return enter[definition_index] <= enter[use_index] && leave[use_index] <= leave[definition_index];
	};
	const auto dominates = [&](const Block* definition, const Block* use) {
		return dominates_index(block_indices.at(definition), block_indices.at(use));
	};
	const auto control_dominates = [&](Value value, const Block* use) {
		const auto* definition = value.TryInstruction();
		return definition == nullptr || definition->Parent() == use ||
		       dominates(definition->Parent(), use);
	};

	// Every definition is in the program and in its parent block (checked above), so the
	// recorded ordinals give its block and position without further lookups.
	ordinal = 0;
	for (size_t block_index = 0; block_index < program.blocks.size(); block_index++) {
		const auto* block = program.blocks[block_index];
		for (const auto& inst: *block) {
			const auto inst_ordinal = ordinal++;
			for (size_t arg_index = 0; arg_index < inst.NumArgs(); arg_index++) {
				const auto definition = arg_definitions[arg_offsets[inst_ordinal] + arg_index];
				if (definition == SIZE_MAX) {
					continue;
				}
				const auto definition_block = block_of[definition];
				if (inst.GetOpcode() == ValueOpcode::Phi) {
					const auto predecessor = block_indices.at(inst.PhiBlock(arg_index));
					if (definition_block != predecessor && !dominates_index(definition_block, predecessor)) {
						return Fail("value IR Phi incoming definition does not dominate its edge");
					}
				} else if (definition_block == block_index) {
					if (positions[definition] >= positions[inst_ordinal]) {
						return Fail("value IR instruction uses a same-block definition before it");
					}
				} else if (!dominates_index(definition_block, block_index)) {
					return Fail("value IR instruction definition does not dominate its use");
				}
			}
		}
		const auto& info = *program.blocks[block_index];
		if (!info.condition.IsEmpty() && !control_dominates(info.condition, block)) {
			return Fail("value IR branch condition definition does not dominate its use");
		}
		if (!info.indirect_target.IsEmpty() && !control_dominates(info.indirect_target, block)) {
			return Fail("value IR indirect selector definition does not dominate its use");
		}
	}
}

} // namespace

std::string ProgramToString(const Program& program) {
	std::map<const Inst*, size_t> ids;
	size_t                        next_id = 1;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			ids.emplace(&inst, next_id++);
		}
	}
	const auto value_string = [&](Value value) {
		if (value.IsEmpty()) {
			return std::string("<null>");
		}
		if (const auto* inst = value.TryInstruction(); inst != nullptr) {
			return fmt::format("%{}", ids.at(inst));
		}
		switch (value.GetType()) {
			case Type::ScalarReg: return fmt::format("s{}", RegIndex(value.ScalarRegister()));
			case Type::VectorReg: return fmt::format("v{}", RegIndex(value.VectorRegister()));
			case Type::U1: return std::string(value.U1() ? "true" : "false");
			case Type::U8: return fmt::format("{}u8", value.U8());
			case Type::U16: return fmt::format("{}u16", value.U16());
			case Type::U32: return fmt::format("0x{:08x}", value.U32());
			case Type::U64: return fmt::format("0x{:016x}", value.U64());
			case Type::F16: return fmt::format("f16(0x{:04x})", value.F16Bits());
			case Type::F32: return fmt::format("{}f", value.F32Value());
			default: return fmt::format("<{}>", TypeName(value.GetType()));
		}
	};

	std::string text;
	for (size_t block_index = 0; block_index < program.blocks.size(); block_index++) {
		text +=
		    fmt::format("Block ${} pc=0x{:08x}..0x{:08x}\n", block_index,
		                program.blocks[block_index]->start_pc, program.blocks[block_index]->end_pc);
		for (const auto& inst: *program.blocks[block_index]) {
			const auto type = inst.GetType();
			if (type != Type::Void) {
				text +=
				    fmt::format("  %{:<5} = {}", ids.at(&inst), ValueOpcodeName(inst.GetOpcode()));
			} else {
				text += fmt::format("          {}", ValueOpcodeName(inst.GetOpcode()));
			}
			for (size_t index = 0; index < inst.NumArgs(); index++) {
				text += index == 0 ? " " : ", ";
				if (inst.GetOpcode() == ValueOpcode::Phi) {
					const auto predecessor =
					    std::ranges::find(program.blocks, inst.PhiBlock(index));
					text += fmt::format("[{}, ${}]", value_string(inst.Arg(index)),
					                    std::distance(program.blocks.begin(), predecessor));
				} else {
					text += value_string(inst.Arg(index));
				}
			}
			text +=
			    fmt::format(" ({}; uses={})\n", TypeName(Value(const_cast<Inst*>(&inst)).GetType()),
			                inst.UseCount());
		}
	}
	return text;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
