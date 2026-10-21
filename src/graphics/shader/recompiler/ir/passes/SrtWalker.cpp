#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include "common/abi.h"
#include "common/assert.h"
#include "common/profiler.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <xbyak/xbyak.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <optional>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <cstdlib>

namespace Libs::Graphics::ShaderRecompiler::IR {

SrtRuntime SrtReadCapture::ObservedRuntime() {
	auto runtime = m_source;
	runtime.userdata = this;
	runtime.read_specialization_memory = m_source.read_specialization_memory != nullptr
	                                         ? ReadStrict : nullptr;
	runtime.read_memory = ReadOrdinary;
	return runtime;
}

bool SrtReadCapture::ReadStrict(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<SrtReadCapture*>(userdata);
	if (!capture.m_source.read_specialization_memory(capture.m_source.userdata, address, values)) {
		return false;
	}
	capture.m_ranges.emplace_back(address, values.size_bytes());
	return true;
}

bool SrtReadCapture::ReadOrdinary(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<SrtReadCapture*>(userdata);
	if (capture.m_source.read_memory != nullptr) {
		if (!capture.m_source.read_memory(capture.m_source.userdata, address, values)) return false;
	} else {
		std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
	}
	capture.m_ranges.emplace_back(address, values.size_bytes());
	return true;
}

SrtRuntime CleanRuntime(SrtRuntime runtime) {
	runtime.read_memory = runtime.read_specialization_memory != nullptr
	                          ? runtime.read_specialization_memory
	                          : +[](void*, uint64_t, std::span<uint32_t>) { return false; };
	return runtime;
}

namespace {

constexpr uint64_t AddressMask = 0x0000ffffffffffffull;

// Perf (uncommitted, SRT reachability memo): while a control-flow reachability
// pass runs, record every input it consumes so the next walk can replay-check
// them and skip the fixpoint when they are all unchanged. Thread-local: the
// evaluation happens on whichever walker instance evaluates the condition.
struct SrtMemoRecorder {
	std::vector<ResourcePlan::SrtMemoRead>* reads  = nullptr;
	bool                                   active = false;
};
thread_local SrtMemoRecorder g_srt_memo_recorder;

bool IsRawRead(const ResourcePlan& values, const Inst& inst);

inline void SrtMemoRecord(uint32_t kind, uint32_t index, uint64_t address, uint32_t value) {
	ResourcePlan::SrtMemoRead entry {};
	entry.kind    = kind;
	entry.index   = index;
	entry.address = address;
	entry.value   = value;
	g_srt_memo_recorder.reads->push_back(entry);
}

// Perf (uncommitted): classify a control-flow branch condition by its leaves.
// Cacheable when every input is a user-data dword or the shader base: such a
// condition repeats whenever those words do, with no evaluation at all
// (idea: pengui07's DirectCfCondition). Any other leaf (raw read, Phi, ...)
// disqualifies. Returns false for not-cacheable.
bool CollectConditionLeaves(const ResourcePlan& program, Value root,
                            std::vector<ResourcePlan::SrtCfLeaf>& leaves) {
	constexpr size_t MaxLeaves = 16;
	std::vector<const Inst*> stack;
	std::vector<const Inst*> visited;
	const auto push = [&](Value value) {
		value = value.Resolve();
		if (value.IsImmediate()) return true;
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return false;
		if (std::find(visited.begin(), visited.end(), inst) != visited.end()) return true;
		visited.push_back(inst);
		stack.push_back(inst);
		return visited.size() <= 256;
	};
	const auto add = [&](ResourcePlan::SrtCfLeaf leaf) {
		for (const auto& known: leaves) {
			if (known.kind == leaf.kind && known.index == leaf.index) return true;
		}
		if (leaves.size() >= MaxLeaves) return false;
		leaves.push_back(leaf);
		return true;
	};
	if (!push(root)) return false;
	while (!stack.empty()) {
		const auto* inst = stack.back();
		stack.pop_back();
		switch (inst->GetOpcode()) {
			case ValueOpcode::GetUserData: {
				const auto reg = RegIndex(inst->Arg(0).ScalarRegister());
				if (reg < program.user_data_base ||
				    !add({0u, reg - program.user_data_base})) {
					return false;
				}
				break;
			}
			case ValueOpcode::GetShaderBase:
				if (!add({1u, 0u})) return false;
				break;
			case ValueOpcode::LoadAddressU32: {
				// A read the plan owns: its word is (or will be) in the flat
				// buffer, fetched by this walk; anything else disqualifies.
				if (!IsRawRead(program, *inst)) return false;
				for (const auto& read: program.srt_reads) {
					if (read.value.Resolve().TryInstruction() == inst) {
						if (!add({2u, read.flat_offset})) return false;
						break;
					}
				}
				break;
			}
			case ValueOpcode::Phi:
			case ValueOpcode::ReadFirstLane:
			case ValueOpcode::ReadConstBuffer:
			case ValueOpcode::LoadBufferU32:
			case ValueOpcode::UndefU1:
			case ValueOpcode::UndefU32: return false;
			default:
				for (size_t i = 0; i < inst->NumArgs(); ++i) {
					if (!push(inst->Arg(i))) return false;
				}
				break;
		}
	}
	return true;
}

// Perf (uncommitted): classify one SRT read slot as a one-level direct read
// (address = user-data register pair + dword-aligned immediate offset), like
// the non-control-flow JIT's root analysis. Slots that do not qualify keep
// valid=false and fall back to the IR evaluation in the direct walk.
void ClassifyDirectRead(const ResourcePlan& program, uint32_t slot,
                        ResourcePlan::SrtDirectRead& out) {
	const auto& read = program.srt_reads[slot];
	const auto* inst = read.value.Resolve().TryInstruction();
	if (inst == nullptr || inst->GetOpcode() != ValueOpcode::LoadAddressU32 ||
	    !IsRawRead(program, *inst)) {
		return;
	}
	const auto offset_value = inst->Arg(1).Resolve();
	if (!offset_value.IsImmediate() || offset_value.GetType() != Type::U32) return;
	const auto* handle = inst->Arg(0).ResolveInstruction();
	if (handle == nullptr || handle->GetOpcode() != ValueOpcode::GetAddressResource ||
	    handle->NumArgs() != 2) {
		return;
	}
	const auto* low = handle->Arg(0).Resolve().TryInstruction();
	const auto* high = handle->Arg(1).Resolve().TryInstruction();
	if (low == nullptr || high == nullptr ||
	    low->GetOpcode() != ValueOpcode::GetUserData ||
	    high->GetOpcode() != ValueOpcode::GetUserData || low->NumArgs() != 1 ||
	    high->NumArgs() != 1 || low->Arg(0).GetType() != Type::ScalarReg ||
	    high->Arg(0).GetType() != Type::ScalarReg) {
		return;
	}
	const auto low_reg  = RegIndex(low->Arg(0).ScalarRegister());
	const auto high_reg = RegIndex(high->Arg(0).ScalarRegister());
	if (low_reg < program.user_data_base || high_reg != low_reg + 1u) return;
	const auto& memory = program.memory_info[inst->Flags<SrtReadFlags>().index];
	const auto relative =
	    (static_cast<int64_t>(static_cast<int32_t>(memory.offset)) & ~int64_t {3}) +
	    static_cast<int64_t>(offset_value.U32() & ~3u);
	if (relative < INT32_MIN || relative > INT32_MAX) return;
	out.low_register     = low_reg - program.user_data_base;
	out.high_register    = high_reg - program.user_data_base;
	out.relative         = static_cast<int32_t>(relative);
	out.evaluation_index = inst->EvaluationIndex(program.evaluation_value_count);
	out.clean            = inst->Flags<SrtReadFlags>().clean != 0u;
	out.valid            = out.evaluation_index < program.evaluation_value_count;
}

// Perf (uncommitted): build the per-block run table and the entry active-source
// state for the direct control-flow walk. Returns false when the plan cannot
// be served (caller keeps the fixpoint walk).
bool BuildDirectCfPlan(const ResourcePlan& program) {
	if (program.control_flow.empty() || program.srt_reads.empty()) return false;
	auto& direct = program.srt_direct_reads;
	if (direct.size() != program.srt_reads.size()) {
		direct.assign(program.srt_reads.size(), {});
		for (uint32_t slot = 0; slot < program.srt_reads.size(); ++slot) {
			ClassifyDirectRead(program, slot, direct[slot]);
		}
	}
	auto& runs  = program.srt_cf_runs;
	auto& begin = program.srt_cf_run_begin;
	runs.clear();
	begin.clear();
	begin.reserve(program.control_flow.size() + 1);
	for (const auto& block: program.control_flow) {
		begin.push_back(static_cast<uint32_t>(runs.size()));
		for (size_t position = 0; position < block.srt_reads.size();) {
			const auto slot = block.srt_reads[position];
			if (slot >= program.srt_reads.size()) return false;
			uint32_t count = 1;
			const auto first_flat = program.srt_reads[slot].flat_offset;
			const auto clean      = direct[slot].clean;
			while (position + count < block.srt_reads.size() &&
			       count < 16u) {
				const auto next = block.srt_reads[position + count];
				if (next != slot + count || next >= program.srt_reads.size()) break;
				if (!direct[next].valid || direct[next].clean != clean) break;
				if (program.srt_reads[next].flat_offset != first_flat + count) break;
				++count;
			}
			runs.push_back({slot, count});
			position += count;
		}
		for (const auto successor: block.successors) {
			if (successor >= program.control_flow.size()) return false;
		}
		if (!block.condition.IsEmpty() && block.successors.size() < 2) return false;
	}
	begin.push_back(static_cast<uint32_t>(runs.size()));
	auto& active = program.srt_cf_entry_active;
	active.assign(program.descriptor_sources.size(), 1u);
	for (const auto& block: program.control_flow) {
		for (const auto source: block.sources) {
			if (source >= active.size()) return false;
			active[source] = 0u;
		}
	}
	return true;
}

bool AddSignedAddress(uint64_t base, int64_t offset, uint64_t& result) {
	if (base > AddressMask) {
		return false;
	}
	if (offset < 0) {
		const auto magnitude = uint64_t {0} - static_cast<uint64_t>(offset);
		if (magnitude > base) {
			return false;
		}
		result = base - magnitude;
		return true;
	}
	const auto magnitude = static_cast<uint64_t>(offset);
	if (magnitude > AddressMask - base) {
		return false;
	}
	result = base + magnitude;
	return true;
}

bool IsRawRead(const ResourcePlan& values, const Inst& inst) {
	const auto op = inst.GetOpcode();
	if (op != ValueOpcode::LoadAddressU32 && op != ValueOpcode::ReadConstBuffer &&
	    op != ValueOpcode::LoadBufferU32) {
		return false;
	}
	const auto index = inst.Flags<MemoryFlags>().index;
	if (index >= values.memory_info.size()) {
		return false;
	}
	const auto& memory = values.memory_info[index];
	if (op != ValueOpcode::LoadBufferU32) {
		return (op == ValueOpcode::LoadAddressU32 && memory.kind == ResourceKind::ScalarAddress) ||
		       (op == ValueOpcode::ReadConstBuffer && memory.kind == ResourceKind::ScalarBuffer);
	}
	if (inst.NumArgs() != 5u || memory.kind != ResourceKind::Buffer || memory.typed ||
	    memory.formatted || memory.coherent || memory.data_bits != 32u ||
	    memory.data_dwords != 1u || memory.idxen || memory.offen || memory.offset != 0u ||
	    inst.Arg(4).GetType() != Type::U1) {
		return false;
	}
	for (uint32_t arg = 1; arg <= 3; ++arg) {
		if (inst.Arg(arg).Resolve() != Value(0u)) return false;
	}
	return true;
}

bool IsDescriptorHandle(ValueOpcode opcode) {
	switch (opcode) {
		case ValueOpcode::GetBufferResource:
		case ValueOpcode::GetAddressResource:
		case ValueOpcode::GetImageResource:
		case ValueOpcode::GetSamplerResource: return true;
		default: return false;
	}
}

bool IsRuntimeSelect(ValueOpcode op) {
	return op == ValueOpcode::SelectU1 || op == ValueOpcode::SelectU32 ||
	       op == ValueOpcode::SelectF32;
}

bool IsRuntimeUniformOp(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::ConditionRef:
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32:
		case ValueOpcode::ConvertU32F32:
		case ValueOpcode::ConvertF32U32:
		case ValueOpcode::CompositeConstructU64:
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeConstructU32x2:
		case ValueOpcode::CompositeExtractU32x2:
		case ValueOpcode::BitFieldInsert:
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract:
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
		case ValueOpcode::UMulHi:
		case ValueOpcode::UMin32:
		case ValueOpcode::UMax32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftLeftLogical64:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::ShiftRightLogical64:
		case ValueOpcode::ShiftRightArithmetic32:
		case ValueOpcode::ShiftRightArithmetic64:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseAnd64:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
		case ValueOpcode::BitwiseNot32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::ULessThanEqual32:
		case ValueOpcode::IEqual32:
		case ValueOpcode::IEqual64:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::INotEqual64:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalXor:
		case ValueOpcode::LogicalNot:
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
		case ValueOpcode::FPIsNan32:
		case ValueOpcode::FPMul32:
		case ValueOpcode::FPRecipIFlag32:
		case ValueOpcode::FPTrunc32: return true;
		default: return false;
	}
}

class RuntimeValidator {
public:
	explicit RuntimeValidator(const ResourcePlan& program, RuntimeValueType type,
	                          bool wave_uniform = false)
	    : m_program(program), m_type(type), m_wave_uniform(wave_uniform) {}

	bool Run(Value value) { return Validate(value); }

private:
	bool ValidateArguments(const Inst& inst, bool require_uniform) {
		for (size_t index = 0; index < inst.NumArgs(); index++) {
			if (!Validate(inst.Arg(index), require_uniform)) return false;
		}
		return true;
	}

	bool Validate(Value value, bool require_uniform = true) {
		value = value.Resolve();
		// Host floating-point evaluation does not model shader rounding/denormal modes.
		if (m_type != RuntimeValueType::Any &&
		    TypesOverlap(value.GetType(), Type::F16 | Type::F32 | Type::F32x2)) {
			return false;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			if (!require_uniform) return true;
			switch (value.GetType()) {
				case Type::U1:
				case Type::U8:
				case Type::U16:
				case Type::U32:
				case Type::U64:
				case Type::F32: return true;
				default: return false;
			}
		}
		if (require_uniform && !m_active_mask.IsEmpty() && value == m_active_mask) return true;
		// Integer-only dependency checks do not depend on the active EXEC mask.
		if (!require_uniform && m_validated_dependencies.contains(inst)) return true;
		if (!m_visiting.insert(inst).second) {
			return !require_uniform;
		}
		const auto finish = [&](bool valid) {
			m_visiting.erase(inst);
			if (valid && !require_uniform) m_validated_dependencies.insert(inst);
			return valid;
		};
		const auto op = inst->GetOpcode();
		if (m_type == RuntimeValueType::ImmutableInteger &&
		    (op == ValueOpcode::ReadConst || BufferAccessOf(op) != BufferAccess::None ||
		     AddressOpcodeInfoOf(op).access != AddressAccess::None ||
		     ImageOpcodeInfoOf(op).access != ImageAccess::None ||
		     SharedAccessOf(op) != SharedAccess::None)) {
			return finish(false);
		}
		if (op == ValueOpcode::ReadConst) {
			const auto slot = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (inst->NumArgs() != 2 || inst->Arg(0).Resolve().TryInstruction() == nullptr ||
			    inst->Arg(0).Resolve().TryInstruction()->GetOpcode() !=
			        ValueOpcode::GetSrtResource ||
			    !slot.IsImmediate() || slot.GetType() != Type::U32 ||
			    slot.U32() >= m_program.srt_reads.size()) {
				return finish(false);
			}
			if (m_type != RuntimeValueType::Any) {
				const auto active_mask = m_active_mask;
				m_active_mask          = {};
				const bool valid       = Validate(m_program.srt_reads[slot.U32()].value);
				m_active_mask          = active_mask;
				if (!valid) return finish(false);
			}
		}
		if (!require_uniform) return finish(ValidateArguments(*inst, false));
		if (!m_active_mask.IsEmpty() && IsRuntimeSelect(op) && inst->NumArgs() == 3 &&
		    inst->Arg(0).Resolve() == m_active_mask) {
			// Empty EXEC reads lane zero, so ignored operands still require integer types.
			if (m_type != RuntimeValueType::Any && !Validate(inst->Arg(2), false)) {
				return finish(false);
			}
			return finish(Validate(inst->Arg(1)));
		}
		if (op == ValueOpcode::UndefU1 || op == ValueOpcode::UndefU32 || op == ValueOpcode::Void) {
			return finish(false);
		}
		if (op == ValueOpcode::GetUserData) {
			if (inst->NumArgs() != 1 || inst->Arg(0).GetType() != Type::ScalarReg) {
				return finish(false);
			}
			const auto reg = RegIndex(inst->Arg(0).ScalarRegister());
			if (reg < m_program.user_data_base ||
			    reg - m_program.user_data_base >= m_program.user_data_count) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::GetShaderBase) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (m_wave_uniform && op == ValueOpcode::GetBuiltin) {
			const auto axis = inst->NumArgs() == 2u ? inst->Arg(1).Resolve() : Value {};
			return finish(m_program.stage == ShaderType::Compute && inst->NumArgs() == 2u &&
			              inst->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::WorkgroupId)) &&
			              axis.IsImmediate() && axis.GetType() == Type::U32 && axis.U32() < 3u);
		}
		if (op == ValueOpcode::Phi) {
			if (m_type != RuntimeValueType::Any && !ValidateArguments(*inst, false)) {
				return finish(false);
			}
			const auto invariant = ResolveInvariantPhi(m_program, value);
			if (invariant.IsEmpty()) {
				return finish(false);
			}
			return finish(Validate(invariant));
		}
		if (op == ValueOpcode::ReadFirstLane) {
			if (inst->NumArgs() != 2 || inst->Arg(0).GetType() != Type::U32 ||
			    inst->Arg(1).GetType() != Type::U1) {
				return finish(false);
			}
			if (m_type != RuntimeValueType::Any && !Validate(inst->Arg(1), false)) {
				return finish(false);
			}
			const auto active_mask = m_active_mask;
			m_active_mask          = inst->Arg(1).Resolve();
			const bool valid       = Validate(inst->Arg(0));
			m_active_mask          = active_mask;
			return finish(valid);
		}
		if (op == ValueOpcode::GetSrtResource) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer ||
		    op == ValueOpcode::LoadBufferU32) {
			const auto  expected = op == ValueOpcode::LoadAddressU32
			                           ? ValueOpcode::GetAddressResource
			                           : ValueOpcode::GetBufferResource;
			const auto* handle = inst->NumArgs() != 0 ? inst->Arg(0).ResolveInstruction() : nullptr;
			if (!IsRawRead(m_program, *inst) || handle == nullptr ||
			    handle->GetOpcode() != expected) {
				return finish(false);
			}
			if (op == ValueOpcode::LoadBufferU32) {
				const auto guard = inst->Arg(4).Resolve();
				return finish(Validate(inst->Arg(0)) &&
				              ((!m_active_mask.IsEmpty() && guard == m_active_mask) || Validate(guard)));
			}
		} else if (op == ValueOpcode::CompositeExtractU64) {
			const auto index = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (!index.IsImmediate() || index.GetType() != Type::U32 || index.U32() >= 2u) {
				return finish(false);
			}
		} else if (op == ValueOpcode::CompositeExtractU32x2) {
			const auto* source = inst->NumArgs() == 2 ? inst->Arg(0).ResolveInstruction() : nullptr;
			const auto  index  = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (source == nullptr || !index.IsImmediate() || index.GetType() != Type::U32 ||
			    index.U32() >= 2u || source->GetOpcode() != ValueOpcode::CompositeConstructU32x2) {
				return finish(false);
			}
		}
		if (IsDescriptorHandle(op)) {
			size_t expected = 4u;
			if (op == ValueOpcode::GetImageResource) {
				expected = 8u;
			} else if (op == ValueOpcode::GetAddressResource) {
				expected = 2u;
			}
			if (inst->NumArgs() != expected) {
				return finish(false);
			}
		} else if (op != ValueOpcode::ReadConst && op != ValueOpcode::ReadConstBuffer &&
		           op != ValueOpcode::LoadAddressU32 && !IsRuntimeUniformOp(op)) {
			return finish(false);
		}
		return finish(ValidateArguments(*inst, true));
	}

	const ResourcePlan&             m_program;
	RuntimeValueType                m_type;
	bool                            m_wave_uniform;
	Value                           m_active_mask;
	std::unordered_set<const Inst*> m_visiting;
	std::unordered_set<const Inst*> m_validated_dependencies;
};


} // namespace

void CompileSrtWalker(ResourcePlan& program) {
	if (!program.resource_tracking_complete || !program.srt_plan_complete ||
	    program.capture_specialization_reads || !program.control_flow.empty() ||
	    program.srt_reads.empty()) return;

	using Reads = std::map<int32_t, const SrtRead*>;
	std::map<uint32_t, Reads> roots;
	std::unordered_map<const Inst*, Reads> pointer_uses;
	std::unordered_map<const Inst*, int32_t> offsets;
	for (const auto& read: program.srt_reads) {
		const auto* inst = read.value.Resolve().TryInstruction();
		if (inst == nullptr || inst->GetOpcode() != ValueOpcode::LoadAddressU32 ||
		    inst->NumArgs() < 2 || !IsRawRead(program, *inst) ||
		    read.flat_offset >= program.srt_reads.size()) return;
		const auto offset = inst->Arg(1).Resolve();
		if (!offset.IsImmediate() || offset.GetType() != Type::U32) return;
		const auto& memory = program.memory_info[inst->Flags<SrtReadFlags>().index];
		const auto relative =
		    (static_cast<int64_t>(static_cast<int32_t>(memory.offset)) & ~int64_t {3}) +
		    static_cast<int64_t>(offset.U32() & ~3u);
		if (relative < INT32_MIN || relative > INT32_MAX ||
		    !offsets.emplace(inst, static_cast<int32_t>(relative)).second) return;
	}
	uint32_t user_data_count = 0;
	for (const auto& read: program.srt_reads) {
		const auto* inst = read.value.ResolveInstruction();
		const auto* handle = inst->Arg(0).Resolve().TryInstruction();
		if (handle == nullptr || handle->GetOpcode() != ValueOpcode::GetAddressResource ||
		    handle->NumArgs() != 2) return;
		const auto* low = handle->Arg(0).Resolve().TryInstruction();
		const auto* high = handle->Arg(1).Resolve().TryInstruction();
		if (low == nullptr || high == nullptr) return;
		Reads* uses = nullptr;
		if (low->GetOpcode() == ValueOpcode::GetUserData &&
		    high->GetOpcode() == ValueOpcode::GetUserData && low->NumArgs() == 1 &&
		    high->NumArgs() == 1 && low->Arg(0).GetType() == Type::ScalarReg &&
		    high->Arg(0).GetType() == Type::ScalarReg) {
			const auto reg = RegIndex(low->Arg(0).ScalarRegister());
			if (reg < program.user_data_base ||
			    RegIndex(high->Arg(0).ScalarRegister()) != reg + 1u) return;
			const auto base = reg - program.user_data_base;
			user_data_count = std::max(user_data_count, base + 2u);
			uses = &roots[base];
		} else {
			const auto lo = offsets.find(low);
			const auto hi = offsets.find(high);
			if (lo == offsets.end() || hi == offsets.end() ||
			    static_cast<int64_t>(lo->second) + 4 != hi->second ||
			    !EquivalentValue(program, low->Arg(0), high->Arg(0))) return;
			uses = &pointer_uses[low];
		}
		if (!uses->emplace(offsets.at(inst), &read).second) return;
	}
	std::unordered_set<const Inst*> visited;
	const auto reachable = [&](auto&& self, const Reads& reads) -> bool {
		for (const auto& [offset, read]: reads) {
			const auto* inst = read->value.ResolveInstruction();
			const auto child = pointer_uses.find(inst);
			if (child != pointer_uses.end() &&
			    (!visited.insert(inst).second || !self(self, child->second))) return false;
		}
		return true;
	};
	for (const auto& [reg, reads]: roots) {
		if (!reachable(reachable, reads)) return;
	}
	if (visited.size() != pointer_uses.size()) return;

	program.srt_code = std::make_unique<Xbyak::CodeGenerator>(4096, Xbyak::AutoGrow);
	auto& code = *program.srt_code;
	Xbyak::Label failed, finished;
	code.push(code.rbp);
	code.mov(code.rbp, code.rsp);
	code.cmp(code.r8, user_data_count);
	code.jb(failed, Xbyak::CodeGenerator::T_NEAR);
	code.mov(code.r9, AddressMask & ~uint64_t {3});
	// Follow shadPS4's VisitPointer: copy one table level in offset order, then
	// descend through its child pointers. Seed the descriptor evaluator's memo.
	const auto visit = [&](auto&& self, int32_t pointer_offset, const Reads& reads) -> void {
		code.push(code.rdi);
		code.mov(code.rdi, code.qword[code.rdi + pointer_offset]);
		code.and_(code.rdi, code.r9);
		const auto first = reads.begin()->first;
		const auto last = reads.rbegin()->first;
		if (first < 0) {
			code.mov(code.r10, -static_cast<int64_t>(first));
			code.cmp(code.rdi, code.r10);
			code.jb(failed, Xbyak::CodeGenerator::T_NEAR);
		}
		if (last > 0) {
			code.mov(code.r10, AddressMask - last);
			code.cmp(code.rdi, code.r10);
			code.ja(failed, Xbyak::CodeGenerator::T_NEAR);
		}
		for (const auto& [offset, read]: reads) {
			const auto index = read->value.ResolveInstruction()->EvaluationIndex(program.evaluation_value_count);
			using Entry = ResourcePlan::EvaluationContext::Entry;
			const auto slot = code.rdx + index * sizeof(Entry);
			code.mov(code.r10d, code.dword[code.rdi + offset]);
			code.mov(code.dword[code.rsi + read->flat_offset * sizeof(uint32_t)], code.r10d);
			code.mov(code.qword[slot + offsetof(Entry, value)], code.r10);
			code.mov(code.qword[slot + offsetof(Entry, generation)], code.rcx);
		}
		for (const auto& [offset, read]: reads) {
			const auto child = pointer_uses.find(read->value.ResolveInstruction());
			if (child == pointer_uses.end()) continue;
			self(self, offset, child->second);
		}
		code.pop(code.rdi);
	};
	for (const auto& [reg, reads]: roots) {
		visit(visit, reg * sizeof(uint32_t), reads);
	}
	code.mov(code.eax, 1);
	code.jmp(finished, Xbyak::CodeGenerator::T_NEAR);
	code.L(failed);
	code.xor_(code.eax, code.eax);
	code.L(finished);
	code.mov(code.rsp, code.rbp);
	code.pop(code.rbp);
	code.ret();
	code.readyRE();
	EXIT_IF(Xbyak::GetError() != 0);
}

SrtWalker::SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
                     SrtWalker* clean_evaluator)
    : m_program(program), m_runtime(runtime), m_clean_evaluator(clean_evaluator),
      m_session(this),
      m_read_policy(clean_evaluator != nullptr ? 2u : 0u), m_context(AcquireContext()) {}

SrtWalker::SrtWalker(SrtWalker& parent, Value active_mask, bool clean,
                     SrtWalker* clean_evaluator)
    : m_program(parent.m_program), m_runtime(clean ? CleanRuntime(parent.m_runtime) : parent.m_runtime),
      m_clean_evaluator(clean_evaluator), m_active_mask(active_mask.Resolve()),
      m_reachable_blocks(parent.m_reachable_blocks), m_session(parent.m_session),
      m_read_policy((parent.m_read_policy & 1u) | (clean ? 1u : 0u) |
                    (clean_evaluator != nullptr ? 2u : 0u)), m_context(AcquireContext()) {}

SrtWalker::~SrtWalker() {
	if (m_session != this) return;
	for (auto& context: m_program.evaluation_contexts)
		if (context.session == this) context.session = nullptr;
}

bool SrtWalker::Evaluate(Value value, uint32_t& result) {
	uint64_t wide = 0;
	if (!EvaluateWide(value, wide)) {
		return false;
	}
	result = static_cast<uint32_t>(wide);
	return true;
}

// Perf switches (uncommitted, both on by default; =0 for A/B): KYTY_SRT_CONTEXT_INDEX,
// KYTY_SRT_SLOT_CACHE.
static bool SrtSwitch(const char* name) {
	const char* value = std::getenv(name);
	return value == nullptr || value[0] != '0';
}

ResourcePlan::EvaluationContext& SrtWalker::AcquireContext() {
	// The session's index holds every context it acquired; the scan below finds the same one (a
	// key is registered when its context is first taken, and no other context matches it).
	static const bool context_index = SrtSwitch("KYTY_SRT_CONTEXT_INDEX");
	for (const auto& [mask, policy, context]: m_session->m_context_index) {
		if (!context_index) break;
		if (policy == m_read_policy && context->session == m_session && mask == m_active_mask) {
			return *context;
		}
	}
	ResourcePlan::EvaluationContext* unused = nullptr;
	for (auto& context: m_program.evaluation_contexts) {
		if (context.session == m_session && context.active_mask == m_active_mask &&
		    context.read_policy == m_read_policy) {
			m_session->m_context_index.emplace_back(m_active_mask, m_read_policy, &context);
			return context;
		}
		if (context.session == nullptr && unused == nullptr) unused = &context;
	}
	auto& context = unused != nullptr ? *unused : m_program.evaluation_contexts.emplace_back();
	context.session = m_session;
	context.active_mask = m_active_mask;
	context.read_policy = m_read_policy;
	context.generation += 4;
	context.unknowns.clear();
	m_session->m_context_index.emplace_back(m_active_mask, m_read_policy, &context);
	return context;
}

bool SrtWalker::TryMemoized(uint32_t index, uint32_t& result) const {
	if (!m_active_mask.IsEmpty() || index >= m_context.values.size() ||
	    m_context.values[index].generation != m_context.generation) {
		return false;
	}
	result = static_cast<uint32_t>(m_context.values[index].value);
	return true;
}

void SrtWalker::ResetUnknowns() {
	KYTY_PROFILER_BLOCK("Srt:ResetUnknowns");
	// Perf (uncommitted): NHL 27 spent ~0.9 us per call scanning every value of every context
	// (5+ calls per control-flow walk). Only EvaluateWide marks values unknown, and it records
	// them; the JIT walker writes known values only. KYTY_SRT_FAST_RESET=0 restores the scan.
	static const bool fast_reset = [] {
		const char* value = std::getenv("KYTY_SRT_FAST_RESET");
		return value == nullptr || value[0] != '0';
	}();
	for (auto& context: m_program.evaluation_contexts) {
		if (context.session != m_session) continue;
		if (fast_reset) {
			for (const auto index: context.unknowns) {
				if (index < context.values.size() &&
				    context.values[index].generation == (context.generation | 2u)) {
					context.values[index].generation = 0;
				}
			}
		} else {
			for (auto& entry: context.values)
				if (entry.generation == (context.generation | 2u)) entry.generation = 0;
		}
		context.unknowns.clear();
	}
}

float SrtWalker::Float32(uint64_t bits) {
	return std::bit_cast<float>(static_cast<uint32_t>(bits));
}

bool SrtWalker::EvaluateWide(Value value, uint64_t& result) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		switch (value.GetType()) {
			case Type::U1: result = value.U1(); return true;
			case Type::U8: result = value.U8(); return true;
			case Type::U16: result = value.U16(); return true;
			case Type::U32: result = value.U32(); return true;
			case Type::U64: result = value.U64(); return true;
			case Type::F32: result = std::bit_cast<uint32_t>(value.F32Value()); return true;
			default: return false;
		}
	}
	if (!m_active_mask.IsEmpty() && value == m_active_mask) {
		result = 1u;
		return true;
	}
	auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return false;
	}
	if (!m_active_mask.IsEmpty() && IsRuntimeSelect(inst->GetOpcode()) &&
	    inst->NumArgs() == 3 && inst->Arg(0).Resolve() == m_active_mask) {
		return EvaluateWide(inst->Arg(1), result);
	}
	const auto index = inst->EvaluationIndex(m_program.evaluation_value_count);
	if (index >= m_context.values.size()) {
		m_context.values.resize(m_program.evaluation_value_count);
	}
	if (m_context.values[index].generation == m_context.generation) {
		result = m_context.values[index].value;
		return true;
	}
	// Visiting and completed-unknown entries both stop repeated evaluation in this
	// context. Only completed unknowns are retried when reachability becomes tighter.
	if (m_context.values[index].generation == (m_context.generation | 1u) ||
	    m_context.values[index].generation == (m_context.generation | 2u)) {
		return false;
	}
	m_context.values[index].generation = m_context.generation | 1u;
	uint64_t out = 0;
	const bool evaluated = EvaluateInst(*inst, out);
	// Recursive evaluation may grow the dense memo vector.
	auto& memo = m_context.values[index];
	if (!evaluated) {
		memo.generation = m_context.generation | 2u;
		m_context.unknowns.push_back(static_cast<uint32_t>(index));
		return false;
	}
	memo.value      = out;
	memo.generation = m_context.generation;
	result = out;
	return true;
}

bool SrtWalker::Arg(const Inst& inst, size_t index, uint64_t& result) {
	return EvaluateWide(inst.Arg(index), result);
}

bool SrtWalker::EvaluatePhi(const Inst& inst, uint64_t& result) {
	if (const auto control = m_program.control_phi_predecessors.find(&inst);
	    control != m_program.control_phi_predecessors.end()) {
		bool found = false;
		for (size_t index = 0; index < inst.NumArgs(); ++index) {
			if (m_reachable_blocks != nullptr &&
			    !m_reachable_blocks->at(control->second[index])) continue;
			uint64_t incoming = 0;
			if (!Arg(inst, index, incoming) || (found && incoming != result)) return false;
			result = incoming;
			found = true;
		}
		return found;
	}
	const auto value = ResolveInvariantPhi(m_program, Value(const_cast<Inst*>(&inst)));
	return !value.IsEmpty() && EvaluateWide(value, result);
}

bool SrtWalker::EvaluateExtract(const Inst& inst, uint64_t& result) {
	const auto index = inst.Arg(1).Resolve();
	if (!index.IsImmediate() || index.GetType() != Type::U32) {
		return false;
	}
	const auto component = index.U32();
	if (component >= 2u) {
		return false;
	}
	if (inst.GetOpcode() == ValueOpcode::CompositeExtractU64) {
		uint64_t packed = 0;
		if (!Arg(inst, 0, packed)) {
			return false;
		}
		result = static_cast<uint32_t>(packed >> (component * 32u));
		return true;
	}
	const auto* source = inst.Arg(0).ResolveInstruction();
	if (source == nullptr) {
		return false;
	}
	if (source->GetOpcode() == ValueOpcode::CompositeConstructU32x2) {
		return EvaluateWide(source->Arg(component), result);
	}
	return false;
}

bool SrtWalker::EvaluateRawRead(const Inst& inst, uint64_t& result) {
	const auto flags = inst.Flags<SrtReadFlags>();
	if (flags.index >= m_program.memory_info.size()) {
		return false;
	}
	const auto& mem    = m_program.memory_info[flags.index];
	const bool vector = inst.GetOpcode() == ValueOpcode::LoadBufferU32;
	const auto guard = vector ? inst.Arg(4).Resolve() : Value {};
	if (vector && (guard.IsImmediate() || guard != m_active_mask)) {
		uint64_t enabled = 0;
		if (!Arg(inst, 4, enabled)) return false;
		if (enabled == 0u) {
			result = 0u;
			return true;
		}
	}
	const auto* handle = inst.Arg(0).ResolveInstruction();
	if (handle == nullptr) {
		return false;
	}
	uint64_t low    = 0;
	uint64_t high   = 0;
	uint64_t offset = 0;
	if (!Arg(*handle, 0, low) || !Arg(*handle, 1, high) || !Arg(inst, 1, offset)) {
		return false;
	}
	const auto base      = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
	const auto immediate = static_cast<int64_t>(static_cast<int32_t>(mem.offset));
	uint64_t   address   = 0;
	if (inst.GetOpcode() == ValueOpcode::ReadConstBuffer || vector) {
		uint64_t records = 0;
		uint64_t word3   = 0;
		if (handle->NumArgs() != 4u || !Arg(*handle, 2, records) || !Arg(*handle, 3, word3)) {
			return false;
		}
		if (immediate < 0) {
			return false;
		}
		const auto byte_offset =
		    (static_cast<uint64_t>(immediate) & ~uint64_t {3}) + (static_cast<uint32_t>(offset) & ~3u);
		const auto stride  = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
		if (vector) {
			// Only the uniform, unswizzled structured DWORD address is evaluated on the host.
			if ((high & (1u << 31u)) != 0u || (word3 & ((1u << 23u) | 0xf0000000u)) != 0u)
				return false;
			if (stride == 0u || records == 0u || ((word3 >> 12u) & 0x7fu) == 0u) {
				result = 0u;
				return true;
			}
		}
		const auto size = stride == 0u
		                      ? static_cast<uint64_t>(static_cast<uint32_t>(records))
		                      : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
		if (byte_offset > size || size - byte_offset < sizeof(uint32_t)) {
			result = 0u;
			return true;
		}
		address = (base & ~uint64_t {3}) + byte_offset;
	} else {
		const auto relative = (immediate & ~int64_t {3}) +
		                      static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
		if (!AddSignedAddress(base & ~uint64_t {3}, relative, address)) {
			return false;
		}
	}
	uint32_t word = 0;
	const auto reader = vector ? m_runtime.read_specialization_memory : m_runtime.read_memory;
	if (reader != nullptr) {
		if (!reader(m_runtime.userdata, address, {&word, 1})) {
			if (g_srt_memo_recorder.active) {
				// Record the decline; the memo replay requires the same decline.
				SrtMemoRecord(vector ? 4u : 3u, 0u, address, 0u);
			}
			return false;
		}
		if (g_srt_memo_recorder.active) {
			SrtMemoRecord(vector ? 2u : 1u, 0u, address, word);
		}
	} else {
		if (vector) {
			if (g_srt_memo_recorder.active) {
				// No specialization reader: replay matches when there is still none.
				SrtMemoRecord(4u, 0u, address, 0u);
			}
			return false;
		}
		std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
		if (g_srt_memo_recorder.active) {
			SrtMemoRecord(5u, 0u, address, word);
		}
	}
	result = word;
	return true;
}

bool SrtWalker::EvaluateInst(const Inst& inst, uint64_t& result) {
	uint64_t   a       = 0;
	uint64_t   b       = 0;
	uint64_t   c       = 0;
	const auto binary  = [&]() { return Arg(inst, 0, a) && Arg(inst, 1, b); };
	const auto ternary = [&]() {
		return Arg(inst, 0, a) && Arg(inst, 1, b) && Arg(inst, 2, c);
	};
	switch (inst.GetOpcode()) {
		case ValueOpcode::GetUserData: {
			const auto reg = RegIndex(inst.Arg(0).ScalarRegister());
			if (reg < m_program.user_data_base ||
			    reg - m_program.user_data_base >= m_runtime.user_data.size()) {
				// SRT reachability memo: a would-be user-data read beyond the
				// current span fails structurally; reg < base is a constant.
				if (reg >= m_program.user_data_base && g_srt_memo_recorder.active) {
					SrtMemoRecord(6u, reg - m_program.user_data_base, 0u, 0u);
				}
				return false;
			}
			result = m_runtime.user_data[reg - m_program.user_data_base];
			if (g_srt_memo_recorder.active) {
				SrtMemoRecord(0u, reg - m_program.user_data_base, 0u, static_cast<uint32_t>(result));
			}
			return true;
		}
		case ValueOpcode::GetShaderBase: result = m_runtime.shader_base; return true;
		case ValueOpcode::Phi: return EvaluatePhi(inst, result);
		case ValueOpcode::ReadFirstLane: {
			if (!m_active_mask.IsEmpty() && inst.Arg(1).Resolve() == m_active_mask) {
				return EvaluateWide(inst.Arg(0), result);
			}
			SrtWalker clean_active(*this, inst.Arg(1), true);
			SrtWalker active(*this, inst.Arg(1), false, &clean_active);
			return active.EvaluateWide(inst.Arg(0), result);
		}
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32: return Arg(inst, 0, result);
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeExtractU32x2: return EvaluateExtract(inst, result);
		case ValueOpcode::CompositeConstructU64:
			if (!binary()) {
				return false;
			}
			result = static_cast<uint32_t>(a) |
			         (static_cast<uint64_t>(static_cast<uint32_t>(b)) << 32u);
			return true;
		case ValueOpcode::LoadAddressU32:
		case ValueOpcode::ReadConstBuffer:
		case ValueOpcode::LoadBufferU32:
			if (IsRawRead(m_program, inst)) {
				if (m_clean_evaluator != nullptr &&
				    (inst.Flags<SrtReadFlags>().clean != 0u ||
				     (inst.GetOpcode() == ValueOpcode::LoadBufferU32 &&
				      m_clean_evaluator->m_active_mask == m_active_mask))) {
					return m_clean_evaluator->EvaluateWide(Value(const_cast<Inst*>(&inst)), result);
				}
				return EvaluateRawRead(inst, result);
			}
			break;
		case ValueOpcode::IAdd32:
			if (binary()) {
				result = static_cast<uint32_t>(a + b);
				return true;
			}
			return false;
		case ValueOpcode::IAdd64:
			if (binary()) {
				result = a + b;
				return true;
			}
			return false;
		case ValueOpcode::ISub32:
			if (binary()) {
				result = static_cast<uint32_t>(a - b);
				return true;
			}
			return false;
		case ValueOpcode::ISub64:
			if (binary()) {
				result = a - b;
				return true;
			}
			return false;
		case ValueOpcode::IMul32:
			if (binary()) {
				result = static_cast<uint32_t>(a * b);
				return true;
			}
			return false;
		case ValueOpcode::UMulHi:
			if (binary()) {
				result = (static_cast<uint64_t>(static_cast<uint32_t>(a)) * static_cast<uint32_t>(b)) >> 32u;
				return true;
			}
			return false;
		case ValueOpcode::UMin32:
		case ValueOpcode::UMax32:
			if (binary()) {
				result = inst.GetOpcode() == ValueOpcode::UMin32
				    ? std::min(static_cast<uint32_t>(a), static_cast<uint32_t>(b))
				    : std::max(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
				return true;
			}
			return false;
		case ValueOpcode::ConvertF32U32:
			if (Arg(inst, 0, a)) {
				result = std::bit_cast<uint32_t>(static_cast<float>(static_cast<uint32_t>(a)));
				return true;
			}
			return false;
		case ValueOpcode::ConvertU32F32:
			if (Arg(inst, 0, a)) {
				// The conversion saturates and maps NaN to zero.
				const auto value = Float32(a);
				result           = std::isnan(value) || value <= 0.0f ? 0u
				                   : value >= 4294967296.0f ? UINT32_MAX
				                                            : static_cast<uint32_t>(value);
				return true;
			}
			return false;
		case ValueOpcode::FPMul32:
			if (binary()) {
				result = std::bit_cast<uint32_t>(Float32(a) * Float32(b));
				return true;
			}
			return false;
		case ValueOpcode::FPTrunc32:
			if (Arg(inst, 0, a)) {
				result = std::bit_cast<uint32_t>(std::trunc(Float32(a)));
				return true;
			}
			return false;
		case ValueOpcode::FPRecipIFlag32:
			if (Arg(inst, 0, a)) {
				const auto exponent = (a >> 23u) & 0xffu;
				// Normal positive powers of two have exact normal reciprocals in every FP mode.
				if ((a & 0x807fffffu) != 0u || exponent == 0u || exponent >= 254u) return false;
				result = (254u - exponent) << 23u;
				return true;
			}
			return false;
		case ValueOpcode::FPIsNan32:
			if (Arg(inst, 0, a)) {
				result = std::isnan(Float32(a));
				return true;
			}
			return false;
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
			if (binary()) {
				const auto operand = [&](uint64_t bits) {
					if (inst.Flags<FPCompareFlags>().flush_input_denorms &&
					    (bits & 0x7fffffffu) < 0x00800000u) {
						bits &= 0x80000000u;
					}
					return Float32(bits);
				};
				result = inst.GetOpcode() == ValueOpcode::FPOrdLessThanEqual32
				             ? operand(a) <= operand(b)
				             : operand(a) >= operand(b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseAnd32:
			if (binary()) {
				result = static_cast<uint32_t>(a & b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseAnd64:
			if (binary()) {
				result = a & b;
				return true;
			}
			return false;
		case ValueOpcode::BitwiseOr32:
			if (binary()) {
				result = static_cast<uint32_t>(a | b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseXor32:
			if (binary()) {
				result = static_cast<uint32_t>(a ^ b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseNot32:
			if (Arg(inst, 0, a)) {
				result = ~static_cast<uint32_t>(a);
				return true;
			}
			return false;
		case ValueOpcode::ShiftLeftLogical32:
			if (binary()) {
				result = static_cast<uint32_t>(a) << (b & 31u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftLeftLogical64:
			if (binary()) {
				result = a << (b & 63u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightLogical32:
			if (binary()) {
				result = static_cast<uint32_t>(a) >> (b & 31u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightLogical64:
			if (binary()) {
				result = a >> (b & 63u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightArithmetic32:
			if (binary()) {
				result = static_cast<uint32_t>(
				    std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >> (b & 31u));
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightArithmetic64:
			if (binary()) {
				result = static_cast<uint64_t>(std::bit_cast<int64_t>(a) >> (b & 63u));
				return true;
			}
			return false;
		case ValueOpcode::BitFieldUExtract:
			if (ternary()) {
				const auto offset = static_cast<uint32_t>(b);
				const auto width  = static_cast<uint32_t>(c);
				if (offset > 32u || width > 32u - offset) {
					return false;
				}
				const auto mask = width == 32u  ? UINT32_MAX
				                  : width == 0u ? 0u
				                                : (uint32_t {1} << width) - 1u;
				result = width == 0u ? 0u : (static_cast<uint32_t>(a) >> offset) & mask;
				return true;
			}
			return false;
		case ValueOpcode::BitFieldSExtract:
			if (ternary()) {
				const auto offset = static_cast<uint32_t>(b);
				const auto width  = static_cast<uint32_t>(c);
				if (offset > 32u || width > 32u - offset) {
					return false;
				}
				if (width == 0u) {
					result = 0;
					return true;
				}
				const auto mask = width == 32u ? UINT32_MAX : (uint32_t {1} << width) - 1u;
				auto       bits = (static_cast<uint32_t>(a) >> offset) & mask;
				if (width < 32u && (bits & (uint32_t {1} << (width - 1u))) != 0u) {
					bits |= ~mask;
				}
				result = bits;
				return true;
			}
			return false;
		case ValueOpcode::BitFieldInsert: {
			uint64_t d = 0;
			if (!ternary() || !Arg(inst, 3, d)) {
				return false;
			}
			const auto offset = static_cast<uint32_t>(c);
			const auto width  = static_cast<uint32_t>(d);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			if (width == 0u) {
				result = static_cast<uint32_t>(a);
				return true;
			}
			const auto mask =
			    width == 32u ? UINT32_MAX : ((uint32_t {1} << width) - 1u) << offset;
			result = (static_cast<uint32_t>(a) & ~mask) |
			         ((static_cast<uint32_t>(b) << offset) & mask);
			return true;
		}
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectF32: {
			auto& predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
			if (predicate.EvaluateWide(inst.Arg(0), a)) {
				return Arg(inst, a != 0u ? 1u : 2u, result);
			}
			return false;
		}
		case ValueOpcode::IEqual32:
			if (binary()) {
				result = static_cast<uint32_t>(a) == static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::INotEqual32:
			if (binary()) {
				result = static_cast<uint32_t>(a) != static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::IEqual64:
		case ValueOpcode::INotEqual64:
			if (!binary()) return false;
			result = inst.GetOpcode() == ValueOpcode::IEqual64 ? a == b : a != b;
			return true;
		case ValueOpcode::ULessThan32:
			// No U32 is below zero, including an unknown loop-carried value.
			if (!Arg(inst, 1, b)) return false;
			if (static_cast<uint32_t>(b) == 0u) {
				result = 0u;
				return true;
			}
			if (Arg(inst, 0, a)) {
				result = static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::ULessThanEqual32:
			// A zero lower bound accepts every U32, including a varying lane value.
			if (!Arg(inst, 0, a)) return false;
			if (a == 0u) {
				result = 1u;
				return true;
			}
			if (Arg(inst, 1, b)) {
				result = static_cast<uint32_t>(a) <= static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::UGreaterThan32:
			if (binary()) {
				result = static_cast<uint32_t>(a) > static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::SGreaterThanEqual32:
			if (binary()) {
				result = std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >=
				         std::bit_cast<int32_t>(static_cast<uint32_t>(b));
				return true;
			}
			return false;
		case ValueOpcode::LogicalAnd: {
			const bool left = Arg(inst, 0, a);
			if (left && a == 0u) {
				result = 0u;
				return true;
			}
			if (Arg(inst, 1, b) && (b == 0u || left)) {
				result = b != 0u;
				return true;
			}
			if (!left) {
				// Only lanes satisfying the left EXEC factor can make the conjunction
				// true. Use a separate memo so this assumption cannot escape its RHS.
				SrtWalker active(m_clean_evaluator != nullptr ? *m_clean_evaluator : *this,
				                 inst.Arg(0));
				if (active.EvaluateWide(inst.Arg(1), b) && b == 0u) {
					result = 0u;
					return true;
				}
			}
			return false;
		}
		case ValueOpcode::LogicalOr: {
			const bool left = Arg(inst, 0, a);
			if (left && a != 0u) {
				result = 1u;
				return true;
			}
			if (!Arg(inst, 1, b) || (b == 0u && !left)) return false;
			result = b != 0u;
			return true;
		}
		case ValueOpcode::LogicalXor:
			if (binary()) {
				result = (a != 0u) != (b != 0u);
				return true;
			}
			return false;
		case ValueOpcode::ConditionRef: return Arg(inst, 0, result);
		case ValueOpcode::LogicalNot:
			if (Arg(inst, 0, a)) {
				result = a == 0u;
				return true;
			}
			return false;
		case ValueOpcode::UndefU1:
		case ValueOpcode::UndefU32: return false;
		default: break;
	}
	return false;
}
bool SrtWalker::EvaluateDescriptor(uint32_t source, DescriptorValue& result) {
	if (source >= m_program.descriptor_sources.size()) {
		return false;
	}
	const auto& descriptor = m_program.descriptor_sources[source];
	result = {};
	result.dword_count = descriptor.dword_count;
	for (uint32_t index = 0; index < descriptor.dword_count; ++index) {
		if (!Evaluate(descriptor.dwords[index], result.dwords[index])) {
			return false;
		}
	}
	return true;
}

// =============================================================================
// Perf (uncommitted, §8.26): the DENSE EVALUATOR — the control-flow walk's
// expression graph compiled once per plan into a compact node array, with the
// SAME lazy algorithm run on it by DenseWalker below. Exactness mirrors the
// Inst-based walker case by case; KYTY_SRT_DENSE_VERIFY runs both and compares.
// =============================================================================

namespace {

// Value::operator== comparison bits for an immediate, mirrored by the dense
// immediate dedupe (so "node == mask node" equals "value == mask value").
uint64_t DenseImmediateBits(const Value& value) {
	switch (value.GetType()) {
		case Type::U1: return value.U1();
		case Type::U8: return value.U8();
		case Type::U16: return value.U16();
		case Type::F16: return value.F16Bits();
		case Type::U32: return value.U32();
		case Type::F32: return std::bit_cast<uint32_t>(value.F32Value());
		case Type::U64: return value.U64();
		case Type::ScalarReg: return RegIndex(value.ScalarRegister());
		case Type::VectorReg: return RegIndex(value.VectorRegister());
		default: return 0; // Void and Opaque never reach the immediate form
	}
}

// An immediate node evaluates exactly like EvaluateWide's immediate switch:
// the six evaluable types return their bits, everything else returns false.
bool DenseImmediateEvaluates(uint8_t type) {
	return type == static_cast<uint8_t>(Type::U1) || type == static_cast<uint8_t>(Type::U8) ||
	       type == static_cast<uint8_t>(Type::U16) || type == static_cast<uint8_t>(Type::U32) ||
	       type == static_cast<uint8_t>(Type::U64) || type == static_cast<uint8_t>(Type::F32);
}

class DenseCompiler {
public:
	explicit DenseCompiler(const ResourcePlan& program): m_program(program) {}

	bool Run() {
		auto& nodes = m_program.dense_nodes;
		if (nodes.empty()) {
			if (!Compile()) return false;
		}
		return true;
	}

private:
	const ResourcePlan& m_program;
	// Nodes are built in a deque: FillNode keeps a reference to its node while CompileValue
	// appends children, and a vector would move it (use-after-free found by the compute tests'
	// debug heap, 2026-10-21). Copied into plan.dense_nodes when compilation ends.
	std::deque<ResourcePlan::DenseNode> m_nodes;
	std::unordered_map<const Inst*, uint32_t> m_inst_nodes;
	std::map<std::pair<uint16_t, uint64_t>, uint32_t> m_imm_nodes;
	std::vector<const Inst*> m_pending;

	uint32_t NewNode() {
		m_nodes.emplace_back();
		return static_cast<uint32_t>(m_nodes.size() - 1u);
	}

	void Fail(ResourcePlan::DenseNode& node) {
		node = {};
		node.opcode = ResourcePlan::kDenseFail;
	}

	uint32_t CompileValue(Value value) {
		value = value.Resolve();
		if (value.IsImmediate()) {
			const auto type = static_cast<uint16_t>(value.GetType());
			const auto bits  = DenseImmediateBits(value);
			if (const auto found = m_imm_nodes.find({type, bits}); found != m_imm_nodes.end()) {
				return found->second;
			}
			const auto node = NewNode();
			auto&      entry = m_nodes[node];
			entry.opcode = ResourcePlan::kDenseImm;
			entry.flags  = static_cast<uint8_t>(value.GetType());
			entry.imm64  = bits;
			m_imm_nodes.emplace(std::pair {type, bits}, node);
			return node;
		}
		auto* inst = value.TryInstruction();
		if (const auto found = m_inst_nodes.find(inst); found != m_inst_nodes.end()) {
			return found->second;
		}
		const auto node = NewNode();
		m_inst_nodes.emplace(inst, node);
		m_pending.push_back(inst);
		return node;
	}

	void FlushPending() {
		while (!m_pending.empty()) {
			const auto* inst = m_pending.back();
			m_pending.pop_back();
			FillNode(const_cast<Inst*>(inst), m_inst_nodes.at(inst));
		}
	}

	uint32_t CompileArg(const Inst& inst, size_t index) {
		return CompileValue(inst.Arg(index));
	}

	static void SetArg(ResourcePlan::DenseNode& node, uint8_t slot, uint32_t child) {
		node.args[slot] = child;
		if (slot >= node.num_args) node.num_args = slot + 1u;
	}

	void FillNode(Inst* inst, uint32_t node_index) {
		auto&      node    = m_nodes[node_index];
		const auto opcode = inst->GetOpcode();
		node.opcode = static_cast<uint16_t>(opcode);
		switch (opcode) {
			case ValueOpcode::Identity:
				// Resolved away by CompileValue; an Identity here means the plan
				// was mutated after compilation started — refuse the whole form.
				Fail(node);
				return;
			case ValueOpcode::GetUserData: {
				const auto reg = RegIndex(inst->Arg(0).ScalarRegister());
				if (reg < m_program.user_data_base) {
					Fail(node); // constant structural failure, like the old switch
					return;
				}
				node.imm = reg - m_program.user_data_base;
				return;
			}
			case ValueOpcode::Phi: {
				const auto control = m_program.control_phi_predecessors.find(inst);
				if (control == m_program.control_phi_predecessors.end()) {
					const auto invariant = ResolveInvariantPhi(m_program, Value(inst));
					if (invariant.IsEmpty()) {
						Fail(node);
						return;
					}
					node.opcode = ResourcePlan::kDensePhiInvariant;
					SetArg(node, 0, CompileValue(invariant));
					return;
				}
				const auto count  = inst->NumArgs();
				const auto offset = static_cast<uint32_t>(m_program.dense_phi_args.size());
				m_program.dense_phi_args.resize(offset + count);
				m_program.dense_phi_preds.resize(offset + count);
				for (size_t index = 0; index < count; ++index) {
					m_program.dense_phi_args[offset + index]  = CompileValue(inst->Arg(index));
					m_program.dense_phi_preds[offset + index] = control->second[index];
				}
				node.opcode = ResourcePlan::kDensePhiControl;
				node.imm   = offset;
				node.num_args = static_cast<uint8_t>(count);
				return;
			}
			case ValueOpcode::ReadFirstLane:
				SetArg(node, 0, CompileArg(*inst, 0));
				SetArg(node, 1, CompileArg(*inst, 1));
				return;
			case ValueOpcode::CompositeExtractU64: {
				const auto index = inst->Arg(1).Resolve();
				if (!index.IsImmediate() || index.GetType() != Type::U32) {
					Fail(node);
					return;
				}
				node.opcode = ResourcePlan::kDenseExtractU64;
				node.imm    = index.U32();
				SetArg(node, 0, CompileArg(*inst, 0));
				return;
			}
			case ValueOpcode::CompositeExtractU32x2: {
				const auto index = inst->Arg(1).Resolve();
				if (!index.IsImmediate() || index.GetType() != Type::U32) {
					Fail(node);
					return;
				}
				const auto component = index.U32();
				if (component >= 2u) {
					Fail(node);
					return;
				}
				const auto* source = inst->Arg(0).ResolveInstruction();
				if (source == nullptr) {
					Fail(node);
					return;
				}
				if (source->GetOpcode() == ValueOpcode::CompositeConstructU32x2) {
					node.opcode = ResourcePlan::kDenseExtractConstruct;
					node.imm    = component;
					SetArg(node, 0, CompileValue(source->Arg(component)));
					return;
				}
				Fail(node);
				return;
			}
			case ValueOpcode::LoadAddressU32:
			case ValueOpcode::ReadConstBuffer:
			case ValueOpcode::LoadBufferU32: {
				if (!IsRawRead(m_program, *inst)) {
					Fail(node);
					return;
				}
				const auto flags = inst->Flags<SrtReadFlags>();
				const auto* handle = inst->Arg(0).ResolveInstruction();
				if (handle == nullptr) {
					Fail(node);
					return;
				}
				const bool is_vector   = opcode == ValueOpcode::LoadBufferU32;
				const bool is_constbuf = opcode == ValueOpcode::ReadConstBuffer;
				const bool handle4     = handle->NumArgs() == 4u;
				if ((is_vector || is_constbuf) && !handle4) {
					Fail(node);
					return;
				}
				node.opcode = ResourcePlan::kDenseRawRead;
				node.imm    = flags.index;
				node.flags |= is_vector ? ResourcePlan::kDenseReadVector : 0;
				node.flags |= is_constbuf ? ResourcePlan::kDenseReadConstBuf : 0;
				node.flags |= flags.clean != 0u ? ResourcePlan::kDenseReadClean : 0;
				node.flags |= handle4 ? ResourcePlan::kDenseReadHandle4 : 0;
				SetArg(node, 0, CompileValue(handle->Arg(0)));
				SetArg(node, 1, CompileValue(handle->Arg(1)));
				SetArg(node, 2, CompileArg(*inst, 1));
				if (is_vector || is_constbuf) {
					SetArg(node, 3, CompileValue(handle->Arg(2)));
					SetArg(node, 4, CompileValue(handle->Arg(3)));
				}
				if (is_vector) {
					SetArg(node, 5, CompileArg(*inst, 4));
				}
				return;
			}
			case ValueOpcode::FPOrdLessThanEqual32:
			case ValueOpcode::FPOrdGreaterThanEqual32:
				if (inst->Flags<FPCompareFlags>().flush_input_denorms) {
					node.flags |= ResourcePlan::kDenseFlushDenorms;
				}
				SetArg(node, 0, CompileArg(*inst, 0));
				SetArg(node, 1, CompileArg(*inst, 1));
				return;
			// One-argument forms.
			case ValueOpcode::BitCastU32F32:
			case ValueOpcode::BitCastF32U32:
			case ValueOpcode::ConvertF32U32:
			case ValueOpcode::ConvertU32F32:
			case ValueOpcode::FPTrunc32:
			case ValueOpcode::FPRecipIFlag32:
			case ValueOpcode::FPIsNan32:
			case ValueOpcode::BitwiseNot32:
			case ValueOpcode::ConditionRef:
			case ValueOpcode::LogicalNot:
			case ValueOpcode::UndefU1:
			case ValueOpcode::UndefU32:
				if (opcode == ValueOpcode::UndefU1 || opcode == ValueOpcode::UndefU32) {
					Fail(node); // the old switch returns false for these
					return;
				}
				SetArg(node, 0, CompileArg(*inst, 0));
				return;
			// Two-argument forms.
			case ValueOpcode::CompositeConstructU64:
			case ValueOpcode::IAdd32:
			case ValueOpcode::IAdd64:
			case ValueOpcode::ISub32:
			case ValueOpcode::ISub64:
			case ValueOpcode::IMul32:
			case ValueOpcode::UMulHi:
			case ValueOpcode::UMin32:
			case ValueOpcode::UMax32:
			case ValueOpcode::FPMul32:
			case ValueOpcode::BitwiseAnd32:
			case ValueOpcode::BitwiseAnd64:
			case ValueOpcode::BitwiseOr32:
			case ValueOpcode::BitwiseXor32:
			case ValueOpcode::ShiftLeftLogical32:
			case ValueOpcode::ShiftLeftLogical64:
			case ValueOpcode::ShiftRightLogical32:
			case ValueOpcode::ShiftRightLogical64:
			case ValueOpcode::ShiftRightArithmetic32:
			case ValueOpcode::ShiftRightArithmetic64:
			case ValueOpcode::IEqual32:
			case ValueOpcode::INotEqual32:
			case ValueOpcode::IEqual64:
			case ValueOpcode::INotEqual64:
			case ValueOpcode::UGreaterThan32:
			case ValueOpcode::SGreaterThanEqual32:
			case ValueOpcode::ULessThan32:
			case ValueOpcode::ULessThanEqual32:
			case ValueOpcode::LogicalAnd:
			case ValueOpcode::LogicalOr:
			case ValueOpcode::LogicalXor:
				SetArg(node, 0, CompileArg(*inst, 0));
				SetArg(node, 1, CompileArg(*inst, 1));
				return;
			// Three-argument forms.
			case ValueOpcode::BitFieldUExtract:
			case ValueOpcode::BitFieldSExtract:
			case ValueOpcode::SelectU32:
			case ValueOpcode::SelectU1:
			case ValueOpcode::SelectF32:
				SetArg(node, 0, CompileArg(*inst, 0));
				SetArg(node, 1, CompileArg(*inst, 1));
				SetArg(node, 2, CompileArg(*inst, 2));
				return;
			case ValueOpcode::BitFieldInsert:
				SetArg(node, 0, CompileArg(*inst, 0));
				SetArg(node, 1, CompileArg(*inst, 1));
				SetArg(node, 2, CompileArg(*inst, 2));
				SetArg(node, 3, CompileArg(*inst, 3));
				return;
			case ValueOpcode::GetShaderBase:
			case ValueOpcode::GetAddressResource:
			case ValueOpcode::GetBufferResource:
			case ValueOpcode::GetImageResource:
			case ValueOpcode::GetSamplerResource:
				// GetShaderBase evaluates; the resource getters fall to the old
				// switch's default (false) when evaluated directly.
				if (opcode != ValueOpcode::GetShaderBase) Fail(node);
				return;
			default:
				// Every opcode EvaluateInst does not handle evaluates false.
				Fail(node);
				return;
		}
	}

	bool Compile() {
		auto& plan = m_program;
		plan.dense_nodes.clear();
		m_nodes.clear();
		plan.dense_phi_args.clear();
		plan.dense_phi_preds.clear();
		plan.dense_block_conditions.assign(plan.control_flow.size(), ResourcePlan::kDenseNoNode);
		plan.dense_slot_nodes.assign(plan.srt_reads.size(), ResourcePlan::kDenseNoNode);
		plan.dense_slot_clean.assign(plan.srt_reads.size(), 0u);
		try {
			for (uint32_t block = 0; block < plan.control_flow.size(); ++block) {
				if (!plan.control_flow[block].condition.IsEmpty()) {
					plan.dense_block_conditions[block] = CompileValue(plan.control_flow[block].condition);
				}
			}
			for (uint32_t slot = 0; slot < plan.srt_reads.size(); ++slot) {
				const auto& read  = plan.srt_reads[slot];
				const auto* inst = read.value.Resolve().TryInstruction();
				plan.dense_slot_clean[slot] = inst != nullptr && inst->Flags<SrtReadFlags>().clean != 0u;
				plan.dense_slot_nodes[slot]  = CompileValue(read.value);
			}
			FlushPending();
		} catch (const std::bad_alloc&) {
			return false;
		}
		if (m_nodes.size() >= ResourcePlan::kDenseNoNode) {
			return false; // node indices would not be representable
		}
		plan.dense_nodes.assign(m_nodes.begin(), m_nodes.end());
		return true;
	}
};

float DenseFloat32(uint64_t bits) {
	return std::bit_cast<float>(static_cast<uint32_t>(bits));
}

} // namespace

// The dense walker: the SrtWalker's lazy evaluation over plan.dense_nodes.
// Mirrors EvaluateWide/EvaluateInst exactly, including short-circuits, the
// generation memo, child walkers for active masks and the clean-evaluator
// routing; context pools are the plan's dense_contexts.
class DenseWalker {
public:
	DenseWalker(const ResourcePlan& program, const SrtRuntime& runtime,
	            DenseWalker* clean_evaluator = nullptr)
	    : m_program(program), m_runtime(runtime), m_clean_evaluator(clean_evaluator),
	      m_session(this),
	      m_read_policy(clean_evaluator != nullptr ? 2u : 0u), m_context(AcquireContext()) {}

	DenseWalker(DenseWalker& parent, uint32_t active_mask_node, bool clean = false,
	            DenseWalker* clean_evaluator = nullptr)
	    : m_program(parent.m_program),
	      m_runtime(clean ? CleanRuntime(parent.m_runtime) : parent.m_runtime),
	      m_clean_evaluator(clean_evaluator), m_active_mask_node(active_mask_node),
	      m_reachable_blocks(parent.m_reachable_blocks), m_session(parent.m_session),
	      m_read_policy((parent.m_read_policy & 1u) | (clean ? 1u : 0u) |
	                    (clean_evaluator != nullptr ? 2u : 0u)),
	      m_context(AcquireContext()) {}

	~DenseWalker() {
		if (m_session != this) return;
		for (auto& context: m_program.dense_contexts) {
			if (context.session == this) context.session = nullptr;
		}
	}

	DenseWalker(const DenseWalker&)            = delete;
	DenseWalker& operator=(const DenseWalker&) = delete;

	bool Evaluate(uint32_t node, uint32_t& result) {
		uint64_t wide = 0;
		if (!EvaluateNode(node, wide)) {
			return false;
		}
		result = static_cast<uint32_t>(wide);
		return true;
	}

	void ResetUnknowns() {
		for (auto& context: m_program.dense_contexts) {
			if (context.session != m_session) continue;
			for (const auto index: context.unknowns) {
				if (index < context.values.size() &&
				    context.values[index].generation == (context.generation | 2u)) {
					context.values[index].generation = 0;
				}
			}
			context.unknowns.clear();
		}
	}

	const std::vector<uint8_t>* m_reachable_blocks = nullptr;

private:
	ResourcePlan::DenseContext& AcquireContext() {
		for (const auto& [mask, policy, context]: m_session->m_context_index) {
			if (policy == m_read_policy && context->session == m_session &&
			    mask == m_active_mask_node) {
				return *context;
			}
		}
		ResourcePlan::DenseContext* unused = nullptr;
		for (auto& context: m_program.dense_contexts) {
			if (context.session == m_session && context.active_mask_node == m_active_mask_node &&
			    context.read_policy == m_read_policy) {
				m_session->m_context_index.emplace_back(m_active_mask_node, m_read_policy, &context);
				return context;
			}
			if (context.session == nullptr && unused == nullptr) unused = &context;
		}
		auto& context = unused != nullptr ? *unused : m_program.dense_contexts.emplace_back();
		context.session          = m_session;
		context.active_mask_node = m_active_mask_node;
		context.read_policy      = m_read_policy;
		context.generation += 4;
		context.unknowns.clear();
		m_session->m_context_index.emplace_back(m_active_mask_node, m_read_policy, &context);
		return context;
	}

	bool Arg(const ResourcePlan::DenseNode& node, uint8_t index, uint64_t& result) {
		return EvaluateNode(node.args[index], result);
	}

	bool EvaluateNode(uint32_t node_index, uint64_t& result) {
		const auto& node = m_program.dense_nodes[node_index];
		if (node.opcode == ResourcePlan::kDenseImm) {
			if (DenseImmediateEvaluates(node.flags)) {
				result = node.imm64;
				return true;
			}
			return false;
		}
		if (m_active_mask_node != ResourcePlan::kDenseNoNode &&
		    node_index == m_active_mask_node) {
			result = 1u;
			return true;
		}
		if (m_active_mask_node != ResourcePlan::kDenseNoNode &&
		    IsRuntimeSelect(static_cast<ValueOpcode>(node.opcode)) && node.num_args == 3u &&
		    node.args[0] == m_active_mask_node) {
			return EvaluateNode(node.args[1], result);
		}
		if (node.opcode == ResourcePlan::kDenseFail) {
			return false;
		}
		if (node_index >= m_context.values.size()) {
			m_context.values.resize(m_program.dense_nodes.size());
		}
		if (m_context.values[node_index].generation == m_context.generation) {
			result = m_context.values[node_index].value;
			return true;
		}
		if (m_context.values[node_index].generation == (m_context.generation | 1u) ||
		    m_context.values[node_index].generation == (m_context.generation | 2u)) {
			return false;
		}
		m_context.values[node_index].generation = m_context.generation | 1u;
		uint64_t out = 0;
		const bool evaluated = EvaluateDense(node_index, out);
		auto& memo = m_context.values[node_index];
		if (!evaluated) {
			memo.generation = m_context.generation | 2u;
			m_context.unknowns.push_back(node_index);
			return false;
		}
		memo.value      = out;
		memo.generation = m_context.generation;
		result = out;
		return true;
	}

	bool EvaluateDense(uint32_t node_index, uint64_t& result) {
		const auto& node = m_program.dense_nodes[node_index];
		switch (node.opcode) {
			case ResourcePlan::kDensePhiControl: {
				bool found = false;
				for (uint8_t index = 0; index < node.num_args; ++index) {
					const auto predecessor = m_program.dense_phi_preds[node.imm + index];
					if (m_reachable_blocks != nullptr && !m_reachable_blocks->at(predecessor)) {
						continue;
					}
					uint64_t incoming = 0;
					if (!EvaluateNode(m_program.dense_phi_args[node.imm + index], incoming) ||
					    (found && incoming != result)) {
						return false;
					}
					result = incoming;
					found  = true;
				}
				return found;
			}
			case ResourcePlan::kDensePhiInvariant:
				return Arg(node, 0, result);
			case ResourcePlan::kDenseExtractU64: {
				uint64_t packed = 0;
				if (!Arg(node, 0, packed)) {
					return false;
				}
				result = static_cast<uint32_t>(packed >> (node.imm * 32u));
				return true;
			}
			case ResourcePlan::kDenseExtractConstruct:
				return Arg(node, 0, result);
			case ResourcePlan::kDenseExtractCarry: {
				uint64_t lhs = 0;
				uint64_t rhs = 0;
				if (!Arg(node, 0, lhs) || !Arg(node, 1, rhs)) {
					return false;
				}
				const auto sum =
				    static_cast<uint64_t>(static_cast<uint32_t>(lhs)) + static_cast<uint32_t>(rhs);
				result = node.imm == 0u ? static_cast<uint32_t>(sum)
				                        : static_cast<uint32_t>(sum >> 32u);
				return true;
			}
			case ResourcePlan::kDenseRawRead: {
				// Mirror of the old IsRawRead routing: clean reads (and vector
				// reads under a matching active mask) evaluate in the clean
				// evaluator's own memo context with the strict runtime.
				const bool vector = (node.flags & ResourcePlan::kDenseReadVector) != 0u;
				const bool clean  = (node.flags & ResourcePlan::kDenseReadClean) != 0u;
				if (m_clean_evaluator != nullptr &&
				    (clean ||
				     (vector &&
				      m_clean_evaluator->m_active_mask_node == m_active_mask_node))) {
					return m_clean_evaluator->EvaluateNode(node_index, result);
				}
				return EvaluateDenseRawRead(node, result);
			}
			default: break;
		}
		const auto opcode = static_cast<ValueOpcode>(node.opcode);
		uint64_t a = 0;
		uint64_t b = 0;
		uint64_t c = 0;
		const auto binary  = [&]() { return Arg(node, 0, a) && Arg(node, 1, b); };
		const auto ternary = [&]() { return Arg(node, 0, a) && Arg(node, 1, b) && Arg(node, 2, c); };
		switch (opcode) {
			case ValueOpcode::GetUserData:
				// The label was missing (statements before the first case never run), so every
				// user-data value was unknown in dense mode (found by the verify, 2026-10-21).
				if (node.imm >= m_runtime.user_data.size()) {
					return false;
				}
				result = m_runtime.user_data[node.imm];
				return true;
			case ValueOpcode::GetShaderBase:
				result = m_runtime.shader_base;
				return true;
			case ValueOpcode::ReadFirstLane: {
				if (m_active_mask_node != ResourcePlan::kDenseNoNode &&
				    node.args[1] == m_active_mask_node) {
					return Arg(node, 0, result);
				}
				DenseWalker clean_active(*this, node.args[1], true);
				DenseWalker active(*this, node.args[1], false, &clean_active);
				return active.EvaluateNode(node.args[0], result);
			}
			case ValueOpcode::BitCastU32F32:
			case ValueOpcode::BitCastF32U32:
				return Arg(node, 0, result);
			case ValueOpcode::CompositeConstructU64:
				if (!binary()) {
					return false;
				}
				result = static_cast<uint32_t>(a) |
				         (static_cast<uint64_t>(static_cast<uint32_t>(b)) << 32u);
				return true;
			case ValueOpcode::IAdd32:
				if (binary()) {
					result = static_cast<uint32_t>(a + b);
					return true;
				}
				return false;
			case ValueOpcode::IAdd64:
				if (binary()) {
					result = a + b;
					return true;
				}
				return false;
			case ValueOpcode::ISub32:
				if (binary()) {
					result = static_cast<uint32_t>(a - b);
					return true;
				}
				return false;
			case ValueOpcode::ISub64:
				if (binary()) {
					result = a - b;
					return true;
				}
				return false;
			case ValueOpcode::IMul32:
				if (binary()) {
					result = static_cast<uint32_t>(a * b);
					return true;
				}
				return false;
			case ValueOpcode::UMulHi:
				if (binary()) {
					result =
					    (static_cast<uint64_t>(static_cast<uint32_t>(a)) * static_cast<uint32_t>(b)) >> 32u;
					return true;
				}
				return false;
			case ValueOpcode::UMin32:
			case ValueOpcode::UMax32:
				if (binary()) {
					result = opcode == ValueOpcode::UMin32
					             ? std::min(static_cast<uint32_t>(a), static_cast<uint32_t>(b))
					             : std::max(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
					return true;
				}
				return false;
			case ValueOpcode::ConvertF32U32:
				if (Arg(node, 0, a)) {
					result = std::bit_cast<uint32_t>(static_cast<float>(static_cast<uint32_t>(a)));
					return true;
				}
				return false;
			case ValueOpcode::ConvertU32F32:
				if (Arg(node, 0, a)) {
					// The conversion saturates and maps NaN to zero (as EvaluateInst, 2d00e13b).
					const auto value = DenseFloat32(a);
					result           = std::isnan(value) || value <= 0.0f ? 0u
					                   : value >= 4294967296.0f ? UINT32_MAX
					                                            : static_cast<uint32_t>(value);
					return true;
				}
				return false;
			case ValueOpcode::FPMul32:
				if (binary()) {
					result = std::bit_cast<uint32_t>(DenseFloat32(a) * DenseFloat32(b));
					return true;
				}
				return false;
			case ValueOpcode::FPTrunc32:
				if (Arg(node, 0, a)) {
					result = std::bit_cast<uint32_t>(std::trunc(DenseFloat32(a)));
					return true;
				}
				return false;
			case ValueOpcode::FPRecipIFlag32:
				if (Arg(node, 0, a)) {
					const auto exponent = (a >> 23u) & 0xffu;
					if ((a & 0x807fffffu) != 0u || exponent == 0u || exponent >= 254u) return false;
					result = (254u - exponent) << 23u;
					return true;
				}
				return false;
			case ValueOpcode::FPIsNan32:
				if (Arg(node, 0, a)) {
					result = std::isnan(DenseFloat32(a));
					return true;
				}
				return false;
			case ValueOpcode::FPOrdLessThanEqual32:
			case ValueOpcode::FPOrdGreaterThanEqual32:
				if (binary()) {
					const auto operand = [&](uint64_t bits) {
						if ((node.flags & ResourcePlan::kDenseFlushDenorms) != 0u &&
						    (bits & 0x7fffffffu) < 0x00800000u) {
							bits &= 0x80000000u;
						}
						return DenseFloat32(bits);
					};
					result = opcode == ValueOpcode::FPOrdLessThanEqual32
					             ? operand(a) <= operand(b)
					             : operand(a) >= operand(b);
					return true;
				}
				return false;
			case ValueOpcode::BitwiseAnd32:
				if (binary()) {
					result = static_cast<uint32_t>(a & b);
					return true;
				}
				return false;
			case ValueOpcode::BitwiseAnd64:
				if (binary()) {
					result = a & b;
					return true;
				}
				return false;
			case ValueOpcode::BitwiseOr32:
				if (binary()) {
					result = static_cast<uint32_t>(a | b);
					return true;
				}
				return false;
			case ValueOpcode::BitwiseXor32:
				if (binary()) {
					result = static_cast<uint32_t>(a ^ b);
					return true;
				}
				return false;
			case ValueOpcode::BitwiseNot32:
				if (Arg(node, 0, a)) {
					result = ~static_cast<uint32_t>(a);
					return true;
				}
				return false;
			case ValueOpcode::ShiftLeftLogical32:
				if (binary()) {
					result = static_cast<uint32_t>(a) << (b & 31u);
					return true;
				}
				return false;
			case ValueOpcode::ShiftLeftLogical64:
				if (binary()) {
					result = a << (b & 63u);
					return true;
				}
				return false;
			case ValueOpcode::ShiftRightLogical32:
				if (binary()) {
					result = static_cast<uint32_t>(a) >> (b & 31u);
					return true;
				}
				return false;
			case ValueOpcode::ShiftRightLogical64:
				if (binary()) {
					result = a >> (b & 63u);
					return true;
				}
				return false;
			case ValueOpcode::ShiftRightArithmetic32:
				if (binary()) {
					result = static_cast<uint32_t>(
					    std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >> (b & 31u));
					return true;
				}
				return false;
			case ValueOpcode::ShiftRightArithmetic64:
				if (binary()) {
					result = static_cast<uint64_t>(std::bit_cast<int64_t>(a) >> (b & 63u));
					return true;
				}
				return false;
			case ValueOpcode::BitFieldUExtract:
				if (ternary()) {
					const auto offset = static_cast<uint32_t>(b);
					const auto width  = static_cast<uint32_t>(c);
					if (offset > 32u || width > 32u - offset) {
						return false;
					}
					const auto mask = width == 32u  ? UINT32_MAX
					                  : width == 0u ? 0u
					                                : (uint32_t {1} << width) - 1u;
					result = width == 0u ? 0u : (static_cast<uint32_t>(a) >> offset) & mask;
					return true;
				}
				return false;
			case ValueOpcode::BitFieldSExtract:
				if (ternary()) {
					const auto offset = static_cast<uint32_t>(b);
					const auto width  = static_cast<uint32_t>(c);
					if (offset > 32u || width > 32u - offset) {
						return false;
					}
					if (width == 0u) {
						result = 0;
						return true;
					}
					const auto mask = width == 32u ? UINT32_MAX : (uint32_t {1} << width) - 1u;
					auto       bits = (static_cast<uint32_t>(a) >> offset) & mask;
					if (width < 32u && (bits & (uint32_t {1} << (width - 1u))) != 0u) {
						bits |= ~mask;
					}
					result = bits;
					return true;
				}
				return false;
			case ValueOpcode::BitFieldInsert: {
				uint64_t d = 0;
				if (!ternary() || !Arg(node, 3, d)) {
					return false;
				}
				const auto offset = static_cast<uint32_t>(c);
				const auto width  = static_cast<uint32_t>(d);
				if (offset > 32u || width > 32u - offset) {
					return false;
				}
				if (width == 0u) {
					result = static_cast<uint32_t>(a);
					return true;
				}
				const auto mask =
				    width == 32u ? UINT32_MAX : ((uint32_t {1} << width) - 1u) << offset;
				result = (static_cast<uint32_t>(a) & ~mask) |
				         ((static_cast<uint32_t>(b) << offset) & mask);
				return true;
			}
			case ValueOpcode::SelectU32:
			case ValueOpcode::SelectU1:
			case ValueOpcode::SelectF32: {
				auto& predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
				if (predicate.EvaluateNode(node.args[0], a)) {
					return Arg(node, a != 0u ? 1u : 2u, result);
				}
				return false;
			}
			case ValueOpcode::IEqual32:
				if (binary()) {
					result = static_cast<uint32_t>(a) == static_cast<uint32_t>(b);
					return true;
				}
				return false;
			case ValueOpcode::INotEqual32:
				if (binary()) {
					result = static_cast<uint32_t>(a) != static_cast<uint32_t>(b);
					return true;
				}
				return false;
			case ValueOpcode::IEqual64:
			case ValueOpcode::INotEqual64:
				if (!binary()) return false;
				result = opcode == ValueOpcode::IEqual64 ? a == b : a != b;
				return true;
			case ValueOpcode::ULessThan32:
				if (!Arg(node, 1, b)) return false;
				if (static_cast<uint32_t>(b) == 0u) {
					result = 0u;
					return true;
				}
				if (Arg(node, 0, a)) {
					result = static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
					return true;
				}
				return false;
			case ValueOpcode::ULessThanEqual32:
				if (!Arg(node, 0, a)) return false;
				if (a == 0u) {
					result = 1u;
					return true;
				}
				if (Arg(node, 1, b)) {
					result = static_cast<uint32_t>(a) <= static_cast<uint32_t>(b);
					return true;
				}
				return false;
			case ValueOpcode::UGreaterThan32:
				if (binary()) {
					result = static_cast<uint32_t>(a) > static_cast<uint32_t>(b);
					return true;
				}
				return false;
			case ValueOpcode::SGreaterThanEqual32:
				if (binary()) {
					result = std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >=
					         std::bit_cast<int32_t>(static_cast<uint32_t>(b));
					return true;
				}
				return false;
			case ValueOpcode::LogicalAnd: {
				const bool left = Arg(node, 0, a);
				if (left && a == 0u) {
					result = 0u;
					return true;
				}
				if (Arg(node, 1, b) && (b == 0u || left)) {
					result = b != 0u;
					return true;
				}
				if (!left) {
					DenseWalker active(m_clean_evaluator != nullptr ? *m_clean_evaluator : *this,
					                   node.args[0]);
					if (active.EvaluateNode(node.args[1], b) && b == 0u) {
						result = 0u;
						return true;
					}
				}
				return false;
			}
			case ValueOpcode::LogicalOr: {
				const bool left = Arg(node, 0, a);
				if (left && a != 0u) {
					result = 1u;
					return true;
				}
				if (!Arg(node, 1, b) || (b == 0u && !left)) return false;
				result = b != 0u;
				return true;
			}
			case ValueOpcode::LogicalXor:
				if (binary()) {
					result = (a != 0u) != (b != 0u);
					return true;
				}
				return false;
			case ValueOpcode::ConditionRef:
				return Arg(node, 0, result);
			case ValueOpcode::LogicalNot:
				if (Arg(node, 0, a)) {
					result = a == 0u;
					return true;
				}
				return false;
			default:
				return false;
		}
	}

	bool EvaluateDenseRawRead(const ResourcePlan::DenseNode& node, uint64_t& result) {
		if (node.imm >= m_program.memory_info.size()) {
			return false;
		}
		const auto& mem      = m_program.memory_info[node.imm];
		const bool  vector   = (node.flags & ResourcePlan::kDenseReadVector) != 0u;
		if (vector) {
			// Old: guard.IsImmediate() || guard != m_active_mask (an empty mask
			// never equals a real guard, matching kDenseNoNode != any node).
			const auto& guard_node = m_program.dense_nodes[node.args[5]];
			if (guard_node.opcode == ResourcePlan::kDenseImm ||
			    node.args[5] != m_active_mask_node) {
				uint64_t enabled = 0;
				if (!EvaluateNode(node.args[5], enabled)) return false;
				if (enabled == 0u) {
					result = 0u;
					return true;
				}
			}
		}
		uint64_t low    = 0;
		uint64_t high   = 0;
		uint64_t offset = 0;
		if (!EvaluateNode(node.args[0], low) || !EvaluateNode(node.args[1], high) ||
		    !EvaluateNode(node.args[2], offset)) {
			return false;
		}
		const auto base      = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
		const auto immediate = static_cast<int64_t>(static_cast<int32_t>(mem.offset));
		uint64_t   address   = 0;
		if (vector || (node.flags & ResourcePlan::kDenseReadConstBuf) != 0u) {
			uint64_t records = 0;
			uint64_t word3   = 0;
			if (!EvaluateNode(node.args[3], records) || !EvaluateNode(node.args[4], word3)) {
				return false;
			}
			if (immediate < 0) {
				return false;
			}
			const auto byte_offset =
			    (static_cast<uint64_t>(immediate) & ~uint64_t {3}) + (static_cast<uint32_t>(offset) & ~3u);
			const auto stride = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
			if (vector) {
				if ((high & (1u << 31u)) != 0u || (word3 & ((1u << 23u) | 0xf0000000u)) != 0u)
					return false;
				if (stride == 0u || records == 0u || ((word3 >> 12u) & 0x7fu) == 0u) {
					result = 0u;
					return true;
				}
			}
			const auto size = stride == 0u
			                      ? static_cast<uint64_t>(static_cast<uint32_t>(records))
			                      : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
			if (byte_offset > size || size - byte_offset < sizeof(uint32_t)) {
				result = 0u;
				return true;
			}
			address = (base & ~uint64_t {3}) + byte_offset;
		} else {
			const auto relative = (immediate & ~int64_t {3}) +
			                      static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
			if (!AddSignedAddress(base & ~uint64_t {3}, relative, address)) {
				return false;
			}
		}
		uint32_t word = 0;
		const auto reader = vector ? m_runtime.read_specialization_memory : m_runtime.read_memory;
		if (reader != nullptr) {
			if (!reader(m_runtime.userdata, address, {&word, 1})) {
				return false;
			}
		} else {
			if (vector) {
				return false;
			}
			std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
		}
		result = word;
		return true;
	}

	const ResourcePlan&                m_program;
	SrtRuntime                         m_runtime;
	DenseWalker*                       m_clean_evaluator = nullptr;
	uint32_t                           m_active_mask_node = ResourcePlan::kDenseNoNode;
	DenseWalker*                       m_session;
	uint8_t                            m_read_policy;
	// The session's contexts by (active mask, read policy), used through m_session like
	// SrtWalker's. Declared before m_context: AcquireContext runs in the member initializers
	// (declared after it, the vector was used before construction: heap corruption, 2026-10-21).
	struct ContextIndexEntry {
		uint32_t                     mask;
		uint8_t                      policy;
		ResourcePlan::DenseContext* context;
	};
	std::vector<ContextIndexEntry> m_context_index;
	ResourcePlan::DenseContext&         m_context;
};

// The dense evaluator's outputs for verification / hand-off.
struct DenseWalkOutput {
	std::vector<uint32_t> flat;
	std::vector<uint8_t>  visited;
	std::vector<uint8_t>  active;
	bool                 ok = false;
};

// Runs the SAME fixpoint and visited-reads phase as SrtWalker::RefreshFlatBuffer's
// control-flow branch, on the plan's dense node array. `runtime` must already be
// stripped of capture wrappers when replaying for verification.
bool RunDenseControlFlow(const ResourcePlan& program, const SrtRuntime& runtime,
                         bool has_clean_evaluator, DenseWalkOutput& out) {
	if (program.control_flow.empty() || !program.srt_plan_complete) return false;
	if (program.dense_state == 0u) {
		DenseCompiler compiler(program);
		program.dense_state = compiler.Run() ? 1u : 2u;
		static uint64_t compiled = 0, unsupported = 0;
		program.dense_state == 1u ? compiled++ : unsupported++;
		if ((compiled + unsupported) % 128u == 0u) {
			std::fprintf(stderr, "[dense] compiled %llu plans, %llu unsupported\n",
			             static_cast<unsigned long long>(compiled),
			             static_cast<unsigned long long>(unsupported));
			std::fflush(stderr);
		}
	}
	if (program.dense_state != 1u) return false;

	DenseWalker clean(program, CleanRuntime(runtime));
	DenseWalker walker(program, runtime, has_clean_evaluator ? &clean : nullptr);
	DenseWalker* predicate = has_clean_evaluator ? &clean : &walker;

	auto& visited = out.visited;
	visited.assign(program.control_flow.size(), 1u);
	std::vector<uint8_t> reachable(program.control_flow.size());
	std::vector<uint32_t> pending;
	walker.ResetUnknowns();
	if (has_clean_evaluator) clean.ResetUnknowns();
	walker.m_reachable_blocks = &visited;
	if (has_clean_evaluator) clean.m_reachable_blocks = &visited;
	for (uint32_t rounds = 0;; ++rounds) {
		reachable.assign(program.control_flow.size(), 0u);
		pending.clear();
		pending.push_back(0u);
		while (!pending.empty()) {
			const auto index = pending.back();
			pending.pop_back();
			if (reachable.at(index) != 0u) continue;
			reachable[index] = 1u;
			const auto& block = program.control_flow[index];
			uint32_t condition = 0;
			bool     condition_known = false;
			const auto condition_node = program.dense_block_conditions[index];
			if (condition_node != ResourcePlan::kDenseNoNode &&
			    (runtime.read_specialization_memory != nullptr ||
			     !program.capture_specialization_reads)) {
				g_srt_read_role = 1;
				condition_known = predicate->Evaluate(condition_node, condition);
				g_srt_read_role = 0;
			}
			if (condition_known) {
				pending.push_back(block.successors[condition != 0u ? 0u : 1u]);
			} else {
				pending.insert(pending.end(), block.successors.begin(), block.successors.end());
			}
		}
		if (reachable == visited) break;
		visited.swap(reachable);
		if (program.control_phi_predecessors.empty()) break;
		walker.ResetUnknowns();
		if (has_clean_evaluator) clean.ResetUnknowns();
	}
	// Experiment (KYTY_SRT_BRANCH_BOTH, as in the old fixpoint).
	static const bool branch_both = [] {
		const char* value = std::getenv("KYTY_SRT_BRANCH_BOTH");
		return value != nullptr && value[0] == '1';
	}();
	if (branch_both) {
		walker.ResetUnknowns();
		if (has_clean_evaluator) clean.ResetUnknowns();
	}
	auto& flat  = out.flat;
	auto& active = out.active;
	flat.assign(program.srt_reads.size(), 0u);
	active.assign(program.descriptor_sources.size(), 1u);
	for (const auto& block: program.control_flow) {
		for (const auto source: block.sources) active.at(source) = 0u;
	}
	const auto refresh = [&](uint32_t slot) {
		if (slot >= program.srt_reads.size()) return false;
		const bool clean_slot = program.dense_slot_clean[slot] != 0u;
		if (clean_slot && (!has_clean_evaluator ||
		                   runtime.read_specialization_memory == nullptr)) {
			return false;
		}
		auto& evaluator = clean_slot ? clean : walker;
		const auto& read = program.srt_reads[slot];
		return read.flat_offset < flat.size() &&
		       evaluator.Evaluate(program.dense_slot_nodes[slot], flat[read.flat_offset]);
	};
	for (uint32_t index = 0; index < visited.size(); ++index) {
		if (visited[index] == 0u) continue;
		const auto& block = program.control_flow[index];
		for (const auto source: block.sources) active.at(source) = 1u;
		for (const auto slot: block.srt_reads) {
			if (!refresh(slot)) return false;
		}
	}
	out.ok = true;
	return true;
}


// Perf (uncommitted): fetch one direct read run (one-level user-data pointer +
// fixed offset) into the flat buffer and seed the evaluator's memo so later
// condition evaluations of the same words skip the reader. The reader choice
// matches EvaluateRawRead: the clean evaluator's reader for clean reads, the
// ordinary reader otherwise; a decline fails the run and the caller falls
// back to the fixpoint walk.
bool SrtWalker::RefreshDirectControlFlow(std::vector<uint32_t>& flat) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SRT_DIRECT_CF");
		return value != nullptr && value[0] == '1';
	}();
	if (!enabled) return false;
	if (m_program.srt_direct_state == 0u) {
		m_program.srt_direct_state = BuildDirectCfPlan(m_program) ? 1u : 2u;
	}
	if (m_program.srt_direct_state != 1u) return false;
	if (m_program.control_flow.empty() || m_program.srt_reads.empty()) return false;

	const auto read_run = [&](uint32_t slot, uint32_t count) -> bool {
		const auto& direct = m_program.srt_direct_reads[slot];
		if (!direct.valid) return false;
		if (direct.clean &&
		    (m_clean_evaluator == nullptr || m_runtime.read_specialization_memory == nullptr)) {
			return false;
		}
		auto&       evaluator = direct.clean ? *m_clean_evaluator : *this;
		const auto& user_data = evaluator.m_runtime.user_data;
		if (direct.low_register >= user_data.size() ||
		    direct.high_register >= user_data.size()) {
			return false;
		}
		const auto base = ((static_cast<uint64_t>(user_data[direct.high_register]) << 32u) |
		                   user_data[direct.low_register]) &
		                  AddressMask;
		uint64_t address = 0;
		const auto bytes = static_cast<uint64_t>(count) * 4u;
		if (!AddSignedAddress(base & ~uint64_t {3}, direct.relative, address) ||
		    address > AddressMask || bytes - 1 > AddressMask - address) {
			return false;
		}
		const auto first = m_program.srt_reads[slot].flat_offset;
		if (first > flat.size() || count > flat.size() - first) return false;
		const auto words = std::span(flat).subspan(first, count);
		if (evaluator.m_runtime.read_memory != nullptr) {
			if (!evaluator.m_runtime.read_memory(evaluator.m_runtime.userdata, address,
			                                     words)) {
				return false;
			}
		} else {
			std::memcpy(words.data(), reinterpret_cast<const void*>(address),
			            words.size_bytes());
		}
		auto& values = evaluator.m_context.values;
		if (values.size() < m_program.evaluation_value_count) {
			values.resize(m_program.evaluation_value_count);
		}
		for (uint32_t i = 0; i < count; ++i) {
			const auto index = m_program.srt_direct_reads[slot + i].evaluation_index;
			if (index < values.size()) {
				values[index].value      = words[i];
				values[index].generation = evaluator.m_context.generation;
			}
		}
		return true;
	};

	flat.assign(m_program.srt_reads.size(), 0u);
	auto& filled = m_program.srt_cf_filled;
	filled.assign(m_program.srt_reads.size(), 0u);
	auto& active = m_program.active_sources;
	active        = m_program.srt_cf_entry_active;
	auto& visited = m_program.visited_blocks;
	auto& pending = m_program.pending_blocks;
	visited.assign(m_program.control_flow.size(), 0u);
	pending.clear();
	pending.push_back(0u);
	static uint64_t direct_walks = 0, direct_fallbacks = 0, direct_cond_cacheable = 0,
	                direct_cond_hits = 0, direct_cond_total = 0;
	direct_walks++;
	while (!pending.empty()) {
		const auto index = pending.back();
		pending.pop_back();
		if (visited.at(index) != 0u) continue;
		visited[index] = 1u;
		const auto& block = m_program.control_flow[index];
		for (const auto source: block.sources) active.at(source) = 1u;
		for (uint32_t r = m_program.srt_cf_run_begin[index];
		     r < m_program.srt_cf_run_begin[index + 1]; ++r) {
			const auto run = m_program.srt_cf_runs[r];
			const auto slot = run.slot;
			if (m_program.srt_direct_reads[slot].valid) {
				if (!read_run(slot, run.count)) {
					direct_fallbacks++;
					return false;
				}
				for (uint32_t i = 0; i < run.count; ++i) {
					const auto offset = m_program.srt_reads[slot + i].flat_offset;
					if (offset < filled.size()) filled[offset] = 1u;
				}
			} else {
				// Not a direct read: evaluate the slots through the IR path.
				for (uint32_t i = 0; i < run.count; ++i) {
					const auto this_slot = slot + i;
					const auto& read     = m_program.srt_reads[this_slot];
					const auto* inst     = read.value.ResolveInstruction();
					const bool  clean    = inst != nullptr &&
				                           inst->Flags<SrtReadFlags>().clean != 0u;
					if (clean &&
					    (m_clean_evaluator == nullptr ||
					     m_runtime.read_specialization_memory == nullptr)) {
						direct_fallbacks++;
						return false;
					}
					auto& evaluator = clean ? *m_clean_evaluator : *this;
					if (read.flat_offset >= flat.size() ||
					    !evaluator.Evaluate(read.value, flat[read.flat_offset])) {
						direct_fallbacks++;
						return false;
					}
					if (read.flat_offset < filled.size()) filled[read.flat_offset] = 1u;
				}
			}
		}
		uint32_t condition = 0;
		auto&    predicate  = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
		bool     condition_known = false;
		if (!block.condition.IsEmpty() &&
		    (m_runtime.read_specialization_memory != nullptr ||
		     !m_program.capture_specialization_reads)) {
			auto& cache = m_program.srt_cf_conditions;
			if (cache.size() < m_program.control_flow.size()) {
				cache.resize(m_program.control_flow.size());
			}
			auto& entry = cache[index];
			if (entry.classified == 0u) {
				entry.classified = CollectConditionLeaves(m_program, block.condition,
				                                           entry.leaves)
				                      ? 1u
				                      : 2u;
				if (entry.classified == 1u) {
					entry.last_values.resize(entry.leaves.size());
				}
			}
			direct_cond_total++;
			if (entry.classified == 1u) direct_cond_cacheable++;
			bool ready = entry.classified == 1u && entry.has_last;
			if (ready) {
				for (size_t i = 0; i < entry.leaves.size(); ++i) {
					const auto& leaf = entry.leaves[i];
					uint64_t    now  = 0;
					if (leaf.kind == 0u) {
						if (leaf.index >= m_runtime.user_data.size()) {
							ready = false;
							break;
						}
						now = m_runtime.user_data[leaf.index];
					} else if (leaf.kind == 2u) {
						if (leaf.index >= flat.size() || filled.at(leaf.index) == 0u) {
							// Not fetched (yet): cannot compare, evaluate instead.
							ready = false;
							break;
						}
						now = flat[leaf.index];
					} else {
						now = m_runtime.shader_base;
					}
					if (now != entry.last_values[i]) {
						ready = false;
						break;
					}
				}
			}
			if (ready) {
				condition       = entry.last_result;
				condition_known = true;
				direct_cond_hits++;
			} else {
				g_srt_read_role = 1;
				condition_known = predicate.Evaluate(block.condition, condition);
				g_srt_read_role = 0;
				if (entry.classified == 1u && condition_known) {
					bool leaves_ready = true;
					for (size_t i = 0; i < entry.leaves.size(); ++i) {
						const auto& leaf = entry.leaves[i];
						if (leaf.kind == 0u) {
							if (leaf.index >= m_runtime.user_data.size()) {
								leaves_ready = false;
								break;
							}
							entry.last_values[i] = m_runtime.user_data[leaf.index];
						} else if (leaf.kind == 2u) {
							if (leaf.index >= flat.size() || filled.at(leaf.index) == 0u) {
								leaves_ready = false;
								break;
							}
							entry.last_values[i] = flat[leaf.index];
						} else {
							entry.last_values[i] = m_runtime.shader_base;
						}
					}
					if (leaves_ready) {
						entry.last_result = condition;
						entry.has_last    = true;
					}
				}
			}
		}
		if (condition_known) {
			pending.push_back(block.successors[condition != 0u ? 0u : 1u]);
		} else {
			pending.insert(pending.end(), block.successors.begin(), block.successors.end());
		}
	}
	if ((direct_walks & 0xffffu) == 0u) {
		std::fprintf(stderr,
		             "[srtcf-direct] %llu walks, %llu fallbacks | cond-cache: %llu of %llu "
		             "cacheable, %llu hits\n",
		             static_cast<unsigned long long>(direct_walks),
		             static_cast<unsigned long long>(direct_fallbacks),
		             static_cast<unsigned long long>(direct_cond_cacheable),
		             static_cast<unsigned long long>(direct_cond_total),
		             static_cast<unsigned long long>(direct_cond_hits));
		std::fflush(stderr);
	}
	return true;
}

bool SrtWalker::RefreshFlatBuffer(std::vector<uint32_t>& flat) {
	if (!m_program.srt_plan_complete) return false;
	if (m_program.srt_code && m_runtime.read_memory == nullptr) {
		flat.resize(m_program.srt_reads.size());
		m_program.active_sources.clear();
		m_context.values.resize(m_program.evaluation_value_count);
		using Walker = bool KYTY_SYSV_ABI (*)(const uint32_t*, uint32_t*,
		    ResourcePlan::EvaluationContext::Entry*, uint64_t, size_t);
		return m_program.srt_code->getCode<Walker>()(m_runtime.user_data.data(), flat.data(),
		    m_context.values.data(), m_context.generation, m_runtime.user_data.size());
	}
	// Perf (uncommitted): a slot's clean flag and evaluation index are plan constants; values
	// the conditions already evaluated are taken from the memo without touching the instruction.
	static const bool slot_cache_enabled = SrtSwitch("KYTY_SRT_SLOT_CACHE");
	auto&             slot_cache         = m_program.srt_read_cache;
	if (slot_cache.size() != m_program.srt_reads.size()) {
		slot_cache.assign(m_program.srt_reads.size(), {});
	}
	const auto refresh = [&](uint32_t slot) {
		if (slot >= m_program.srt_reads.size()) return false;
		const auto& read  = m_program.srt_reads[slot];
		auto&       entry = slot_cache[slot];
		if (!entry.prepared || !slot_cache_enabled) {
			const auto* inst       = read.value.ResolveInstruction();
			entry.clean            = inst != nullptr && inst->Flags<SrtReadFlags>().clean != 0u;
			entry.evaluation_index = inst != nullptr
			                             ? inst->EvaluationIndex(m_program.evaluation_value_count)
			                             : UINT32_MAX;
			entry.prepared         = true;
		}
		const bool clean = entry.clean;
		if (clean && (m_clean_evaluator == nullptr || m_runtime.read_specialization_memory == nullptr))
			return false;
		auto& evaluator = clean ? *m_clean_evaluator : *this;
		if (read.flat_offset >= flat.size()) return false;
		return (slot_cache_enabled &&
		        evaluator.TryMemoized(entry.evaluation_index, flat[read.flat_offset])) ||
		       evaluator.Evaluate(read.value, flat[read.flat_offset]);
	};
	auto& active = m_program.active_sources;
	if (m_program.control_flow.empty()) {
		active.clear();
		flat.resize(m_program.srt_reads.size());
		for (uint32_t slot = 0; slot < m_program.srt_reads.size(); ++slot) {
			if (!refresh(slot)) return false;
		}
		return true;
	}
	// Perf (uncommitted): the direct control-flow walk (data-first single
	// pass, conditions comparable against fetched words). Any failure falls
	// back to the fixpoint below.
	if (RefreshDirectControlFlow(flat)) {
		return true;
	}
	// Perf (uncommitted, §8.26 dense evaluator): KYTY_SRT_DENSE=1 runs the
	// control-flow walk on the plan's compiled dense node array; any failure
	// falls back to the fixpoint below. KYTY_SRT_DENSE_VERIFY=1 additionally
	// replays the dense walk (capture-stripped) and compares its outputs with
	// the old fixpoint's at the end of this function; the old result is the
	// one used.
	static const bool dense_enabled = [] {
		const char* value = std::getenv("KYTY_SRT_DENSE");
		return value != nullptr && value[0] == '1';
	}();
	static const bool dense_verify_env = [] {
		const char* value = std::getenv("KYTY_SRT_DENSE_VERIFY");
		return value != nullptr && value[0] == '1';
	}();
	// Experiment (uncommitted, §8.26): KYTY_SRT_DENSE_ALTERNATE=<seconds> switches the dense walk
	// on in even and off in odd windows of steady_clock time, an A/B inside one session (same
	// scenes for both); tools\frame_trace.py --alternate <seconds> splits the intervals likewise.
	static const int64_t alternate_us = [] {
		const char* value = std::getenv("KYTY_SRT_DENSE_ALTERNATE");
		return value != nullptr ? static_cast<int64_t>(std::atof(value) * 1e6) : int64_t {0};
	}();
	bool dense_now = dense_enabled;
	if (alternate_us > 0) {
		const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
		                    std::chrono::steady_clock::now().time_since_epoch())
		                    .count();
		dense_now = (us / alternate_us) % 2 == 0;
	}
	// MaterializeResources' full verification (KYTY_SRT_DENSE_VERIFY=full) forces one mode per call.
	const bool      dense_forced  = g_srt_dense_override >= 0;
	const bool      dense_wanted  = dense_forced ? g_srt_dense_override == 1 : dense_now;
	const bool      dense_verify  = !dense_forced && dense_verify_env;
	DenseWalkOutput dense_output;
	bool            dense_ran = false;
	static uint64_t dense_walks = 0, dense_fails = 0;
	static double   dense_us = 0.0;
	if (dense_wanted || dense_verify) {
		SrtRuntime dense_runtime = m_runtime;
		if (dense_verify && m_runtime.read_memory == &SrtReadCapture::ReadOrdinary) {
			// Verification replays the walk without appending a second set of
			// capture ranges: the old fixpoint records the real ones.
			dense_runtime = static_cast<const SrtReadCapture*>(m_runtime.userdata)->Source();
		}
		const auto dense_begin = std::chrono::steady_clock::now();
		dense_ran = RunDenseControlFlow(m_program, dense_runtime,
		                                 m_clean_evaluator != nullptr, dense_output);
		dense_walks++;
		dense_us += std::chrono::duration<double, std::micro>(
		                std::chrono::steady_clock::now() - dense_begin)
		                .count();
		if (!dense_ran) dense_fails++;
		if ((dense_walks & 0xffffu) == 0u) {
			std::fprintf(stderr,
			             "[dense] %llu walks (%llu failed), %.2f us/walk, %llu nodes\n",
			             static_cast<unsigned long long>(dense_walks),
			             static_cast<unsigned long long>(dense_fails),
			             dense_us / static_cast<double>(dense_walks),
			             static_cast<unsigned long long>(m_program.dense_nodes.size()));
			std::fflush(stderr);
		}
		if (dense_ran && dense_wanted && !dense_verify) {
			flat                     = std::move(dense_output.flat);
			m_program.active_sources = std::move(dense_output.active);
			m_program.visited_blocks = std::move(dense_output.visited);
			// As the fixpoint below leaves them: descriptors evaluated after the walk see Phi
			// predecessors through the final visited set.
			m_reachable_blocks = &m_program.visited_blocks;
			if (m_clean_evaluator != nullptr) {
				m_clean_evaluator->m_reachable_blocks = &m_program.visited_blocks;
			}
			return true;
		}
	}
	// Profiling (uncommitted): reachability rounds vs the reads of the visited blocks.
	std::optional<tracy::ScopedZone> cf_zone;
	{
		static constexpr tracy::SourceLocationData location {
		    "Srt:Reachability", TracyFunction, TracyFile, static_cast<uint32_t>(__LINE__), 0};
		cf_zone.emplace(&location, TRACY_CALLSTACK, tracy::ProfilerAvailable());
	}
	static uint64_t cf_walks = 0, cf_rounds = 0, cf_blocks = 0, cf_visited = 0, cf_values = 0;
	static uint64_t cf_cacheable = 0, cf_cache_hits = 0, cf_conditions = 0;
	cf_walks++;
	// Diagnostic (uncommitted): wall time per control-flow walk, printed as us/walk in [srtcf].
	struct CfTimer {
		std::chrono::steady_clock::time_point start;
		double&                               total;
		~CfTimer() {
			total += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start)
			             .count();
		}
	};
	static double cf_us = 0.0;
	CfTimer       cf_timer {std::chrono::steady_clock::now(), cf_us};
	cf_blocks += m_program.control_flow.size();
	cf_values += m_program.evaluation_value_count;
	auto& visited = m_program.visited_blocks;
	auto& reachable = m_program.reachable_blocks;
	auto& pending = m_program.pending_blocks;

	// Perf (uncommitted, SRT reachability memo): replay the previous pass's
	// inputs; if every user-data dword and raw read is unchanged, reuse its
	// visited-set and skip the fixpoint. Soundness: the recorded list covers
	// EVERY input the evaluation consumed, in order (later addresses may be
	// derived from earlier recorded values), declines included. Toggle at
	// runtime by writing "0"/"1" into _Build\windows\srt_memo.flag (checked
	// every 4096 walks); KYTY_SRT_MEMO sets the initial value.
	static uint64_t memo_hits = 0, memo_misses = 0, memo_reads = 0, memo_declines = 0;
	static bool memo_enabled = [] {
		const char* value = std::getenv("KYTY_SRT_MEMO");
		return value != nullptr && value[0] == '1';
	}();
	bool memo_hit = false;
	{
		auto& memo = m_program.srt_reachability_memo;
		if ((cf_walks & 0xfffu) == 0u) {
			if (auto* flag = std::fopen("srt_memo.flag", "rb")) {
				char buffer[2] {0, 0};
				const auto read = std::fread(buffer, 1, 1, flag);
				std::fclose(flag);
				if (read == 1u) memo_enabled = buffer[0] != '0';
			}
		}
		if (memo_enabled && memo.valid &&
		    memo.visited.size() == m_program.control_flow.size() &&
		    memo.shader_base == m_runtime.shader_base) {
			auto&  predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
			bool   ok        = true;
			uint32_t abort_index = 0;
			for (const auto& read: memo.reads) {
				abort_index++;
				switch (read.kind) {
					case 0u:
						if (read.index >= m_runtime.user_data.size() ||
						    m_runtime.user_data[read.index] != read.value) {
							ok = false;
						}
						break;
					case 1u: case 3u: {
						uint32_t word = 0;
						if (predicate.m_runtime.read_memory == nullptr) {
							ok = false;
						} else {
							const bool succeeded =
							    predicate.m_runtime.read_memory(predicate.m_runtime.userdata,
							                                    read.address, {&word, 1});
							if (succeeded != (read.kind == 1u) ||
							    (succeeded && word != read.value)) {
								ok = false;
							}
						}
						break;
					}
					case 2u: case 4u: {
						if (predicate.m_runtime.read_specialization_memory == nullptr) {
							ok = read.kind == 4u;
						} else {
							uint32_t word = 0;
							const bool succeeded =
							    predicate.m_runtime.read_specialization_memory(
							        predicate.m_runtime.userdata, read.address, {&word, 1});
							if (succeeded != (read.kind == 2u) ||
							    (succeeded && word != read.value)) {
								ok = false;
							}
						}
						break;
					}
					case 5u: {
						uint32_t word = 0;
						std::memcpy(&word, reinterpret_cast<const void*>(read.address), sizeof(word));
						if (word != read.value) {
							ok = false;
						}
						break;
					}
					case 6u:
						if (read.index < m_runtime.user_data.size()) {
							ok = false;
						}
						break;
					default: ok = false; break;
				}
				if (!ok) break;
			}
			if (ok) {
				visited.assign(memo.visited.begin(), memo.visited.end());
				memo_hit = true;
				memo_hits++;
				if ((memo_hits & 0xffffu) == 0u) {
					std::fprintf(stderr, "[srtmemo] hit streak\n");
					std::fflush(stderr);
				}
			} else {
				memo_misses++;
				if ((memo_misses & 0xffffu) == 0u) {
					std::fprintf(stderr,
					             "[srtmemo] miss at read %u of %llu (hits %llu, misses %llu)\n",
					             abort_index,
					             static_cast<unsigned long long>(memo.reads.size()),
					             static_cast<unsigned long long>(memo_hits),
					             static_cast<unsigned long long>(memo_misses));
					std::fflush(stderr);
				}
			}
		} else if (memo_enabled) {
			memo_misses++;
		}
	}

	if (!memo_hit) {
	visited.assign(m_program.control_flow.size(), 1u);
	ResetUnknowns();
	m_reachable_blocks = &visited;
	if (m_clean_evaluator != nullptr) {
		m_clean_evaluator->ResetUnknowns();
		m_clean_evaluator->m_reachable_blocks = &visited;
	}
	thread_local std::vector<ResourcePlan::SrtMemoRead> memo_scratch;
	memo_scratch.clear();
	g_srt_memo_recorder.reads  = &memo_scratch;
	g_srt_memo_recorder.active = true;
	for (;;) {
		cf_rounds++;
		reachable.assign(m_program.control_flow.size(), 0u);
		pending.clear();
		pending.push_back(0u);
		while (!pending.empty()) {
			const auto index = pending.back();
			pending.pop_back();
			if (reachable.at(index)) continue;
			reachable[index] = 1u;
			const auto& block = m_program.control_flow[index];
			uint32_t condition = 0;
			auto& predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
			bool condition_known = false;
			if (!block.condition.IsEmpty() &&
			    (m_runtime.read_specialization_memory != nullptr || !m_program.capture_specialization_reads)) {
				// Perf (uncommitted, condition leaf cache): classify once, then
				// reuse the result while the condition's user-data/shader-base
				// leaves are unchanged (skips the whole IR evaluation).
				auto& cache = m_program.srt_cf_conditions;
				if (cache.size() < m_program.control_flow.size()) {
					cache.resize(m_program.control_flow.size());
				}
				auto& entry = cache[index];
				if (entry.classified == 0u) {
					entry.classified = CollectConditionLeaves(m_program, block.condition, entry.leaves)
					                      ? 1u
					                      : 2u;
					if (entry.classified == 1u) {
						entry.last_values.resize(entry.leaves.size());
						cf_cacheable++;
					}
					cf_conditions++;
				}
				bool ready = entry.classified == 1u && entry.has_last;
				if (ready) {
					for (size_t i = 0; i < entry.leaves.size(); ++i) {
						const auto& leaf = entry.leaves[i];
						if (leaf.kind == 2u) {
							// Flat leaves are only comparable in the direct walk
							// (the fixpoint has not fetched them yet).
							ready = false;
							break;
						}
						if (leaf.kind == 0u && leaf.index >= m_runtime.user_data.size()) {
							// The span shrank below a leaf: the read that produced the
							// cached result would now fail structurally. Not unchanged.
							ready = false;
							break;
						}
						const auto now = leaf.kind == 0u ? m_runtime.user_data[leaf.index]
						                                 : m_runtime.shader_base;
						if (now != entry.last_values[i]) {
							ready = false;
							break;
						}
					}
				}
				if (ready) {
					condition       = entry.last_result;
					condition_known = true;
					cf_cache_hits++;
				} else {
					g_srt_read_role = 1;
					condition_known = predicate.Evaluate(block.condition, condition);
					g_srt_read_role = 0;
					if (entry.classified == 1u && condition_known) {
						for (size_t i = 0; i < entry.leaves.size(); ++i) {
							const auto& leaf = entry.leaves[i];
							entry.last_values[i] = leaf.kind == 0u
							                          ? (leaf.index < m_runtime.user_data.size()
							                                 ? m_runtime.user_data[leaf.index]
							                                 : 0u)
							                          : m_runtime.shader_base;
						}
						entry.last_result = condition;
						entry.has_last    = true;
					}
				}
			}
			if (condition_known) {
				pending.push_back(block.successors[condition != 0u ? 0u : 1u]);
			} else {
				pending.insert(pending.end(), block.successors.begin(), block.successors.end());
			}
		}
		if (reachable == visited) break;
		visited.swap(reachable);
		if (m_program.control_phi_predecessors.empty()) break;
		// Only unknown values need another evaluation: successful Phi facts remain
		// true as their possible predecessors shrink. Keep uniform-read memos stable.
		ResetUnknowns();
		if (m_clean_evaluator != nullptr) m_clean_evaluator->ResetUnknowns();
	}
	g_srt_memo_recorder.active = false;
	{
		auto& memo = m_program.srt_reachability_memo;
		if (memo_enabled && memo_scratch.size() <= 4096u) {
			memo.reads.assign(memo_scratch.begin(), memo_scratch.end());
			memo.visited.assign(visited.begin(), visited.end());
			memo.shader_base = m_runtime.shader_base;
			memo.valid       = true;
			memo_reads += memo.reads.size();
			for (const auto& read: memo.reads) {
				if (read.kind == 3u || read.kind == 4u) memo_declines++;
			}
		}
	}
	} // !memo_hit
	// Experiment (uncommitted, KYTY_SRT_BRANCH_BOTH): a condition read that was declined left its
	// values memoized as unknown; later strict evaluations must read them again.
	static const bool branch_both = [] {
		const char* value = std::getenv("KYTY_SRT_BRANCH_BOTH");
		return value != nullptr && value[0] == '1';
	}();
	if (branch_both) {
		ResetUnknowns();
		if (m_clean_evaluator != nullptr) m_clean_evaluator->ResetUnknowns();
	}
	cf_zone.reset();
	for (const auto v: visited) cf_visited += v;
	if (cf_walks % (1u << 18u) == 0) {
		std::fprintf(stderr,
		             "[srtcf] %llu control-flow walks: avg %.2f reachability rounds, %.1f blocks "
		             "(%.1f visited), %.0f evaluation values, %.1f contexts | memo: %llu hits, "
		             "%llu misses, %.1f reads/walk, %llu declined | cond-cache: %llu of %llu "
		             "classified cacheable, %llu hits | %.2f us/walk\n",
		             static_cast<unsigned long long>(cf_walks),
		             static_cast<double>(cf_rounds) / static_cast<double>(cf_walks),
		             static_cast<double>(cf_blocks) / static_cast<double>(cf_walks),
		             static_cast<double>(cf_visited) / static_cast<double>(cf_walks),
		             static_cast<double>(cf_values) / static_cast<double>(cf_walks),
		             static_cast<double>(m_program.evaluation_contexts.size()),
		             static_cast<unsigned long long>(memo_hits),
		             static_cast<unsigned long long>(memo_misses),
		             memo_hits + memo_misses != 0
		                 ? static_cast<double>(memo_reads) /
		                       static_cast<double>(memo_hits + memo_misses)
		                 : 0.0,
		             static_cast<unsigned long long>(memo_declines),
		             static_cast<unsigned long long>(cf_cacheable),
		             static_cast<unsigned long long>(cf_conditions),
		             static_cast<unsigned long long>(cf_cache_hits),
		             cf_us / static_cast<double>(cf_walks));
		std::fflush(stderr);
	}
	KYTY_PROFILER_BLOCK("Srt:VisitedReads");
	flat.assign(m_program.srt_reads.size(), 0u);
	active.assign(m_program.descriptor_sources.size(), 1u);
	for (const auto& block: m_program.control_flow)
		for (const auto source: block.sources) active.at(source) = 0u;
	for (uint32_t index = 0; index < visited.size(); ++index) {
		if (!visited[index]) continue;
		const auto& block = m_program.control_flow[index];
		for (const auto source: block.sources) active[source] = 1u;
		for (const auto slot: block.srt_reads)
			if (!refresh(slot)) return false;
	}
	// Dense evaluator verification (§8.26): compare the replayed dense walk's
	// outputs with this (old) walk's results. The old result is authoritative.
	if (dense_ran && dense_output.ok) {
		static uint64_t v_walks = 0, v_flat_diff = 0, v_visited_diff = 0, v_active_diff = 0,
		                v_size_diff = 0;
		v_walks++;
		const auto compare_u32 = [](const std::vector<uint32_t>& a, const std::vector<uint32_t>& b,
		                            uint64_t& diffs) {
			if (a.size() != b.size()) {
				diffs += a.size() < b.size() ? b.size() - a.size() : a.size() - b.size();
				return;
			}
			for (size_t i = 0; i < a.size(); ++i) {
				if (a[i] != b[i]) diffs++;
			}
		};
		const auto compare_u8 = [](const std::vector<uint8_t>& a, const std::vector<uint8_t>& b,
		                           uint64_t& diffs) {
			if (a.size() != b.size()) {
				diffs += a.size() < b.size() ? b.size() - a.size() : a.size() - b.size();
				return;
			}
			for (size_t i = 0; i < a.size(); ++i) {
				if (a[i] != b[i]) diffs++;
			}
		};
		compare_u32(flat, dense_output.flat, v_flat_diff);
		compare_u8(m_program.visited_blocks, dense_output.visited, v_visited_diff);
		compare_u8(m_program.active_sources, dense_output.active, v_active_diff);
		if (flat.size() != dense_output.flat.size() ||
		    m_program.visited_blocks.size() != dense_output.visited.size() ||
		    m_program.active_sources.size() != dense_output.active.size()) {
			v_size_diff++;
		}
		if ((v_walks & 0xfffu) == 0u || v_flat_diff != 0u || v_visited_diff != 0u ||
		    v_active_diff != 0u) {
			if (v_walks <= 4u || v_flat_diff != 0u || v_visited_diff != 0u ||
			    v_active_diff != 0u) {
				std::fprintf(stderr,
				             "[denseverify] %llu walks: flat diffs %llu, visited diffs %llu, "
				             "active diffs %llu, size diffs %llu\n",
				             static_cast<unsigned long long>(v_walks),
				             static_cast<unsigned long long>(v_flat_diff),
				             static_cast<unsigned long long>(v_visited_diff),
				             static_cast<unsigned long long>(v_active_diff),
				             static_cast<unsigned long long>(v_size_diff));
				std::fflush(stderr);
			}
		}
	}
	return true;
}

bool ValidateRuntimeValue(const ResourcePlan& program, Value value, RuntimeValueType type) {
	return RuntimeValidator(program, type).Run(value);
}

bool ValidateWaveUniformValue(const ResourcePlan& program, Value value) {
	return RuntimeValidator(program, RuntimeValueType::Any, true).Run(value);
}


} // namespace Libs::Graphics::ShaderRecompiler::IR
