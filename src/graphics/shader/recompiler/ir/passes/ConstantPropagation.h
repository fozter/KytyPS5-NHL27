#pragma once

#include "graphics/shader/recompiler/ir/Block.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

// Diagnostic (uncommitted): work counters over all passes of the session.
struct ConstantPropagationStats {
	uint64_t passes = 0, instructions = 0, folds = 0, changed = 0, rounds = 0, flushes = 0,
	         affected_seeds = 0, affected_visited = 0;
};
inline ConstantPropagationStats g_cp_stats;

void ConstantPropagationPass(const BlockList& blocks, uint32_t wave_size = 64);

} // namespace Libs::Graphics::ShaderRecompiler::IR
