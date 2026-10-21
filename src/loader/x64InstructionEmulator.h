#ifndef KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_
#define KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_

#include <cstdint>

namespace Loader::X64InstructionEmulator {

[[nodiscard]] bool TryEmulate(void* native_context);

// Emulates one integer load (MOV/MOVZX/MOVSX/MOVSXD from memory into a general-purpose register)
// whose memory operand covers fault_vaddr, reading its bytes through `read` instead of repeating
// the faulting access. False, with the context unchanged, when the instruction is not such a load
// or `read` declines the bytes.
using LoadReader = bool (*)(uint64_t vaddr, void* data, uint64_t size);
[[nodiscard]] bool TryEmulateLoad(void* native_context, uint64_t fault_vaddr, LoadReader read);

} // namespace Loader::X64InstructionEmulator

#endif /* KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_ */
