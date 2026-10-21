#ifndef EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_POOLALLOCATOR_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_POOLALLOCATOR_H_

#include <cstddef>
#include <new>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Thread-local size-class pool for IR nodes (perf, uncommitted). The recompiler allocates and
// frees millions of small nodes per boot (instruction list nodes, use lists, Phi operands) and
// the process heap was ~24% of NHL 27's first-use compile time. Blocks up to MaxBytes are
// recycled by the thread that frees them and never returned to the OS; larger requests use
// operator new.
class NodePool {
public:
	static constexpr size_t Granule  = 16;
	static constexpr size_t MaxBytes = 512;

	static void* Allocate(size_t bytes) {
		if (bytes == 0 || bytes > MaxBytes) {
			return ::operator new(bytes);
		}
		auto&      state = Local();
		const auto index = (bytes - 1) / Granule;
		if (auto* node = state.free[index]; node != nullptr) {
			state.free[index] = node->next;
			return node;
		}
		return state.Carve((index + 1) * Granule);
	}

	static void Free(void* pointer, size_t bytes) noexcept {
		if (pointer == nullptr) {
			return;
		}
		if (bytes == 0 || bytes > MaxBytes) {
			::operator delete(pointer);
			return;
		}
		auto&      state = Local();
		const auto index = (bytes - 1) / Granule;
		auto*      node  = static_cast<Node*>(pointer);
		node->next       = state.free[index];
		state.free[index] = node;
	}

private:
	struct Node {
		Node* next;
	};
	struct State {
		Node* free[MaxBytes / Granule] = {};
		char* cursor                   = nullptr;
		char* end                      = nullptr;

		void* Carve(size_t size) {
			if (static_cast<size_t>(end - cursor) < size) {
				constexpr size_t ChunkBytes = 256 * 1024;
				// Never freed: blocks of a chunk may be live in any pool.
				cursor = static_cast<char*>(::operator new(ChunkBytes));
				end    = cursor + ChunkBytes;
			}
			void* result = cursor;
			cursor += size;
			return result;
		}
	};
	static State& Local() {
		thread_local State state;
		return state;
	}
};

template <typename T>
struct PoolAllocator {
	static_assert(alignof(T) <= NodePool::Granule);
	using value_type = T;

	PoolAllocator() noexcept = default;
	template <typename U>
	PoolAllocator(const PoolAllocator<U>& /*other*/) noexcept {}

	[[nodiscard]] T* allocate(size_t count) {
		return static_cast<T*>(NodePool::Allocate(count * sizeof(T)));
	}
	void deallocate(T* pointer, size_t count) noexcept { NodePool::Free(pointer, count * sizeof(T)); }

	template <typename U>
	bool operator==(const PoolAllocator<U>& /*other*/) const noexcept {
		return true;
	}
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_POOLALLOCATOR_H_ */
