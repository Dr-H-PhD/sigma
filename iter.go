package sigma

import "iter"

// All returns an iterator over all elements in row-major order.
// Yields (flat_index, value) pairs.
func (a *NDArray[T]) All() iter.Seq2[int, T] {
	return func(yield func(int, T) bool) {
		size := totalSize(a.shape)
		if size == 0 {
			return
		}

		// Fast path: contiguous with zero offset
		if isContiguous(a.shape, a.strides) && a.offset == 0 {
			for i, v := range a.data[:size] {
				if !yield(i, v) {
					return
				}
			}
			return
		}

		// General path: stride-based iteration
		indices := make([]int, len(a.shape))
		for i := 0; i < size; i++ {
			off := a.offset
			for d := 0; d < len(a.shape); d++ {
				off += indices[d] * a.strides[d]
			}
			if !yield(i, a.data[off]) {
				return
			}
			// Increment indices
			for d := len(a.shape) - 1; d >= 0; d-- {
				indices[d]++
				if indices[d] < a.shape[d] {
					break
				}
				indices[d] = 0
			}
		}
	}
}

// NDIter returns an iterator yielding (N-D index, value) pairs.
func (a *NDArray[T]) NDIter() iter.Seq2[[]int, T] {
	return func(yield func([]int, T) bool) {
		size := totalSize(a.shape)
		if size == 0 {
			return
		}
		indices := make([]int, len(a.shape))
		for i := 0; i < size; i++ {
			off := a.offset
			for d := 0; d < len(a.shape); d++ {
				off += indices[d] * a.strides[d]
			}
			// Return a copy of indices so caller can store it
			idxCopy := make([]int, len(indices))
			copy(idxCopy, indices)
			if !yield(idxCopy, a.data[off]) {
				return
			}
			for d := len(a.shape) - 1; d >= 0; d-- {
				indices[d]++
				if indices[d] < a.shape[d] {
					break
				}
				indices[d] = 0
			}
		}
	}
}
