package sigma

// Copy returns a deep copy of the array with contiguous layout.
func (a *NDArray[T]) Copy() *NDArray[T] {
	size := totalSize(a.shape)
	data := make([]T, size)
	shapeCopy := make([]int, len(a.shape))
	copy(shapeCopy, a.shape)
	strides := rowMajorStrides(shapeCopy)

	// Iterate over all elements
	indices := make([]int, len(a.shape))
	for i := 0; i < size; i++ {
		// Read from source using strides
		srcIdx := a.offset
		for d := 0; d < len(a.shape); d++ {
			srcIdx += indices[d] * a.strides[d]
		}
		data[i] = a.data[srcIdx]

		// Increment indices (row-major order)
		for d := len(a.shape) - 1; d >= 0; d-- {
			indices[d]++
			if indices[d] < a.shape[d] {
				break
			}
			indices[d] = 0
		}
	}

	return &NDArray[T]{
		data:    data,
		shape:   shapeCopy,
		strides: strides,
	}
}

// Contiguous returns the array itself if already contiguous, or a
// contiguous copy otherwise.
func (a *NDArray[T]) Contiguous() *NDArray[T] {
	if isContiguous(a.shape, a.strides) && a.offset == 0 {
		return a
	}
	return a.Copy()
}
