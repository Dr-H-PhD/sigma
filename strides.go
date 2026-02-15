package sigma

// rowMajorStrides computes C-order strides for the given shape.
// Each stride value represents the number of elements to skip.
func rowMajorStrides(shape []int) []int {
	n := len(shape)
	if n == 0 {
		return nil
	}
	strides := make([]int, n)
	strides[n-1] = 1
	for i := n - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * shape[i+1]
	}
	return strides
}

// flatIndex converts an N-dimensional index to a flat offset using strides.
func flatIndex(indices []int, strides []int, offset int) int {
	idx := offset
	for i, v := range indices {
		idx += v * strides[i]
	}
	return idx
}

// isContiguous returns true if the array has standard row-major layout
// with no gaps or overlaps between elements.
func isContiguous(shape, strides []int) bool {
	if len(shape) == 0 {
		return true
	}
	expected := 1
	for i := len(shape) - 1; i >= 0; i-- {
		if shape[i] == 0 {
			return true
		}
		if strides[i] != expected {
			return false
		}
		expected *= shape[i]
	}
	return true
}

// totalSize returns the product of all dimensions in shape.
func totalSize(shape []int) int {
	size := 1
	for _, s := range shape {
		size *= s
	}
	return size
}
