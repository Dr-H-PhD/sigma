package sigma

// Shape returns a copy of the array's dimensions.
func (a *NDArray[T]) Shape() []int {
	s := make([]int, len(a.shape))
	copy(s, a.shape)
	return s
}

// Strides returns a copy of the array's strides.
func (a *NDArray[T]) Strides() []int {
	s := make([]int, len(a.strides))
	copy(s, a.strides)
	return s
}

// Ndim returns the number of dimensions.
func (a *NDArray[T]) Ndim() int {
	return len(a.shape)
}

// Size returns the total number of elements.
func (a *NDArray[T]) Size() int {
	return totalSize(a.shape)
}

// IsContiguous returns true if the array has a standard row-major layout.
func (a *NDArray[T]) IsContiguous() bool {
	return isContiguous(a.shape, a.strides)
}
