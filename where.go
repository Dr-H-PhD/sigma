package sigma

// Where selects elements from x where condition is non-zero,
// or from y where condition is zero. All three must be broadcastable.
func Where[T Number](cond, x, y *NDArray[T]) (*NDArray[T], error) {
	// First broadcast cond and x
	shape, err := broadcastShape(cond.shape, x.shape)
	if err != nil {
		return nil, err
	}
	// Then broadcast result with y
	shape, err = broadcastShape(shape, y.shape)
	if err != nil {
		return nil, err
	}

	size := totalSize(shape)
	result := make([]T, size)
	strides := rowMajorStrides(shape)

	condStrides := broadcastStrides(cond.shape, shape, cond.strides)
	xStrides := broadcastStrides(x.shape, shape, x.strides)
	yStrides := broadcastStrides(y.shape, shape, y.strides)

	ndim := len(shape)
	indices := make([]int, ndim)
	for i := 0; i < size; i++ {
		cOff := cond.offset
		xOff := x.offset
		yOff := y.offset
		for d := 0; d < ndim; d++ {
			cOff += indices[d] * condStrides[d]
			xOff += indices[d] * xStrides[d]
			yOff += indices[d] * yStrides[d]
		}
		if cond.data[cOff] != 0 {
			result[i] = x.data[xOff]
		} else {
			result[i] = y.data[yOff]
		}

		for d := ndim - 1; d >= 0; d-- {
			indices[d]++
			if indices[d] < shape[d] {
				break
			}
			indices[d] = 0
		}
	}

	return &NDArray[T]{data: result, shape: shape, strides: strides}, nil
}
