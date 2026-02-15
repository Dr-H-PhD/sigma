package sigma

// broadcastShape computes the result shape from broadcasting two shapes.
// Rules: right-align, each dim must be equal or one must be 1.
func broadcastShape(a, b []int) ([]int, error) {
	na, nb := len(a), len(b)
	ndim := na
	if nb > ndim {
		ndim = nb
	}
	result := make([]int, ndim)
	for i := 0; i < ndim; i++ {
		da := 1
		if i < na {
			da = a[na-1-i]
		}
		db := 1
		if i < nb {
			db = b[nb-1-i]
		}
		if da == db {
			result[ndim-1-i] = da
		} else if da == 1 {
			result[ndim-1-i] = db
		} else if db == 1 {
			result[ndim-1-i] = da
		} else {
			return nil, &BroadcastError{ShapeA: a, ShapeB: b}
		}
	}
	return result, nil
}

// broadcastStrides returns strides for broadcasting array with shape `from`
// into the result shape `to`. Dimensions of size 1 get stride 0.
func broadcastStrides(from, to []int, strides []int) []int {
	nf := len(from)
	nt := len(to)
	result := make([]int, nt)
	for i := 0; i < nt; i++ {
		fi := nf - 1 - i
		ti := nt - 1 - i
		if fi >= 0 && from[fi] == to[ti] {
			result[ti] = strides[fi]
		}
		// else result[ti] stays 0 (broadcast)
	}
	return result
}

// applyBinaryOp applies a binary operation element-wise with broadcasting.
func applyBinaryOp[T Number](a, b *NDArray[T], op func(T, T) T) (*NDArray[T], error) {
	shape, err := broadcastShape(a.shape, b.shape)
	if err != nil {
		return nil, err
	}

	size := totalSize(shape)
	result := make([]T, size)
	resultStrides := rowMajorStrides(shape)

	// Compute broadcast strides for each input
	aStrides := broadcastStrides(a.shape, shape, a.strides)
	bStrides := broadcastStrides(b.shape, shape, b.strides)

	// Fast path: both contiguous, same shape, no broadcasting
	if sliceEqualInt(a.shape, b.shape) && isContiguous(a.shape, a.strides) && isContiguous(b.shape, b.strides) {
		for i := 0; i < size; i++ {
			result[i] = op(a.data[a.offset+i], b.data[b.offset+i])
		}
		return &NDArray[T]{data: result, shape: shape, strides: resultStrides}, nil
	}

	// General path: stride-based iteration
	ndim := len(shape)
	indices := make([]int, ndim)
	for i := 0; i < size; i++ {
		aOff := a.offset
		bOff := b.offset
		for d := 0; d < ndim; d++ {
			aOff += indices[d] * aStrides[d]
			bOff += indices[d] * bStrides[d]
		}
		result[i] = op(a.data[aOff], b.data[bOff])

		// Increment indices
		for d := ndim - 1; d >= 0; d-- {
			indices[d]++
			if indices[d] < shape[d] {
				break
			}
			indices[d] = 0
		}
	}
	return &NDArray[T]{data: result, shape: shape, strides: resultStrides}, nil
}

// applyUnaryOp applies a unary operation element-wise.
func applyUnaryOp[T Number](a *NDArray[T], op func(T) T) *NDArray[T] {
	size := totalSize(a.shape)
	result := make([]T, size)
	shapeCopy := make([]int, len(a.shape))
	copy(shapeCopy, a.shape)
	strides := rowMajorStrides(shapeCopy)

	// Fast path
	if isContiguous(a.shape, a.strides) && a.offset == 0 {
		for i := 0; i < size; i++ {
			result[i] = op(a.data[i])
		}
		return &NDArray[T]{data: result, shape: shapeCopy, strides: strides}
	}

	// General path
	indices := make([]int, len(a.shape))
	for i := 0; i < size; i++ {
		off := a.offset
		for d := 0; d < len(a.shape); d++ {
			off += indices[d] * a.strides[d]
		}
		result[i] = op(a.data[off])
		for d := len(a.shape) - 1; d >= 0; d-- {
			indices[d]++
			if indices[d] < a.shape[d] {
				break
			}
			indices[d] = 0
		}
	}
	return &NDArray[T]{data: result, shape: shapeCopy, strides: strides}
}

// applyScalarOp applies a scalar operation element-wise.
func applyScalarOp[T Number](a *NDArray[T], scalar T, op func(T, T) T) *NDArray[T] {
	return applyUnaryOp(a, func(v T) T { return op(v, scalar) })
}

func sliceEqualInt(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
