package sigma

import "fmt"

// Concatenate joins arrays along the given axis.
// All arrays must have the same shape except along the concatenation axis.
func Concatenate[T Number](arrays []*NDArray[T], axis int) (*NDArray[T], error) {
	if len(arrays) == 0 {
		return nil, &ShapeError{Msg: "need at least one array to concatenate"}
	}

	ndim := arrays[0].Ndim()
	ax, err := normalizeAxis(axis, ndim)
	if err != nil {
		return nil, err
	}

	// Validate shapes
	for i := 1; i < len(arrays); i++ {
		if arrays[i].Ndim() != ndim {
			return nil, &ShapeError{
				Msg: fmt.Sprintf("all arrays must have same number of dimensions"),
			}
		}
		for d := 0; d < ndim; d++ {
			if d != ax && arrays[i].shape[d] != arrays[0].shape[d] {
				return nil, &ShapeError{
					Msg: fmt.Sprintf("shape mismatch on axis %d: %v vs %v", d, arrays[0].shape, arrays[i].shape),
				}
			}
		}
	}

	// Compute output shape
	outShape := make([]int, ndim)
	copy(outShape, arrays[0].shape)
	for i := 1; i < len(arrays); i++ {
		outShape[ax] += arrays[i].shape[ax]
	}

	outSize := totalSize(outShape)
	outData := make([]T, outSize)
	outStrides := rowMajorStrides(outShape)

	// Copy data
	axOffset := 0
	for _, arr := range arrays {
		size := totalSize(arr.shape)
		indices := make([]int, ndim)
		for i := 0; i < size; i++ {
			// Read source
			srcOff := arr.offset
			for d := 0; d < ndim; d++ {
				srcOff += indices[d] * arr.strides[d]
			}

			// Write to dest
			dstIdx := 0
			for d := 0; d < ndim; d++ {
				idx := indices[d]
				if d == ax {
					idx += axOffset
				}
				dstIdx += idx * outStrides[d]
			}
			outData[dstIdx] = arr.data[srcOff]

			// Increment
			for d := ndim - 1; d >= 0; d-- {
				indices[d]++
				if indices[d] < arr.shape[d] {
					break
				}
				indices[d] = 0
			}
		}
		axOffset += arr.shape[ax]
	}

	return &NDArray[T]{data: outData, shape: outShape, strides: outStrides}, nil
}

// Stack joins arrays along a new axis.
func Stack[T Number](arrays []*NDArray[T], axis int) (*NDArray[T], error) {
	if len(arrays) == 0 {
		return nil, &ShapeError{Msg: "need at least one array to stack"}
	}

	// All arrays must have same shape
	refShape := arrays[0].Shape()
	for i := 1; i < len(arrays); i++ {
		if !sliceEqualInt(arrays[i].Shape(), refShape) {
			return nil, &ShapeError{
				Msg: fmt.Sprintf("all arrays must have same shape for stacking"),
			}
		}
	}

	ndim := len(refShape) + 1
	if axis < 0 {
		axis += ndim
	}
	if axis < 0 || axis >= ndim {
		return nil, &AxisError{Axis: axis, Ndim: ndim}
	}

	// Unsqueeze each array at the given axis, then concatenate
	expanded := make([]*NDArray[T], len(arrays))
	for i, arr := range arrays {
		e, err := arr.Unsqueeze(axis)
		if err != nil {
			return nil, err
		}
		expanded[i] = e
	}
	return Concatenate(expanded, axis)
}

// Split divides an array into n equal parts along the given axis.
func Split[T Number](a *NDArray[T], n int, axis int) ([]*NDArray[T], error) {
	ax, err := normalizeAxis(axis, len(a.shape))
	if err != nil {
		return nil, err
	}
	axSize := a.shape[ax]
	if axSize%n != 0 {
		return nil, &ShapeError{
			Msg: fmt.Sprintf("array of size %d on axis %d cannot be split into %d equal parts", axSize, ax, n),
		}
	}
	chunkSize := axSize / n
	result := make([]*NDArray[T], n)
	for i := 0; i < n; i++ {
		ranges := make([]SliceRange, len(a.shape))
		for d := 0; d < len(a.shape); d++ {
			if d == ax {
				ranges[d] = S(i*chunkSize, (i+1)*chunkSize)
			} else {
				ranges[d] = All()
			}
		}
		part, err := a.Slice(ranges...)
		if err != nil {
			return nil, err
		}
		result[i] = part.Copy()
	}
	return result, nil
}

// VStack stacks arrays vertically (along axis 0).
func VStack[T Number](arrays []*NDArray[T]) (*NDArray[T], error) {
	if len(arrays) == 0 {
		return nil, &ShapeError{Msg: "need at least one array for VStack"}
	}
	// If 1-D, reshape to (1, n) first
	expanded := make([]*NDArray[T], len(arrays))
	for i, arr := range arrays {
		if arr.Ndim() == 1 {
			r, err := arr.Reshape(1, arr.Size())
			if err != nil {
				return nil, err
			}
			expanded[i] = r
		} else {
			expanded[i] = arr
		}
	}
	return Concatenate(expanded, 0)
}

// HStack stacks arrays horizontally (along axis 1 for 2-D, axis 0 for 1-D).
func HStack[T Number](arrays []*NDArray[T]) (*NDArray[T], error) {
	if len(arrays) == 0 {
		return nil, &ShapeError{Msg: "need at least one array for HStack"}
	}
	if arrays[0].Ndim() == 1 {
		return Concatenate(arrays, 0)
	}
	return Concatenate(arrays, 1)
}
