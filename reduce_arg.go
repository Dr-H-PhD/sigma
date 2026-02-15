package sigma

// ArgMin returns the index of the minimum along the given axis.
func ArgMin[T Number](a *NDArray[T], axis int) (*NDArray[int64], error) {
	ax, err := normalizeAxis(axis, len(a.shape))
	if err != nil {
		return nil, err
	}
	if a.shape[ax] == 0 {
		return nil, &ShapeError{Msg: "cannot argmin empty axis"}
	}

	outShape := make([]int, 0, len(a.shape)-1)
	for i, s := range a.shape {
		if i != ax {
			outShape = append(outShape, s)
		}
	}
	if len(outShape) == 0 {
		outShape = []int{1}
	}
	outSize := totalSize(outShape)
	outData := make([]int64, outSize)
	outStrides := rowMajorStrides(outShape)
	minVals := make([]T, outSize)
	initialised := make([]bool, outSize)

	size := totalSize(a.shape)
	indices := make([]int, len(a.shape))
	for i := 0; i < size; i++ {
		off := a.offset
		for d := 0; d < len(a.shape); d++ {
			off += indices[d] * a.strides[d]
		}
		outIdx := 0
		oi := 0
		for d := 0; d < len(a.shape); d++ {
			if d != ax {
				outIdx += indices[d] * outStrides[oi]
				oi++
			}
		}
		val := a.data[off]
		if !initialised[outIdx] || val < minVals[outIdx] {
			minVals[outIdx] = val
			outData[outIdx] = int64(indices[ax])
			initialised[outIdx] = true
		}
		for d := len(a.shape) - 1; d >= 0; d-- {
			indices[d]++
			if indices[d] < a.shape[d] {
				break
			}
			indices[d] = 0
		}
	}
	return &NDArray[int64]{data: outData, shape: outShape, strides: outStrides}, nil
}

// ArgMinAll returns the flat index of the minimum element.
func ArgMinAll[T Number](a *NDArray[T]) int {
	idx := 0
	var min T
	first := true
	for i, v := range a.All() {
		if first || v < min {
			min = v
			idx = i
			first = false
		}
	}
	return idx
}

// ArgMax returns the index of the maximum along the given axis.
func ArgMax[T Number](a *NDArray[T], axis int) (*NDArray[int64], error) {
	ax, err := normalizeAxis(axis, len(a.shape))
	if err != nil {
		return nil, err
	}
	if a.shape[ax] == 0 {
		return nil, &ShapeError{Msg: "cannot argmax empty axis"}
	}

	outShape := make([]int, 0, len(a.shape)-1)
	for i, s := range a.shape {
		if i != ax {
			outShape = append(outShape, s)
		}
	}
	if len(outShape) == 0 {
		outShape = []int{1}
	}
	outSize := totalSize(outShape)
	outData := make([]int64, outSize)
	outStrides := rowMajorStrides(outShape)
	maxVals := make([]T, outSize)
	initialised := make([]bool, outSize)

	size := totalSize(a.shape)
	indices := make([]int, len(a.shape))
	for i := 0; i < size; i++ {
		off := a.offset
		for d := 0; d < len(a.shape); d++ {
			off += indices[d] * a.strides[d]
		}
		outIdx := 0
		oi := 0
		for d := 0; d < len(a.shape); d++ {
			if d != ax {
				outIdx += indices[d] * outStrides[oi]
				oi++
			}
		}
		val := a.data[off]
		if !initialised[outIdx] || val > maxVals[outIdx] {
			maxVals[outIdx] = val
			outData[outIdx] = int64(indices[ax])
			initialised[outIdx] = true
		}
		for d := len(a.shape) - 1; d >= 0; d-- {
			indices[d]++
			if indices[d] < a.shape[d] {
				break
			}
			indices[d] = 0
		}
	}
	return &NDArray[int64]{data: outData, shape: outShape, strides: outStrides}, nil
}

// ArgMaxAll returns the flat index of the maximum element.
func ArgMaxAll[T Number](a *NDArray[T]) int {
	idx := 0
	var max T
	first := true
	for i, v := range a.All() {
		if first || v > max {
			max = v
			idx = i
			first = false
		}
	}
	return idx
}
