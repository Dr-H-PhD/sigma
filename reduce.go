package sigma

import "math"

// normalizeAxis resolves a possibly negative axis index for an ndim array.
func normalizeAxis(axis, ndim int) (int, error) {
	orig := axis
	if axis < 0 {
		axis += ndim
	}
	if axis < 0 || axis >= ndim {
		return 0, &AxisError{Axis: orig, Ndim: ndim}
	}
	return axis, nil
}

// reduceAlongAxis applies a reduction function along the specified axis.
// init is the initial value; fn(acc, val) returns the new accumulator.
func reduceAlongAxis[T Number](a *NDArray[T], axis int, init T, fn func(T, T) T) (*NDArray[T], error) {
	axis, err := normalizeAxis(axis, len(a.shape))
	if err != nil {
		return nil, err
	}

	// Build output shape (remove the reduced axis)
	outShape := make([]int, 0, len(a.shape)-1)
	for i, s := range a.shape {
		if i != axis {
			outShape = append(outShape, s)
		}
	}
	if len(outShape) == 0 {
		outShape = []int{1}
	}

	outSize := totalSize(outShape)
	outData := make([]T, outSize)
	for i := range outData {
		outData[i] = init
	}
	outStrides := rowMajorStrides(outShape)

	// Iterate over all elements
	size := totalSize(a.shape)
	indices := make([]int, len(a.shape))
	for i := 0; i < size; i++ {
		off := a.offset
		for d := 0; d < len(a.shape); d++ {
			off += indices[d] * a.strides[d]
		}

		// Compute output index (skip the reduced axis)
		outIdx := 0
		oi := 0
		for d := 0; d < len(a.shape); d++ {
			if d != axis {
				outIdx += indices[d] * outStrides[oi]
				oi++
			}
		}
		outData[outIdx] = fn(outData[outIdx], a.data[off])

		// Increment indices
		for d := len(a.shape) - 1; d >= 0; d-- {
			indices[d]++
			if indices[d] < a.shape[d] {
				break
			}
			indices[d] = 0
		}
	}

	return &NDArray[T]{data: outData, shape: outShape, strides: outStrides}, nil
}

// reduceAll applies a reduction across all elements.
func reduceAll[T Number](a *NDArray[T], init T, fn func(T, T) T) T {
	result := init
	for _, v := range a.All() {
		result = fn(result, v)
	}
	return result
}

// Sum reduces along the given axis. If axis is nil, sums all elements.
func Sum[T Number](a *NDArray[T], axis int) (*NDArray[T], error) {
	return reduceAlongAxis(a, axis, 0, func(acc, v T) T { return acc + v })
}

// SumAll sums all elements and returns a scalar.
func SumAll[T Number](a *NDArray[T]) T {
	return reduceAll(a, 0, func(acc, v T) T { return acc + v })
}

// Mean computes the mean along the given axis.
func Mean[T Float](a *NDArray[T], axis int) (*NDArray[T], error) {
	s, err := Sum(a, axis)
	if err != nil {
		return nil, err
	}
	ax, _ := normalizeAxis(axis, len(a.shape))
	n := T(a.shape[ax])
	return applyUnaryOp(s, func(v T) T { return v / n }), nil
}

// MeanAll computes the mean of all elements.
func MeanAll[T Float](a *NDArray[T]) T {
	s := SumAll(a)
	return s / T(a.Size())
}

// Var computes the variance along the given axis (population variance, ddof=0).
func Var[T Float](a *NDArray[T], axis int) (*NDArray[T], error) {
	ax, err := normalizeAxis(axis, len(a.shape))
	if err != nil {
		return nil, err
	}
	mean, err := Mean(a, ax)
	if err != nil {
		return nil, err
	}
	n := T(a.shape[ax])

	// Build output shape
	outShape := mean.Shape()
	outSize := totalSize(outShape)
	outData := make([]T, outSize)
	outStrides := rowMajorStrides(outShape)

	// Accumulate squared deviations
	size := totalSize(a.shape)
	indices := make([]int, len(a.shape))
	for i := 0; i < size; i++ {
		off := a.offset
		for d := 0; d < len(a.shape); d++ {
			off += indices[d] * a.strides[d]
		}

		// Compute mean index
		outIdx := 0
		oi := 0
		for d := 0; d < len(a.shape); d++ {
			if d != ax {
				outIdx += indices[d] * outStrides[oi]
				oi++
			}
		}
		diff := a.data[off] - mean.data[outIdx]
		outData[outIdx] += diff * diff

		for d := len(a.shape) - 1; d >= 0; d-- {
			indices[d]++
			if indices[d] < a.shape[d] {
				break
			}
			indices[d] = 0
		}
	}

	// Divide by n
	for i := range outData {
		outData[i] /= n
	}

	return &NDArray[T]{data: outData, shape: outShape, strides: outStrides}, nil
}

// VarAll computes the variance of all elements.
func VarAll[T Float](a *NDArray[T]) T {
	mean := MeanAll(a)
	var sum T
	for _, v := range a.All() {
		d := v - mean
		sum += d * d
	}
	return sum / T(a.Size())
}

// Std computes the standard deviation along the given axis.
func Std[T Float](a *NDArray[T], axis int) (*NDArray[T], error) {
	v, err := Var(a, axis)
	if err != nil {
		return nil, err
	}
	return Sqrt(v), nil
}

// StdAll computes the standard deviation of all elements.
func StdAll[T Float](a *NDArray[T]) T {
	return T(math.Sqrt(float64(VarAll(a))))
}

// Min reduces along the given axis, returning the minimum.
func Min[T Number](a *NDArray[T], axis int) (*NDArray[T], error) {
	ax, err := normalizeAxis(axis, len(a.shape))
	if err != nil {
		return nil, err
	}
	if a.shape[ax] == 0 {
		return nil, &ShapeError{Msg: "cannot reduce empty axis"}
	}

	// Initialise with first slice along the axis
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
	outData := make([]T, outSize)
	outStrides := rowMajorStrides(outShape)
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
		if !initialised[outIdx] || val < outData[outIdx] {
			outData[outIdx] = val
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
	return &NDArray[T]{data: outData, shape: outShape, strides: outStrides}, nil
}

// MinAll returns the minimum of all elements.
func MinAll[T Number](a *NDArray[T]) T {
	var min T
	first := true
	for _, v := range a.All() {
		if first || v < min {
			min = v
			first = false
		}
	}
	return min
}

// Max reduces along the given axis, returning the maximum.
func Max[T Number](a *NDArray[T], axis int) (*NDArray[T], error) {
	ax, err := normalizeAxis(axis, len(a.shape))
	if err != nil {
		return nil, err
	}
	if a.shape[ax] == 0 {
		return nil, &ShapeError{Msg: "cannot reduce empty axis"}
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
	outData := make([]T, outSize)
	outStrides := rowMajorStrides(outShape)
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
		if !initialised[outIdx] || val > outData[outIdx] {
			outData[outIdx] = val
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
	return &NDArray[T]{data: outData, shape: outShape, strides: outStrides}, nil
}

// MaxAll returns the maximum of all elements.
func MaxAll[T Number](a *NDArray[T]) T {
	var max T
	first := true
	for _, v := range a.All() {
		if first || v > max {
			max = v
			first = false
		}
	}
	return max
}
