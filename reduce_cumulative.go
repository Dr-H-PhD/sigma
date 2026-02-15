package sigma

// CumSum computes the cumulative sum along the given axis.
func CumSum[T Number](a *NDArray[T], axis int) (*NDArray[T], error) {
	ax, err := normalizeAxis(axis, len(a.shape))
	if err != nil {
		return nil, err
	}

	result := a.Copy()
	size := totalSize(a.shape)
	indices := make([]int, len(a.shape))

	// For each position, if it's not the first along the axis,
	// add the previous position's value.
	// We iterate in row-major order, which means for axis != last dim
	// we need to carefully accumulate. Instead, iterate over all
	// "rows" along the axis.

	// Build iteration over outer dimensions
	outerShape := make([]int, 0, len(a.shape)-1)
	for i, s := range a.shape {
		if i != ax {
			outerShape = append(outerShape, s)
		}
	}

	if len(outerShape) == 0 {
		// 1-D case
		var acc T
		for i := 0; i < a.shape[ax]; i++ {
			indices[ax] = i
			off := result.offset
			for d := 0; d < len(result.shape); d++ {
				off += indices[d] * result.strides[d]
			}
			acc += result.data[off]
			result.data[off] = acc
		}
		return result, nil
	}

	// General case: iterate over all outer positions
	outerSize := totalSize(outerShape)
	_ = size // suppress unused
	outerIndices := make([]int, len(outerShape))
	for o := 0; o < outerSize; o++ {
		// Map outer indices to full indices
		oi := 0
		for d := 0; d < len(a.shape); d++ {
			if d != ax {
				indices[d] = outerIndices[oi]
				oi++
			}
		}

		// Accumulate along axis
		var acc T
		for k := 0; k < a.shape[ax]; k++ {
			indices[ax] = k
			off := result.offset
			for d := 0; d < len(result.shape); d++ {
				off += indices[d] * result.strides[d]
			}
			acc += result.data[off]
			result.data[off] = acc
		}

		// Increment outer indices
		for d := len(outerIndices) - 1; d >= 0; d-- {
			outerIndices[d]++
			if outerIndices[d] < outerShape[d] {
				break
			}
			outerIndices[d] = 0
		}
	}
	return result, nil
}

// CumProd computes the cumulative product along the given axis.
func CumProd[T Number](a *NDArray[T], axis int) (*NDArray[T], error) {
	ax, err := normalizeAxis(axis, len(a.shape))
	if err != nil {
		return nil, err
	}

	result := a.Copy()
	indices := make([]int, len(a.shape))

	outerShape := make([]int, 0, len(a.shape)-1)
	for i, s := range a.shape {
		if i != ax {
			outerShape = append(outerShape, s)
		}
	}

	if len(outerShape) == 0 {
		var acc T = 1
		for i := 0; i < a.shape[ax]; i++ {
			indices[ax] = i
			off := result.offset
			for d := 0; d < len(result.shape); d++ {
				off += indices[d] * result.strides[d]
			}
			acc *= result.data[off]
			result.data[off] = acc
		}
		return result, nil
	}

	outerSize := totalSize(outerShape)
	outerIndices := make([]int, len(outerShape))
	for o := 0; o < outerSize; o++ {
		oi := 0
		for d := 0; d < len(a.shape); d++ {
			if d != ax {
				indices[d] = outerIndices[oi]
				oi++
			}
		}

		var acc T = 1
		for k := 0; k < a.shape[ax]; k++ {
			indices[ax] = k
			off := result.offset
			for d := 0; d < len(result.shape); d++ {
				off += indices[d] * result.strides[d]
			}
			acc *= result.data[off]
			result.data[off] = acc
		}

		for d := len(outerIndices) - 1; d >= 0; d-- {
			outerIndices[d]++
			if outerIndices[d] < outerShape[d] {
				break
			}
			outerIndices[d] = 0
		}
	}
	return result, nil
}
