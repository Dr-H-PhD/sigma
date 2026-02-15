package sigma

import "fmt"

// SliceRange represents a range for slicing: [Start, Stop) with Step.
// Missing values use sentinel: Start=-MaxInt means from beginning,
// Stop=-MaxInt means to end.
type SliceRange struct {
	Start int
	Stop  int
	Step  int
}

const sliceMissing = -1 << 62 // sentinel for "not specified"

// S creates a SliceRange. Usage:
//
//	S(stop)             → [0, stop)
//	S(start, stop)      → [start, stop)
//	S(start, stop, step) → [start, stop) with step
//
// Use a large negative for "all": S(sliceMissing, sliceMissing) selects everything.
func S(args ...int) SliceRange {
	switch len(args) {
	case 0:
		return SliceRange{Start: sliceMissing, Stop: sliceMissing, Step: 1}
	case 1:
		return SliceRange{Start: 0, Stop: args[0], Step: 1}
	case 2:
		return SliceRange{Start: args[0], Stop: args[1], Step: 1}
	case 3:
		return SliceRange{Start: args[0], Stop: args[1], Step: args[2]}
	default:
		panic("sigma: S() accepts 0-3 arguments")
	}
}

// All returns a SliceRange that selects all elements along an axis.
func All() SliceRange {
	return SliceRange{Start: sliceMissing, Stop: sliceMissing, Step: 1}
}

// normalizeIndex handles negative indexing. Returns the resolved index
// or an error if out of bounds.
func normalizeIndex(idx, size int) (int, error) {
	orig := idx
	if idx < 0 {
		idx += size
	}
	if idx < 0 || idx >= size {
		return 0, &IndexError{Index: orig, Axis: 0, Size: size}
	}
	return idx, nil
}

// Get returns the element at the given N-dimensional index.
// Supports negative indexing.
func (a *NDArray[T]) Get(indices ...int) (T, error) {
	var zero T
	if len(indices) != len(a.shape) {
		return zero, &ShapeError{
			Msg: fmt.Sprintf("expected %d indices, got %d", len(a.shape), len(indices)),
		}
	}
	flat := a.offset
	for i, idx := range indices {
		ni, err := normalizeIndex(idx, a.shape[i])
		if err != nil {
			return zero, &IndexError{Index: idx, Axis: i, Size: a.shape[i]}
		}
		flat += ni * a.strides[i]
	}
	return a.data[flat], nil
}

// Set sets the element at the given N-dimensional index.
// Supports negative indexing.
func (a *NDArray[T]) Set(val T, indices ...int) error {
	if len(indices) != len(a.shape) {
		return &ShapeError{
			Msg: fmt.Sprintf("expected %d indices, got %d", len(a.shape), len(indices)),
		}
	}
	flat := a.offset
	for i, idx := range indices {
		ni, err := normalizeIndex(idx, a.shape[i])
		if err != nil {
			return &IndexError{Index: idx, Axis: i, Size: a.shape[i]}
		}
		flat += ni * a.strides[i]
	}
	a.data[flat] = val
	return nil
}

// Slice returns a view into the array defined by the given slice ranges.
// This is O(1) — no data is copied.
func (a *NDArray[T]) Slice(ranges ...SliceRange) (*NDArray[T], error) {
	if len(ranges) > len(a.shape) {
		return nil, &ShapeError{
			Msg: fmt.Sprintf("too many slice ranges: %d for %d dimensions", len(ranges), len(a.shape)),
		}
	}

	newShape := make([]int, len(a.shape))
	newStrides := make([]int, len(a.shape))
	newOffset := a.offset

	for i := 0; i < len(a.shape); i++ {
		dimSize := a.shape[i]

		var sr SliceRange
		if i < len(ranges) {
			sr = ranges[i]
		} else {
			sr = All()
		}

		step := sr.Step
		if step == 0 {
			return nil, &ShapeError{Msg: "slice step cannot be zero"}
		}

		start := sr.Start
		stop := sr.Stop

		// Resolve missing values
		if step > 0 {
			if start == sliceMissing {
				start = 0
			}
			if stop == sliceMissing {
				stop = dimSize
			}
		} else {
			if start == sliceMissing {
				start = dimSize - 1
			}
			if stop == sliceMissing {
				stop = -(dimSize + 1)
			}
		}

		// Handle negative indices
		if start < 0 {
			start += dimSize
		}
		if stop < 0 {
			stop += dimSize
		}

		// Clamp
		if start < 0 {
			start = 0
		}
		if start > dimSize {
			start = dimSize
		}
		if stop < 0 {
			stop = 0
		}
		if stop > dimSize {
			stop = dimSize
		}

		// Compute number of elements
		var n int
		if step > 0 {
			if stop > start {
				n = (stop - start + step - 1) / step
			}
		} else {
			if start > stop {
				n = (start - stop - step - 1) / (-step)
			}
		}

		newOffset += start * a.strides[i]
		newShape[i] = n
		newStrides[i] = a.strides[i] * step
	}

	return &NDArray[T]{
		data:    a.data,
		shape:   newShape,
		strides: newStrides,
		offset:  newOffset,
		base:    a.viewBase(),
	}, nil
}

// At returns the element at a flat index (for 1-D arrays or contiguous iteration).
func (a *NDArray[T]) At(i int) (T, error) {
	var zero T
	if i < 0 || i >= a.Size() {
		return zero, &IndexError{Index: i, Axis: 0, Size: a.Size()}
	}
	// Convert flat index to N-D index
	indices := make([]int, len(a.shape))
	rem := i
	for d := len(a.shape) - 1; d >= 0; d-- {
		indices[d] = rem % a.shape[d]
		rem /= a.shape[d]
	}
	flat := a.offset
	for d, idx := range indices {
		flat += idx * a.strides[d]
	}
	return a.data[flat], nil
}
