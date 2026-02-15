package sigma

import "fmt"

// Reshape returns a new view with the given shape if contiguous,
// or a copy with the new shape otherwise. One dimension may be -1
// to infer its size.
func (a *NDArray[T]) Reshape(shape ...int) (*NDArray[T], error) {
	newShape := make([]int, len(shape))
	copy(newShape, shape)

	// Handle -1 inference
	unknownIdx := -1
	known := 1
	for i, s := range newShape {
		if s == -1 {
			if unknownIdx != -1 {
				return nil, &ShapeError{Msg: "can only specify one unknown dimension (-1)"}
			}
			unknownIdx = i
		} else if s < 0 {
			return nil, &ShapeError{Msg: fmt.Sprintf("negative dimension: %d", s)}
		} else {
			known *= s
		}
	}
	size := totalSize(a.shape)
	if unknownIdx != -1 {
		if known == 0 {
			return nil, &ShapeError{Msg: "cannot infer dimension with zero-size shape"}
		}
		if size%known != 0 {
			return nil, &ShapeError{
				Msg: fmt.Sprintf("cannot reshape array of size %d into shape %v", size, shape),
			}
		}
		newShape[unknownIdx] = size / known
	}

	if totalSize(newShape) != size {
		return nil, &ShapeError{
			Msg: fmt.Sprintf("cannot reshape array of size %d into shape %v", size, newShape),
		}
	}

	// If contiguous, return a view (O(1))
	if isContiguous(a.shape, a.strides) {
		return &NDArray[T]{
			data:    a.data,
			shape:   newShape,
			strides: rowMajorStrides(newShape),
			offset:  a.offset,
			base:    a.viewBase(),
		}, nil
	}

	// Non-contiguous: copy first
	c := a.Contiguous()
	return &NDArray[T]{
		data:    c.data,
		shape:   newShape,
		strides: rowMajorStrides(newShape),
	}, nil
}

// Transpose returns a view with reversed axes (O(1) — no data copy).
// For 2-D arrays this is the matrix transpose.
func (a *NDArray[T]) Transpose(axes ...int) (*NDArray[T], error) {
	ndim := len(a.shape)
	if len(axes) == 0 {
		// Default: reverse all axes
		axes = make([]int, ndim)
		for i := range axes {
			axes[i] = ndim - 1 - i
		}
	}
	if len(axes) != ndim {
		return nil, &ShapeError{
			Msg: fmt.Sprintf("axes length %d doesn't match ndim %d", len(axes), ndim),
		}
	}
	// Validate axes
	seen := make([]bool, ndim)
	for _, ax := range axes {
		if ax < 0 || ax >= ndim {
			return nil, &AxisError{Axis: ax, Ndim: ndim}
		}
		if seen[ax] {
			return nil, &ShapeError{Msg: fmt.Sprintf("repeated axis %d in transpose", ax)}
		}
		seen[ax] = true
	}

	newShape := make([]int, ndim)
	newStrides := make([]int, ndim)
	for i, ax := range axes {
		newShape[i] = a.shape[ax]
		newStrides[i] = a.strides[ax]
	}

	return &NDArray[T]{
		data:    a.data,
		shape:   newShape,
		strides: newStrides,
		offset:  a.offset,
		base:    a.viewBase(),
	}, nil
}

// T is shorthand for Transpose with no arguments (reverse all axes).
func (a *NDArray[T]) T() (*NDArray[T], error) {
	return a.Transpose()
}

// Flatten returns a contiguous 1-D copy of the array.
func (a *NDArray[T]) Flatten() *NDArray[T] {
	c := a.Contiguous()
	size := totalSize(c.shape)
	return &NDArray[T]{
		data:    c.data,
		shape:   []int{size},
		strides: []int{1},
	}
}

// Squeeze removes all dimensions of size 1.
func (a *NDArray[T]) Squeeze() *NDArray[T] {
	var newShape []int
	var newStrides []int
	for i, s := range a.shape {
		if s != 1 {
			newShape = append(newShape, s)
			newStrides = append(newStrides, a.strides[i])
		}
	}
	if len(newShape) == 0 {
		// Scalar case
		newShape = []int{1}
		newStrides = []int{1}
	}
	return &NDArray[T]{
		data:    a.data,
		shape:   newShape,
		strides: newStrides,
		offset:  a.offset,
		base:    a.viewBase(),
	}
}

// Unsqueeze inserts a dimension of size 1 at the given axis.
func (a *NDArray[T]) Unsqueeze(axis int) (*NDArray[T], error) {
	ndim := len(a.shape) + 1
	if axis < 0 {
		axis = ndim + axis
	}
	if axis < 0 || axis >= ndim {
		return nil, &AxisError{Axis: axis, Ndim: ndim}
	}
	newShape := make([]int, ndim)
	newStrides := make([]int, ndim)

	copy(newShape, a.shape[:axis])
	newShape[axis] = 1
	copy(newShape[axis+1:], a.shape[axis:])

	copy(newStrides, a.strides[:axis])
	if axis < len(a.strides) {
		newStrides[axis] = a.strides[axis] // stride doesn't matter for size-1 dim
	} else {
		newStrides[axis] = 1
	}
	copy(newStrides[axis+1:], a.strides[axis:])

	return &NDArray[T]{
		data:    a.data,
		shape:   newShape,
		strides: newStrides,
		offset:  a.offset,
		base:    a.viewBase(),
	}, nil
}
