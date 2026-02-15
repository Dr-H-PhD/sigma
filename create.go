package sigma

import "fmt"

// New creates an NDArray from a flat slice with the given shape.
// The data is copied into the array.
func New[T Number](data []T, shape ...int) (*NDArray[T], error) {
	size := totalSize(shape)
	if len(data) != size {
		return nil, &ShapeError{
			Msg: fmt.Sprintf("data length %d does not match shape %v (size %d)", len(data), shape, size),
		}
	}
	for _, s := range shape {
		if s < 0 {
			return nil, &ShapeError{Msg: fmt.Sprintf("negative dimension: %d", s)}
		}
	}
	buf := make([]T, size)
	copy(buf, data)
	shapeCopy := make([]int, len(shape))
	copy(shapeCopy, shape)
	return &NDArray[T]{
		data:    buf,
		shape:   shapeCopy,
		strides: rowMajorStrides(shapeCopy),
	}, nil
}

// NewFromSlice creates a 1-D NDArray from a slice.
func NewFromSlice[T Number](data []T) *NDArray[T] {
	buf := make([]T, len(data))
	copy(buf, data)
	shape := []int{len(data)}
	return &NDArray[T]{
		data:    buf,
		shape:   shape,
		strides: rowMajorStrides(shape),
	}
}

// Zeros creates an NDArray filled with zeros.
func Zeros[T Number](shape ...int) *NDArray[T] {
	for _, s := range shape {
		if s < 0 {
			panic(fmt.Sprintf("sigma: negative dimension: %d", s))
		}
	}
	size := totalSize(shape)
	shapeCopy := make([]int, len(shape))
	copy(shapeCopy, shape)
	return &NDArray[T]{
		data:    make([]T, size),
		shape:   shapeCopy,
		strides: rowMajorStrides(shapeCopy),
	}
}

// Ones creates an NDArray filled with ones.
func Ones[T Number](shape ...int) *NDArray[T] {
	a := Zeros[T](shape...)
	for i := range a.data {
		a.data[i] = 1
	}
	return a
}

// Full creates an NDArray filled with the given value.
func Full[T Number](val T, shape ...int) *NDArray[T] {
	a := Zeros[T](shape...)
	for i := range a.data {
		a.data[i] = val
	}
	return a
}

// Empty creates an uninitialized NDArray (in Go, zero-valued).
func Empty[T Number](shape ...int) *NDArray[T] {
	return Zeros[T](shape...)
}

// Arange creates a 1-D NDArray with values in [start, stop) with the given step.
func Arange[T Number](start, stop, step T) (*NDArray[T], error) {
	if step == 0 {
		return nil, &ShapeError{Msg: "step cannot be zero"}
	}
	var n int
	if step > 0 {
		if stop <= start {
			n = 0
		} else {
			n = int((stop - start + step - 1) / step)
		}
	} else {
		if stop >= start {
			n = 0
		} else {
			n = int((start - stop - step - 1) / (-step))
		}
	}
	data := make([]T, n)
	v := start
	for i := 0; i < n; i++ {
		data[i] = v
		v += step
	}
	shape := []int{n}
	return &NDArray[T]{
		data:    data,
		shape:   shape,
		strides: rowMajorStrides(shape),
	}, nil
}

// Linspace creates a 1-D NDArray with n evenly spaced values in [start, stop].
func Linspace[T Float](start, stop T, n int) (*NDArray[T], error) {
	if n < 0 {
		return nil, &ShapeError{Msg: fmt.Sprintf("number of samples must be non-negative, got %d", n)}
	}
	data := make([]T, n)
	if n == 0 {
		return &NDArray[T]{data: data, shape: []int{0}, strides: []int{1}}, nil
	}
	if n == 1 {
		data[0] = start
		return &NDArray[T]{data: data, shape: []int{1}, strides: []int{1}}, nil
	}
	step := (stop - start) / T(n-1)
	for i := 0; i < n; i++ {
		data[i] = start + T(i)*step
	}
	data[n-1] = stop // exact endpoint
	shape := []int{n}
	return &NDArray[T]{data: data, shape: shape, strides: rowMajorStrides(shape)}, nil
}

// Eye creates a 2-D identity matrix of size n×n.
func Eye[T Number](n int) *NDArray[T] {
	a := Zeros[T](n, n)
	for i := 0; i < n; i++ {
		a.data[i*n+i] = 1
	}
	return a
}
