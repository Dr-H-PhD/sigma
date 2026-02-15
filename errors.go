package sigma

import "fmt"

// ShapeError indicates an invalid or incompatible shape.
type ShapeError struct {
	Msg string
}

func (e *ShapeError) Error() string {
	return fmt.Sprintf("sigma: shape error: %s", e.Msg)
}

// IndexError indicates an out-of-bounds index.
type IndexError struct {
	Index int
	Axis  int
	Size  int
}

func (e *IndexError) Error() string {
	return fmt.Sprintf("sigma: index %d is out of bounds for axis %d with size %d", e.Index, e.Axis, e.Size)
}

// BroadcastError indicates shapes that cannot be broadcast together.
type BroadcastError struct {
	ShapeA []int
	ShapeB []int
}

func (e *BroadcastError) Error() string {
	return fmt.Sprintf("sigma: cannot broadcast shapes %v and %v", e.ShapeA, e.ShapeB)
}

// AxisError indicates an invalid axis for the given number of dimensions.
type AxisError struct {
	Axis int
	Ndim int
}

func (e *AxisError) Error() string {
	return fmt.Sprintf("sigma: axis %d is out of bounds for array with %d dimensions", e.Axis, e.Ndim)
}

// DimensionError indicates an operation received arrays with wrong dimensionality.
type DimensionError struct {
	Msg string
}

func (e *DimensionError) Error() string {
	return fmt.Sprintf("sigma: dimension error: %s", e.Msg)
}

// SingularMatrixError indicates a matrix is singular and cannot be inverted.
type SingularMatrixError struct{}

func (e *SingularMatrixError) Error() string {
	return "sigma: singular matrix"
}
