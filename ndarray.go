package sigma

import (
	"fmt"
	"strings"
)

// NDArray is an N-dimensional array with stride-based views.
type NDArray[T Number] struct {
	data    []T
	shape   []int
	strides []int
	offset  int
	base    *NDArray[T] // non-nil if this is a view
}

// viewBase returns a pointer to the root array for view tracking.
// If arr is already a view, returns its base. Otherwise returns arr.
func (a *NDArray[T]) viewBase() *NDArray[T] {
	if a.base != nil {
		return a.base
	}
	return a
}

// IsView returns true if this array shares data with another array.
func (a *NDArray[T]) IsView() bool {
	return a.base != nil
}

// Data returns the raw backing slice. Use with care — modifying it
// affects all views.
func (a *NDArray[T]) Data() []T {
	return a.data
}

// String returns a human-readable representation of the array.
func (a *NDArray[T]) String() string {
	if len(a.shape) == 0 {
		return "NDArray[]"
	}
	size := totalSize(a.shape)
	if size == 0 {
		return fmt.Sprintf("NDArray(%v, [])", a.shape)
	}

	var buf strings.Builder
	a.writeString(&buf, 0, a.offset, 0)
	return buf.String()
}

func (a *NDArray[T]) writeString(buf *strings.Builder, dim int, off int, depth int) {
	if dim == len(a.shape)-1 {
		buf.WriteByte('[')
		for i := 0; i < a.shape[dim]; i++ {
			if i > 0 {
				buf.WriteString(", ")
			}
			fmt.Fprintf(buf, "%v", a.data[off+i*a.strides[dim]])
		}
		buf.WriteByte(']')
		return
	}

	buf.WriteByte('[')
	for i := 0; i < a.shape[dim]; i++ {
		if i > 0 {
			buf.WriteString(",\n")
			for j := 0; j <= depth; j++ {
				buf.WriteByte(' ')
			}
		}
		a.writeString(buf, dim+1, off+i*a.strides[dim], depth+1)
	}
	buf.WriteByte(']')
}
