package sigma

import (
	"math"
	"testing"
)

// ── strides ──

func TestRowMajorStrides(t *testing.T) {
	tests := []struct {
		shape []int
		want  []int
	}{
		{[]int{3}, []int{1}},
		{[]int{2, 3}, []int{3, 1}},
		{[]int{2, 3, 4}, []int{12, 4, 1}},
		{[]int{5, 1}, []int{1, 1}},
	}
	for _, tc := range tests {
		got := rowMajorStrides(tc.shape)
		if !sliceEqual(got, tc.want) {
			t.Errorf("rowMajorStrides(%v) = %v, want %v", tc.shape, got, tc.want)
		}
	}
}

func TestTotalSize(t *testing.T) {
	tests := []struct {
		shape []int
		want  int
	}{
		{[]int{3}, 3},
		{[]int{2, 3}, 6},
		{[]int{2, 3, 4}, 24},
		{[]int{1}, 1},
	}
	for _, tc := range tests {
		got := totalSize(tc.shape)
		if got != tc.want {
			t.Errorf("totalSize(%v) = %d, want %d", tc.shape, got, tc.want)
		}
	}
}

func TestIsContiguous(t *testing.T) {
	if !isContiguous([]int{2, 3}, []int{3, 1}) {
		t.Error("expected contiguous for shape [2,3] strides [3,1]")
	}
	if isContiguous([]int{2, 3}, []int{1, 2}) {
		t.Error("expected non-contiguous for shape [2,3] strides [1,2]")
	}
}

func TestFlatIndex(t *testing.T) {
	// 2x3 row-major
	idx := flatIndex([]int{1, 2}, []int{3, 1}, 0)
	if idx != 5 {
		t.Errorf("flatIndex([1,2], [3,1], 0) = %d, want 5", idx)
	}
}

// ── create ──

func TestNew(t *testing.T) {
	a, err := New[float64]([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	if !sliceEqual(a.Shape(), []int{2, 3}) {
		t.Errorf("shape = %v, want [2,3]", a.Shape())
	}
	v, _ := a.Get(1, 2)
	if v != 6 {
		t.Errorf("a[1,2] = %v, want 6", v)
	}
}

func TestNewSizeMismatch(t *testing.T) {
	_, err := New[float64]([]float64{1, 2, 3}, 2, 3)
	if err == nil {
		t.Error("expected error for size mismatch")
	}
}

func TestNewFromSlice(t *testing.T) {
	a := NewFromSlice([]int64{10, 20, 30})
	if a.Size() != 3 {
		t.Errorf("size = %d, want 3", a.Size())
	}
	v, _ := a.Get(1)
	if v != 20 {
		t.Errorf("a[1] = %v, want 20", v)
	}
}

func TestZeros(t *testing.T) {
	a := Zeros[float64](3, 4)
	if a.Size() != 12 {
		t.Errorf("size = %d, want 12", a.Size())
	}
	for i, v := range a.All() {
		if v != 0 {
			t.Errorf("element %d = %v, want 0", i, v)
		}
	}
}

func TestOnes(t *testing.T) {
	a := Ones[int32](2, 2)
	for _, v := range a.All() {
		if v != 1 {
			t.Errorf("expected 1, got %v", v)
		}
	}
}

func TestFull(t *testing.T) {
	a := Full[float64](7.5, 3)
	for _, v := range a.All() {
		if v != 7.5 {
			t.Errorf("expected 7.5, got %v", v)
		}
	}
}

func TestArange(t *testing.T) {
	a, err := Arange[int64](0, 5, 1)
	if err != nil {
		t.Fatal(err)
	}
	if a.Size() != 5 {
		t.Errorf("size = %d, want 5", a.Size())
	}
	v, _ := a.Get(3)
	if v != 3 {
		t.Errorf("a[3] = %v, want 3", v)
	}
}

func TestArangeStep2(t *testing.T) {
	a, err := Arange[int64](0, 10, 2)
	if err != nil {
		t.Fatal(err)
	}
	expected := []int64{0, 2, 4, 6, 8}
	if a.Size() != len(expected) {
		t.Fatalf("size = %d, want %d", a.Size(), len(expected))
	}
	for i, want := range expected {
		v, _ := a.Get(i)
		if v != want {
			t.Errorf("a[%d] = %v, want %v", i, v, want)
		}
	}
}

func TestArangeNegativeStep(t *testing.T) {
	a, err := Arange[int64](5, 0, -1)
	if err != nil {
		t.Fatal(err)
	}
	expected := []int64{5, 4, 3, 2, 1}
	if a.Size() != len(expected) {
		t.Fatalf("size = %d, want %d", a.Size(), len(expected))
	}
	for i, want := range expected {
		v, _ := a.Get(i)
		if v != want {
			t.Errorf("a[%d] = %v, want %v", i, v, want)
		}
	}
}

func TestArangeZeroStep(t *testing.T) {
	_, err := Arange[float64](0, 5, 0)
	if err == nil {
		t.Error("expected error for zero step")
	}
}

func TestLinspace(t *testing.T) {
	a, err := Linspace[float64](0, 1, 5)
	if err != nil {
		t.Fatal(err)
	}
	if a.Size() != 5 {
		t.Fatalf("size = %d, want 5", a.Size())
	}
	expected := []float64{0, 0.25, 0.5, 0.75, 1.0}
	for i, want := range expected {
		v, _ := a.Get(i)
		if math.Abs(v-want) > 1e-10 {
			t.Errorf("a[%d] = %v, want %v", i, v, want)
		}
	}
}

func TestLinspaceSingle(t *testing.T) {
	a, err := Linspace[float64](5, 10, 1)
	if err != nil {
		t.Fatal(err)
	}
	v, _ := a.Get(0)
	if v != 5 {
		t.Errorf("single point = %v, want 5", v)
	}
}

func TestEye(t *testing.T) {
	a := Eye[float64](3)
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			v, _ := a.Get(i, j)
			if i == j && v != 1 {
				t.Errorf("a[%d,%d] = %v, want 1", i, j, v)
			}
			if i != j && v != 0 {
				t.Errorf("a[%d,%d] = %v, want 0", i, j, v)
			}
		}
	}
}

// ── shape methods ──

func TestShapeMethods(t *testing.T) {
	a := Zeros[float64](2, 3, 4)
	if a.Ndim() != 3 {
		t.Errorf("ndim = %d, want 3", a.Ndim())
	}
	if a.Size() != 24 {
		t.Errorf("size = %d, want 24", a.Size())
	}
	if !a.IsContiguous() {
		t.Error("expected contiguous")
	}
}

// ── reshape ──

func TestReshape(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b, err := a.Reshape(3, 2)
	if err != nil {
		t.Fatal(err)
	}
	if !sliceEqual(b.Shape(), []int{3, 2}) {
		t.Errorf("shape = %v, want [3,2]", b.Shape())
	}
	// Should be a view
	if !b.IsView() {
		t.Error("expected reshape to return a view")
	}
	v, _ := b.Get(1, 0)
	if v != 3 {
		t.Errorf("b[1,0] = %v, want 3", v)
	}
}

func TestReshapeInfer(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6}, 6)
	b, err := a.Reshape(2, -1)
	if err != nil {
		t.Fatal(err)
	}
	if !sliceEqual(b.Shape(), []int{2, 3}) {
		t.Errorf("shape = %v, want [2,3]", b.Shape())
	}
}

func TestReshapeIncompatible(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6}, 6)
	_, err := a.Reshape(2, 4)
	if err == nil {
		t.Error("expected error for incompatible reshape")
	}
}

func TestFlatten(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := a.Flatten()
	if !sliceEqual(b.Shape(), []int{6}) {
		t.Errorf("shape = %v, want [6]", b.Shape())
	}
}

// ── transpose ──

func TestTranspose2D(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b, err := a.T()
	if err != nil {
		t.Fatal(err)
	}
	if !sliceEqual(b.Shape(), []int{3, 2}) {
		t.Errorf("shape = %v, want [3,2]", b.Shape())
	}
	// b[0,1] should be a[1,0] = 4
	v, _ := b.Get(0, 1)
	if v != 4 {
		t.Errorf("b[0,1] = %v, want 4", v)
	}
	// b[2,0] should be a[0,2] = 3
	v, _ = b.Get(2, 0)
	if v != 3 {
		t.Errorf("b[2,0] = %v, want 3", v)
	}
}

func TestTransposeIsView(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b, _ := a.T()
	// Modify through b
	b.Set(99, 0, 1)
	// Should be visible in a
	v, _ := a.Get(1, 0)
	if v != 99 {
		t.Errorf("view not shared: a[1,0] = %v, want 99", v)
	}
}

func TestTransposeCustomAxes(t *testing.T) {
	a := Zeros[float64](2, 3, 4)
	b, err := a.Transpose(2, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !sliceEqual(b.Shape(), []int{4, 2, 3}) {
		t.Errorf("shape = %v, want [4,2,3]", b.Shape())
	}
}

// ── squeeze / unsqueeze ──

func TestSqueeze(t *testing.T) {
	a := Zeros[float64](1, 3, 1, 4)
	b := a.Squeeze()
	if !sliceEqual(b.Shape(), []int{3, 4}) {
		t.Errorf("shape = %v, want [3,4]", b.Shape())
	}
}

func TestUnsqueeze(t *testing.T) {
	a := Zeros[float64](3, 4)
	b, err := a.Unsqueeze(0)
	if err != nil {
		t.Fatal(err)
	}
	if !sliceEqual(b.Shape(), []int{1, 3, 4}) {
		t.Errorf("shape = %v, want [1,3,4]", b.Shape())
	}
}

func TestUnsqueezeEnd(t *testing.T) {
	a := Zeros[float64](3, 4)
	b, err := a.Unsqueeze(2)
	if err != nil {
		t.Fatal(err)
	}
	if !sliceEqual(b.Shape(), []int{3, 4, 1}) {
		t.Errorf("shape = %v, want [3,4,1]", b.Shape())
	}
}

// ── indexing ──

func TestGetSet(t *testing.T) {
	a := Zeros[float64](3, 4)
	a.Set(42, 1, 2)
	v, err := a.Get(1, 2)
	if err != nil {
		t.Fatal(err)
	}
	if v != 42 {
		t.Errorf("got %v, want 42", v)
	}
}

func TestNegativeIndex(t *testing.T) {
	a, _ := New[float64]([]float64{10, 20, 30}, 3)
	v, err := a.Get(-1)
	if err != nil {
		t.Fatal(err)
	}
	if v != 30 {
		t.Errorf("a[-1] = %v, want 30", v)
	}
}

func TestIndexOutOfBounds(t *testing.T) {
	a := Zeros[float64](3)
	_, err := a.Get(5)
	if err == nil {
		t.Error("expected error for out-of-bounds index")
	}
}

func TestWrongNumberOfIndices(t *testing.T) {
	a := Zeros[float64](3, 4)
	_, err := a.Get(1)
	if err == nil {
		t.Error("expected error for wrong number of indices")
	}
}

// ── slicing ──

func TestSliceBasic(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	// Slice rows [0:2], cols [1:3]
	b, err := a.Slice(S(0, 2), S(1, 3))
	if err != nil {
		t.Fatal(err)
	}
	if !sliceEqual(b.Shape(), []int{2, 2}) {
		t.Errorf("shape = %v, want [2,2]", b.Shape())
	}
	// b[0,0] = a[0,1] = 2
	v, _ := b.Get(0, 0)
	if v != 2 {
		t.Errorf("b[0,0] = %v, want 2", v)
	}
	// b[1,1] = a[1,2] = 6
	v, _ = b.Get(1, 1)
	if v != 6 {
		t.Errorf("b[1,1] = %v, want 6", v)
	}
}

func TestSliceIsView(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b, _ := a.Slice(All(), S(1, 3))
	b.Set(99, 0, 0)
	// Should modify a[0,1]
	v, _ := a.Get(0, 1)
	if v != 99 {
		t.Errorf("slice not a view: a[0,1] = %v, want 99", v)
	}
}

func TestSliceWithStep(t *testing.T) {
	a, _ := New[float64]([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 10)
	b, _ := a.Slice(S(1, 8, 2))
	// Elements: 1, 3, 5, 7
	if b.Size() != 4 {
		t.Fatalf("size = %d, want 4", b.Size())
	}
	expected := []float64{1, 3, 5, 7}
	for i, want := range expected {
		v, _ := b.Get(i)
		if v != want {
			t.Errorf("b[%d] = %v, want %v", i, v, want)
		}
	}
}

func TestSliceNegativeIndex(t *testing.T) {
	a, _ := New[float64]([]float64{0, 1, 2, 3, 4}, 5)
	b, _ := a.Slice(S(1, -1))
	// [1, 2, 3]
	if b.Size() != 3 {
		t.Fatalf("size = %d, want 3", b.Size())
	}
	v, _ := b.Get(2)
	if v != 3 {
		t.Errorf("b[2] = %v, want 3", v)
	}
}

func TestSliceAll(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b, _ := a.Slice(All(), All())
	if b.Size() != 6 {
		t.Errorf("size = %d, want 6", b.Size())
	}
}

// ── copy ──

func TestCopy(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := a.Copy()
	b.Set(99, 0, 0)
	v, _ := a.Get(0, 0)
	if v != 1 {
		t.Errorf("copy modified original: a[0,0] = %v, want 1", v)
	}
}

func TestContiguousOfView(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	// Transpose creates non-contiguous view
	b, _ := a.T()
	c := b.Contiguous()
	if !c.IsContiguous() {
		t.Error("expected contiguous after Contiguous()")
	}
	// Check values preserved
	v, _ := c.Get(0, 1)
	if v != 4 {
		t.Errorf("c[0,1] = %v, want 4", v)
	}
}

// ── iter ──

func TestAll(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4}, 2, 2)
	var vals []float64
	for _, v := range a.All() {
		vals = append(vals, v)
	}
	expected := []float64{1, 2, 3, 4}
	if !floatSliceEqual(vals, expected) {
		t.Errorf("All() = %v, want %v", vals, expected)
	}
}

func TestAllTransposed(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4}, 2, 2)
	b, _ := a.T()
	var vals []float64
	for _, v := range b.All() {
		vals = append(vals, v)
	}
	// Transposed iteration: [1,3,2,4]
	expected := []float64{1, 3, 2, 4}
	if !floatSliceEqual(vals, expected) {
		t.Errorf("All() transposed = %v, want %v", vals, expected)
	}
}

func TestNDIter(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	count := 0
	for idx, v := range a.NDIter() {
		if len(idx) != 2 {
			t.Fatalf("index has %d dims, want 2", len(idx))
		}
		got, _ := a.Get(idx[0], idx[1])
		if got != v {
			t.Errorf("NDIter at %v: got %v, Get returns %v", idx, v, got)
		}
		count++
	}
	if count != 6 {
		t.Errorf("count = %d, want 6", count)
	}
}

// ── At ──

func TestAt(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	v, err := a.At(4)
	if err != nil {
		t.Fatal(err)
	}
	if v != 5 {
		t.Errorf("At(4) = %v, want 5", v)
	}
}

// ── String ──

func TestString1D(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3}, 3)
	s := a.String()
	if s != "[1, 2, 3]" {
		t.Errorf("String() = %q, want %q", s, "[1, 2, 3]")
	}
}

func TestString2D(t *testing.T) {
	a, _ := New[int64]([]int64{1, 2, 3, 4}, 2, 2)
	s := a.String()
	expected := "[[1, 2],\n [3, 4]]"
	if s != expected {
		t.Errorf("String() = %q, want %q", s, expected)
	}
}

// ── helpers ──

func sliceEqual(a, b []int) bool {
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

func floatSliceEqual(a, b []float64) bool {
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
