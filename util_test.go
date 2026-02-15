package sigma

import (
	"math"
	"testing"
)

// ── Concatenate ──

func TestConcatenateAxis0(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4}, 2, 2)
	b, _ := New[float64]([]float64{5, 6, 7, 8}, 2, 2)
	c, err := Concatenate([]*NDArray[float64]{a, b}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !sliceEqual(c.Shape(), []int{4, 2}) {
		t.Errorf("shape = %v, want [4,2]", c.Shape())
	}
	v, _ := c.Get(2, 0)
	if v != 5 {
		t.Errorf("c[2,0] = %v, want 5", v)
	}
}

func TestConcatenateAxis1(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4}, 2, 2)
	b, _ := New[float64]([]float64{5, 6, 7, 8}, 2, 2)
	c, err := Concatenate([]*NDArray[float64]{a, b}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !sliceEqual(c.Shape(), []int{2, 4}) {
		t.Errorf("shape = %v, want [2,4]", c.Shape())
	}
	v, _ := c.Get(0, 2)
	if v != 5 {
		t.Errorf("c[0,2] = %v, want 5", v)
	}
}

func TestConcatenateShapeMismatch(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4}, 2, 2)
	b, _ := New[float64]([]float64{5, 6, 7, 8, 9, 10}, 2, 3)
	_, err := Concatenate([]*NDArray[float64]{a, b}, 0)
	if err == nil {
		t.Error("expected error for shape mismatch")
	}
}

// ── Stack ──

func TestStack(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3}, 3)
	b, _ := New[float64]([]float64{4, 5, 6}, 3)
	c, err := Stack([]*NDArray[float64]{a, b}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !sliceEqual(c.Shape(), []int{2, 3}) {
		t.Errorf("shape = %v, want [2,3]", c.Shape())
	}
	v, _ := c.Get(1, 2)
	if v != 6 {
		t.Errorf("c[1,2] = %v, want 6", v)
	}
}

func TestStackAxis1(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3}, 3)
	b, _ := New[float64]([]float64{4, 5, 6}, 3)
	c, err := Stack([]*NDArray[float64]{a, b}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !sliceEqual(c.Shape(), []int{3, 2}) {
		t.Errorf("shape = %v, want [3,2]", c.Shape())
	}
	v, _ := c.Get(2, 1) // 6
	if v != 6 {
		t.Errorf("c[2,1] = %v, want 6", v)
	}
}

// ── Split ──

func TestSplit(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6}, 6)
	parts, err := Split(a, 3, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(parts) != 3 {
		t.Fatalf("got %d parts, want 3", len(parts))
	}
	v, _ := parts[0].Get(0)
	if v != 1 {
		t.Errorf("parts[0][0] = %v, want 1", v)
	}
	v, _ = parts[2].Get(1)
	if v != 6 {
		t.Errorf("parts[2][1] = %v, want 6", v)
	}
}

func TestSplit2D(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 4, 3)
	parts, err := Split(a, 2, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(parts) != 2 {
		t.Fatalf("got %d parts, want 2", len(parts))
	}
	if !sliceEqual(parts[0].Shape(), []int{2, 3}) {
		t.Errorf("parts[0] shape = %v, want [2,3]", parts[0].Shape())
	}
	v, _ := parts[1].Get(0, 0)
	if v != 7 {
		t.Errorf("parts[1][0,0] = %v, want 7", v)
	}
}

func TestSplitUnevenError(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5}, 5)
	_, err := Split(a, 2, 0)
	if err == nil {
		t.Error("expected error for uneven split")
	}
}

// ── VStack / HStack ──

func TestVStack(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3}, 3)
	b, _ := New[float64]([]float64{4, 5, 6}, 3)
	c, err := VStack([]*NDArray[float64]{a, b})
	if err != nil {
		t.Fatal(err)
	}
	if !sliceEqual(c.Shape(), []int{2, 3}) {
		t.Errorf("shape = %v, want [2,3]", c.Shape())
	}
}

func TestVStack2D(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4}, 2, 2)
	b, _ := New[float64]([]float64{5, 6, 7, 8}, 2, 2)
	c, err := VStack([]*NDArray[float64]{a, b})
	if err != nil {
		t.Fatal(err)
	}
	if !sliceEqual(c.Shape(), []int{4, 2}) {
		t.Errorf("shape = %v, want [4,2]", c.Shape())
	}
}

func TestHStack(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3}, 3)
	b, _ := New[float64]([]float64{4, 5, 6}, 3)
	c, err := HStack([]*NDArray[float64]{a, b})
	if err != nil {
		t.Fatal(err)
	}
	if !sliceEqual(c.Shape(), []int{6}) {
		t.Errorf("shape = %v, want [6]", c.Shape())
	}
}

func TestHStack2D(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4}, 2, 2)
	b, _ := New[float64]([]float64{5, 6, 7, 8}, 2, 2)
	c, err := HStack([]*NDArray[float64]{a, b})
	if err != nil {
		t.Fatal(err)
	}
	if !sliceEqual(c.Shape(), []int{2, 4}) {
		t.Errorf("shape = %v, want [2,4]", c.Shape())
	}
}

// ── Where ──

func TestWhere(t *testing.T) {
	cond, _ := New[float64]([]float64{1, 0, 1, 0}, 4)
	x, _ := New[float64]([]float64{10, 20, 30, 40}, 4)
	y, _ := New[float64]([]float64{-1, -2, -3, -4}, 4)
	result, err := Where(cond, x, y)
	if err != nil {
		t.Fatal(err)
	}
	expected := []float64{10, -2, 30, -4}
	for i, want := range expected {
		v, _ := result.Get(i)
		if v != want {
			t.Errorf("result[%d] = %v, want %v", i, v, want)
		}
	}
}

func TestWhereBroadcast(t *testing.T) {
	cond, _ := New[float64]([]float64{1, 0, 1}, 3)
	x := Full[float64](99, 3)
	y := Full[float64](-1, 3)
	result, err := Where(cond, x, y)
	if err != nil {
		t.Fatal(err)
	}
	expected := []float64{99, -1, 99}
	for i, want := range expected {
		v, _ := result.Get(i)
		if v != want {
			t.Errorf("result[%d] = %v, want %v", i, v, want)
		}
	}
}

// ── Sort ──

func TestSort(t *testing.T) {
	a, _ := New[float64]([]float64{3, 1, 4, 1, 5, 9}, 6)
	b, err := Sort(a)
	if err != nil {
		t.Fatal(err)
	}
	expected := []float64{1, 1, 3, 4, 5, 9}
	for i, want := range expected {
		v, _ := b.Get(i)
		if v != want {
			t.Errorf("b[%d] = %v, want %v", i, v, want)
		}
	}
}

func TestSortDoesNotModifyOriginal(t *testing.T) {
	a, _ := New[float64]([]float64{3, 1, 2}, 3)
	Sort(a)
	v, _ := a.Get(0)
	if v != 3 {
		t.Errorf("original modified: a[0] = %v, want 3", v)
	}
}

// ── Argsort ──

func TestArgsort(t *testing.T) {
	a, _ := New[float64]([]float64{30, 10, 20}, 3)
	idx, err := Argsort(a)
	if err != nil {
		t.Fatal(err)
	}
	// Sorted order: 10(1), 20(2), 30(0)
	expected := []int64{1, 2, 0}
	for i, want := range expected {
		v, _ := idx.Get(i)
		if v != want {
			t.Errorf("idx[%d] = %v, want %v", i, v, want)
		}
	}
}

// ── Clip ──

func TestClip(t *testing.T) {
	a, _ := New[float64]([]float64{-5, 0, 5, 10, 15}, 5)
	b := Clip(a, 0, 10)
	expected := []float64{0, 0, 5, 10, 10}
	for i, want := range expected {
		v, _ := b.Get(i)
		if v != want {
			t.Errorf("b[%d] = %v, want %v", i, v, want)
		}
	}
}

// ── Round / Floor / Ceil ──

func TestRound(t *testing.T) {
	a, _ := New[float64]([]float64{1.4, 1.5, 2.6, -1.5}, 4)
	b := Round(a)
	expected := []float64{1, 2, 3, -2}
	for i, want := range expected {
		v, _ := b.Get(i)
		if v != want {
			t.Errorf("Round[%d] = %v, want %v", i, v, want)
		}
	}
}

func TestFloor(t *testing.T) {
	a, _ := New[float64]([]float64{1.7, -1.7, 2.0}, 3)
	b := Floor(a)
	expected := []float64{1, -2, 2}
	for i, want := range expected {
		v, _ := b.Get(i)
		if v != want {
			t.Errorf("Floor[%d] = %v, want %v", i, v, want)
		}
	}
}

func TestCeil(t *testing.T) {
	a, _ := New[float64]([]float64{1.1, -1.1, 2.0}, 3)
	b := Ceil(a)
	expected := []float64{2, -1, 2}
	for i, want := range expected {
		v, _ := b.Get(i)
		if v != want {
			t.Errorf("Ceil[%d] = %v, want %v", i, v, want)
		}
	}
}

// ── Integration: Where with comparison ──

func TestWhereWithComparison(t *testing.T) {
	a, _ := New[float64]([]float64{1, 5, 3, 8, 2}, 5)
	threshold := Full[float64](4, 5)
	cond, _ := Greater(a, threshold)
	ones := Full[float64](1, 5)
	zeros := Zeros[float64](5)
	result, err := Where(cond, ones, zeros)
	if err != nil {
		t.Fatal(err)
	}
	// Elements > 4: 5, 8 → indices 1, 3
	expected := []float64{0, 1, 0, 1, 0}
	for i, want := range expected {
		v, _ := result.Get(i)
		if v != want {
			t.Errorf("result[%d] = %v, want %v", i, v, want)
		}
	}
}

// ── Integration: end-to-end pipeline ──

func TestEndToEndPipeline(t *testing.T) {
	// Create, reshape, transpose, slice, add, sum
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 3, 4)
	b, _ := a.Slice(S(0, 2), S(1, 3)) // [[2,3],[6,7]]
	c, _ := b.T()                      // [[2,6],[3,7]]
	d := AddScalar(c, 10)              // [[12,16],[13,17]]
	total := SumAll(d)
	if total != 58 {
		t.Errorf("pipeline total = %v, want 58", total)
	}
}

func TestEndToEndStats(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 10)
	mean := MeanAll(a)
	if math.Abs(mean-5.5) > 1e-10 {
		t.Errorf("mean = %v, want 5.5", mean)
	}
	s := StdAll(a)
	expected := math.Sqrt(8.25) // population std
	if math.Abs(s-expected) > 1e-10 {
		t.Errorf("std = %v, want %v", s, expected)
	}
}
