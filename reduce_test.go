package sigma

import (
	"math"
	"testing"
)

// ── Sum ──

func TestSumAxis1(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	s, err := Sum(a, 1)
	if err != nil {
		t.Fatal(err)
	}
	// Row sums: [6, 15]
	if !sliceEqual(s.Shape(), []int{2}) {
		t.Errorf("shape = %v, want [2]", s.Shape())
	}
	v, _ := s.Get(0)
	if v != 6 {
		t.Errorf("sum[0] = %v, want 6", v)
	}
	v, _ = s.Get(1)
	if v != 15 {
		t.Errorf("sum[1] = %v, want 15", v)
	}
}

func TestSumAxis0(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	s, err := Sum(a, 0)
	if err != nil {
		t.Fatal(err)
	}
	// Column sums: [5, 7, 9]
	if !sliceEqual(s.Shape(), []int{3}) {
		t.Errorf("shape = %v, want [3]", s.Shape())
	}
	expected := []float64{5, 7, 9}
	for i, want := range expected {
		v, _ := s.Get(i)
		if v != want {
			t.Errorf("sum[%d] = %v, want %v", i, v, want)
		}
	}
}

func TestSumAll(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	s := SumAll(a)
	if s != 21 {
		t.Errorf("SumAll = %v, want 21", s)
	}
}

func TestSumNegativeAxis(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	s, err := Sum(a, -1) // axis -1 = axis 1
	if err != nil {
		t.Fatal(err)
	}
	v, _ := s.Get(0)
	if v != 6 {
		t.Errorf("sum[0] = %v, want 6", v)
	}
}

func TestSum3D(t *testing.T) {
	a, _ := New[float64]([]float64{
		1, 2, 3, 4, 5, 6,
		7, 8, 9, 10, 11, 12,
	}, 2, 2, 3)
	// Sum along axis 2 → shape [2,2]
	s, err := Sum(a, 2)
	if err != nil {
		t.Fatal(err)
	}
	if !sliceEqual(s.Shape(), []int{2, 2}) {
		t.Errorf("shape = %v, want [2,2]", s.Shape())
	}
	// [0,0]: 1+2+3=6, [0,1]: 4+5+6=15, [1,0]: 7+8+9=24, [1,1]: 10+11+12=33
	v, _ := s.Get(1, 1)
	if v != 33 {
		t.Errorf("s[1,1] = %v, want 33", v)
	}
}

// ── Mean ──

func TestMean(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	m, err := Mean(a, 1)
	if err != nil {
		t.Fatal(err)
	}
	v, _ := m.Get(0)
	if v != 2 {
		t.Errorf("mean[0] = %v, want 2", v)
	}
	v, _ = m.Get(1)
	if v != 5 {
		t.Errorf("mean[1] = %v, want 5", v)
	}
}

func TestMeanAll(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6}, 6)
	m := MeanAll(a)
	if m != 3.5 {
		t.Errorf("MeanAll = %v, want 3.5", m)
	}
}

// ── Var / Std ──

func TestVar(t *testing.T) {
	a, _ := New[float64]([]float64{2, 4, 4, 4, 5, 5, 7, 9}, 8)
	v := VarAll(a)
	if math.Abs(v-4) > 1e-10 {
		t.Errorf("VarAll = %v, want 4", v)
	}
}

func TestStd(t *testing.T) {
	a, _ := New[float64]([]float64{2, 4, 4, 4, 5, 5, 7, 9}, 8)
	s := StdAll(a)
	if math.Abs(s-2) > 1e-10 {
		t.Errorf("StdAll = %v, want 2", s)
	}
}

func TestVarAxis(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	v, err := Var(a, 1)
	if err != nil {
		t.Fatal(err)
	}
	// Row 0: mean=2, var=(1+0+1)/3 = 2/3
	val, _ := v.Get(0)
	if math.Abs(val-2.0/3.0) > 1e-10 {
		t.Errorf("var[0] = %v, want %v", val, 2.0/3.0)
	}
}

func TestStdAxis(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	s, err := Std(a, 1)
	if err != nil {
		t.Fatal(err)
	}
	val, _ := s.Get(0)
	expected := math.Sqrt(2.0 / 3.0)
	if math.Abs(val-expected) > 1e-10 {
		t.Errorf("std[0] = %v, want %v", val, expected)
	}
}

// ── Min / Max ──

func TestMin(t *testing.T) {
	a, _ := New[float64]([]float64{3, 1, 4, 1, 5, 9}, 2, 3)
	m, err := Min(a, 1)
	if err != nil {
		t.Fatal(err)
	}
	v, _ := m.Get(0)
	if v != 1 {
		t.Errorf("min[0] = %v, want 1", v)
	}
	v, _ = m.Get(1)
	if v != 1 {
		t.Errorf("min[1] = %v, want 1", v)
	}
}

func TestMinAll(t *testing.T) {
	a, _ := New[float64]([]float64{5, 3, 8, 1, 7}, 5)
	m := MinAll(a)
	if m != 1 {
		t.Errorf("MinAll = %v, want 1", m)
	}
}

func TestMax(t *testing.T) {
	a, _ := New[float64]([]float64{3, 1, 4, 1, 5, 9}, 2, 3)
	m, err := Max(a, 1)
	if err != nil {
		t.Fatal(err)
	}
	v, _ := m.Get(0)
	if v != 4 {
		t.Errorf("max[0] = %v, want 4", v)
	}
	v, _ = m.Get(1)
	if v != 9 {
		t.Errorf("max[1] = %v, want 9", v)
	}
}

func TestMaxAll(t *testing.T) {
	a, _ := New[float64]([]float64{5, 3, 8, 1, 7}, 5)
	m := MaxAll(a)
	if m != 8 {
		t.Errorf("MaxAll = %v, want 8", m)
	}
}

// ── ArgMin / ArgMax ──

func TestArgMin(t *testing.T) {
	a, _ := New[float64]([]float64{3, 1, 4, 9, 5, 2}, 2, 3)
	r, err := ArgMin(a, 1)
	if err != nil {
		t.Fatal(err)
	}
	v, _ := r.Get(0)
	if v != 1 { // min of [3,1,4] at index 1
		t.Errorf("argmin[0] = %v, want 1", v)
	}
	v, _ = r.Get(1)
	if v != 2 { // min of [9,5,2] at index 2
		t.Errorf("argmin[1] = %v, want 2", v)
	}
}

func TestArgMax(t *testing.T) {
	a, _ := New[float64]([]float64{3, 1, 4, 9, 5, 2}, 2, 3)
	r, err := ArgMax(a, 1)
	if err != nil {
		t.Fatal(err)
	}
	v, _ := r.Get(0)
	if v != 2 { // max of [3,1,4] at index 2
		t.Errorf("argmax[0] = %v, want 2", v)
	}
	v, _ = r.Get(1)
	if v != 0 { // max of [9,5,2] at index 0
		t.Errorf("argmax[1] = %v, want 0", v)
	}
}

func TestArgMinAll(t *testing.T) {
	a, _ := New[float64]([]float64{5, 3, 8, 1, 7}, 5)
	idx := ArgMinAll(a)
	if idx != 3 {
		t.Errorf("ArgMinAll = %d, want 3", idx)
	}
}

func TestArgMaxAll(t *testing.T) {
	a, _ := New[float64]([]float64{5, 3, 8, 1, 7}, 5)
	idx := ArgMaxAll(a)
	if idx != 2 {
		t.Errorf("ArgMaxAll = %d, want 2", idx)
	}
}

// ── CumSum / CumProd ──

func TestCumSum1D(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4}, 4)
	c, err := CumSum(a, 0)
	if err != nil {
		t.Fatal(err)
	}
	expected := []float64{1, 3, 6, 10}
	for i, want := range expected {
		v, _ := c.Get(i)
		if v != want {
			t.Errorf("cumsum[%d] = %v, want %v", i, v, want)
		}
	}
}

func TestCumSum2D(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	c, err := CumSum(a, 1)
	if err != nil {
		t.Fatal(err)
	}
	// Row 0: [1, 3, 6]
	// Row 1: [4, 9, 15]
	v, _ := c.Get(0, 2)
	if v != 6 {
		t.Errorf("cumsum[0,2] = %v, want 6", v)
	}
	v, _ = c.Get(1, 2)
	if v != 15 {
		t.Errorf("cumsum[1,2] = %v, want 15", v)
	}
}

func TestCumSumAxis0(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	c, err := CumSum(a, 0)
	if err != nil {
		t.Fatal(err)
	}
	// Col 0: [1, 5], Col 1: [2, 7], Col 2: [3, 9]
	v, _ := c.Get(1, 0)
	if v != 5 {
		t.Errorf("cumsum[1,0] = %v, want 5", v)
	}
	v, _ = c.Get(1, 2)
	if v != 9 {
		t.Errorf("cumsum[1,2] = %v, want 9", v)
	}
}

func TestCumProd1D(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4}, 4)
	c, err := CumProd(a, 0)
	if err != nil {
		t.Fatal(err)
	}
	expected := []float64{1, 2, 6, 24}
	for i, want := range expected {
		v, _ := c.Get(i)
		if v != want {
			t.Errorf("cumprod[%d] = %v, want %v", i, v, want)
		}
	}
}

func TestCumProd2D(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	c, err := CumProd(a, 1)
	if err != nil {
		t.Fatal(err)
	}
	// Row 0: [1, 2, 6]
	// Row 1: [4, 20, 120]
	v, _ := c.Get(0, 2)
	if v != 6 {
		t.Errorf("cumprod[0,2] = %v, want 6", v)
	}
	v, _ = c.Get(1, 2)
	if v != 120 {
		t.Errorf("cumprod[1,2] = %v, want 120", v)
	}
}

// ── Reduce on views ──

func TestSumOnView(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3)
	b, _ := a.Slice(S(0, 2), S(0, 2)) // [[1,2],[4,5]]
	s := SumAll(b)
	if s != 12 {
		t.Errorf("SumAll of slice = %v, want 12", s)
	}
}

func TestSumOnTransposed(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b, _ := a.T() // [[1,4],[2,5],[3,6]]
	s, err := Sum(b, 1) // Row sums: [5, 7, 9]
	if err != nil {
		t.Fatal(err)
	}
	expected := []float64{5, 7, 9}
	for i, want := range expected {
		v, _ := s.Get(i)
		if v != want {
			t.Errorf("sum[%d] = %v, want %v", i, v, want)
		}
	}
}
