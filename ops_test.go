package sigma

import (
	"math"
	"testing"
)

// ── broadcast ──

func TestBroadcastShapeSame(t *testing.T) {
	s, err := broadcastShape([]int{3, 4}, []int{3, 4})
	if err != nil {
		t.Fatal(err)
	}
	if !sliceEqual(s, []int{3, 4}) {
		t.Errorf("got %v, want [3,4]", s)
	}
}

func TestBroadcastShapeExpand(t *testing.T) {
	s, err := broadcastShape([]int{3, 4}, []int{4})
	if err != nil {
		t.Fatal(err)
	}
	if !sliceEqual(s, []int{3, 4}) {
		t.Errorf("got %v, want [3,4]", s)
	}
}

func TestBroadcastShapeBoth(t *testing.T) {
	s, err := broadcastShape([]int{3, 1}, []int{1, 4})
	if err != nil {
		t.Fatal(err)
	}
	if !sliceEqual(s, []int{3, 4}) {
		t.Errorf("got %v, want [3,4]", s)
	}
}

func TestBroadcastShapeError(t *testing.T) {
	_, err := broadcastShape([]int{3, 4}, []int{3, 5})
	if err == nil {
		t.Error("expected broadcast error")
	}
}

// ── arithmetic ──

func TestAddSameShape(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3}, 3)
	b, _ := New[float64]([]float64{4, 5, 6}, 3)
	c, err := Add(a, b)
	if err != nil {
		t.Fatal(err)
	}
	expected := []float64{5, 7, 9}
	for i, want := range expected {
		v, _ := c.Get(i)
		if v != want {
			t.Errorf("c[%d] = %v, want %v", i, v, want)
		}
	}
}

func TestAddBroadcast(t *testing.T) {
	a := Ones[float64](3, 4)
	b, _ := New[float64]([]float64{1, 2, 3, 4}, 4)
	c, err := Add(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if !sliceEqual(c.Shape(), []int{3, 4}) {
		t.Errorf("shape = %v, want [3,4]", c.Shape())
	}
	// Row 0: [1+1, 1+2, 1+3, 1+4] = [2, 3, 4, 5]
	v, _ := c.Get(0, 0)
	if v != 2 {
		t.Errorf("c[0,0] = %v, want 2", v)
	}
	v, _ = c.Get(0, 3)
	if v != 5 {
		t.Errorf("c[0,3] = %v, want 5", v)
	}
	// Row 2: same values
	v, _ = c.Get(2, 1)
	if v != 3 {
		t.Errorf("c[2,1] = %v, want 3", v)
	}
}

func TestAddBroadcastColumnVector(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b, _ := New[float64]([]float64{10, 20}, 2, 1)
	c, err := Add(a, b)
	if err != nil {
		t.Fatal(err)
	}
	// Row 0: [1+10, 2+10, 3+10] = [11, 12, 13]
	v, _ := c.Get(0, 2)
	if v != 13 {
		t.Errorf("c[0,2] = %v, want 13", v)
	}
	// Row 1: [4+20, 5+20, 6+20] = [24, 25, 26]
	v, _ = c.Get(1, 0)
	if v != 24 {
		t.Errorf("c[1,0] = %v, want 24", v)
	}
}

func TestSub(t *testing.T) {
	a, _ := New[float64]([]float64{5, 10, 15}, 3)
	b, _ := New[float64]([]float64{1, 2, 3}, 3)
	c, _ := Sub(a, b)
	expected := []float64{4, 8, 12}
	for i, want := range expected {
		v, _ := c.Get(i)
		if v != want {
			t.Errorf("c[%d] = %v, want %v", i, v, want)
		}
	}
}

func TestMul(t *testing.T) {
	a, _ := New[float64]([]float64{2, 3, 4}, 3)
	b, _ := New[float64]([]float64{5, 6, 7}, 3)
	c, _ := Mul(a, b)
	expected := []float64{10, 18, 28}
	for i, want := range expected {
		v, _ := c.Get(i)
		if v != want {
			t.Errorf("c[%d] = %v, want %v", i, v, want)
		}
	}
}

func TestDiv(t *testing.T) {
	a, _ := New[float64]([]float64{10, 20, 30}, 3)
	b, _ := New[float64]([]float64{2, 4, 5}, 3)
	c, _ := Div(a, b)
	expected := []float64{5, 5, 6}
	for i, want := range expected {
		v, _ := c.Get(i)
		if v != want {
			t.Errorf("c[%d] = %v, want %v", i, v, want)
		}
	}
}

func TestPow(t *testing.T) {
	a, _ := New[float64]([]float64{2, 3, 4}, 3)
	b, _ := New[float64]([]float64{3, 2, 0.5}, 3)
	c, _ := Pow(a, b)
	expected := []float64{8, 9, 2}
	for i, want := range expected {
		v, _ := c.Get(i)
		if math.Abs(v-want) > 1e-10 {
			t.Errorf("c[%d] = %v, want %v", i, v, want)
		}
	}
}

func TestMod(t *testing.T) {
	a, _ := New[int64]([]int64{10, 11, 12}, 3)
	b, _ := New[int64]([]int64{3, 3, 3}, 3)
	c, _ := Mod(a, b)
	expected := []int64{1, 2, 0}
	for i, want := range expected {
		v, _ := c.Get(i)
		if v != want {
			t.Errorf("c[%d] = %v, want %v", i, v, want)
		}
	}
}

func TestNeg(t *testing.T) {
	a, _ := New[float64]([]float64{1, -2, 3}, 3)
	b := Neg(a)
	expected := []float64{-1, 2, -3}
	for i, want := range expected {
		v, _ := b.Get(i)
		if v != want {
			t.Errorf("b[%d] = %v, want %v", i, v, want)
		}
	}
}

func TestAbs(t *testing.T) {
	a, _ := New[float64]([]float64{-1, 2, -3}, 3)
	b := Abs(a)
	expected := []float64{1, 2, 3}
	for i, want := range expected {
		v, _ := b.Get(i)
		if v != want {
			t.Errorf("b[%d] = %v, want %v", i, v, want)
		}
	}
}

// ── math ──

func TestSqrt(t *testing.T) {
	a, _ := New[float64]([]float64{1, 4, 9, 16}, 4)
	b := Sqrt(a)
	expected := []float64{1, 2, 3, 4}
	for i, want := range expected {
		v, _ := b.Get(i)
		if math.Abs(v-want) > 1e-10 {
			t.Errorf("b[%d] = %v, want %v", i, v, want)
		}
	}
}

func TestExpLog(t *testing.T) {
	a, _ := New[float64]([]float64{0, 1, 2}, 3)
	b := Exp(a)
	c := Log(b)
	// Log(Exp(x)) should return x
	for i := 0; i < 3; i++ {
		v, _ := c.Get(i)
		w, _ := a.Get(i)
		if math.Abs(v-w) > 1e-10 {
			t.Errorf("Log(Exp(a[%d])) = %v, want %v", i, v, w)
		}
	}
}

func TestSinCos(t *testing.T) {
	a, _ := New[float64]([]float64{0, math.Pi / 2, math.Pi}, 3)
	s := Sin(a)
	c := Cos(a)

	// sin(0) = 0
	v, _ := s.Get(0)
	if math.Abs(v) > 1e-10 {
		t.Errorf("sin(0) = %v", v)
	}
	// sin(π/2) = 1
	v, _ = s.Get(1)
	if math.Abs(v-1) > 1e-10 {
		t.Errorf("sin(π/2) = %v", v)
	}
	// cos(0) = 1
	v, _ = c.Get(0)
	if math.Abs(v-1) > 1e-10 {
		t.Errorf("cos(0) = %v", v)
	}
}

// ── comparison ──

func TestEqual(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3}, 3)
	b, _ := New[float64]([]float64{1, 5, 3}, 3)
	c, _ := Equal(a, b)
	expected := []float64{1, 0, 1}
	for i, want := range expected {
		v, _ := c.Get(i)
		if v != want {
			t.Errorf("c[%d] = %v, want %v", i, v, want)
		}
	}
}

func TestLess(t *testing.T) {
	a, _ := New[float64]([]float64{1, 5, 3}, 3)
	b, _ := New[float64]([]float64{2, 2, 3}, 3)
	c, _ := Less(a, b)
	expected := []float64{1, 0, 0}
	for i, want := range expected {
		v, _ := c.Get(i)
		if v != want {
			t.Errorf("c[%d] = %v, want %v", i, v, want)
		}
	}
}

func TestGreater(t *testing.T) {
	a, _ := New[float64]([]float64{1, 5, 3}, 3)
	b, _ := New[float64]([]float64{2, 2, 3}, 3)
	c, _ := Greater(a, b)
	expected := []float64{0, 1, 0}
	for i, want := range expected {
		v, _ := c.Get(i)
		if v != want {
			t.Errorf("c[%d] = %v, want %v", i, v, want)
		}
	}
}

func TestComparisonBroadcast(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b, _ := New[float64]([]float64{3}, 1)
	c, _ := Greater(a, b)
	if !sliceEqual(c.Shape(), []int{2, 3}) {
		t.Errorf("shape = %v, want [2,3]", c.Shape())
	}
	// Elements > 3: 4, 5, 6 → row 1 indices 0,1,2
	v, _ := c.Get(0, 2) // 3 > 3 = false
	if v != 0 {
		t.Errorf("c[0,2] = %v, want 0", v)
	}
	v, _ = c.Get(1, 0) // 4 > 3 = true
	if v != 1 {
		t.Errorf("c[1,0] = %v, want 1", v)
	}
}

// ── scalar ops ──

func TestAddScalar(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3}, 3)
	b := AddScalar(a, 10)
	expected := []float64{11, 12, 13}
	for i, want := range expected {
		v, _ := b.Get(i)
		if v != want {
			t.Errorf("b[%d] = %v, want %v", i, v, want)
		}
	}
}

func TestMulScalar(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3}, 3)
	b := MulScalar(a, 3)
	expected := []float64{3, 6, 9}
	for i, want := range expected {
		v, _ := b.Get(i)
		if v != want {
			t.Errorf("b[%d] = %v, want %v", i, v, want)
		}
	}
}

func TestPowScalar(t *testing.T) {
	a, _ := New[float64]([]float64{2, 3, 4}, 3)
	b := PowScalar(a, 2)
	expected := []float64{4, 9, 16}
	for i, want := range expected {
		v, _ := b.Get(i)
		if math.Abs(v-want) > 1e-10 {
			t.Errorf("b[%d] = %v, want %v", i, v, want)
		}
	}
}

// ── ops on views ──

func TestAddOnTransposedView(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4}, 2, 2)
	b, _ := a.T() // [[1,3],[2,4]]
	c, err := Add(b, b)
	if err != nil {
		t.Fatal(err)
	}
	// [[2,6],[4,8]]
	v, _ := c.Get(0, 1)
	if v != 6 {
		t.Errorf("c[0,1] = %v, want 6", v)
	}
}

func TestAddOnSlicedView(t *testing.T) {
	a, _ := New[float64]([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b, _ := a.Slice(All(), S(1, 3))   // [[2,3],[5,6]]
	c := Ones[float64](2, 2)
	d, err := Add(b, c)
	if err != nil {
		t.Fatal(err)
	}
	v, _ := d.Get(0, 0)
	if v != 3 {
		t.Errorf("d[0,0] = %v, want 3", v)
	}
}
