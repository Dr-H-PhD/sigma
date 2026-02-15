package rand

import (
	"math"
	"testing"

	"github.com/Dr-H-PhD/sigma"
)

func TestUniform(t *testing.T) {
	s := NewSource(42)
	a := Uniform[float64](s, 0, 1, 100)
	if a.Size() != 100 {
		t.Fatalf("size = %d, want 100", a.Size())
	}
	for i := 0; i < 100; i++ {
		v, _ := a.Get(i)
		if v < 0 || v >= 1 {
			t.Errorf("value %v out of [0, 1)", v)
		}
	}
}

func TestUniform2D(t *testing.T) {
	s := NewSource(42)
	a := Uniform[float64](s, -5, 5, 3, 4)
	shape := a.Shape()
	if shape[0] != 3 || shape[1] != 4 {
		t.Errorf("shape = %v, want [3,4]", shape)
	}
}

func TestNormal(t *testing.T) {
	s := NewSource(42)
	a := Normal[float64](s, 0, 1, 10000)
	// Check mean ≈ 0 and std ≈ 1
	mean := sigma.MeanAll(a)
	std := sigma.StdAll(a)
	if math.Abs(mean) > 0.1 {
		t.Errorf("mean = %v, expected ≈ 0", mean)
	}
	if math.Abs(std-1) > 0.1 {
		t.Errorf("std = %v, expected ≈ 1", std)
	}
}

func TestNormalWithParams(t *testing.T) {
	s := NewSource(42)
	a := Normal[float64](s, 5, 2, 10000)
	mean := sigma.MeanAll(a)
	if math.Abs(mean-5) > 0.2 {
		t.Errorf("mean = %v, expected ≈ 5", mean)
	}
}

func TestRandint(t *testing.T) {
	s := NewSource(42)
	a := Randint(s, 0, 10, 100)
	for i := 0; i < 100; i++ {
		v, _ := a.Get(i)
		if v < 0 || v >= 10 {
			t.Errorf("value %v out of [0, 10)", v)
		}
	}
}

func TestShuffle(t *testing.T) {
	s := NewSource(42)
	a, _ := sigma.New[float64]([]float64{1, 2, 3, 4, 5}, 5)
	b := Shuffle[float64](s, a)
	if b.Size() != 5 {
		t.Fatalf("size = %d, want 5", b.Size())
	}
	// Original should be unchanged
	v, _ := a.Get(0)
	if v != 1 {
		t.Error("original was modified")
	}
	// All elements should still be present
	sum := sigma.SumAll(b)
	if sum != 15 {
		t.Errorf("sum = %v, want 15 (elements lost)", sum)
	}
}

func TestChoiceWithReplacement(t *testing.T) {
	s := NewSource(42)
	a, _ := sigma.New[float64]([]float64{10, 20, 30}, 3)
	b, err := Choice[float64](s, a, 5, true)
	if err != nil {
		t.Fatal(err)
	}
	if b.Size() != 5 {
		t.Fatalf("size = %d, want 5", b.Size())
	}
	for i := 0; i < 5; i++ {
		v, _ := b.Get(i)
		if v != 10 && v != 20 && v != 30 {
			t.Errorf("unexpected value %v", v)
		}
	}
}

func TestChoiceWithoutReplacement(t *testing.T) {
	s := NewSource(42)
	a, _ := sigma.New[float64]([]float64{10, 20, 30, 40, 50}, 5)
	b, err := Choice[float64](s, a, 3, false)
	if err != nil {
		t.Fatal(err)
	}
	if b.Size() != 3 {
		t.Fatalf("size = %d, want 3", b.Size())
	}
	// All values must be unique
	seen := map[float64]bool{}
	for i := 0; i < 3; i++ {
		v, _ := b.Get(i)
		if seen[v] {
			t.Errorf("duplicate value %v in sample without replacement", v)
		}
		seen[v] = true
	}
}

func TestChoiceTooManyWithout(t *testing.T) {
	s := NewSource(42)
	a, _ := sigma.New[float64]([]float64{1, 2, 3}, 3)
	_, err := Choice[float64](s, a, 5, false)
	if err == nil {
		t.Error("expected error when n > population without replacement")
	}
}

func TestPermutation(t *testing.T) {
	s := NewSource(42)
	p := Permutation(s, 10)
	if p.Size() != 10 {
		t.Fatalf("size = %d, want 10", p.Size())
	}
	// All values 0-9 must be present
	seen := map[int64]bool{}
	for i := 0; i < 10; i++ {
		v, _ := p.Get(i)
		if v < 0 || v >= 10 {
			t.Errorf("value %v out of range", v)
		}
		if seen[v] {
			t.Errorf("duplicate value %v", v)
		}
		seen[v] = true
	}
}

func TestReproducibility(t *testing.T) {
	a1 := Uniform[float64](NewSource(42), 0, 1, 10)
	a2 := Uniform[float64](NewSource(42), 0, 1, 10)
	for i := 0; i < 10; i++ {
		v1, _ := a1.Get(i)
		v2, _ := a2.Get(i)
		if v1 != v2 {
			t.Errorf("not reproducible: a1[%d]=%v, a2[%d]=%v", i, v1, i, v2)
		}
	}
}
