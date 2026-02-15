package linalg

import (
	"math"
	"testing"

	"github.com/Dr-H-PhD/sigma"
)

const tol = 1e-8

func approxEq(a, b float64) bool {
	return math.Abs(a-b) < tol
}

// ── Dot ──

func TestDot(t *testing.T) {
	a, _ := sigma.New[float64]([]float64{1, 2, 3}, 3)
	b, _ := sigma.New[float64]([]float64{4, 5, 6}, 3)
	v, err := Dot(a, b)
	if err != nil {
		t.Fatal(err)
	}
	// 1*4 + 2*5 + 3*6 = 32
	if v != 32 {
		t.Errorf("Dot = %v, want 32", v)
	}
}

// ── MatMul ──

func TestMatMul(t *testing.T) {
	a, _ := sigma.New[float64]([]float64{1, 2, 3, 4}, 2, 2)
	b, _ := sigma.New[float64]([]float64{5, 6, 7, 8}, 2, 2)
	c, err := MatMul(a, b)
	if err != nil {
		t.Fatal(err)
	}
	// [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
	expected := [][]float64{{19, 22}, {43, 50}}
	for i, row := range expected {
		for j, want := range row {
			v, _ := c.Get(i, j)
			if !approxEq(v, want) {
				t.Errorf("c[%d,%d] = %v, want %v", i, j, v, want)
			}
		}
	}
}

func TestMatMulRectangular(t *testing.T) {
	a, _ := sigma.New[float64]([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b, _ := sigma.New[float64]([]float64{7, 8, 9, 10, 11, 12}, 3, 2)
	c, err := MatMul(a, b)
	if err != nil {
		t.Fatal(err)
	}
	shape := c.Shape()
	if shape[0] != 2 || shape[1] != 2 {
		t.Errorf("shape = %v, want [2,2]", shape)
	}
	// [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
	v, _ := c.Get(0, 0)
	if !approxEq(v, 58) {
		t.Errorf("c[0,0] = %v, want 58", v)
	}
}

func TestMatMulDimMismatch(t *testing.T) {
	a, _ := sigma.New[float64]([]float64{1, 2, 3, 4}, 2, 2)
	b, _ := sigma.New[float64]([]float64{1, 2, 3, 4, 5, 6}, 3, 2)
	_, err := MatMul(a, b)
	if err == nil {
		t.Error("expected error for dimension mismatch")
	}
}

// ── Outer ──

func TestOuter(t *testing.T) {
	a, _ := sigma.New[float64]([]float64{1, 2, 3}, 3)
	b, _ := sigma.New[float64]([]float64{4, 5}, 2)
	c, err := Outer(a, b)
	if err != nil {
		t.Fatal(err)
	}
	shape := c.Shape()
	if shape[0] != 3 || shape[1] != 2 {
		t.Errorf("shape = %v, want [3,2]", shape)
	}
	v, _ := c.Get(2, 1) // 3 * 5 = 15
	if v != 15 {
		t.Errorf("c[2,1] = %v, want 15", v)
	}
}

// ── LU ──

func TestLU(t *testing.T) {
	a, _ := sigma.New[float64]([]float64{2, 1, 5, 3}, 2, 2)
	l, u, piv, err := LU(a)
	if err != nil {
		t.Fatal(err)
	}

	// Verify PA = LU
	lu, _ := MatMul(l, u)
	n := 2
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			pa, _ := a.Get(piv[i], j)
			luv, _ := lu.Get(i, j)
			if !approxEq(float64(pa), float64(luv)) {
				t.Errorf("PA[%d,%d]=%v != LU[%d,%d]=%v", i, j, pa, i, j, luv)
			}
		}
	}
}

// ── Solve ──

func TestSolve(t *testing.T) {
	// Solve [[2,1],[5,3]] x = [4, 7]
	a, _ := sigma.New[float64]([]float64{2, 1, 5, 3}, 2, 2)
	b, _ := sigma.New[float64]([]float64{4, 7}, 2)
	x, err := Solve(a, b)
	if err != nil {
		t.Fatal(err)
	}
	// x = [5, -6]
	v0, _ := x.Get(0)
	v1, _ := x.Get(1)
	if !approxEq(v0, 5) || !approxEq(v1, -6) {
		t.Errorf("x = [%v, %v], want [5, -6]", v0, v1)
	}
}

func TestSolve3x3(t *testing.T) {
	// [[1,2,3],[0,1,4],[5,6,0]] x = [1,2,3]
	a, _ := sigma.New[float64]([]float64{1, 2, 3, 0, 1, 4, 5, 6, 0}, 3, 3)
	b, _ := sigma.New[float64]([]float64{1, 2, 3}, 3)
	x, err := Solve(a, b)
	if err != nil {
		t.Fatal(err)
	}
	// Verify Ax = b
	ax, _ := MatMul(a, func() *sigma.NDArray[float64] { r, _ := x.Reshape(3, 1); return r }())
	for i := 0; i < 3; i++ {
		want, _ := b.Get(i)
		got, _ := ax.Get(i, 0)
		if !approxEq(got, want) {
			t.Errorf("Ax[%d] = %v, want %v", i, got, want)
		}
	}
}

// ── Inv ──

func TestInv(t *testing.T) {
	a, _ := sigma.New[float64]([]float64{4, 7, 2, 6}, 2, 2)
	inv, err := Inv(a)
	if err != nil {
		t.Fatal(err)
	}
	// A * A^-1 should be identity
	prod, _ := MatMul(a, inv)
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			v, _ := prod.Get(i, j)
			want := 0.0
			if i == j {
				want = 1.0
			}
			if !approxEq(v, want) {
				t.Errorf("(A*inv)[%d,%d] = %v, want %v", i, j, v, want)
			}
		}
	}
}

// ── Det ──

func TestDet(t *testing.T) {
	a, _ := sigma.New[float64]([]float64{1, 2, 3, 4}, 2, 2)
	d, err := Det(a)
	if err != nil {
		t.Fatal(err)
	}
	// det = 1*4 - 2*3 = -2
	if !approxEq(float64(d), -2) {
		t.Errorf("Det = %v, want -2", d)
	}
}

func TestDet3x3(t *testing.T) {
	a, _ := sigma.New[float64]([]float64{6, 1, 1, 4, -2, 5, 2, 8, 7}, 3, 3)
	d, err := Det(a)
	if err != nil {
		t.Fatal(err)
	}
	// Known det = -306
	if !approxEq(float64(d), -306) {
		t.Errorf("Det = %v, want -306", d)
	}
}

// ── Trace ──

func TestTrace(t *testing.T) {
	a, _ := sigma.New[float64]([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3)
	tr, err := Trace(a)
	if err != nil {
		t.Fatal(err)
	}
	if tr != 15 {
		t.Errorf("Trace = %v, want 15", tr)
	}
}

// ── Norm ──

func TestNormVector(t *testing.T) {
	a, _ := sigma.New[float64]([]float64{3, 4}, 2)
	n, err := Norm(a, 2)
	if err != nil {
		t.Fatal(err)
	}
	if !approxEq(float64(n), 5) {
		t.Errorf("L2 norm = %v, want 5", n)
	}
}

func TestNormL1(t *testing.T) {
	a, _ := sigma.New[float64]([]float64{-3, 4}, 2)
	n, err := Norm(a, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !approxEq(float64(n), 7) {
		t.Errorf("L1 norm = %v, want 7", n)
	}
}

func TestNormInf(t *testing.T) {
	a, _ := sigma.New[float64]([]float64{-3, 4}, 2)
	n, err := Norm(a, math.Inf(1))
	if err != nil {
		t.Fatal(err)
	}
	if !approxEq(float64(n), 4) {
		t.Errorf("Inf norm = %v, want 4", n)
	}
}

func TestNormFrobenius(t *testing.T) {
	a, _ := sigma.New[float64]([]float64{1, 2, 3, 4}, 2, 2)
	n, err := Norm(a, 0)
	if err != nil {
		t.Fatal(err)
	}
	// sqrt(1+4+9+16) = sqrt(30)
	if !approxEq(float64(n), math.Sqrt(30)) {
		t.Errorf("Frobenius norm = %v, want %v", n, math.Sqrt(30))
	}
}

// ── QR ──

func TestQR(t *testing.T) {
	a, _ := sigma.New[float64]([]float64{12, -51, 4, 6, 167, -68, -4, 24, -41}, 3, 3)
	q, r, err := QR(a)
	if err != nil {
		t.Fatal(err)
	}

	// Verify Q*R ≈ A
	qr, _ := MatMul(q, r)
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			want, _ := a.Get(i, j)
			got, _ := qr.Get(i, j)
			if !approxEq(got, want) {
				t.Errorf("QR[%d,%d] = %v, want %v", i, j, got, want)
			}
		}
	}

	// Verify Q is orthogonal: Q^T Q ≈ I
	qt, _ := q.T()
	qtq, _ := MatMul(qt, q)
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			v, _ := qtq.Get(i, j)
			want := 0.0
			if i == j {
				want = 1.0
			}
			if !approxEq(v, want) {
				t.Errorf("Q^TQ[%d,%d] = %v, want %v", i, j, v, want)
			}
		}
	}
}

// ── SVD ──

func TestSVD(t *testing.T) {
	// [[3,0],[0,4]] → singular values should be 4, 3
	a, _ := sigma.New[float64]([]float64{3, 0, 0, 4}, 2, 2)
	u, s, v, err := SVD(a)
	if err != nil {
		t.Fatal(err)
	}

	// Check singular values (sorted descending)
	s0, _ := s.Get(0)
	s1, _ := s.Get(1)
	if !approxEq(s0, 4) || !approxEq(s1, 3) {
		t.Errorf("singular values = [%v, %v], want [4, 3]", s0, s1)
	}

	// Verify U * diag(S) * V^T ≈ A
	// Build diagonal matrix from S
	smat := sigma.Zeros[float64](2, 2)
	smat.Set(s0, 0, 0)
	smat.Set(s1, 1, 1)
	vt, _ := v.T()
	usv, _ := MatMul(u, smat)
	result, _ := MatMul(usv, vt)
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			want, _ := a.Get(i, j)
			got, _ := result.Get(i, j)
			if !approxEq(got, want) {
				t.Errorf("USV^T[%d,%d] = %v, want %v", i, j, got, want)
			}
		}
	}

	_ = u
}

// ── Eig ──

func TestEig(t *testing.T) {
	// Symmetric: [[2,1],[1,2]] → eigenvalues 3, 1
	a, _ := sigma.New[float64]([]float64{2, 1, 1, 2}, 2, 2)
	vals, _, err := Eig(a)
	if err != nil {
		t.Fatal(err)
	}

	// Sort eigenvalues for comparison
	v0, _ := vals.Get(0)
	v1, _ := vals.Get(1)
	if v0 < v1 {
		v0, v1 = v1, v0
	}
	if !approxEq(v0, 3) || !approxEq(v1, 1) {
		t.Errorf("eigenvalues = [%v, %v], want [3, 1]", v0, v1)
	}
}

// ── Cholesky ──

func TestCholesky(t *testing.T) {
	// [[4,12,-16],[12,37,-43],[-16,-43,98]]
	a, _ := sigma.New[float64]([]float64{4, 12, -16, 12, 37, -43, -16, -43, 98}, 3, 3)
	l, err := Cholesky(a)
	if err != nil {
		t.Fatal(err)
	}

	// Verify L L^T = A
	lt, _ := l.T()
	llt, _ := MatMul(l, lt)
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			want, _ := a.Get(i, j)
			got, _ := llt.Get(i, j)
			if !approxEq(got, want) {
				t.Errorf("LL^T[%d,%d] = %v, want %v", i, j, got, want)
			}
		}
	}
}

func TestCholeskyNotPositiveDefinite(t *testing.T) {
	a, _ := sigma.New[float64]([]float64{1, 2, 2, 1}, 2, 2)
	_, err := Cholesky(a)
	if err == nil {
		t.Error("expected error for non-positive-definite matrix")
	}
}

// ── Rank ──

func TestRank(t *testing.T) {
	// Rank 2 matrix (3x3)
	a, _ := sigma.New[float64]([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3)
	r, err := Rank(a, 1e-8)
	if err != nil {
		t.Fatal(err)
	}
	if r != 2 {
		t.Errorf("Rank = %d, want 2", r)
	}
}

func TestRankFull(t *testing.T) {
	a := sigma.Eye[float64](3)
	r, err := Rank(a, 1e-8)
	if err != nil {
		t.Fatal(err)
	}
	if r != 3 {
		t.Errorf("Rank = %d, want 3", r)
	}
}

// ── Cond ──

func TestCond(t *testing.T) {
	a := sigma.Eye[float64](3)
	c, err := Cond(a)
	if err != nil {
		t.Fatal(err)
	}
	if !approxEq(float64(c), 1) {
		t.Errorf("Cond(I) = %v, want 1", c)
	}
}

// ── LstSq ──

func TestLstSq(t *testing.T) {
	// Overdetermined: fit y = mx + c
	// Points: (0,1), (1,3), (2,5) → y = 2x + 1
	a, _ := sigma.New[float64]([]float64{0, 1, 1, 1, 2, 1}, 3, 2)
	b, _ := sigma.New[float64]([]float64{1, 3, 5}, 3)
	x, err := LstSq(a, b)
	if err != nil {
		t.Fatal(err)
	}
	// x[0] = 2 (slope), x[1] = 1 (intercept)
	v0, _ := x.Get(0)
	v1, _ := x.Get(1)
	if !approxEq(v0, 2) || !approxEq(v1, 1) {
		t.Errorf("LstSq = [%v, %v], want [2, 1]", v0, v1)
	}
}
