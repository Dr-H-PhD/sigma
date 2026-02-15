package linalg

import (
	"fmt"
	"math"

	"github.com/Dr-H-PhD/sigma"
)

// LU computes the LU decomposition with partial pivoting.
// Returns L, U, P (permutation vector) such that PA = LU.
func LU[T sigma.Float](a *sigma.NDArray[T]) (l, u *sigma.NDArray[T], piv []int, err error) {
	if a.Ndim() != 2 {
		return nil, nil, nil, &sigma.DimensionError{Msg: "LU requires a 2-D array"}
	}
	shape := a.Shape()
	n := shape[0]
	if n != shape[1] {
		return nil, nil, nil, &sigma.ShapeError{Msg: "LU requires a square matrix"}
	}

	// Work on a copy
	work := a.Copy()
	piv = make([]int, n)
	for i := range piv {
		piv[i] = i
	}

	for k := 0; k < n; k++ {
		// Find pivot
		maxVal := T(0)
		maxIdx := k
		for i := k; i < n; i++ {
			v, _ := work.Get(i, k)
			av := T(math.Abs(float64(v)))
			if av > maxVal {
				maxVal = av
				maxIdx = i
			}
		}

		if maxVal < T(1e-15) {
			return nil, nil, nil, &sigma.SingularMatrixError{}
		}

		// Swap rows
		if maxIdx != k {
			piv[k], piv[maxIdx] = piv[maxIdx], piv[k]
			for j := 0; j < n; j++ {
				vk, _ := work.Get(k, j)
				vm, _ := work.Get(maxIdx, j)
				work.Set(vm, k, j)
				work.Set(vk, maxIdx, j)
			}
		}

		// Eliminate below
		pivot, _ := work.Get(k, k)
		for i := k + 1; i < n; i++ {
			vik, _ := work.Get(i, k)
			factor := vik / pivot
			work.Set(factor, i, k)
			for j := k + 1; j < n; j++ {
				vij, _ := work.Get(i, j)
				vkj, _ := work.Get(k, j)
				work.Set(vij-factor*vkj, i, j)
			}
		}
	}

	// Extract L and U
	l = sigma.Eye[T](n)
	u = sigma.Zeros[T](n, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			v, _ := work.Get(i, j)
			if j >= i {
				u.Set(v, i, j)
			} else {
				l.Set(v, i, j)
			}
		}
	}
	return l, u, piv, nil
}

// Solve solves the linear system Ax = b using LU decomposition.
// A must be square (n,n) and b must be (n,) or (n,m).
func Solve[T sigma.Float](a, b *sigma.NDArray[T]) (*sigma.NDArray[T], error) {
	if a.Ndim() != 2 {
		return nil, &sigma.DimensionError{Msg: "Solve: A must be 2-D"}
	}
	aShape := a.Shape()
	n := aShape[0]
	if n != aShape[1] {
		return nil, &sigma.ShapeError{Msg: "Solve: A must be square"}
	}

	// Handle 1-D b
	bIs1D := b.Ndim() == 1
	var bMat *sigma.NDArray[T]
	if bIs1D {
		if b.Shape()[0] != n {
			return nil, &sigma.ShapeError{
				Msg: fmt.Sprintf("Solve: b length %d doesn't match A size %d", b.Shape()[0], n),
			}
		}
		bMat, _ = b.Reshape(n, 1)
	} else if b.Ndim() == 2 {
		if b.Shape()[0] != n {
			return nil, &sigma.ShapeError{
				Msg: fmt.Sprintf("Solve: b rows %d doesn't match A size %d", b.Shape()[0], n),
			}
		}
		bMat = b
	} else {
		return nil, &sigma.DimensionError{Msg: "Solve: b must be 1-D or 2-D"}
	}

	l, u, piv, err := LU(a)
	if err != nil {
		return nil, err
	}

	nrhs := bMat.Shape()[1]
	x := sigma.Zeros[T](n, nrhs)

	for col := 0; col < nrhs; col++ {
		// Apply permutation: Pb
		pb := make([]T, n)
		for i := 0; i < n; i++ {
			pb[i], _ = bMat.Get(piv[i], col)
		}

		// Forward substitution: Ly = Pb
		y := make([]T, n)
		for i := 0; i < n; i++ {
			y[i] = pb[i]
			for j := 0; j < i; j++ {
				lij, _ := l.Get(i, j)
				y[i] -= lij * y[j]
			}
		}

		// Back substitution: Ux = y
		xCol := make([]T, n)
		for i := n - 1; i >= 0; i-- {
			xCol[i] = y[i]
			for j := i + 1; j < n; j++ {
				uij, _ := u.Get(i, j)
				xCol[i] -= uij * xCol[j]
			}
			uii, _ := u.Get(i, i)
			xCol[i] /= uii
		}

		for i := 0; i < n; i++ {
			x.Set(xCol[i], i, col)
		}
	}

	if bIs1D {
		return x.Reshape(n)
	}
	return x, nil
}

// LstSq computes the least-squares solution to Ax ≈ b using the
// normal equations: x = (A^T A)^{-1} A^T b.
func LstSq[T sigma.Float](a, b *sigma.NDArray[T]) (*sigma.NDArray[T], error) {
	if a.Ndim() != 2 {
		return nil, &sigma.DimensionError{Msg: "LstSq: A must be 2-D"}
	}
	at, err := a.T()
	if err != nil {
		return nil, err
	}
	ata, err := MatMul(at, a)
	if err != nil {
		return nil, err
	}

	// Handle 1-D b
	bIs1D := b.Ndim() == 1
	var bMat *sigma.NDArray[T]
	if bIs1D {
		bMat, _ = b.Reshape(b.Shape()[0], 1)
	} else {
		bMat = b
	}

	atb, err := MatMul(at, bMat)
	if err != nil {
		return nil, err
	}
	result, err := Solve(ata, atb)
	if err != nil {
		return nil, err
	}
	if bIs1D {
		return result.Reshape(result.Shape()[0])
	}
	return result, nil
}
