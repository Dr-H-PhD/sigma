package linalg

import (
	"math"

	"github.com/Dr-H-PhD/sigma"
)

// Inv computes the inverse of a square matrix using LU decomposition.
func Inv[T sigma.Float](a *sigma.NDArray[T]) (*sigma.NDArray[T], error) {
	if a.Ndim() != 2 {
		return nil, &sigma.DimensionError{Msg: "Inv requires a 2-D array"}
	}
	shape := a.Shape()
	n := shape[0]
	if n != shape[1] {
		return nil, &sigma.ShapeError{Msg: "Inv requires a square matrix"}
	}
	identity := sigma.Eye[T](n)
	return Solve(a, identity)
}

// Det computes the determinant of a square matrix.
func Det[T sigma.Float](a *sigma.NDArray[T]) (T, error) {
	if a.Ndim() != 2 {
		return 0, &sigma.DimensionError{Msg: "Det requires a 2-D array"}
	}
	shape := a.Shape()
	n := shape[0]
	if n != shape[1] {
		return 0, &sigma.ShapeError{Msg: "Det requires a square matrix"}
	}

	_, u, piv, err := LU(a)
	if err != nil {
		// Singular → det = 0
		if _, ok := err.(*sigma.SingularMatrixError); ok {
			return 0, nil
		}
		return 0, err
	}

	det := T(1)
	for i := 0; i < n; i++ {
		v, _ := u.Get(i, i)
		det *= v
	}

	// Count swaps in permutation
	swaps := 0
	seen := make([]bool, n)
	for i := 0; i < n; i++ {
		if !seen[i] {
			seen[i] = true
			j := piv[i]
			for j != i {
				swaps++
				seen[j] = true
				j = piv[j]
			}
		}
	}
	if swaps%2 == 1 {
		det = -det
	}
	return det, nil
}

// Trace computes the sum of diagonal elements.
func Trace[T sigma.Float](a *sigma.NDArray[T]) (T, error) {
	if a.Ndim() != 2 {
		return 0, &sigma.DimensionError{Msg: "Trace requires a 2-D array"}
	}
	shape := a.Shape()
	n := shape[0]
	if shape[1] < n {
		n = shape[1]
	}
	var sum T
	for i := 0; i < n; i++ {
		v, _ := a.Get(i, i)
		sum += v
	}
	return sum, nil
}

// Norm computes the matrix or vector norm.
// For vectors (1-D):
//
//	ord=1: L1, ord=2: L2 (Euclidean), ord=math.Inf(1): max(|x|)
//
// For matrices (2-D):
//
//	ord="fro" (Frobenius) is the default.
//
// The ord parameter is float64 for simplicity.
func Norm[T sigma.Float](a *sigma.NDArray[T], ord float64) (T, error) {
	if a.Ndim() == 1 {
		return vectorNorm(a, ord)
	}
	if a.Ndim() == 2 {
		return matrixNorm(a, ord)
	}
	return 0, &sigma.DimensionError{Msg: "Norm requires a 1-D or 2-D array"}
}

func vectorNorm[T sigma.Float](a *sigma.NDArray[T], ord float64) (T, error) {
	n := a.Shape()[0]
	if math.IsInf(ord, 1) {
		var max T
		for i := 0; i < n; i++ {
			v, _ := a.Get(i)
			av := T(math.Abs(float64(v)))
			if av > max {
				max = av
			}
		}
		return max, nil
	}
	if math.IsInf(ord, -1) {
		var min T
		first := true
		for i := 0; i < n; i++ {
			v, _ := a.Get(i)
			av := T(math.Abs(float64(v)))
			if first || av < min {
				min = av
				first = false
			}
		}
		return min, nil
	}
	var sum float64
	for i := 0; i < n; i++ {
		v, _ := a.Get(i)
		sum += math.Pow(math.Abs(float64(v)), ord)
	}
	return T(math.Pow(sum, 1.0/ord)), nil
}

func matrixNorm[T sigma.Float](a *sigma.NDArray[T], ord float64) (T, error) {
	shape := a.Shape()
	m, n := shape[0], shape[1]

	// Frobenius norm (default when ord == 0 or unspecified)
	if ord == 0 || (ord != 1 && !math.IsInf(ord, 1)) {
		var sum float64
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				v, _ := a.Get(i, j)
				sum += float64(v) * float64(v)
			}
		}
		return T(math.Sqrt(sum)), nil
	}

	if ord == 1 {
		// Max column sum
		var maxSum float64
		for j := 0; j < n; j++ {
			var colSum float64
			for i := 0; i < m; i++ {
				v, _ := a.Get(i, j)
				colSum += math.Abs(float64(v))
			}
			if colSum > maxSum {
				maxSum = colSum
			}
		}
		return T(maxSum), nil
	}

	if math.IsInf(ord, 1) {
		// Max row sum
		var maxSum float64
		for i := 0; i < m; i++ {
			var rowSum float64
			for j := 0; j < n; j++ {
				v, _ := a.Get(i, j)
				rowSum += math.Abs(float64(v))
			}
			if rowSum > maxSum {
				maxSum = rowSum
			}
		}
		return T(maxSum), nil
	}

	return 0, &sigma.ShapeError{Msg: "unsupported norm order"}
}
