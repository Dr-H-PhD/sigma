package linalg

import (
	"math"

	"github.com/Dr-H-PhD/sigma"
)

// Rank computes the rank of a matrix via SVD.
// Singular values below tol are treated as zero.
func Rank[T sigma.Float](a *sigma.NDArray[T], tol float64) (int, error) {
	_, s, _, err := SVD(a)
	if err != nil {
		return 0, err
	}
	rank := 0
	n := s.Shape()[0]
	for i := 0; i < n; i++ {
		v, _ := s.Get(i)
		if math.Abs(float64(v)) > tol {
			rank++
		}
	}
	return rank, nil
}

// Cond computes the condition number of a matrix (ratio of largest
// to smallest singular value).
func Cond[T sigma.Float](a *sigma.NDArray[T]) (T, error) {
	_, s, _, err := SVD(a)
	if err != nil {
		return 0, err
	}
	n := s.Shape()[0]
	if n == 0 {
		return 0, &sigma.ShapeError{Msg: "cannot compute condition number of empty matrix"}
	}
	sMax, _ := s.Get(0)
	sMin, _ := s.Get(n - 1)
	if math.Abs(float64(sMin)) < 1e-15 {
		return T(math.Inf(1)), nil
	}
	return sMax / sMin, nil
}
