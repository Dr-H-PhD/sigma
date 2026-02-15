package linalg

import (
	"fmt"

	"github.com/Dr-H-PhD/sigma"
)

// Dot computes the dot product of two 1-D arrays.
func Dot[T sigma.Float](a, b *sigma.NDArray[T]) (T, error) {
	if a.Ndim() != 1 || b.Ndim() != 1 {
		return 0, &sigma.DimensionError{Msg: "Dot requires 1-D arrays"}
	}
	if a.Shape()[0] != b.Shape()[0] {
		return 0, &sigma.ShapeError{Msg: fmt.Sprintf("dot product: shape mismatch %v vs %v", a.Shape(), b.Shape())}
	}
	var sum T
	n := a.Shape()[0]
	for i := 0; i < n; i++ {
		va, _ := a.Get(i)
		vb, _ := b.Get(i)
		sum += va * vb
	}
	return sum, nil
}

// MatMul computes the matrix product of two 2-D arrays.
// A is (m,k), B is (k,n), result is (m,n).
func MatMul[T sigma.Float](a, b *sigma.NDArray[T]) (*sigma.NDArray[T], error) {
	if a.Ndim() != 2 || b.Ndim() != 2 {
		return nil, &sigma.DimensionError{Msg: "MatMul requires 2-D arrays"}
	}
	aShape := a.Shape()
	bShape := b.Shape()
	m, k := aShape[0], aShape[1]
	k2, n := bShape[0], bShape[1]
	if k != k2 {
		return nil, &sigma.ShapeError{
			Msg: fmt.Sprintf("matmul: inner dimensions mismatch: (%d,%d) x (%d,%d)", m, k, k2, n),
		}
	}

	result := sigma.Zeros[T](m, n)

	// Tiled multiplication for cache friendliness
	const tileSize = 64
	for ii := 0; ii < m; ii += tileSize {
		for jj := 0; jj < n; jj += tileSize {
			for kk := 0; kk < k; kk += tileSize {
				iEnd := ii + tileSize
				if iEnd > m {
					iEnd = m
				}
				jEnd := jj + tileSize
				if jEnd > n {
					jEnd = n
				}
				kEnd := kk + tileSize
				if kEnd > k {
					kEnd = k
				}
				for i := ii; i < iEnd; i++ {
					for j := jj; j < jEnd; j++ {
						var sum T
						for p := kk; p < kEnd; p++ {
							va, _ := a.Get(i, p)
							vb, _ := b.Get(p, j)
							sum += va * vb
						}
						cur, _ := result.Get(i, j)
						result.Set(cur+sum, i, j)
					}
				}
			}
		}
	}
	return result, nil
}

// Inner computes the inner product of two 1-D arrays (same as Dot).
func Inner[T sigma.Float](a, b *sigma.NDArray[T]) (T, error) {
	return Dot(a, b)
}

// Outer computes the outer product of two 1-D arrays.
// Result is an (m,n) matrix where result[i,j] = a[i] * b[j].
func Outer[T sigma.Float](a, b *sigma.NDArray[T]) (*sigma.NDArray[T], error) {
	if a.Ndim() != 1 || b.Ndim() != 1 {
		return nil, &sigma.DimensionError{Msg: "Outer requires 1-D arrays"}
	}
	m := a.Shape()[0]
	n := b.Shape()[0]
	result := sigma.Zeros[T](m, n)
	for i := 0; i < m; i++ {
		va, _ := a.Get(i)
		for j := 0; j < n; j++ {
			vb, _ := b.Get(j)
			result.Set(va*vb, i, j)
		}
	}
	return result, nil
}
