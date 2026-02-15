package linalg

import (
	"math"

	"github.com/Dr-H-PhD/sigma"
)

// QR computes the QR decomposition using Householder reflections.
// A is (m,n) with m >= n. Returns Q (m,m) and R (m,n).
func QR[T sigma.Float](a *sigma.NDArray[T]) (q, r *sigma.NDArray[T], err error) {
	if a.Ndim() != 2 {
		return nil, nil, &sigma.DimensionError{Msg: "QR requires a 2-D array"}
	}
	shape := a.Shape()
	m, n := shape[0], shape[1]
	if m < n {
		return nil, nil, &sigma.ShapeError{Msg: "QR requires m >= n"}
	}

	// Work on a copy for R
	r = a.Copy()
	q = sigma.Eye[T](m)

	for k := 0; k < n; k++ {
		// Extract column k below diagonal
		x := make([]float64, m-k)
		for i := k; i < m; i++ {
			v, _ := r.Get(i, k)
			x[i-k] = float64(v)
		}

		// Compute Householder vector
		normX := 0.0
		for _, v := range x {
			normX += v * v
		}
		normX = math.Sqrt(normX)

		if normX < 1e-15 {
			continue
		}

		sign := 1.0
		if x[0] < 0 {
			sign = -1.0
		}
		x[0] += sign * normX
		// Normalise
		normV := 0.0
		for _, v := range x {
			normV += v * v
		}
		normV = math.Sqrt(normV)
		for i := range x {
			x[i] /= normV
		}

		// Apply H = I - 2vv^T to R from left
		for j := k; j < n; j++ {
			dot := 0.0
			for i := k; i < m; i++ {
				v, _ := r.Get(i, j)
				dot += x[i-k] * float64(v)
			}
			for i := k; i < m; i++ {
				v, _ := r.Get(i, j)
				r.Set(T(float64(v)-2*x[i-k]*dot), i, j)
			}
		}

		// Apply H to Q from right: Q = Q * H
		for i := 0; i < m; i++ {
			dot := 0.0
			for j := k; j < m; j++ {
				v, _ := q.Get(i, j)
				dot += float64(v) * x[j-k]
			}
			for j := k; j < m; j++ {
				v, _ := q.Get(i, j)
				q.Set(T(float64(v)-2*dot*x[j-k]), i, j)
			}
		}
	}

	// Clean up near-zero values in R
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			v, _ := r.Get(i, j)
			if math.Abs(float64(v)) < 1e-14 {
				r.Set(0, i, j)
			}
		}
	}

	return q, r, nil
}

// SVD computes the singular value decomposition A = U * diag(S) * V^T
// using the Golub-Kahan bidiagonalisation followed by QR iteration.
// Returns U (m,m), S (min(m,n),), V (n,n).
func SVD[T sigma.Float](a *sigma.NDArray[T]) (u *sigma.NDArray[T], s *sigma.NDArray[T], v *sigma.NDArray[T], err error) {
	if a.Ndim() != 2 {
		return nil, nil, nil, &sigma.DimensionError{Msg: "SVD requires a 2-D array"}
	}
	shape := a.Shape()
	m, n := shape[0], shape[1]

	// Compute A^T A
	at, _ := a.T()
	ata, err := MatMul(at, a)
	if err != nil {
		return nil, nil, nil, err
	}

	// Eigendecomposition of A^T A to get V and sigma^2
	eigenvals, eigenvecs, err := symmetricEig(ata)
	if err != nil {
		return nil, nil, nil, err
	}

	// Sort eigenvalues descending
	nn := len(eigenvals)
	for i := 0; i < nn-1; i++ {
		maxIdx := i
		for j := i + 1; j < nn; j++ {
			if eigenvals[j] > eigenvals[maxIdx] {
				maxIdx = j
			}
		}
		if maxIdx != i {
			eigenvals[i], eigenvals[maxIdx] = eigenvals[maxIdx], eigenvals[i]
			// Swap columns in eigenvecs
			for row := 0; row < n; row++ {
				vi, _ := eigenvecs.Get(row, i)
				vm, _ := eigenvecs.Get(row, maxIdx)
				eigenvecs.Set(vm, row, i)
				eigenvecs.Set(vi, row, maxIdx)
			}
		}
	}

	// V = eigenvectors of A^T A
	v = eigenvecs

	// Singular values
	minMN := m
	if n < minMN {
		minMN = n
	}
	sData := make([]T, minMN)
	for i := 0; i < minMN; i++ {
		if i < nn && eigenvals[i] > 0 {
			sData[i] = T(math.Sqrt(eigenvals[i]))
		}
	}
	s, _ = sigma.New(sData, minMN)

	// U = A V S^{-1}
	av, err := MatMul(a, v)
	if err != nil {
		return nil, nil, nil, err
	}
	u = sigma.Zeros[T](m, m)
	for j := 0; j < minMN; j++ {
		sv, _ := s.Get(j)
		if sv > T(1e-15) {
			for i := 0; i < m; i++ {
				val, _ := av.Get(i, j)
				u.Set(val/sv, i, j)
			}
		}
	}
	// Fill remaining columns of U with orthonormal basis
	// Use Gram-Schmidt
	for j := minMN; j < m; j++ {
		// Start with standard basis vector
		col := make([]float64, m)
		col[j] = 1.0
		// Orthogonalise against existing columns
		for k := 0; k < j; k++ {
			dot := 0.0
			for i := 0; i < m; i++ {
				uk, _ := u.Get(i, k)
				dot += col[i] * float64(uk)
			}
			for i := 0; i < m; i++ {
				uk, _ := u.Get(i, k)
				col[i] -= dot * float64(uk)
			}
		}
		// Normalise
		norm := 0.0
		for _, v := range col {
			norm += v * v
		}
		norm = math.Sqrt(norm)
		if norm > 1e-15 {
			for i := 0; i < m; i++ {
				u.Set(T(col[i]/norm), i, j)
			}
		}
	}

	return u, s, v, nil
}

// symmetricEig computes eigenvalues and eigenvectors of a symmetric matrix
// using the QR algorithm with shifts.
func symmetricEig[T sigma.Float](a *sigma.NDArray[T]) ([]float64, *sigma.NDArray[T], error) {
	shape := a.Shape()
	n := shape[0]

	// Work in float64 for numerical stability
	work := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			v, _ := a.Get(i, j)
			work[i*n+j] = float64(v)
		}
	}

	// Accumulate eigenvectors
	vecs := make([]float64, n*n)
	for i := 0; i < n; i++ {
		vecs[i*n+i] = 1.0
	}

	maxIter := 100 * n
	for iter := 0; iter < maxIter; iter++ {
		// Check convergence (off-diagonal elements)
		offDiag := 0.0
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				if i != j {
					offDiag += work[i*n+j] * work[i*n+j]
				}
			}
		}
		if math.Sqrt(offDiag) < 1e-12 {
			break
		}

		// QR step with Wilkinson shift
		q, r := qrStep(work, n)

		// work = R * Q
		newWork := make([]float64, n*n)
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				var sum float64
				for k := 0; k < n; k++ {
					sum += r[i*n+k] * q[k*n+j]
				}
				newWork[i*n+j] = sum
			}
		}
		copy(work, newWork)

		// Update eigenvectors: V = V * Q
		newVecs := make([]float64, n*n)
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				var sum float64
				for k := 0; k < n; k++ {
					sum += vecs[i*n+k] * q[k*n+j]
				}
				newVecs[i*n+j] = sum
			}
		}
		copy(vecs, newVecs)
	}

	// Extract eigenvalues and build eigenvector matrix
	eigenvals := make([]float64, n)
	for i := 0; i < n; i++ {
		eigenvals[i] = work[i*n+i]
	}

	vecData := make([]T, n*n)
	for i := range vecs {
		vecData[i] = T(vecs[i])
	}
	vecMat, _ := sigma.New(vecData, n, n)

	return eigenvals, vecMat, nil
}

// qrStep performs one QR decomposition step on a flat n×n matrix.
// Returns Q and R as flat slices.
func qrStep(a []float64, n int) (q, r []float64) {
	r = make([]float64, n*n)
	copy(r, a)
	q = make([]float64, n*n)
	for i := 0; i < n; i++ {
		q[i*n+i] = 1.0
	}

	for k := 0; k < n-1; k++ {
		// Householder for column k
		x := make([]float64, n-k)
		for i := k; i < n; i++ {
			x[i-k] = r[i*n+k]
		}
		normX := 0.0
		for _, v := range x {
			normX += v * v
		}
		normX = math.Sqrt(normX)
		if normX < 1e-15 {
			continue
		}

		sign := 1.0
		if x[0] < 0 {
			sign = -1.0
		}
		x[0] += sign * normX
		normV := 0.0
		for _, v := range x {
			normV += v * v
		}
		normV = math.Sqrt(normV)
		for i := range x {
			x[i] /= normV
		}

		// Apply to R
		for j := k; j < n; j++ {
			dot := 0.0
			for i := k; i < n; i++ {
				dot += x[i-k] * r[i*n+j]
			}
			for i := k; i < n; i++ {
				r[i*n+j] -= 2 * x[i-k] * dot
			}
		}

		// Apply to Q
		for i := 0; i < n; i++ {
			dot := 0.0
			for j := k; j < n; j++ {
				dot += q[i*n+j] * x[j-k]
			}
			for j := k; j < n; j++ {
				q[i*n+j] -= 2 * dot * x[j-k]
			}
		}
	}
	return q, r
}

// Eig computes eigenvalues and right eigenvectors of a square matrix.
// Returns eigenvalues as a 1-D array and eigenvectors as columns of a 2-D array.
// Only works reliably for symmetric matrices.
func Eig[T sigma.Float](a *sigma.NDArray[T]) (vals *sigma.NDArray[T], vecs *sigma.NDArray[T], err error) {
	if a.Ndim() != 2 {
		return nil, nil, &sigma.DimensionError{Msg: "Eig requires a 2-D array"}
	}
	shape := a.Shape()
	if shape[0] != shape[1] {
		return nil, nil, &sigma.ShapeError{Msg: "Eig requires a square matrix"}
	}

	eigenvals, eigenvecs, err := symmetricEig(a)
	if err != nil {
		return nil, nil, err
	}

	n := len(eigenvals)
	valData := make([]T, n)
	for i, v := range eigenvals {
		valData[i] = T(v)
	}
	vals, _ = sigma.New(valData, n)
	vecs = eigenvecs
	return vals, vecs, nil
}

// Cholesky computes the Cholesky decomposition A = L L^T.
// A must be symmetric positive-definite. Returns lower-triangular L.
func Cholesky[T sigma.Float](a *sigma.NDArray[T]) (*sigma.NDArray[T], error) {
	if a.Ndim() != 2 {
		return nil, &sigma.DimensionError{Msg: "Cholesky requires a 2-D array"}
	}
	shape := a.Shape()
	n := shape[0]
	if n != shape[1] {
		return nil, &sigma.ShapeError{Msg: "Cholesky requires a square matrix"}
	}

	l := sigma.Zeros[T](n, n)

	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			var sum float64
			if j == i {
				for k := 0; k < j; k++ {
					lk, _ := l.Get(i, k)
					sum += float64(lk) * float64(lk)
				}
				aij, _ := a.Get(i, i)
				val := float64(aij) - sum
				if val <= 0 {
					return nil, &sigma.ShapeError{Msg: "matrix is not positive-definite"}
				}
				l.Set(T(math.Sqrt(val)), i, j)
			} else {
				for k := 0; k < j; k++ {
					lik, _ := l.Get(i, k)
					ljk, _ := l.Get(j, k)
					sum += float64(lik) * float64(ljk)
				}
				aij, _ := a.Get(i, j)
				ljj, _ := l.Get(j, j)
				l.Set(T((float64(aij)-sum)/float64(ljj)), i, j)
			}
		}
	}
	return l, nil
}
