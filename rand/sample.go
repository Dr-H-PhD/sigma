package rand

import (
	"github.com/Dr-H-PhD/sigma"
)

// Shuffle returns a new 1-D array with elements randomly reordered.
func Shuffle[T sigma.Number](s *Source, a *sigma.NDArray[T]) *sigma.NDArray[T] {
	if s == nil {
		s = defaultSource
	}
	result := a.Flatten().Copy()
	n := result.Size()
	data := result.Data()
	s.shuffle(n, func(i, j int) {
		data[i], data[j] = data[j], data[i]
	})
	return result
}

// Choice returns an array of random samples from a 1-D array.
// If replace is true, sampling is with replacement.
func Choice[T sigma.Number](s *Source, a *sigma.NDArray[T], n int, replace bool) (*sigma.NDArray[T], error) {
	if s == nil {
		s = defaultSource
	}
	size := a.Size()
	if !replace && n > size {
		return nil, &sigma.ShapeError{Msg: "cannot take more samples than population without replacement"}
	}

	data := make([]T, n)
	if replace {
		for i := 0; i < n; i++ {
			idx := s.intN(size)
			data[i], _ = a.At(idx)
		}
	} else {
		// Fisher-Yates partial shuffle
		indices := make([]int, size)
		for i := range indices {
			indices[i] = i
		}
		for i := 0; i < n; i++ {
			j := i + s.intN(size-i)
			indices[i], indices[j] = indices[j], indices[i]
			data[i], _ = a.At(indices[i])
		}
	}
	result, _ := sigma.New(data, n)
	return result, nil
}

// Permutation returns a random permutation of integers [0, n).
func Permutation(s *Source, n int) *sigma.NDArray[int64] {
	if s == nil {
		s = defaultSource
	}
	data := make([]int64, n)
	for i := range data {
		data[i] = int64(i)
	}
	s.shuffle(n, func(i, j int) {
		data[i], data[j] = data[j], data[i]
	})
	result, _ := sigma.New(data, n)
	return result
}
