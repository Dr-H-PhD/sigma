package rand

import (
	"github.com/Dr-H-PhD/sigma"
)

// Uniform returns an NDArray of uniform random values in [low, high).
func Uniform[T sigma.Float](s *Source, low, high T, shape ...int) *sigma.NDArray[T] {
	if s == nil {
		s = defaultSource
	}
	return fillArray[T](shape, func() T {
		return low + T(s.float64())*(high-low)
	})
}

// Normal returns an NDArray of normally distributed values
// with given mean and standard deviation.
func Normal[T sigma.Float](s *Source, mean, std T, shape ...int) *sigma.NDArray[T] {
	if s == nil {
		s = defaultSource
	}
	return fillArray[T](shape, func() T {
		return mean + T(s.normFloat64())*std
	})
}

// Randint returns an NDArray of random integers in [low, high).
func Randint(s *Source, low, high int64, shape ...int) *sigma.NDArray[int64] {
	if s == nil {
		s = defaultSource
	}
	rang := high - low
	return fillArray[int64](shape, func() int64 {
		return low + int64(s.intN(int(rang)))
	})
}
