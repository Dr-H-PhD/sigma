package rand

import (
	"math/rand/v2"

	"github.com/Dr-H-PhD/sigma"
)

// Source wraps a random number generator for reproducible array generation.
type Source struct {
	rng *rand.Rand
}

// NewSource creates a new random source with the given seed.
func NewSource(seed uint64) *Source {
	return &Source{rng: rand.New(rand.NewPCG(seed, seed^0xDEADBEEF))}
}

// defaultSource uses the global rand for non-seeded operations.
var defaultSource = &Source{}

func (s *Source) float64() float64 {
	if s.rng != nil {
		return s.rng.Float64()
	}
	return rand.Float64()
}

func (s *Source) normFloat64() float64 {
	if s.rng != nil {
		return s.rng.NormFloat64()
	}
	return rand.NormFloat64()
}

func (s *Source) intN(n int) int {
	if s.rng != nil {
		return s.rng.IntN(n)
	}
	return rand.IntN(n)
}

func (s *Source) shuffle(n int, swap func(i, j int)) {
	if s.rng != nil {
		for i := n - 1; i > 0; i-- {
			j := s.rng.IntN(i + 1)
			swap(i, j)
		}
		return
	}
	for i := n - 1; i > 0; i-- {
		j := rand.IntN(i + 1)
		swap(i, j)
	}
}

// fillArray creates an NDArray with the given shape and fills it using fn.
func fillArray[T sigma.Number](shape []int, fn func() T) *sigma.NDArray[T] {
	size := 1
	for _, s := range shape {
		size *= s
	}
	data := make([]T, size)
	for i := range data {
		data[i] = fn()
	}
	a, _ := sigma.New(data, shape...)
	return a
}
