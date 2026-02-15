package sigma

import "math"

// Clip limits values to the range [min, max].
func Clip[T Number](a *NDArray[T], lo, hi T) *NDArray[T] {
	return applyUnaryOp(a, func(v T) T {
		if v < lo {
			return lo
		}
		if v > hi {
			return hi
		}
		return v
	})
}

// Round rounds each element to the nearest integer (Float types).
func Round[T Float](a *NDArray[T]) *NDArray[T] {
	return applyUnaryOp(a, func(v T) T {
		return T(math.Round(float64(v)))
	})
}

// Floor returns the floor of each element (Float types).
func Floor[T Float](a *NDArray[T]) *NDArray[T] {
	return applyUnaryOp(a, func(v T) T {
		return T(math.Floor(float64(v)))
	})
}

// Ceil returns the ceiling of each element (Float types).
func Ceil[T Float](a *NDArray[T]) *NDArray[T] {
	return applyUnaryOp(a, func(v T) T {
		return T(math.Ceil(float64(v)))
	})
}
