package sigma

import "math"

// AddScalar returns a + scalar for each element.
func AddScalar[T Number](a *NDArray[T], s T) *NDArray[T] {
	return applyScalarOp(a, s, func(x, y T) T { return x + y })
}

// SubScalar returns a - scalar for each element.
func SubScalar[T Number](a *NDArray[T], s T) *NDArray[T] {
	return applyScalarOp(a, s, func(x, y T) T { return x - y })
}

// MulScalar returns a * scalar for each element.
func MulScalar[T Number](a *NDArray[T], s T) *NDArray[T] {
	return applyScalarOp(a, s, func(x, y T) T { return x * y })
}

// DivScalar returns a / scalar for each element.
func DivScalar[T Number](a *NDArray[T], s T) *NDArray[T] {
	return applyScalarOp(a, s, func(x, y T) T { return x / y })
}

// PowScalar returns a ** scalar for each element (float types).
func PowScalar[T Float](a *NDArray[T], s T) *NDArray[T] {
	return applyUnaryOp(a, func(v T) T {
		return T(math.Pow(float64(v), float64(s)))
	})
}
