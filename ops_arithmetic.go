package sigma

import "math"

// Add returns element-wise a + b with broadcasting.
func Add[T Number](a, b *NDArray[T]) (*NDArray[T], error) {
	return applyBinaryOp(a, b, func(x, y T) T { return x + y })
}

// Sub returns element-wise a - b with broadcasting.
func Sub[T Number](a, b *NDArray[T]) (*NDArray[T], error) {
	return applyBinaryOp(a, b, func(x, y T) T { return x - y })
}

// Mul returns element-wise a * b with broadcasting.
func Mul[T Number](a, b *NDArray[T]) (*NDArray[T], error) {
	return applyBinaryOp(a, b, func(x, y T) T { return x * y })
}

// Div returns element-wise a / b with broadcasting.
func Div[T Number](a, b *NDArray[T]) (*NDArray[T], error) {
	return applyBinaryOp(a, b, func(x, y T) T { return x / y })
}

// Mod returns element-wise a % b with broadcasting (integer types only).
func Mod(a, b *NDArray[int64]) (*NDArray[int64], error) {
	return applyBinaryOp(a, b, func(x, y int64) int64 { return x % y })
}

// Mod32 returns element-wise a % b with broadcasting (int32).
func Mod32(a, b *NDArray[int32]) (*NDArray[int32], error) {
	return applyBinaryOp(a, b, func(x, y int32) int32 { return x % y })
}

// Pow returns element-wise a ** b with broadcasting (float types).
func Pow[T Float](a, b *NDArray[T]) (*NDArray[T], error) {
	return applyBinaryOp(a, b, func(x, y T) T {
		return T(math.Pow(float64(x), float64(y)))
	})
}

// Neg returns element-wise -a.
func Neg[T Number](a *NDArray[T]) *NDArray[T] {
	return applyUnaryOp(a, func(v T) T { return -v })
}

// Abs returns element-wise |a|.
func Abs[T Number](a *NDArray[T]) *NDArray[T] {
	return applyUnaryOp(a, func(v T) T {
		if v < 0 {
			return -v
		}
		return v
	})
}
