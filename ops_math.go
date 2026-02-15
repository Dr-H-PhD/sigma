package sigma

import "math"

// Sqrt returns element-wise square root (Float types only).
func Sqrt[T Float](a *NDArray[T]) *NDArray[T] {
	return applyUnaryOp(a, func(v T) T { return T(math.Sqrt(float64(v))) })
}

// Exp returns element-wise e^x.
func Exp[T Float](a *NDArray[T]) *NDArray[T] {
	return applyUnaryOp(a, func(v T) T { return T(math.Exp(float64(v))) })
}

// Log returns element-wise natural logarithm.
func Log[T Float](a *NDArray[T]) *NDArray[T] {
	return applyUnaryOp(a, func(v T) T { return T(math.Log(float64(v))) })
}

// Log2 returns element-wise base-2 logarithm.
func Log2[T Float](a *NDArray[T]) *NDArray[T] {
	return applyUnaryOp(a, func(v T) T { return T(math.Log2(float64(v))) })
}

// Log10 returns element-wise base-10 logarithm.
func Log10[T Float](a *NDArray[T]) *NDArray[T] {
	return applyUnaryOp(a, func(v T) T { return T(math.Log10(float64(v))) })
}

// Sin returns element-wise sine.
func Sin[T Float](a *NDArray[T]) *NDArray[T] {
	return applyUnaryOp(a, func(v T) T { return T(math.Sin(float64(v))) })
}

// Cos returns element-wise cosine.
func Cos[T Float](a *NDArray[T]) *NDArray[T] {
	return applyUnaryOp(a, func(v T) T { return T(math.Cos(float64(v))) })
}

// Tan returns element-wise tangent.
func Tan[T Float](a *NDArray[T]) *NDArray[T] {
	return applyUnaryOp(a, func(v T) T { return T(math.Tan(float64(v))) })
}
