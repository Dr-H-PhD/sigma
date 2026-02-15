package sigma

// Equal returns element-wise a == b as 0/1 values.
func Equal[T Number](a, b *NDArray[T]) (*NDArray[T], error) {
	return applyBinaryOp(a, b, func(x, y T) T {
		if x == y {
			return 1
		}
		return 0
	})
}

// NotEqual returns element-wise a != b as 0/1 values.
func NotEqual[T Number](a, b *NDArray[T]) (*NDArray[T], error) {
	return applyBinaryOp(a, b, func(x, y T) T {
		if x != y {
			return 1
		}
		return 0
	})
}

// Less returns element-wise a < b as 0/1 values.
func Less[T Number](a, b *NDArray[T]) (*NDArray[T], error) {
	return applyBinaryOp(a, b, func(x, y T) T {
		if x < y {
			return 1
		}
		return 0
	})
}

// LessEqual returns element-wise a <= b as 0/1 values.
func LessEqual[T Number](a, b *NDArray[T]) (*NDArray[T], error) {
	return applyBinaryOp(a, b, func(x, y T) T {
		if x <= y {
			return 1
		}
		return 0
	})
}

// Greater returns element-wise a > b as 0/1 values.
func Greater[T Number](a, b *NDArray[T]) (*NDArray[T], error) {
	return applyBinaryOp(a, b, func(x, y T) T {
		if x > y {
			return 1
		}
		return 0
	})
}

// GreaterEqual returns element-wise a >= b as 0/1 values.
func GreaterEqual[T Number](a, b *NDArray[T]) (*NDArray[T], error) {
	return applyBinaryOp(a, b, func(x, y T) T {
		if x >= y {
			return 1
		}
		return 0
	})
}
