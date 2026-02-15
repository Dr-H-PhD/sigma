package sigma

// Number is the constraint for all supported numeric types.
type Number interface {
	~int32 | ~int64 | ~float32 | ~float64
}

// Float is the constraint for floating-point types (used by math operations).
type Float interface {
	~float32 | ~float64
}
