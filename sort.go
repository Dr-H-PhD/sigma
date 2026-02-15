package sigma

import "sort"

// Sort returns a sorted copy of a 1-D array in ascending order.
func Sort[T Number](a *NDArray[T]) (*NDArray[T], error) {
	if a.Ndim() != 1 {
		return nil, &DimensionError{Msg: "Sort requires a 1-D array"}
	}
	result := a.Copy()
	data := result.Data()
	sort.Slice(data, func(i, j int) bool {
		return data[i] < data[j]
	})
	return result, nil
}

// Argsort returns the indices that would sort a 1-D array.
func Argsort[T Number](a *NDArray[T]) (*NDArray[int64], error) {
	if a.Ndim() != 1 {
		return nil, &DimensionError{Msg: "Argsort requires a 1-D array"}
	}
	n := a.Size()
	indices := make([]int64, n)
	for i := range indices {
		indices[i] = int64(i)
	}
	sort.Slice(indices, func(i, j int) bool {
		vi, _ := a.At(int(indices[i]))
		vj, _ := a.At(int(indices[j]))
		return vi < vj
	})
	result, _ := New(indices, n)
	return result, nil
}
