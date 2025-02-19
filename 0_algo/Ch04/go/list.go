package main

type List struct {
	Arr      []int
	Capacity int
	Size     int
}

func newList() *List {
	return &List{
		Arr:      []int{},
		Capacity: 10,
		Size:     0,
	}
}

func (l *List) expandCapacity() {
	appendArr := make([]int, l.Capacity)
	l.Arr = append(l.Arr, appendArr...)
	l.Capacity = len(l.Arr)
}
