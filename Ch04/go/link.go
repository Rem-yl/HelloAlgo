package main

import (
	"fmt"
)

type ListNode struct {
	Val  int
	Next *ListNode
}

func (l *ListNode) data() []int {
	if l == nil {
		return nil
	}

	var res []int

	for l != nil {
		res = append(res, l.Val)
		l = l.Next
	}

	return res
}

func (l *ListNode) size() int {
	size := 0

	for l != nil {
		size++
		l = l.Next
	}

	return size
}

func (l *ListNode) insert(idx, num int) {
	size := l.size()
	if idx >= size {
		panic("索引越界")
	}

	newNode := NewLinkNode(num)
	cnt := 0
	for cnt < idx {
		l = l.Next
		cnt++
	}

	if idx == size-1 {
		l.Next = newNode
	} else {
		tmp := l.Next
		l.Next = newNode
		newNode.Next = tmp
	}
}

func (l *ListNode) delete(idx int) {
	size := l.size()
	if idx >= size {
		panic("索引越界")
	}

	cnt := 0
	for cnt < idx-1 {
		l = l.Next
		cnt++
	}

	if idx == size-1 {
		l.Next = nil
	} else {
		tmp := l.Next
		l.Next = tmp.Next
	}
}

func (l *ListNode) get(idx int) *ListNode {
	size := l.size()
	if idx >= size {
		panic("索引越界")
	}

	for i := 0; i < idx; i++ {
		l = l.Next
	}

	return l
}
func (l *ListNode) find(val int) int {
	if l == nil {
		return -1
	}

	cnt := 0
	for l != nil {
		if l.Val == val {
			return cnt
		}
		l = l.Next
		cnt++
	}

	return -1
}

func NewLinkNode(val int) *ListNode {
	return &ListNode{
		Val:  val,
		Next: nil,
	}
}

func NewLinkList(vals []int) *ListNode {
	if len(vals) <= 0 {
		return nil
	}

	head := &ListNode{vals[0], nil}

	curr := head

	for i := 1; i < len(vals); i++ {
		newNode := NewLinkNode(vals[i])
		curr.Next = newNode
		curr = curr.Next
	}

	return head
}

func main() {
	vals := []int{1, 2, 3, 4, 5}
	list := NewLinkList(vals)
	fmt.Println(list.data())

	list.insert(4, 100)
	fmt.Println(list.data())

	list.delete(2)
	fmt.Println(list.data())

	l := list.get(2)
	fmt.Println(l.Val)

	res := list.find(1000)
	fmt.Println(res)
}
