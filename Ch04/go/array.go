package main

import (
	"fmt"
	"math/rand"
)

// 随机访问元素
func randomAccess(arr []int) int {
	idx := rand.Intn(len(arr))
	return arr[idx]
}

// 插入元素
func insert(arr []int, num int, idx int) {
	for i := len(arr) - 1; i > idx; i-- {
		arr[i] = arr[i-1]
	}

	arr[idx] = num
}

// 删除元素
func delete(arr []int, idx int) {
	for i := idx; i < len(arr)-1; i++ {
		arr[i] = arr[i+1]
	}
}

// 查找元素
func find(arr []int, num int) bool {
	for i := range arr {
		if i == num {
			return true
		}
	}

	return false
}

func main() {
	arr := []int{1, 2, 3, 4, 5, 6, 7, 8, 9}
	num1 := randomAccess(arr)
	fmt.Println(num1)

	insert(arr, 100, 3)
	fmt.Println(arr)

	delete(arr, 3)
	fmt.Println(arr)

	res := find(arr, 10)
	fmt.Println(res)

}
