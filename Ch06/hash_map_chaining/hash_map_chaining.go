package main

import "fmt"

type pair struct {
	key   int
	value string
}

var loadRate float32 = 0.75
var defaultBucketSize = 4
var defaultChainSize = 4

type hashMap struct {
	buckets [][]pair

	size     int
	capacity int
}

func newHashMap() *hashMap {
	buckets := make([][]pair, defaultBucketSize)
	for i := 0; i < defaultBucketSize; i++ {
		buckets[i] = make([]pair, 0)
	}

	return &hashMap{
		buckets:  buckets,
		size:     0,
		capacity: defaultBucketSize,
	}
}

func (h *hashMap) hashFunc(key int) int {
	return key % h.capacity
}

func (h *hashMap) loadFactor() float32 {
	return float32(h.size) / float32(h.capacity)
}

func (h *hashMap) get(key int) (string, error) {
	idx := h.hashFunc(key)
	bucket := h.buckets[idx]

	for _, p := range bucket {
		if p.key == key {
			return p.value, nil
		}
	}

	return "", fmt.Errorf("key: %d not found", key)
}

func (h *hashMap) set(key int, value string) error {
	idx := h.hashFunc(key)
	bucket := h.buckets[idx]

	for i := 0; i < len(bucket); i++ {
		if bucket[i].key == key {
			bucket[i].value = value
			return nil
		}
	}

	return fmt.Errorf("key: %d not found", key)
}

func (h *hashMap) add(key int, value string) error {
	if h.size == 0 {
		idx := h.hashFunc(key)
		h.buckets[idx] = append(h.buckets[idx], pair{key: key, value: value})
		h.size += 1
		return nil
	}

	if h.loadFactor() >= loadRate {
		h.extend()
	}

	idx := h.hashFunc(key)
	bucket := h.buckets[idx]
	for _, p := range bucket {
		if p.key == key {
			panic(fmt.Sprintf("key: %d has existed!", key))
		}
	}

	h.buckets[idx] = append(h.buckets[idx], pair{key: key, value: value})
	h.size += 1
	return nil
}

func (h *hashMap) remove(key int) error {
	if h.size == 0 {
		return fmt.Errorf("hash map is empty")
	}

	idx := h.hashFunc(key)
	bucket := h.buckets[idx]

	for i := 0; i < len(bucket); i++ {
		if bucket[i].key == key {
			bucket = append(bucket[:i], bucket[i+1:]...)
		}
	}

	h.size -= 1

	return nil
}

func (h *hashMap) extend() {
	tmp := make([][]pair, h.capacity)
	for i := 0; i < len(h.buckets); i++ {
		tmp[i] = make([]pair, 0)
		copy(tmp[i], h.buckets[i])
	}

	h.capacity = h.capacity * 2
	h.buckets = make([][]pair, h.capacity)
	for _, bucket := range tmp {
		for _, p := range bucket {
			h.add(p.key, p.value)
		}
	}
}

func main() {
	h := newHashMap()
	for i := 0; i < 10; i++ {
		h.add(i, "rem")
	}

	fmt.Print(h)
}
