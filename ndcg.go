package reval

import (
	"math"
	"sort"
)

func NDCG(predicted []string, relevance map[string]int, k int) float64 {
	// ideal
	var ideal []int
	for _, v := range relevance {
		ideal = append(ideal, v)
	}
	sort.Slice(ideal, func(i, j int) bool { return ideal[i] > ideal[j] })

	// top k
	ideal = ideal[:min(k, len(ideal))]

	// max DCG
	dcg := DCG(ideal)
	if IsClose(dcg, 0.0) {
		return 0.0
	}

	// actual
	var relsK []int
	for i := 0; i < len(predicted) && i < k; i++ {
		relsK = append(relsK, relevance[predicted[i]])
	}

	return DCG(relsK) / dcg
}

func DCG(relevance []int) float64 {
	var s float64
	for i, rel := range relevance {
		s += (math.Pow(2, float64(rel)) - 1) / math.Log2(float64(i+2))
	}

	return s
}

func IsClose(a, b float64) bool {
	return isClose(a, b, 1e-08, 1e-05)
}

func isClose(a, b float64, atol, rtol float64) bool {
	return math.Abs(a-b) <= atol+rtol*math.Max(math.Abs(a), math.Abs(b))
}
