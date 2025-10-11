package reval

import (
	"math"
	"sort"
)

func NDCG(predicted []string, relevance map[string]int, k int) float64 {
	// ideal
	var relv []int
	for _, v := range relevance {
		relv = append(relv, v)
	}
	sort.Slice(relv, func(i, j int) bool { return relv[i] > relv[j] })

	// top k
	ideal := relv[:min(k, len(relv))]

	// max DCG
	dcg := DCG(ideal)
	if IsZero(dcg) {
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

const (
	AbsTol = 1e-08
	RelTol = 1e-05
)

func IsZero(a float64) bool {
	return IsClose(a, 0.0)
}

func IsClose(a, b float64) bool {
	return isClose(a, b, AbsTol, RelTol)
}

func isClose(a, b float64, atol, rtol float64) bool {
	return math.Abs(a-b) <= atol+rtol*math.Max(math.Abs(a), math.Abs(b))
}
