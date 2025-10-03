package reval

func Precision(predicted []string, relevance map[string]int, k int) float64 {
	if k == 0 {
		return 0.0
	}

	var hit int
	for i := 0; i < len(predicted) && i < k; i++ {
		v, ok := relevance[predicted[i]]
		if !ok {
			continue
		}

		if v > 0 {
			hit++
		}
	}

	return float64(hit) / float64(k)
}
