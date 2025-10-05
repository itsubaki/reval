package reval

func ROUGE1(candidates, refs []string) (precision, recall, f1 float64) {
	if len(candidates) == 0 || len(refs) == 0 {
		return 0, 0, 0
	}

	count := make(map[string]int, len(refs))
	for _, token := range refs {
		count[token]++
	}

	var overlap int
	for _, token := range candidates {
		if count[token] < 1 {
			continue
		}

		overlap++
		count[token]--
	}

	// score
	precision = float64(overlap) / float64(len(candidates))
	recall = float64(overlap) / float64(len(refs))
	if precision+recall > 0 {
		f1 = FBeta(precision, recall, 1.0)
	}

	return
}

func FBeta(precision, recall, beta float64) float64 {
	if precision == 0 && recall == 0 {
		return 0
	}

	beta2 := beta * beta
	return (1 + beta2) * precision * recall / (beta2*precision + recall)
}
