package reval

// ROUGE1 returns the ROUGE-1 score (unigram overlap)
func ROUGE1(candidates, refs []string) (precision, recall, f1 float64) {
	return rouge(candidates, refs, 1.0, Overlap)
}

// ROUGEL returns the ROUGE-L score (Longest Common Subsequence)
func ROUGEL(candidates, refs []string) (precision, recall, f1 float64) {
	return rouge(candidates, refs, 1.0, LCS)
}

func rouge(candidates, refs []string, beta float64, f func(a, b []string) int) (precision, recall, fbeta float64) {
	if len(candidates) == 0 || len(refs) == 0 {
		return
	}

	// overlap, LCS, ...
	matched := f(candidates, refs)

	// score
	precision = float64(matched) / float64(len(candidates))
	recall = float64(matched) / float64(len(refs))
	fbeta = FBeta(precision, recall, beta)
	return
}

// Overlap returns the count of overlapping tokens between a and b.
func Overlap(a, b []string) int {
	count := make(map[string]int, len(a))
	for _, token := range a {
		count[token]++
	}

	var overlap int
	for _, token := range b {
		if count[token] < 1 {
			continue
		}

		overlap++
		count[token]--
	}

	return overlap
}

// LCS returns the length of the Longest Common Subsequence between a and b.
func LCS(a, b []string) int {
	m, n := len(a), len(b)

	// dynamic programming table
	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}

	// fill dp table
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if a[i-1] == b[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
				continue
			}

			dp[i][j] = max(dp[i-1][j], dp[i][j-1])
		}
	}

	return dp[m][n]
}

// F1 returns the F1 score given precision and recall.
func F1(precision, recall float64) float64 {
	return FBeta(precision, recall, 1.0)
}

// FBeta returns the F-beta score given precision and recall.
func FBeta(precision, recall, beta float64) float64 {
	if precision == 0 && recall == 0 {
		return 0
	}

	beta2 := beta * beta
	return (1 + beta2) * precision * recall / (beta2*precision + recall)
}
