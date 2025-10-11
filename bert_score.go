package reval

import "math"

// BERTScore returns the BERTScore between candidates and refs.
func BERTScore(candidates, refs [][]float64) (precision, recall, f1 float64) {
	if len(candidates) == 0 || len(refs) == 0 {
		return
	}

	precision = maxsumavg(candidates, refs)
	recall = maxsumavg(refs, candidates)
	f1 = F1(precision, recall)
	return
}

// maxsumavg returns the average of maximum similarity scores from each vector in a to vectors in b.
func maxsumavg(a, b [][]float64) float64 {
	var sum float64
	for i := range a {
		max := -math.MaxFloat64
		for j := range b {
			s := DotProduct(a[i], b[j])
			if s > max {
				max = s
			}
		}

		sum += max
	}

	return sum / float64(len(a))
}

// DotProduct returns the dot product of two vectors a and b.
func DotProduct(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}

	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}

	return sum
}

// L2Norm returns the L2 norm (Euclidean norm) of vector a.
func L2Norm(a []float64) float64 {
	var sum float64
	for _, v := range a {
		sum += v * v
	}

	return math.Sqrt(sum)
}

// Normalize returns the L2-normalized version of vector a.
func Normalize(a []float64) []float64 {
	norm := L2Norm(a)
	if IsZero(norm) {
		return a
	}

	normalized := make([]float64, len(a))
	for i, v := range a {
		normalized[i] = v / norm
	}

	return normalized
}
