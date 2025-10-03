package reval_test

import (
	"fmt"
	"testing"

	"github.com/itsubaki/reval"
)

func ExampleNDCG() {
	predicted := []string{"A", "B", "C", "D"}
	relevance := map[string]int{
		"A": 3,
		"B": 2,
		"C": 1,
		"D": 0,
		"E": 3,
	}

	s := reval.NDCG(predicted, relevance, 3)
	fmt.Println("NDCG@3:", s)

	// Output:
	// NDCG@3: 0.8080824371047749
}

func TestNDCG(t *testing.T) {
	cases := []struct {
		predicted []string
		relevance map[string]int
		k         int
		want      float64
	}{
		{
			// Perfect
			predicted: []string{"A", "B", "C"},
			relevance: map[string]int{"A": 3, "B": 2, "C": 1},
			k:         3,
			want:      1.0,
		},
		{
			// Worst case
			predicted: []string{"C", "B", "A"},
			relevance: map[string]int{"A": 3, "B": 2, "C": 1},
			k:         3,
			want:      0.7899980042460358,
		},
		{
			// Partial match in top-2
			predicted: []string{"B", "A"},
			relevance: map[string]int{"A": 3, "B": 2, "C": 1},
			k:         2,
			want:      0.9134015924715544,
		},
		{
			// No relevant items in prediction
			predicted: []string{"X", "Y", "Z"},
			relevance: map[string]int{"A": 3, "B": 2, "C": 1},
			k:         3,
			want:      0.0,
		},
		{
			// Empty input
			predicted: []string{},
			relevance: map[string]int{},
			k:         3,
			want:      0.0,
		},
		{
			// Longer prediction than relevance; top-K is still perfect
			predicted: []string{"A", "B", "C", "D", "E"},
			relevance: map[string]int{"A": 3, "B": 2, "C": 1},
			k:         5,
			want:      1.0,
		},
	}

	for _, c := range cases {
		got := reval.NDCG(c.predicted, c.relevance, c.k)
		if reval.IsClose(got, c.want) {
			continue
		}

		t.Errorf("got=%v, want=%v", got, c.want)
	}
}
