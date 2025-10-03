package reval_test

import (
	"fmt"
	"testing"

	"github.com/itsubaki/reval"
)

func ExampleRecall() {
	predicted := []string{"A", "B", "C", "D"}
	relevance := map[string]int{
		"A": 3,
		"B": 2,
		"C": 1,
		"D": 0,
		"E": 3,
	}

	s := reval.Recall(predicted, relevance, 3)
	fmt.Println("Recall@3:", s)

	// Output:
	// Recall@3: 0.75
}

func TestRecall(t *testing.T) {
	cases := []struct {
		predicted []string
		relevance map[string]int
		k         int
		want      float64
	}{
		{
			// Perfect
			predicted: []string{"A", "B", "C"},
			relevance: map[string]int{"A": 1, "B": 1, "C": 1},
			k:         3,
			want:      1.0,
		},
		{
			// Partial hit
			predicted: []string{"A", "X"},
			relevance: map[string]int{"A": 1, "B": 1, "C": 1},
			k:         2,
			want:      1.0 / 3.0,
		},
		{
			// No relevant items found
			predicted: []string{"X", "Y", "Z"},
			relevance: map[string]int{"A": 1, "B": 1},
			k:         3,
			want:      0.0,
		},
		{
			// More predicted than relevant, all found
			predicted: []string{"A", "B", "C", "D", "E"},
			relevance: map[string]int{"A": 1, "C": 1},
			k:         5,
			want:      1.0,
		},
		{
			// No relevant items in ground truth
			predicted: []string{"A", "B"},
			relevance: map[string]int{"A": 0, "B": 0},
			k:         2,
			want:      0.0,
		},
		{
			// Empty input
			predicted: []string{},
			relevance: map[string]int{},
			k:         3,
			want:      0.0,
		},
	}

	for i, c := range cases {
		got := reval.Recall(c.predicted, c.relevance, c.k)
		if !reval.IsClose(got, c.want) {
			t.Errorf("case %d: got=%v, want=%v", i, got, c.want)
		}
	}
}
