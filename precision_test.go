package reval_test

import (
	"fmt"
	"testing"

	"github.com/itsubaki/reval"
)

func ExamplePrecision() {
	predicted := []string{"A", "B", "C", "D"}
	relevance := map[string]int{
		"A": 3,
		"B": 2,
		"C": 0,
		"D": 0,
		"E": 3,
	}

	s := reval.Precision(predicted, relevance, 3)
	fmt.Println("Precision@3:", s)

	// Output:
	// Precision@3: 0.6666666666666666
}

func TestPrecision(t *testing.T) {
	cases := []struct {
		predicted []string
		relevance map[string]int
		k         int
		want      float64
	}{
		{
			// Perfect precision
			predicted: []string{"A", "B", "C"},
			relevance: map[string]int{"A": 1, "B": 1, "C": 1},
			k:         3,
			want:      1.0,
		},
		{
			// Partial precision
			predicted: []string{"A", "B", "X"},
			relevance: map[string]int{"A": 1, "B": 1, "C": 1},
			k:         3,
			want:      2.0 / 3.0,
		},
		{
			// More predicted than k
			predicted: []string{"A", "B", "C", "D"},
			relevance: map[string]int{"A": 1, "C": 1},
			k:         2,
			want:      0.5,
		},
		{
			// k is zero, precision should be 0
			predicted: []string{"A", "B", "C", "D"},
			relevance: map[string]int{"A": 1, "C": 1},
			k:         0,
			want:      0,
		},
	}

	for _, c := range cases {
		got := reval.Precision(c.predicted, c.relevance, c.k)
		if reval.IsClose(got, c.want) {
			continue
		}

		t.Errorf("got=%v, want=%v", got, c.want)
	}
}
