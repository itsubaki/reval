package reval_test

import (
	"fmt"
	"testing"

	"github.com/itsubaki/reval"
)

func ExampleROUGE1() {
	candidates := []string{"the", "cat", "is", "sitting", "on", "the", "mat"}
	refs := []string{"the", "cat", "sat", "on", "the", "mat"}

	precision, recall, f1 := reval.ROUGE1(candidates, refs)
	fmt.Printf("%.4f, %.4f, %.4f\n", precision, recall, f1)

	// Output:
	// 0.7143, 0.8333, 0.7692
}

func TestFBeta(t *testing.T) {
	cases := []struct {
		precision float64
		recall    float64
		beta      float64
		want      float64
	}{
		{
			// Perfect
			precision: 1.0,
			recall:    1.0,
			beta:      1.0,
			want:      1.0,
		},
		{
			// Both zero
			precision: 0.0,
			recall:    0.0,
			beta:      1.0,
			want:      0.0,
		},
		{
			// Precision zero
			precision: 0.0,
			recall:    1.0,
			beta:      1.0,
			want:      0.0,
		},
		{
			// Recall zero
			precision: 1.0,
			recall:    0.0,
			beta:      1.0,
			want:      0.0,
		},
		{
			// Balanced case
			precision: 0.7,
			recall:    0.5,
			beta:      1.0,
			want:      0.5833333333333334,
		},
		{
			// F2
			precision: 0.7,
			recall:    0.5,
			beta:      2.0,
			want:      0.5303,
		},
		{
			// F0.5
			precision: 0.7,
			recall:    0.5,
			beta:      0.5,
			want:      0.6481481481481481,
		},
	}

	for _, c := range cases {
		got := reval.FBeta(c.precision, c.recall, c.beta)
		if !reval.IsClose(got, c.want) {
			t.Errorf("got=%v, want=%v", got, c.want)
		}
	}
}
