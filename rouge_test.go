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

func ExampleROUGEL() {
	candidates := []string{"the", "cat", "is", "sitting", "on", "the", "mat"}
	refs := []string{"the", "cat", "sat", "on", "the", "mat"}

	precision, recall, f1 := reval.ROUGEL(candidates, refs)
	fmt.Printf("%.4f, %.4f, %.4f\n", precision, recall, f1)

	// Output:
	// 0.7143, 0.8333, 0.7692
}

func ExampleROUGELsum() {
	candidates := [][]string{
		{"the", "cat", "is", "on", "the", "mat"},
		{"it", "is", "cute"},
	}

	refs := [][]string{
		{"the", "dog", "is", "on", "the", "mat"},
		{"the", "animal", "is", "cute"},
		{"the", "pet", "sleeps", "well"},
	}

	precision, recall, f1 := reval.ROUGELsum(candidates, refs)
	fmt.Printf("%.4f, %.4f, %.4f\n", precision, recall, f1)

	// Output:
	// 0.7778, 0.5000, 0.6087
}

func TestROUGE1(t *testing.T) {
	cases := []struct {
		candidates []string
		refs       []string
		precision  float64
		recall     float64
		f1         float64
	}{
		{
			candidates: []string{"a", "b", "c"},
			refs:       []string{"a", "b", "c"},
			precision:  1.0,
			recall:     1.0,
			f1:         1.0,
		},
		{
			candidates: []string{"a", "b", "c"},
			refs:       []string{"b", "c", "d"},
			precision:  2.0 / 3.0,
			recall:     2.0 / 3.0,
			f1:         2.0 / 3.0,
		},
		{
			candidates: []string{"a", "b", "c"},
			refs:       []string{"c", "b", "a"},
			precision:  1.0,
			recall:     1.0,
			f1:         1.0,
		},
		{
			candidates: []string{"a", "b", "a", "c"},
			refs:       []string{"a", "a", "c"},
			precision:  3.0 / 4.0,
			recall:     1.0,
			f1:         0.8571428571428571,
		},
		{
			candidates: []string{"x", "y", "z"},
			refs:       []string{"y", "z"},
			precision:  2.0 / 3.0,
			recall:     1.0,
			f1:         0.8,
		},
		{
			candidates: []string{},
			refs:       []string{"a"},
			precision:  0.0,
			recall:     0.0,
			f1:         0.0,
		},
		{
			candidates: []string{"a"},
			refs:       []string{},
			precision:  0.0,
			recall:     0.0,
			f1:         0.0,
		},
		{
			candidates: []string{"a", "a", "b", "b"},
			refs:       []string{"a", "b", "b", "b"},
			precision:  3.0 / 4.0,
			recall:     3.0 / 4.0,
			f1:         0.75,
		},
		{
			candidates: []string{"a", "b", "c", "a"},
			refs:       []string{"a", "c", "a", "b"},
			precision:  1.0,
			recall:     1.0,
			f1:         1.0,
		},
	}

	for _, c := range cases {
		precision, recall, f1 := reval.ROUGE1(c.candidates, c.refs)
		if !reval.IsClose(precision, c.precision) {
			t.Errorf("got=%v, want=%v", precision, c.precision)
		}

		if !reval.IsClose(recall, c.recall) {
			t.Errorf("got=%v, want=%v", recall, c.recall)
		}

		if !reval.IsClose(f1, c.f1) {
			t.Errorf("got=%v, want=%v", f1, c.f1)
		}
	}
}

func TestROUGEL(t *testing.T) {
	cases := []struct {
		candidates []string
		refs       []string
		precision  float64
		recall     float64
		f1         float64
	}{
		{
			candidates: []string{},
			refs:       []string{},
			precision:  0.0,
			recall:     0.0,
			f1:         0.0,
		},
		{
			candidates: []string{"a", "b", "c"},
			refs:       []string{"a", "b", "c"},
			precision:  1.0,
			recall:     1.0,
			f1:         1.0,
		},
		{
			candidates: []string{"a", "b", "c"},
			refs:       []string{"b", "c", "d"},
			precision:  2.0 / 3.0,
			recall:     2.0 / 3.0,
			f1:         2.0 / 3.0,
		},
		{
			candidates: []string{"a", "b", "c"},
			refs:       []string{"c", "b", "a"},
			precision:  1.0 / 3.0,
			recall:     1.0 / 3.0,
			f1:         1.0 / 3.0,
		},
		{
			candidates: []string{"a", "b", "a", "c"},
			refs:       []string{"a", "a", "c"},
			precision:  3.0 / 4.0,
			recall:     1.0,
			f1:         0.8571428571428571,
		},
		{
			candidates: []string{"x", "y", "z"},
			refs:       []string{"y", "z"},
			precision:  2.0 / 3.0,
			recall:     1.0,
			f1:         0.8,
		},
		{
			candidates: []string{"a"},
			refs:       []string{"a"},
			precision:  1.0,
			recall:     1.0,
			f1:         1.0,
		},
		{
			candidates: []string{"a"},
			refs:       []string{"b"},
			precision:  0.0,
			recall:     0.0,
			f1:         0.0,
		},
		{
			candidates: []string{"a", "b", "c", "a"},
			refs:       []string{"a", "c", "a", "b"},
			precision:  3.0 / 4.0,
			recall:     3.0 / 4.0,
			f1:         0.75,
		},
	}

	for _, c := range cases {
		precision, recall, f1 := reval.ROUGEL(c.candidates, c.refs)
		if !reval.IsClose(precision, c.precision) {
			t.Errorf("got=%v, want=%v", precision, c.precision)
		}

		if !reval.IsClose(recall, c.recall) {
			t.Errorf("got=%v, want=%v", recall, c.recall)
		}

		if !reval.IsClose(f1, c.f1) {
			t.Errorf("got=%v, want=%v", f1, c.f1)
		}
	}
}

func TestROUGELsum(t *testing.T) {
	cases := []struct {
		candidates [][]string
		refs       [][]string
		precision  float64
		recall     float64
		f1         float64
	}{
		{
			candidates: [][]string{},
			refs:       [][]string{},
			precision:  0.0,
			recall:     0.0,
			f1:         0.0,
		},
		{
			candidates: [][]string{
				{},
			},
			refs: [][]string{
				{},
			},
			precision: 0.0,
			recall:    0.0,
			f1:        0.0,
		},
		{
			candidates: [][]string{
				{"a", "b", "c"},
			},
			refs: [][]string{
				{"a", "b", "c"},
			},
			precision: 1.0,
			recall:    1.0,
			f1:        1.0,
		},
	}

	for _, c := range cases {
		precision, recall, f1 := reval.ROUGELsum(c.candidates, c.refs)
		if !reval.IsClose(precision, c.precision) {
			t.Errorf("got=%v, want=%v", precision, c.precision)
		}

		if !reval.IsClose(recall, c.recall) {
			t.Errorf("got=%v, want=%v", recall, c.recall)
		}

		if !reval.IsClose(f1, c.f1) {
			t.Errorf("got=%v, want=%v", f1, c.f1)
		}
	}
}

func TestOverlap(t *testing.T) {
	cases := []struct {
		a    []string
		b    []string
		want int
	}{
		{
			a:    []string{},
			b:    []string{},
			want: 0,
		},
		{
			a:    []string{"a", "b"},
			b:    []string{},
			want: 0,
		},
		{
			a:    []string{},
			b:    []string{"a", "b"},
			want: 0,
		},
		{
			a:    []string{"a", "b", "c"},
			b:    []string{"a", "b", "c"},
			want: 3,
		},
		{
			a:    []string{"a", "b", "c"},
			b:    []string{"b", "c", "d"},
			want: 2,
		},
		{
			a:    []string{"a", "b", "c"},
			b:    []string{"c", "b", "a"},
			want: 3,
		},
		{
			a:    []string{"a", "b", "a", "c"},
			b:    []string{"a", "a", "c"},
			want: 3,
		},
		{
			a:    []string{"x", "y", "z"},
			b:    []string{"y", "z"},
			want: 2,
		},
		{
			a:    []string{"a"},
			b:    []string{"a"},
			want: 1,
		},
		{
			a:    []string{"a"},
			b:    []string{"b"},
			want: 0,
		},
	}

	for _, c := range cases {
		got := reval.Overlap(c.a, c.b)
		if got != c.want {
			t.Errorf("got=%v, want=%v", got, c.want)
		}
	}
}

func TestLCS(t *testing.T) {
	cases := []struct {
		a    []string
		b    []string
		want int
	}{
		{
			a:    []string{},
			b:    []string{},
			want: 0,
		},
		{
			a:    []string{"a", "b"},
			b:    []string{},
			want: 0,
		},
		{
			a:    []string{},
			b:    []string{"a", "b"},
			want: 0,
		},
		{
			a:    []string{"a", "b", "c"},
			b:    []string{"a", "b", "c"},
			want: 3,
		},
		{
			a:    []string{"a", "b", "c"},
			b:    []string{"b", "c", "d"},
			want: 2,
		},
		{
			a:    []string{"a", "b", "c"},
			b:    []string{"c", "b", "a"},
			want: 1,
		},
		{
			a:    []string{"a", "b", "a", "c"},
			b:    []string{"a", "a", "c"},
			want: 3,
		},
		{
			a:    []string{"x", "y", "z"},
			b:    []string{"y", "z"},
			want: 2,
		},
		{
			a:    []string{"a"},
			b:    []string{"a"},
			want: 1,
		},
		{
			a:    []string{"a"},
			b:    []string{"b"},
			want: 0,
		},
	}

	for _, c := range cases {
		got := reval.LCS(c.a, c.b)
		if got != c.want {
			t.Errorf("got=%v, want=%v", got, c.want)
		}
	}
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
