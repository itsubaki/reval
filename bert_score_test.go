package reval_test

import (
	"fmt"
	"testing"

	"github.com/itsubaki/reval"
)

func ExampleBERTScore() {
	candidates := [][]float64{
		{0.1, 0.2, 0.3},
		{0.4, 0.5, 0.6},
	}
	refs := [][]float64{
		{0.1, 0.2, 0.3},
		{0.7, 0.8, 0.9},
	}

	precision, recall, f1 := reval.BERTScore(candidates, refs)
	fmt.Printf("%.4f, %.4f, %.4f\n", precision, recall, f1)

	// Output:
	// 0.8600, 0.7700, 0.8125
}

func TestBERTScore(t *testing.T) {
	cases := []struct {
		candidates [][]float64
		refs       [][]float64
		precision  float64
		recall     float64
		f1         float64
	}{
		{
			candidates: [][]float64{},
			refs: [][]float64{
				{0.1, 0.2, 0.3},
			},
			precision: 0.0,
			recall:    0.0,
			f1:        0.0,
		},
		{
			candidates: [][]float64{
				{0.1, 0.2, 0.3},
			},
			refs:      [][]float64{},
			precision: 0.0,
			recall:    0.0,
			f1:        0.0,
		},
		{
			candidates: [][]float64{
				{0.6, 0.8},
			},
			refs: [][]float64{
				{0.6, 0.8},
			},
			precision: 1.0,
			recall:    1.0,
			f1:        1.0,
		},
	}

	for _, c := range cases {
		precision, recall, f1 := reval.BERTScore(c.candidates, c.refs)
		if !reval.IsClose(precision, c.precision) {
			t.Errorf("precision: got=%v, want=%v", precision, c.precision)
		}

		if !reval.IsClose(recall, c.recall) {
			t.Errorf("recall: got=%v, want=%v", recall, c.recall)
		}

		if !reval.IsClose(f1, c.f1) {
			t.Errorf("f1: got=%v, want=%v", f1, c.f1)
		}
	}
}

func TestDotProduct(t *testing.T) {
	cases := []struct {
		a    []float64
		b    []float64
		want float64
	}{
		{
			a:    []float64{1, 2, 3},
			b:    []float64{4, 5, 6},
			want: 32,
		},
		{
			a:    []float64{1, 2},
			b:    []float64{4, 5, 6},
			want: 0,
		},
	}

	for _, c := range cases {
		got := reval.DotProduct(c.a, c.b)
		if !reval.IsClose(got, c.want) {
			t.Errorf("got=%v, want=%v", got, c.want)
		}
	}
}

func TestL2Norm(t *testing.T) {
	cases := []struct {
		a    []float64
		want float64
	}{
		{
			a:    []float64{3, 4},
			want: 5,
		},
	}

	for _, c := range cases {
		got := reval.L2Norm(c.a)
		if !reval.IsClose(got, c.want) {
			t.Errorf("got=%v, want=%v", got, c.want)
		}
	}
}

func TestNormalize(t *testing.T) {
	cases := []struct {
		a    []float64
		want []float64
	}{
		{
			a:    []float64{3, 4},
			want: []float64{0.6, 0.8},
		},
		{
			a:    []float64{0, 0},
			want: []float64{0, 0},
		},
	}

	for _, c := range cases {
		got := reval.Normalize(c.a)
		for i := range got {
			if !reval.IsClose(got[i], c.want[i]) {
				t.Errorf("got=%v, want=%v", got, c.want)
			}
		}
	}
}
