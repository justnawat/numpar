use pyo3::prelude::{pyfunction, PyResult};
use pyo3::types::PyList;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::prelude::ParallelIterator;

#[pyfunction]
pub fn dot(xs: &PyList, ys: &PyList) -> PyResult<f64> {
    let xs: Vec<f64> = xs.extract()?;
    let ys: Vec<f64> = ys.extract()?;

    Ok(xs.par_iter().zip(ys.par_iter()).map(|(&x, &y)| x * y).sum())
}

mod test {
    #[allow(dead_code)]
    fn generate_vectors() -> (Vec<f64>, Vec<f64>) {
        const VEC_SIZE: usize = 10_000_000;
        let mut xs = Vec::new();
        let mut ys = Vec::new();

        for _ in 1..=VEC_SIZE {
            xs.push(rand::random::<f64>());
            ys.push(rand::random::<f64>());
        }

        (xs, ys)
    }

    #[allow(dead_code)]
    fn dot_sum(xs: &[f64], ys: &[f64]) -> f64 {
        use rayon::prelude::*;
        xs.par_iter().zip(ys.par_iter()).map(|(&x, &y)| x * y).sum()
    }

    #[allow(dead_code)]
    fn dot_reduce(xs: &[f64], ys: &[f64]) -> f64 {
        use rayon::prelude::*;
        xs.par_iter()
            .zip(ys.par_iter())
            .map(|(&x, &y)| x * y)
            .reduce(|| 0f64, |x, y| x + y)
    }

    #[test]
    fn dot_timer() {
        use std::time::Instant;
        let (xs1, ys1) = generate_vectors();
        let (xs2, ys2) = generate_vectors();
        let (xs3, ys3) = generate_vectors();
        let vecs = vec![(xs1, ys1), (xs2, ys2), (xs3, ys3)];

        let now = Instant::now();
        for (xs, ys) in vecs.iter() {
            dot_sum(xs, ys);
        }
        let ds_stop = now.elapsed();

        let now = Instant::now();
        for (xs, ys) in vecs.iter() {
            dot_reduce(xs, ys);
        }
        let dr_stop = now.elapsed();

        dbg!(ds_stop / 3);
        dbg!(dr_stop / 3);
    }
}
