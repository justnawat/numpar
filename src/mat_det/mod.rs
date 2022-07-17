use pyo3::{exceptions::PyTypeError, prelude::pyfunction, types::PyList, PyResult};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::my_util::is_proper_matrix;

#[pyfunction]
pub fn det(matrix: &PyList) -> PyResult<f64> {
    match matrix.extract::<Vec<Vec<f64>>>() {
        Ok(r_matrix) => {
            if !is_proper_matrix(&r_matrix) {
                Err(PyTypeError::new_err("Parameter not a proper matrix."))
            } else {
                Ok(rust_det(&r_matrix))
            }
        }
        _ => Err(PyTypeError::new_err("Parameter not a matrix.")),
    }
}

pub fn rust_det(matrix: &Vec<Vec<f64>>) -> f64 {
    let n = matrix.len();
    if n <= 3 {
        rust_det_leq3(matrix)
    } else {
        0.
    }
}

fn rust_det_leq3(matrix: &Vec<Vec<f64>>) -> f64 {
    let n = matrix.len();

    let to_add: f64 = (0..n)
        .into_par_iter()
        .map(|o_offset| {
            (0..n)
                .into_par_iter()
                .map(|i_offset| {
                    // dbg!(i_offset, (i_offset + o_offset) % n);
                    matrix[i_offset][(i_offset + o_offset) % n]
                })
                .product::<f64>()
        })
        .sum();
    // dbg!(to_add);

    let to_subtract: f64 = (0..n)
        .into_par_iter()
        .rev()
        .map(|o_offset| {
            (0..n)
                .into_par_iter()
                .map(|i_offset| {
                    // dbg!(n - i_offset - 1, (i_offset + n - o_offset - 1) % n);
                    matrix[n - i_offset - 1][(i_offset + n - o_offset - 1) % n]
                })
                .product::<f64>()
        })
        .sum();
    // dbg!(to_subtract);

    to_add - to_subtract
}

mod test {
    #[test]
    fn rd_test() {
        const THRESHOLD: f64 = 1e-8;
        let a = vec![vec![1, 3, 9], vec![6, 3, 8], vec![9, 9, 2]]
            .iter()
            .map(|v| v.iter().map(|e| *e as f64).collect::<Vec<f64>>())
            .collect();
        let res = super::rust_det(&a);
        assert!((357. - res).abs() < THRESHOLD);
    }

    #[test]
    fn rd_test_beeg() {
        const THRESHOLD: f64 = 1e-8;
        let a = vec![
            [-2, 3, -6, 4, 4],
            [2, 0, 1, -3, 2],
            [-2, 5, -5, -5, 5],
            [7, -8, 0, 0, 0],
            [-2, 3, 5, -8, 1],
        ]
        .iter()
        .map(|v| v.iter().map(|e| *e as f64).collect::<Vec<f64>>())
        .collect();
        let res = super::rust_det(&a);
        dbg!(res);
        assert!((2061. - res).abs() < THRESHOLD);
    }
}
