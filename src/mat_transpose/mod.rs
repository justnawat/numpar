use crate::my_util::is_proper_matrix;
use pyo3::exceptions::PyTypeError;
use pyo3::types::PyList;
use pyo3::{pyfunction, PyResult};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

#[pyfunction]
pub fn transpose(matrix: &PyList) -> PyResult<Vec<Vec<f64>>> {
    match matrix.extract::<Vec<Vec<f64>>>() {
        Ok(r_matrix) => {
            if !is_proper_matrix(&r_matrix) {
                Err(PyTypeError::new_err("Parameter not a proper matrix."))
            } else if r_matrix.len() == 0 {
                Ok(vec![vec![]])
            } else {
                Ok(rust_transpose(&r_matrix))
            }
        }
        Err(_) => Err(PyTypeError::new_err("Parameter not a matrix.")),
    }
}

pub fn rust_transpose(r_matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = r_matrix.len();
    let m = r_matrix[0].len();

    (0..m)
        .into_par_iter()
        .map(|t_row| {
            (0..n)
                .into_par_iter()
                .map(|t_col| r_matrix[t_col][t_row])
                .collect()
        })
        .collect()
}

mod test {
    #[test]
    fn trans_test() {
        let b = vec![
            vec![1, 5, 3, 3, 0],
            vec![2, 3, 0, 0, 3],
            vec![7, 8, 9, 9, 10],
        ]
        .iter()
        .map(|row| row.iter().map(|&elm| elm as f64).collect())
        .collect::<Vec<Vec<f64>>>();

        let ans = vec![
            vec![1, 2, 7],
            vec![5, 3, 8],
            vec![3, 0, 9],
            vec![3, 0, 9],
            vec![0, 3, 10],
        ]
        .iter()
        .map(|row| row.iter().map(|&elm| elm as f64).collect())
        .collect::<Vec<Vec<f64>>>();

        assert_eq!(&ans, &super::rust_transpose(&b));
    }
}
