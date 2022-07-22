use crate::cwslice::UnsafeSlice;
use crate::linear_eqn_ops::*;
use crate::my_util::{
    augment, extract_last_n_cols, generate_identity_matrix, is_proper_matrix, is_square_matrix,
    row_major_to_matrix, simplify_soln,
};
use pyo3::exceptions::PyTypeError;
use pyo3::types::PyList;
use pyo3::{pyfunction, PyResult};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

#[pyfunction]
pub fn trace(matrix: &PyList) -> PyResult<f64> {
    match matrix.extract::<Vec<Vec<f64>>>() {
        Ok(r_matrix) => {
            if r_matrix.len() == 0 {
                Ok(0.0)
            } else if !is_square_matrix(&r_matrix) {
                Err(PyTypeError::new_err("Matrix not square."))
            } else {
                Ok(r_matrix.par_iter().enumerate().map(|(i, row)| row[i]).sum())
            }
        }
        _ => Err(PyTypeError::new_err(
            "Parameter cannot be converted to matrix.",
        )),
    }
}

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
    let triangular = fwd_elim(matrix);
    triangular
        .par_iter()
        .enumerate()
        .map(|(i, row)| row[i])
        .product()
}

#[pyfunction]
pub fn transpose(matrix: &PyList) -> PyResult<Vec<Vec<f64>>> {
    match matrix.extract::<Vec<Vec<f64>>>() {
        Ok(r_matrix) => {
            if r_matrix.len() == 0 {
                Ok(vec![vec![]])
            } else if !is_proper_matrix(&r_matrix) {
                Err(PyTypeError::new_err("Parameter not a proper matrix."))
            } else {
                Ok(rust_transpose(&r_matrix))
            }
        }
        _ => Err(PyTypeError::new_err("Parameter not a matrix.")),
    }
}

// still slower than np
pub fn rust_transpose(r_matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let row_len = r_matrix.len();
    let col_len = r_matrix[0].len();

    let mut row_major: Vec<f64> = vec![0.; col_len * row_len];
    let unsafe_rm = UnsafeSlice::new(row_major.as_mut_slice());

    r_matrix.par_iter().enumerate().for_each(|(i_row, row)| {
        row.par_iter().enumerate().for_each(|(i_elm, &elm)| unsafe {
            unsafe_rm.write(i_elm * row_len + i_row, elm);
        })
    });

    row_major_to_matrix(&row_major, row_len)
}

#[pyfunction]
pub fn inv(a: &PyList) -> PyResult<Vec<Vec<f64>>> {
    match a.extract::<Vec<Vec<f64>>>() {
        Ok(a_mat) => {
            if is_square_matrix(&a_mat) {
                Ok(rust_inv(&a_mat))
            } else {
                Err(PyTypeError::new_err("Malformed parameter"))
            }
        }
        _ => Err(PyTypeError::new_err("Malformed parameter")),
    }
}

pub fn rust_inv(a: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = a.len();
    let augmented = augment(a, &generate_identity_matrix(n));
    let fwd = fwd_elim(&augmented);
    let solved = bwd_subs(&fwd);
    let simple = simplify_soln(&solved);
    extract_last_n_cols(&simple, n)
}

mod test {
    #[test]
    fn inv_test() {
        let a = vec![vec![7., 7., 6.], vec![6., 2., 2.], vec![3., 3., 1.]];
        let out = super::rust_inv(&a)
            .iter()
            .map(|row| row.iter().map(|&e| f64::round(e)).collect::<Vec<f64>>())
            .collect::<Vec<Vec<f64>>>();
        let ans = vec![
            vec![-0.0909, 0.25, 0.04545],
            vec![0., -0.25, 0.5],
            vec![0.272727, 0., -0.636363],
        ]
        .iter()
        .map(|row| row.iter().map(|&e| f64::round(e)).collect::<Vec<f64>>())
        .collect::<Vec<Vec<f64>>>();

        assert_eq!(&ans, &out);
    }

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
