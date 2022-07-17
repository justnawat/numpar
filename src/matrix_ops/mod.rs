use crate::cwslice::UnsafeSlice;
use crate::my_util::{is_proper_matrix, is_square_matrix};
use pyo3::exceptions::PyTypeError;
use pyo3::types::PyList;
use pyo3::{pyfunction, PyResult};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rayon::slice::ParallelSlice;

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

    row_major
        .par_chunks(row_len)
        .map(|chunk| chunk.to_vec())
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
