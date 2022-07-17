use crate::cwslice::UnsafeSlice;
use crate::my_util::is_proper_matrix;
use pyo3::exceptions::PyTypeError;
use pyo3::types::PyList;
use pyo3::{pyfunction, PyResult};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rayon::slice::ParallelSlice;

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
}
