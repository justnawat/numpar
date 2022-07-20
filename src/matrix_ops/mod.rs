use crate::cwslice::UnsafeSlice;
use crate::my_util::{
    generate_identity_matrix, is_proper_matrix, is_square_matrix, row_major_to_matrix,
};
use crate::vector_ops::rust_dot;
use pyo3::exceptions::PyTypeError;
use pyo3::types::{PyInt, PyList};
use pyo3::{pyfunction, PyResult};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
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

    row_major_to_matrix(&row_major, row_len)
}

#[pyfunction]
pub fn matmul(a: &PyList, b: &PyList) -> PyResult<Vec<Vec<f64>>> {
    match (a.extract::<Vec<Vec<f64>>>(), b.extract::<Vec<Vec<f64>>>()) {
        (Ok(a_mat), Ok(b_mat)) => {
            if a_mat.len() == 0 || b_mat.len() == 0 {
                Err(PyTypeError::new_err("Malformed parameter(s)"))
            } else if is_proper_matrix(&a_mat)
                && is_proper_matrix(&b_mat)
                && a_mat.len() == b_mat[0].len()
            {
                let m = a_mat.len();
                let n = b_mat.len();
                let p = b_mat[0].len();

                //let a_mat = a_mat.par_iter().flatten().map(|&e| e).collect();
                //let b_mat = rust_transpose(&b_mat);
                //let b_mat = b_mat.par_iter().flatten().map(|&e| e).collect();
                let res = rust_matmul(&a_mat, &b_mat);

                //Ok(row_major_to_matrix(&res, p))
                Ok(res)
            } else {
                Err(PyTypeError::new_err("Malformed parameter(s)"))
            }
        }
        _ => Err(PyTypeError::new_err("Malformed parameter(s)")),
    }
}

// A = m*n, B = n*p, C = m*p
pub fn rust_matmul(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let m = a.len();
    let n = b.len();
    let p = b[0].len();

    let mut c = vec![0.; m * p];
    let unsafe_c = UnsafeSlice::new(c.as_mut_slice());
    let b = rust_transpose(b);

    a.par_iter().enumerate().for_each(|(i, a_row)| {
        b.par_iter().enumerate().for_each(|(j, bt_row)| unsafe {
            unsafe_c.write(i * p + j, rust_dot(a_row, bt_row));
        });
    });

    row_major_to_matrix(&c, p)
}

#[pyfunction]
pub fn matrix_power(a: &PyList, exp: &PyInt) -> PyResult<Vec<Vec<f64>>> {
    match (a.extract::<Vec<Vec<f64>>>(), exp.extract::<u32>()) {
        (Ok(a_mat), Ok(exp)) => {
            if a_mat.len() == 0 {
                Err(PyTypeError::new_err("Malformed parameter(s)"))
            } else if is_square_matrix(&a_mat) {
                Ok(rust_matpow(&a_mat, exp))
            } else {
                Err(PyTypeError::new_err("Malformed parameter(s)"))
            }
        }
        _ => Err(PyTypeError::new_err("Malformed parameter(s)")),
    }
}

pub fn rust_matpow(a: &Vec<Vec<f64>>, exp: u32) -> Vec<Vec<f64>> {
    //let n = a.len();
    //let flattened = a.par_iter().flatten().map(|&e| e).collect();
    let res = rmp_helper(a, exp);
    //row_major_to_matrix(&res, n);
    res
}

fn rmp_helper(a: &Vec<Vec<f64>>, exp: u32) -> Vec<Vec<f64>> {
    const PAR_EXP_MIN: u32 = 8;

    let n = a.len();
    if exp <= PAR_EXP_MIN {
        let mut res = generate_identity_matrix(n);
        for _ in 0..exp {
            res = rust_matmul(&res, a);
        }
        res
    } else {
        let odd = exp % 2;
        let (a1, a2) = rayon::join(|| rmp_helper(a, exp / 2), || rmp_helper(a, exp / 2 + odd));
        rust_matmul(&a1, &a2)
    }
}

mod test {
    #[test]
    fn small_pow_test() {
        let a = super::rust_matpow(
            &vec![
                vec![10.0, 0.0, 0.0],
                vec![0.0, 10.0, 0.0],
                vec![0.0, 0.0, 10.0],
            ],
            129,
        )
        .iter()
        .map(|v| v.iter().sum::<f64>())
        .sum::<f64>();

        let ans = vec![
            vec![1e129, 0.0, 0.0],
            vec![0.0, 1e129, 0.0],
            vec![0.0, 0.0, 1e129],
        ]
        .iter()
        .map(|v| v.iter().sum::<f64>())
        .sum::<f64>();

        dbg!((a - ans).abs());
    }

    #[test]
    fn mul_test() {
        let a = vec![[5, 5, 5], [2, 4, 6], [1, 0, 0]]
            .iter()
            .map(|v| v.iter().map(|x| *x as f64).collect::<Vec<f64>>())
            .collect();
        let b = vec![[7, 1, 7], [7, 2, 4], [7, 3, 1]]
            .iter()
            .map(|v| v.iter().map(|x| *x as f64).collect::<Vec<f64>>())
            .collect();
        let ans: Vec<Vec<f64>> = vec![[105, 30, 60], [84, 28, 36], [7, 1, 7]]
            .iter()
            .map(|v| v.iter().map(|x| *x as f64).collect::<Vec<f64>>())
            .collect();
        let c = super::rust_matmul(&a, &b);
        assert_eq!(&ans, &c);
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