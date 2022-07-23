use crate::my_util::{generate_identity_matrix, is_proper_matrix, is_square_matrix};
use atomic_float::AtomicF64;
use pyo3::exceptions::PyTypeError;
use pyo3::types::{PyInt, PyList};
use pyo3::{pyfunction, PyResult};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rayon::slice::ParallelSlice;
use std::sync::atomic::Ordering::SeqCst;

/// Generally loses to the other implementation
// #[pyfunction]
// pub fn matmul1(a: &PyList, b: &PyList) -> PyResult<Vec<Vec<f64>>> {
//     match (a.extract::<Vec<Vec<f64>>>(), b.extract::<Vec<Vec<f64>>>()) {
//         (Ok(a_mat), Ok(b_mat)) => {
//             if a_mat.len() == 0 || b_mat.len() == 0 {
//                 Err(PyTypeError::new_err("Malformed parameter(s)"))
//             } else if is_proper_matrix(&a_mat)
//                 && is_proper_matrix(&b_mat)
//                 && a_mat.len() == b_mat[0].len()
//             {
//                 let res = rust_matmul1(&a_mat, &b_mat);
//                 Ok(res)
//             } else {
//                 Err(PyTypeError::new_err("Malformed parameter(s)"))
//             }
//         }
//         _ => Err(PyTypeError::new_err("Malformed parameter(s)")),
//     }
// }

// pub fn rust_matmul1(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
//     let m = a.len();
//     let p = b[0].len();
//     let mut c = vec![0.; m * p];
//     let unsafe_c = UnsafeSlice::new(c.as_mut_slice());
//     let b = rust_transpose(b);
//     a.par_iter().enumerate().for_each(|(i, a_row)| {
//         b.par_iter().enumerate().for_each(|(j, bt_row)| unsafe {
//             unsafe_c.write(i * p + j, rust_dot(a_row, bt_row));
//         });
//     });
//     row_major_to_matrix(&c, p)
// }

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
                let res = rust_matmul2(&a_mat, &b_mat);
                Ok(res)
            } else {
                Err(PyTypeError::new_err("Malformed parameter(s)"))
            }
        }
        _ => Err(PyTypeError::new_err("Malformed parameter(s)")),
    }
}

// A = m*n, B = n*p, C = m*p
pub fn rust_matmul2(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let m = a.len();
    let n = b.len();
    let p = b[0].len();

    let c: Vec<AtomicF64> = (0..m * p)
        .into_par_iter()
        .map(|_| AtomicF64::new(0.))
        .collect();

    (0..m).into_par_iter().for_each(|i| {
        (0..p).into_par_iter().for_each(|k| {
            (0..n).into_par_iter().for_each(|j| {
                let to_add = a[i][k] * b[k][j];
                c[i * p + j].fetch_add(to_add, SeqCst);
            })
        })
    });

    // c.par_chunks(p)
    //     .zip(a.par_iter())
    //     .for_each(|(c_row, a_row)| {
    //         a_row
    //             .par_iter()
    //             .zip(b.par_iter())
    //             .for_each(|(a_row_elm, b_row)| {
    //                 c_row
    //                     .par_iter()
    //                     .zip(b_row.par_iter())
    //                     .for_each(|(c_row_elm, b_row_elm)| {
    //                         c_row_elm.fetch_add(a_row_elm * b_row_elm, SeqCst);
    //                     });
    //             });
    //     });

    let c_mat = c
        .par_chunks(p)
        .map(|row| row.par_iter().map(|e| e.load(SeqCst)).collect::<Vec<f64>>())
        .collect::<Vec<Vec<f64>>>();
    c_mat
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
    let res = rmp_helper(a, exp);
    res
}

fn rmp_helper(a: &Vec<Vec<f64>>, exp: u32) -> Vec<Vec<f64>> {
    const PAR_EXP_MIN: u32 = 8;

    let n = a.len();
    if exp <= PAR_EXP_MIN {
        let mut res = generate_identity_matrix(n);
        for _ in 0..exp {
            res = rust_matmul2(&res, a);
        }
        res
    } else {
        let odd = exp % 2;
        let (a1, a2) = rayon::join(|| rmp_helper(a, exp / 2), || rmp_helper(a, exp / 2 + odd));
        rust_matmul2(&a1, &a2)
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
        let c = super::rust_matmul2(&a, &b);
        assert_eq!(&ans, &c);
    }
}
