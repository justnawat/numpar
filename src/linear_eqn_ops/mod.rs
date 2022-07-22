use crate::my_util::{
    augment, is_square_matrix, row_major_to_matrix, simplify_soln, split_last_col,
};
use atomic_float::AtomicF64;
use pyo3::exceptions::PyTypeError;
use pyo3::types::PyList;
use pyo3::{pyfunction, PyResult};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use std::sync::atomic::Ordering::SeqCst;

pub fn fwd_elim(a: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let m = a.len();
    let n = a[0].len();

    let res: Vec<AtomicF64> = a.par_iter().flatten().map(|e| AtomicF64::new(*e)).collect();

    res.chunks_exact(n).enumerate().for_each(|(i, row)| {
        let head = row[i].load(SeqCst);
        let b1 = i * n;

        (i + 1..m).into_par_iter().for_each(|ii| {
            let to_elim = res[(ii) * n + i].load(SeqCst);
            let factor = to_elim / head;
            // dbg!(head, to_elim, factor);

            let b2 = ii * n;
            (i..n).into_par_iter().for_each(|ei| {
                let delta = res[b1 + ei].load(SeqCst) * factor;
                res[b2 + ei].fetch_sub(delta, SeqCst);
            })
        })
    });

    let rm_res: Vec<f64> = res.par_iter().map(|af64| af64.load(SeqCst)).collect();
    let res = row_major_to_matrix(&rm_res, n);
    res
}

pub fn bwd_subs(a: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = a[0].len();

    let res: Vec<AtomicF64> = a.par_iter().flatten().map(|e| AtomicF64::new(*e)).collect();

    res.chunks_exact(n).enumerate().rev().for_each(|(i, row)| {
        // dbg!(row);
        let tail = row[i].load(SeqCst);
        let b1 = i * n;

        (0..i).into_par_iter().for_each(|ii| {
            let to_elim = res[ii * n + i].load(SeqCst);
            let factor = to_elim / tail;

            let b2 = ii * n;
            (i..n).into_par_iter().for_each(|ei| {
                let delta = res[b1 + ei].load(SeqCst) * factor;
                res[b2 + ei].fetch_sub(delta, SeqCst);
            })
        })
    });

    let rm_res: Vec<f64> = res.par_iter().map(|af64| af64.load(SeqCst)).collect();
    let res = row_major_to_matrix(&rm_res, n);
    res
}

#[pyfunction]
pub fn solve(a: &PyList, b: &PyList) -> PyResult<Vec<f64>> {
    match (a.extract::<Vec<Vec<f64>>>(), b.extract::<Vec<f64>>()) {
        (Ok(a_mat), Ok(b_vec)) => {
            if is_square_matrix(&a_mat) {
                Ok(rust_solve(&a_mat, &b_vec))
            } else {
                Err(PyTypeError::new_err("Malformed parameter"))
            }
        }
        _ => Err(PyTypeError::new_err("Malformed parameter")),
    }
}

pub fn rust_solve(a: &Vec<Vec<f64>>, b: &Vec<f64>) -> Vec<f64> {
    let augmented = augment(a, &b.par_iter().map(|&e| vec![e]).collect());
    let fwd = fwd_elim(&augmented);
    let solved = bwd_subs(&fwd);
    let simple = simplify_soln(&solved);
    split_last_col(&simple).1
}

#[pyfunction]
pub fn matrix_rank(a: &PyList) -> PyResult<usize> {
    match a.extract::<Vec<Vec<f64>>>() {
        Ok(a_mat) => {
            if is_square_matrix(&a_mat) {
                Ok(rust_mat_rank(&a_mat))
            } else {
                Err(PyTypeError::new_err("Malformed parameter"))
            }
        }
        _ => Err(PyTypeError::new_err("Malformed parameter")),
    }
}

pub fn rust_mat_rank(a: &Vec<Vec<f64>>) -> usize {
    let fwd = fwd_elim(a);
    let solved = bwd_subs(&fwd);
    let simple = simplify_soln(&solved);
    let first_non_zeros: Vec<f64> = simple
        .par_iter()
        .enumerate()
        .map(|(i, row)| first_non_zero(row, i))
        .collect();
    first_non_zeros
        .par_iter()
        .fold(
            || 0usize,
            |acc, &elm| if elm.abs() > 1e-10 { acc + 1 } else { acc },
        )
        .sum()
}

fn first_non_zero(a: &Vec<f64>, b: usize) -> f64 {
    a[b..]
        .iter()
        .fold(0., |acc, &elm| if elm.abs() > 1e-10 { elm } else { acc })
}

mod test {
    #[test]
    fn solve_test() {
        let a = vec![vec![1., 2., 2.], vec![3., 2., 5.], vec![7., 7., 1.]];
        let b = vec![1., 1., 1.];

        let ans = vec![-0.3333, 0.4444, 0.2222];
        let out: Vec<f64> = super::rust_solve(&a, &b)
            .iter()
            .map(|&e| (e * 10000.).round() / 10000.)
            .collect();
        assert_eq!(&ans, &out);
    }

    #[test]
    fn fwd_test() {
        let a = vec![
            vec![7, 7, 5, 5, 7],
            vec![2, 3, 4, 5, 6],
            vec![2, 2, 3, 3, 4],
            vec![1, 2, 3, 4, 5],
        ]
        .iter()
        .map(|row| {
            row.iter()
                .map(|e| (*e as f64).round())
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>();

        let out = super::fwd_elim(&a)
            .iter()
            .map(|row| row.iter().map(|&e| f64::round(e)).collect::<Vec<f64>>())
            .collect::<Vec<Vec<f64>>>();
        let ans = vec![
            vec![7., 7., 5., 5., 7.],
            vec![0., 1., 2.57142857, 3.57142857, 4.],
            vec![0., 0., 1.57142857, 1.57142857, 2.],
            vec![0., 0., 0., 0., 0.36363636],
        ]
        .iter()
        .map(|row| row.iter().map(|&e| f64::round(e)).collect::<Vec<f64>>())
        .collect::<Vec<Vec<f64>>>();

        assert_eq!(&ans, &out);
    }

    #[test]
    fn bwd_test() {
        let a = super::fwd_elim(&vec![vec![7., 7., 6.], vec![6., 2., 2.], vec![3., 3., 1.]]);
        let out = super::bwd_subs(&a)
            .iter()
            .map(|row| row.iter().map(|&e| f64::round(e)).collect::<Vec<f64>>())
            .collect::<Vec<Vec<f64>>>();
        let ans = vec![vec![7., 0., 0.], vec![0., -4., 0.], vec![0., 0., -1.5714]]
            .iter()
            .map(|row| row.iter().map(|&e| f64::round(e)).collect::<Vec<f64>>())
            .collect::<Vec<Vec<f64>>>();
        assert_eq!(&ans, &out);
    }
}
