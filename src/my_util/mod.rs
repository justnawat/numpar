use crate::cwslice::UnsafeSlice;
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
    },
    slice::ParallelSlice,
};

#[allow(dead_code)]
pub fn split_last_col(a: &Vec<Vec<f64>>) -> (Vec<Vec<f64>>, Vec<f64>) {
    let m = a.len();
    let idx = a[0].len() - 1;

    let mut o1 = vec![vec![]; idx];
    let mut o2 = vec![0.; m];

    let u1 = UnsafeSlice::new(o1.as_mut_slice());
    let u2 = UnsafeSlice::new(o2.as_mut_slice());

    a.par_iter().enumerate().for_each(|(i, row)| {
        let (l, r) = row.split_at(idx);
        unsafe {
            u1.write(i, l.to_vec());
            u2.write(i, r[0]);
        }
    });

    (o1, o2)
}

#[allow(dead_code)]
pub fn augment(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    a.par_iter()
        .zip(b.par_iter())
        .map(|(a_row, b_row)| {
            let mut c = vec![];
            c.extend(a_row);
            c.extend(b_row);
            c
        })
        .collect()
}

#[allow(dead_code)]
pub fn extract_last_n_cols(a: &Vec<Vec<f64>>, n: usize) -> Vec<Vec<f64>> {
    let b = a[0].len() - n;
    a.par_iter().map(|a_row| a_row[b..].to_vec()).collect()
}

#[allow(dead_code)]
pub fn generate_identity_matrix(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .into_par_iter()
        .map(|i| {
            let mut row = vec![0.; n];
            row[i] = 1.;
            row
        })
        .collect()
}

#[allow(dead_code)]
pub fn generate_identity_matrix_row_major(n: usize) -> Vec<f64> {
    let mut res = vec![0.; n * n];
    (0..n).for_each(|i| res[i * n + i] = 1.);
    res
}

#[allow(dead_code)]
pub fn row_major_to_matrix(row_major: &Vec<f64>, cols: usize) -> Vec<Vec<f64>> {
    row_major
        .par_chunks(cols)
        .map(|chunk| chunk.to_vec())
        .collect()
}

pub fn is_square_matrix(matrix: &Vec<Vec<f64>>) -> bool {
    let n = matrix.len();
    let lens = matrix.par_iter().map(|row| row.len()).sum::<usize>();
    n * n == lens
}

pub fn is_proper_matrix(matrix: &Vec<Vec<f64>>) -> bool {
    if matrix.len() == 0 {
        return false;
    }

    let n = matrix.len();
    let m = matrix[0].len();
    let lens = matrix.par_iter().map(|vec| vec.len()).sum::<usize>();
    n * m == lens
}

mod test {
    #[test]
    fn iden_generate_test() {
        let ans = vec![
            vec![1., 0., 0., 0., 0.],
            vec![0., 1., 0., 0., 0.],
            vec![0., 0., 1., 0., 0.],
            vec![0., 0., 0., 1., 0.],
            vec![0., 0., 0., 0., 1.],
        ];
        let res = super::generate_identity_matrix(5);
        assert_eq!(ans, res);
    }

    #[test]
    fn square_test() {
        let sq = vec![
            vec![1., 0., 0., 0., 0.],
            vec![0., 1., 0., 0., 0.],
            vec![0., 0., 1., 0., 0.],
            vec![0., 0., 0., 1., 0.],
            vec![0., 0., 0., 0., 1.],
        ];
        let nsq = vec![vec![7., 7., 7.], vec![5., 5., 5.]];
        assert!(super::is_square_matrix(&sq));
        assert!(!super::is_square_matrix(&nsq));
    }

    #[test]
    fn proper_matrix_test() {
        let proper = vec![vec![1, 2, 3, 4], vec![5, 5, 2, 2]]
            .iter()
            .map(|vec| vec.iter().map(|&elm| elm as f64).collect())
            .collect();
        let improper = vec![vec![1, 5, 5], vec![5, 5, 1, 1]]
            .iter()
            .map(|vec| vec.iter().map(|&elm| elm as f64).collect())
            .collect();
        assert!(super::is_proper_matrix(&proper));
        assert!(!super::is_proper_matrix(&improper));
    }

    #[test]
    fn sl_test() {
        let a = vec![
            vec![1., 2., 3., 4.],
            vec![2., 3., 4., 5.],
            vec![3., 4., 5., 6.],
        ];

        let (a, b) = super::split_last_col(&a);
        let ans_a = vec![vec![1., 2., 3.], vec![2., 3., 4.], vec![3., 4., 5.]];
        let ans_b = vec![4., 5., 6.];
        assert_eq!(&a, &ans_a);
        assert_eq!(&b, &ans_b);
    }

    #[test]
    fn extract_test() {
        use crate::my_util::generate_identity_matrix;
        let a: Vec<Vec<f64>> = vec![
            vec![1, 0, 0, 1, 0, 0],
            vec![0, 1, 0, 0, 1, 0],
            vec![0, 0, 1, 0, 0, 1],
        ]
        .iter()
        .map(|row| row.iter().map(|e| *e as f64).collect())
        .collect();
        let out = super::extract_last_n_cols(&a, 3);
        assert_eq!(&generate_identity_matrix(3), &out);
    }

    #[test]
    fn aug_test() {
        use crate::my_util::generate_identity_matrix;
        let a = generate_identity_matrix(3);
        let b = generate_identity_matrix(3);
        let ans: Vec<Vec<f64>> = vec![
            vec![1, 0, 0, 1, 0, 0],
            vec![0, 1, 0, 0, 1, 0],
            vec![0, 0, 1, 0, 0, 1],
        ]
        .iter()
        .map(|row| row.iter().map(|e| *e as f64).collect())
        .collect();

        assert_eq!(&ans, &super::augment(&a, &b));
    }
}
