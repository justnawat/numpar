use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

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

pub fn is_square_matrix(matrix: &Vec<Vec<f64>>) -> bool {
    let n = matrix.len();
    let lens = matrix.par_iter().map(|row| row.len()).sum::<usize>();
    n * n == lens
}

pub fn is_proper_matrix(matrix: &Vec<Vec<f64>>) -> bool {
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
}
