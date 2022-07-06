use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator},
    prelude::ParallelIterator,
};

pub fn dot_float(xs: &[f64], ys: &[f64]) -> f64 {
    xs.par_iter().zip(ys.par_iter()).map(|(&x, &y)| x * y).sum()
}

pub fn dot_int(xs: &[i64], ys: &[i64]) -> i64 {
    xs.par_iter().zip(ys.par_iter()).map(|(&x, &y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::{dot_float, dot_int};

    fn seq_dotf(xs: &[f64], ys: &[f64]) -> f64 {
        xs.iter().zip(ys.iter()).map(|(&x, &y)| x * y).sum()
    }

    fn seq_doti(xs: &[i64], ys: &[i64]) -> i64 {
        xs.iter().zip(ys.iter()).map(|(&x, &y)| x * y).sum()
    }

    #[test]
    fn float_test() {
        const VEC_SIZE: usize = 10_000_000;
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for _ in 1..=VEC_SIZE {
            xs.push(rand::random::<f64>());
            ys.push(rand::random::<f64>());
        }
        assert!((seq_dotf(&xs, &ys) - dot_float(&xs, &ys)).abs() < 1e-6);
    }

    #[test]
    fn int_test() {
        const VEC_SIZE: usize = 10_000_000;
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for _ in 1..=VEC_SIZE {
            xs.push(rand::thread_rng().gen_range(1..10_000));
            ys.push(rand::thread_rng().gen_range(1..10_000));
        }
        assert_eq!(seq_doti(&xs, &ys), dot_int(&xs, &ys));
    }
}
