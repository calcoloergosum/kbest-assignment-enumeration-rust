#[macro_use]
extern crate float_cmp;
#[macro_use]
extern crate itertools;

#[cfg(test)]
mod tests {
    extern crate ndarray_rand;
    extern crate ordered_float;

    use itertools::Itertools;
    use kbest_lap::lapjv::LapJV;
    use kbest_lap::{get_dual, KBestEnumeration};
    use ndarray::{array, Array};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use num_traits::float::Float;

    #[test]
    fn test_infty() {
        let size = 3;
        let inf = f64::infinity();
        let costs = array![[10., 1., inf], [inf, 0., 0.], [0., inf, inf]];
        let answer = 1.;
        // let costs = array![[18.0, 99.0, 0.0], [inf, 90.0, inf], [inf, 0.0, 0.0]];
        // let answer = 108.0;
        let mut lap = LapJV::new(&costs);
        match lap.solve() {
            Err(_) => panic!("test failed"),
            Ok(_) => {
                let (u, v) = get_dual(&lap);
                assert_eq!(lap.in_row, vec![1, 2, 0]);
                assert!(approx_eq!(
                    f64,
                    (0..size).map(|i| costs[(i, lap.in_row[i])]).sum(),
                    answer,
                    epsilon = f32::epsilon().into()
                ));
                assert!(approx_eq!(
                    f64,
                    u.iter().sum::<f64>() + v.iter().sum::<f64>(),
                    answer,
                    epsilon = f32::epsilon().into()
                ));
            }
        }
    }

    #[test]
    fn test_get_dual() {
        let size = 5;
        let costs = Array::random((size, size), Uniform::new(1_f64, 10.));
        let mut lap = LapJV::new(&costs);
        match lap.solve() {
            Err(_) => panic!("test failed"),
            Ok(_) => {
                let (u, v) = get_dual(&lap);
                for i in 0..size {
                    for j in 0..size {
                        assert!(u[i] + v[j] < (costs[(i, j)] + f32::epsilon() as f64));
                    }
                }
            }
        }
    }

    #[test]
    fn test_permutations() {
        let size: usize = 5;
        // let costs = Array::random((size, size), Uniform::new(1. as f64, 10.));
        // the answers should be
        // 123, 132, 213, 231, 312, 321
        let data = iproduct![0..size, 1..(size + 1)]
            .map(|(i, j)| (j as f64) * (10_f64).powi(i as i32))
            .collect::<Vec<_>>();
        let costs = ndarray::Array2::from_shape_vec((size, size), data).unwrap();
        let factorial: usize = (1..(size + 1)).product();
        // solution count
        let mut kbest = KBestEnumeration::new(costs.clone()).unwrap();
        let mut solutions = vec![];
        for _ in 0..(factorial + 1) {
            match kbest.next() {
                None => break,
                Some(a_solution) => {
                    solutions.push(a_solution);
                }
            };
        }
        // let solutions = KBestEnumeration::new(costs)
        //     .iter()
        //     .collect::<Vec<State<f64>>>();
        // assert_eq!(solutions.len(), factorial);

        // solution sort
        let mut solutions_sorted = solutions.clone();
        solutions_sorted.sort();
        solutions_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        assert_eq!(solutions_sorted, solutions);

        // compare with brute-force
        let mut bf_costs = (0..size)
            .permutations(size)
            .map(|v| v.iter().enumerate().map(|(i, x)| costs[(i, *x)]).sum())
            .collect::<Vec<_>>();
        bf_costs.sort_by(|a: &f64, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let solution_costs = solutions
            .iter()
            .map(|s| s.cost_solution)
            .collect::<Vec<_>>();
        for (x, y) in solution_costs.iter().zip(bf_costs.iter()) {
            println!("{:#?} {:#?}", x.into_inner(), *y);
            assert!(approx_eq!(
                f64,
                x.into_inner(),
                *y,
                epsilon = f32::epsilon().into()
            ));
        }
    }
}
