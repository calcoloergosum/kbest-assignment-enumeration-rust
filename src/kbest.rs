use crate::lapjv::{LapJV, LapJVCost, LapJVError, Matrix};
use num_traits::Float;
use ordered_float::NotNan;
use std::collections::BinaryHeap;

fn get_dual<T>(lap: &LapJV<T>) -> (Vec<T>, Vec<T>)
where
    T: LapJVCost,
{
    let mut u = vec![T::zero(); lap.v.len()];
    let mut v = lap.v.to_owned();
    for (r, c) in lap.in_row.iter().enumerate() {
        if !lap.v[*c].is_finite() {
            v[*c] = T::zero();
        }
        u[r] = lap.costs[(r, *c)] - v[*c];
        if !u[r].is_finite() {
            u[r] = T::zero();
        }
    }
    (u, v)
}

#[derive(Debug)]
pub enum KBestEnumerationError {
    InfeasibleMatrix,
    InternalError(&'static str),
}

impl From<LapJVError> for KBestEnumerationError {
    fn from(err: LapJVError) -> Self {
        KBestEnumerationError::InternalError(err.0)
    }
}

pub type Solution = (Vec<usize>, Vec<usize>);

#[derive(Clone, Debug)]
pub struct State<T> {
    pub cost_solution: NotNan<T>,
    pub costs_reduced: Matrix<T>,
    pub a_solution: Solution,
}

impl<T> State<T> {
    fn new(cost_solution: NotNan<T>, costs_reduced: Matrix<T>, a_solution: Solution) -> Self {
        State {
            cost_solution,
            costs_reduced,
            a_solution,
        }
    }
}

impl<T> PartialOrd for State<T>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // NOTE: reversed to make the binary heap a min heap (not max heap).
        other.cost_solution.partial_cmp(&self.cost_solution)
    }
}

impl<T> Eq for State<T> where T: PartialEq {}

impl<T> PartialEq for State<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.cost_solution == other.cost_solution
    }
}

impl<T> std::cmp::Ord for State<T>
where
    State<T>: PartialOrd + Eq,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(&other).unwrap()
    }
}

pub struct KBestMatchingIterator<T> {
    heap: BinaryHeap<State<T>>,
    last: Option<State<T>>,
}

impl<T> KBestMatchingIterator<T> {
    pub fn new(cost_mat: Matrix<T>) -> Result<Self, KBestEnumerationError>
    where
        T: LapJVCost + From<f32>,
        State<T>: Ord,
    {
        debug_assert_eq!(cost_mat.dim().0, cost_mat.dim().1);
        // # Find first solution
        let state = _reduce_matrix(&cost_mat)?;

        // # keep track of how many branches we searched
        // # to be able to avoid overlapping value in the heap
        let mut heap = BinaryHeap::new();
        heap.push(state);
        Ok(KBestMatchingIterator { heap, last: None })
    }
}

impl<T> Iterator for KBestMatchingIterator<T>
where
    State<T>: Ord,
    T: LapJVCost + From<f32>,
{
    type Item = State<T>;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.last {
            // # split the solution space into n partitions
            None => {}
            Some(parent) => {
                populate_heap(parent, &mut self.heap);
            }
        }

        let last_cost = match &self.last {
            None => T::zero(),
            Some(l) => l.cost_solution.into_inner(),
        };
        self.last = None;

        let state = self.heap.pop()?;
        debug_assert!(state.cost_solution.into_inner() > last_cost - f32::epsilon().into());

        self.last = Some(state.clone());
        Some(state)
    }
}

fn _reduce_matrix<T>(costs: &Matrix<T>) -> Result<State<T>, KBestEnumerationError>
where
    T: LapJVCost + std::ops::Add + From<f32>,
{
    let n = costs.dim().0;
    // if a row or column is filled with infinity, return as infeasible
    for i in 0..n {
        if (0..n).all(|j| costs[(i, j)] == T::infinity()) {
            return Err(KBestEnumerationError::InfeasibleMatrix);
        }
        if (0..n).all(|j| costs[(j, i)] == T::infinity()) {
            return Err(KBestEnumerationError::InfeasibleMatrix);
        }
    }
    let mut lapjv = LapJV::new(costs);
    lapjv.solve()?;

    let (u, v) = get_dual(&lapjv);

    let mut reduced_cost_matrix = costs.clone();
    let n = costs.dim().0;
    for r in 0..n {
        for c in 0..n {
            reduced_cost_matrix[(r, c)] -= u[r] + v[c];
            debug_assert!(
                reduced_cost_matrix[(r, c)].is_nan()
                    || reduced_cost_matrix[(r, lapjv.in_row[r])]
                        > T::zero() - f32::epsilon().into()
            );
        }
    }

    let value: T = lapjv
        .in_row
        .iter()
        .enumerate()
        .map(|(i, j)| costs[(i, *j)])
        .fold(T::zero(), |x, y| x + y);
    if !value.is_finite() {
        return Err(KBestEnumerationError::InfeasibleMatrix);
    };
    let cost = NotNan::new(value);
    match cost {
        Err(_) => Err(KBestEnumerationError::InfeasibleMatrix),
        Ok(_cost) => Ok(State::new(
            _cost,
            reduced_cost_matrix,
            (lapjv.in_row, lapjv.in_col),
        )),
    }
}

fn populate_heap<T>(parent: &State<T>, heap: &mut BinaryHeap<State<T>>)
where
    T: LapJVCost + From<f32>,
    State<T>: Ord,
{
    // # admissible edges
    let n = parent.costs_reduced.dim().0;
    let admissible_edges = iproduct![0..n, 0..n]
        .filter(|(i, j)| parent.costs_reduced[(*i, *j)].abs() < 0.0000001.into())
        .collect::<Vec<_>>();
    let parent_matrix = &parent.costs_reduced;
    for i in 0..n {
        let mut child_matrix = parent_matrix.clone();

        // do nothing for [:i]th rows

        // # remove admissible edges for ith row
        for (_i, _j) in &admissible_edges {
            if *_i != i {
                continue;
            }
            child_matrix[(*_i, *_j)] = T::infinity();
        }

        // # leave admissible edges only for [i+1:]th rows
        for _i in (i + 1)..n {
            for j in 0..n {
                child_matrix[(_i, j)] = T::infinity();
            }
        }
        for (_i, _j) in &admissible_edges {
            if *_i <= i {
                continue;
            }
            child_matrix[(*_i, *_j)] = parent.costs_reduced[(*_i, *_j)];
        }

        // # solve and put into heap
        match _reduce_matrix(&child_matrix) {
            Err(_) => {
                continue;
            }
            Ok(mut s) => {
                if !s.cost_solution.is_finite() {
                    continue;
                }
                s.cost_solution += parent.cost_solution;
                heap.push(s);
            }
        };
    }
}

#[cfg(test)]
mod tests {
    extern crate ndarray_rand;
    extern crate ordered_float;

    use super::{get_dual, KBestMatchingIterator};
    use crate::lapjv::LapJV;
    // use all_lap_rust::bipartite::BipartiteGraph;
    use itertools::Itertools;
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
        // Make matrix of size < 10 as follows:
        // [[1,   2,   3],
        //  [10,  20,  30],
        //  [100, 200, 300]],
        // the cost should be
        // 123, 132, 213, 231, 312, 321
        let data = iproduct![0..size, 1..(size + 1)]
            .map(|(i, j)| (j as f64) * (10_f64).powi(i as i32))
            .collect::<Vec<_>>();
        let costs = ndarray::Array2::from_shape_vec((size, size), data).unwrap();
        let factorial: usize = (1..(size + 1)).product();
        // solution count
        let kbest = KBestMatchingIterator::new(costs.clone()).unwrap();
        let mut solutions = vec![];
        for s in kbest {
            solutions.push(s);
        }
        assert_eq!(solutions.len(), factorial);

        // solution sort
        let mut solutions_sorted = solutions.clone();
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

    #[test]
    fn test_degenerated_enumeration() {
        let size: usize = 6;
        // Make matrix of size < 10 as follows:
        // [[1,   1,   2,   2],
        //  [10,  10,  20,  20],
        //  [100, 100, 200, 200]
        //  [1000, 1000, 2000, 2000]],
        // each of them overlapping 4 times
        // 1122, 1212, 1221, 2112, 2121, 2211
        // in case of 2*n, solution count is n! / 2^n
        let data = iproduct![0..size, 1..(size / 2 + 1)]
            .map(|(i, j)| (j as f64) * (10_f64).powi(i as i32))
            .flat_map(|val| std::iter::repeat(val).take(2))
            .collect::<Vec<_>>();
        let costs = ndarray::Array2::from_shape_vec((size, size), data).unwrap();
        let factorial: usize = (1..(size + 1)).product();
        // solution count
        let kbest = KBestMatchingIterator::new(costs).unwrap();
        let mut solutions = vec![];
        for s in kbest {
            solutions.push(s);
        }
        assert_eq!(
            solutions.len(),
            factorial / (2_usize).pow((size / 2) as u32)
        );

        // solution sort
        let mut solutions_sorted = solutions.clone();
        solutions_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        assert_eq!(solutions_sorted, solutions);
    }
}
