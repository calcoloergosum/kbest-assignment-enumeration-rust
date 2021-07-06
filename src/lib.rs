#![cfg_attr(all(feature = "nightly", test), feature(test))]
#[macro_use]
extern crate itertools;
extern crate ndarray;
extern crate num_traits;
#[cfg(test)]
extern crate rand;
#[macro_use]
extern crate log;

#[cfg(all(feature = "nightly", test))]
extern crate test;
use std::io::{self, Read};
pub mod lapjv;

use lapjv::LapJV;
use lapjv::LapJVCost;
use lapjv::Matrix;
use num_traits::Float;
use ordered_float::NotNan;
use std::collections::BinaryHeap;

pub fn get_dual<'a, T>(lap: &'a LapJV<T>) -> (Vec<T>, Vec<T>)
where
    T: LapJVCost,
{
    let mut u = vec![T::zero(); lap.v.len()];
    let mut v = lap.v.to_owned();
    for r in 0..lap.in_row.len() {
        let c = lap.in_row[r];
        if !lap.v[c].is_finite() {
            v[c] = T::zero();
        }
        u[r] = lap.costs[(r, c)] - v[c];
        if !u[r].is_finite() {
            u[r] = T::zero();
        }
    }
    (u, v)
    // let mut u = vec![T::zero(); lap.v.len()];
    // for r in 0..lap.in_row.len() {
    //     let c = lap.in_row[r];
    //     u[r] = lap.costs[(r, c)] - lap.v[c];
    // }
    // (u, lap.v.to_owned())
}

#[derive(Debug)]
pub enum KBestEnumerationError {
    InfeasibleMatrix,
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

pub struct KBestEnumeration<T> {
    heap: BinaryHeap<State<T>>,
    last: Option<State<T>>,
}

impl<T> KBestEnumeration<T> {
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
        Ok(KBestEnumeration { heap, last: None })
    }
}

impl<T> Iterator for KBestEnumeration<T>
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

        if cfg!(debug_assertions) {
            // println!("{:#?}", state.costs_reduced);
            // println!("{:#?}", state.cost_solution.into_inner());
            // println!("{:#?}", state.a_solution);
            // let mut buffer = String::new();
            // let mut stdin = io::stdin(); // We get `Stdin` here.
            // stdin.read_to_string(&mut buffer);
        }
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
    let err = lapjv.solve();
    if err.is_err() {
        return Err(KBestEnumerationError::InfeasibleMatrix);
    }

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

    let value: T = lapjv.in_row.iter().enumerate().map(|(i, j)| costs[(i, *j)]).fold(T::zero(), |x, y| {x + y});
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
