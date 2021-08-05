use crate::kbest::{KBestMatchingIterator, Solution, State};
use all_lap_rust::bipartite::{BipartiteGraph, Matching, MaximumMatchingsIterator, Node};
use all_lap_rust::contains::Contains;
use float_cmp::ApproxEq;
use num_traits::Float;

fn solution_sparse_to_dense(s: Solution, lsize: usize) -> Matching {
    let mut l2r = std::iter::repeat(None).take(lsize).collect::<Vec<_>>();
    for (l, r) in s.0.into_iter().enumerate() {
        l2r[l] = Some(r);
    }
    Matching::new(l2r)
}

impl<T> From<State<T>> for (BipartiteGraph, Matching)
where
    State<T>: Ord,
    T: ApproxEq + Float + std::fmt::Debug,
    <T as float_cmp::ApproxEq>::Margin: std::convert::From<(f64, i64)>,
{
    fn from(state: State<T>) -> Self {
        let h = state.costs_reduced.nrows();
        let iter_nonzero = state
            .costs_reduced
            .indexed_iter()
            .filter_map(|(index, &item)| {
                if item.approx_eq(T::zero(), (1e-7, 2)) {
                    Some(index)
                } else {
                    None
                }
            });
        let mut adj: Vec<Vec<_>> = (0..h).map(|_| Vec::new()).collect();
        for (i, j) in iter_nonzero {
            adj[i].push(j);
        }
        (
            BipartiteGraph::from_adj(adj),
            solution_sparse_to_dense(state.a_solution, h),
        )
    }
}

pub struct SortedMatchingIterator<'a, T> {
    kbest_enum: KBestMatchingIterator<f64>,
    current_state_iter: Option<MaximumMatchingsIterator<'a, T>>,
    allowed_start_nodes: &'a T,
}

impl<'a, T> SortedMatchingIterator<'a, T> {
    pub fn new(kbest_enum: KBestMatchingIterator<f64>, allowed_start_nodes: &'a T) -> Self {
        Self {
            kbest_enum,
            current_state_iter: None,
            allowed_start_nodes,
        }
    }
}

impl<'a, T> Iterator for SortedMatchingIterator<'a, T>
where
    State<f64>: Ord,
    KBestMatchingIterator<f64>: Iterator,
    T: Contains<Node> + Contains<usize>,
{
    type Item = Matching;
    fn next(&mut self) -> Option<Self::Item> {
        let allowed = self.allowed_start_nodes;

        loop {
            if self.current_state_iter.is_none() {
                let s = self.kbest_enum.next()?;
                let (graph, matching): (BipartiteGraph, Matching) = s.into();
                let digraph = graph.as_directed(&matching);
                self.current_state_iter =
                    MaximumMatchingsIterator::new(graph, matching, digraph, allowed).into();
                continue;
            }
            let iterator = self.current_state_iter.as_mut().unwrap();
            let next = iterator.next();
            if next.is_none() {
                self.current_state_iter = None;
                continue;
            }
            return next;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SortedMatchingIterator;
    use crate::kbest::KBestMatchingIterator;
    use all_lap_rust::bipartite::{Node, NodeGroup, NodeSet};
    use ndarray::array;
    use std::iter::FromIterator;

    #[test]
    fn test_simple_enumeration() {
        let costs = array![[1., 1.], [1., 1.]];
        let kbest = KBestMatchingIterator::new(costs.clone()).unwrap();
        let allowed_start_nodes = NodeSet::new(
            std::collections::HashSet::from_iter(
                std::iter::repeat(NodeGroup::Left)
                    .zip(0..2)
                    .chain(std::iter::repeat(NodeGroup::Right).zip(0..2))
                    .map(|(lr, i)| Node::new(lr, i)),
            ),
            2,
        );
        let allenum = SortedMatchingIterator::new(kbest, &allowed_start_nodes);
        let matchings: Vec<_> = allenum.collect();
        let mut cur_cost = 0.;
        for m in matchings.iter() {
            let mut cost = 0.;
            for (l, r) in m.iter_pairs() {
                cost += costs[(l, r)];
            }
            println!("{:#?}", cost);
            assert!(cur_cost <= cost);
            cur_cost = cost;
        }
        assert_eq!(matchings.len(), 2);
    }

    #[test]
    fn test_complex_enumeration() {
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
        let allowed_start_nodes = NodeSet::new(
            std::collections::HashSet::from_iter(
                std::iter::repeat(NodeGroup::Left)
                    .zip(0..size)
                    .chain(std::iter::repeat(NodeGroup::Right).zip(0..size))
                    .map(|(lr, i)| Node::new(lr, i)),
            ),
            size,
        );
        let kbest = KBestMatchingIterator::new(costs.clone()).unwrap();
        let allenum = SortedMatchingIterator::new(kbest, &allowed_start_nodes);
        let matchings: Vec<_> = allenum.collect();

        // Assert cost ascending order
        let mut cur_cost = 0.;
        for m in matchings.iter() {
            let mut cost = 0.;
            for (l, r) in m.iter_pairs() {
                cost += costs[(l, r)];
            }
            println!("{:#?}", cost);
            assert!(cur_cost <= cost);
            cur_cost = cost;
        }

        // Assert counts
        assert_eq!(matchings.len(), factorial);
    }
}
