use crate::kbest::{KBestMatchingIterator, Solution, State};
use crate::lapjv::Matrix;
use all_lap_rust::bipartite::{BipartiteGraph, MaximumMatchingCalculator, Node};
use all_lap_rust::contains::Contains;
use float_cmp::ApproxEq;
use num_traits::Float;

pub type Matching = all_lap_rust::bipartite::Matching;

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

pub struct SortedMatchingCalculator {
    kbest_enum: KBestMatchingIterator<f64>,
    current_state_iter: Option<MaximumMatchingCalculator>,
}

impl SortedMatchingCalculator {
    pub fn new(kbest_enum: KBestMatchingIterator<f64>) -> Self {
        Self {
            kbest_enum,
            current_state_iter: None as Option<MaximumMatchingCalculator>,
        }
    }

    pub fn next_item(
        &mut self,
        allowed_start_nodes: &'_ (impl Contains<Node> + Contains<usize>),
    ) -> Option<Matching> {
        loop {
            if self.current_state_iter.is_none() {
                let s = self.kbest_enum.next()?;
                let (graph, matching): (BipartiteGraph, Matching) = s.into();
                let digraph = graph.as_directed(&matching);
                self.current_state_iter =
                    MaximumMatchingCalculator::new(graph, matching, digraph).into();
                continue;
            }
            let next = self
                .current_state_iter
                .as_mut()
                .unwrap()
                .next_item(allowed_start_nodes);
            if next.is_none() {
                self.current_state_iter = None;
                continue;
            }
            return next;
        }
    }

    pub fn iter_match<T>(self, allowed_start_nodes: &'_ T) -> impl Iterator<Item = Matching> + '_
    where
        T: Contains<usize> + Contains<Node>,
    {
        SortedMatchingIterator {
            inner: self,
            allowed_start_nodes,
        }
    }

    pub fn from_costs(costs: Matrix<f64>) -> Self {
        let kbest = KBestMatchingIterator::new(costs).unwrap();
        Self::new(kbest)
    }
}

struct SortedMatchingIterator<'a, T> {
    inner: SortedMatchingCalculator,
    allowed_start_nodes: &'a T,
}

impl<'a, T> Iterator for SortedMatchingIterator<'a, T>
where
    T: Contains<usize> + Contains<Node>,
{
    type Item = Matching;
    fn next(&mut self) -> Option<Matching> {
        self.inner.next_item(self.allowed_start_nodes)
    }
}

#[cfg(test)]
mod tests {
    use super::SortedMatchingCalculator;
    use crate::util::make_valid_nodeset;
    use ndarray::array;
    use num_traits::float::Float;

    #[test]
    fn test_simple_enumeration() {
        let costs = array![[1., 1.], [1., 1.]];
        let allowed_start_nodes = make_valid_nodeset(2, 2, 2);
        let matchings: Vec<_> = SortedMatchingCalculator::from_costs(costs.clone())
            .iter_match(&allowed_start_nodes)
            .collect();
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
    fn test_simple_enumeration_with_exclusion() {
        // Imagine there is a detection system.
        // The detection system can overdetect, or underdetect (fail to recall).
        // Let's say we know the number of items to be detected: N_gt (ground truth)
        // And we ran the detection system, and some items were detected: N_ob (observed)
        //               N_ob                     N_gt (dummy)
        //
        //    N_gt   (recall cost)           (underdetection cost)
        //
        //    N_ob   (overdetection cost)            zeros
        //  (dummy)
        //
        // And we enumerate all the possible cases of detection system's behavior.

        // In case where 2 items are expected with 2 detection results,
        let inf = f64::infinity();
        let costs = array![
            // ob1  ob2    (gt1) (gt2)  Observation \ Ground Truth
            [1.0, 2.0, 1e3, inf,], // gt1
            [10., 20., inf, 1e3,], // gt2
            [1e2, inf, 0.0, 0.0,], // (ob1)
            [inf, 1e2, 0.0, 0.0,], // (ob2)
        ];
        // Possible perfect recall case: (12, 21)
        // either one is recalled:  (1100 + (1, 2, 10, 20))
        // Both items were not detected: (2200)
        let allowed_start_nodes = make_valid_nodeset(2, 2, 4);
        let matchings: Vec<_> = SortedMatchingCalculator::from_costs(costs.clone())
            .iter_match(&allowed_start_nodes)
            .collect();
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
        assert_eq!(matchings.len(), 2 + 4 + 1);
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
        let allowed_start_nodes = make_valid_nodeset(size, size, size);
        let matchings: Vec<_> = SortedMatchingCalculator::from_costs(costs.clone())
            .iter_match(&allowed_start_nodes)
            .collect();

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
