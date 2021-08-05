use all_lap_rust::bipartite::{Node, NodeGroup, NodeSet};
use std::collections::HashSet;

pub fn make_valid_hashset(n_valid_left: usize, n_valid_right: usize) -> HashSet<Node> {
    std::iter::repeat(NodeGroup::Left)
        .zip(0..n_valid_left)
        .chain(std::iter::repeat(NodeGroup::Right).zip(0..n_valid_right))
        .map(|(lr, i)| Node::new(lr, i))
        .collect()
}

pub fn make_valid_nodeset(n_valid_left: usize, n_valid_right: usize, lsize: usize) -> NodeSet {
    NodeSet::new(make_valid_hashset(n_valid_left, n_valid_right), lsize)
}
