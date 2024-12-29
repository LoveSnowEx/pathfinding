//! Theta* algorithm.
use indexmap::map::Entry::{Occupied, Vacant};
use num_traits::Zero;
use std::cmp::Ordering;
use std::{collections::BinaryHeap, hash::Hash};

use super::reverse_path;
use crate::FxIndexMap;

/// Theta* algorithm.
#[expect(clippy::missing_panics_doc)]
pub fn theta_star<N, C, FN, IN, FH, FS, FV>(
    start: &N,
    mut successors: FN,
    mut heuristic: FH,
    mut success: FS,
    mut sight: FV,
) -> Option<(Vec<N>, C)>
where
    N: Eq + Hash + Clone,
    C: Zero + Ord + Copy,
    FN: FnMut(&N) -> IN,
    IN: IntoIterator<Item = (N, C)>,
    FH: FnMut(&N) -> C,
    FS: FnMut(&N) -> bool,
    FV: FnMut(&N, &N) -> Option<C>,
{
    let mut to_see = BinaryHeap::new();
    to_see.push(SmallestCostHolder {
        estimated_cost: Zero::zero(),
        cost: Zero::zero(),
        index: 0,
    });
    let mut parents: FxIndexMap<N, (usize, C)> = FxIndexMap::default();
    parents.insert(start.clone(), (usize::MAX, Zero::zero()));
    while let Some(SmallestCostHolder { cost, index, .. }) = to_see.pop() {
        let (successors, parent_index) = {
            let (node, &(parent_index, c)) = parents.get_index(index).unwrap(); // Cannot fail
            if success(node) {
                let path = reverse_path(&parents, |&(p, _)| p, index);
                return Some((path, cost));
            }
            // We may have inserted a node several time into the binary heap if we found
            // a better way to access it. Ensure that we are currently dealing with the
            // best path and discard the others.
            if cost > c {
                continue;
            }
            (successors(node), parent_index)
        };
        for (successor, move_cost) in successors {
            let (index, cost, move_cost) =
                if let Some((parent, (_, parent_cost))) = parents.get_index(parent_index) {
                    if let Some(parent_move_cost) = sight(&successor, parent) {
                        (parent_index, *parent_cost, parent_move_cost)
                    } else {
                        (index, cost, move_cost)
                    }
                } else {
                    (index, cost, move_cost)
                };
            let new_cost = cost + move_cost;
            let h; // heuristic(&successor)
            let n; // index for successor
            match parents.entry(successor) {
                Vacant(e) => {
                    h = heuristic(e.key());
                    n = e.index();
                    e.insert((index, new_cost));
                }
                Occupied(mut e) => {
                    if e.get().1 > new_cost {
                        h = heuristic(e.key());
                        n = e.index();
                        e.insert((index, new_cost));
                    } else {
                        continue;
                    }
                }
            }

            to_see.push(SmallestCostHolder {
                estimated_cost: new_cost + h,
                cost: new_cost,
                index: n,
            });
        }
    }
    None
}

/// This structure is used to implement Rust's max-heap as a min-heap
/// version for A*. The smallest `estimated_cost` (which is the sum of
/// the `cost` and the heuristic) is preferred. For the same
/// `estimated_cost`, the highest `cost` will be favored, as it may
/// indicate that the goal is nearer, thereby requiring fewer
/// exploration steps.
struct SmallestCostHolder<K> {
    estimated_cost: K,
    cost: K,
    index: usize,
}

impl<K: PartialEq> PartialEq for SmallestCostHolder<K> {
    fn eq(&self, other: &Self) -> bool {
        self.estimated_cost.eq(&other.estimated_cost) && self.cost.eq(&other.cost)
    }
}

impl<K: PartialEq> Eq for SmallestCostHolder<K> {}

impl<K: Ord> PartialOrd for SmallestCostHolder<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: Ord> Ord for SmallestCostHolder<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        match other.estimated_cost.cmp(&self.estimated_cost) {
            Ordering::Equal => self.cost.cmp(&other.cost),
            s => s,
        }
    }
}
