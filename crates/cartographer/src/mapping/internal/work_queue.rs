//! Port of `cartographer/mapping/internal/work_queue` (.h).
//!
//! RUST PORT NOTE: the C++ `std::function<Result()>` work items capture the
//! pose graph's `this`. Here a work item is a one-shot closure that receives
//! the pose graph's shared state (`Ctx`, e.g. `pose_graph_2d::Inner`) as an
//! argument instead of capturing it: the queue is stored inside that state,
//! so capturing an `Arc` to it would create a reference cycle. The types are
//! generic over `Ctx` so that the 2D and 3D pose graphs share them.

use std::collections::VecDeque;
use std::time::Instant;

/// Whether the pose graph has to run an optimization after a work item
/// (C++ `WorkItem::Result`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WorkItemResult {
    /// C++ `kDoNotRunOptimization`.
    DoNotRunOptimization,
    /// C++ `kRunOptimization`.
    RunOptimization,
}

/// The work to do for a [`WorkItem`].
pub type WorkItemTask<Ctx> = Box<dyn FnOnce(&Ctx) -> WorkItemResult + Send + 'static>;

/// A deferred task of the pose graph, executed in order on the thread pool.
pub struct WorkItem<Ctx: ?Sized> {
    /// When the item was enqueued (C++ `std::chrono::steady_clock`).
    pub time: Instant,
    /// The work to do.
    pub task: WorkItemTask<Ctx>,
}

impl<Ctx: ?Sized> std::fmt::Debug for WorkItem<Ctx> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkItem")
            .field("time", &self.time)
            .finish_non_exhaustive()
    }
}

/// FIFO of work items (C++ `std::deque<WorkItem>`).
pub type WorkQueue<Ctx> = VecDeque<WorkItem<Ctx>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn work_items_run_in_fifo_order() {
        let mut queue: WorkQueue<Vec<i32>> = WorkQueue::new();
        for i in 0..3 {
            queue.push_back(WorkItem {
                time: Instant::now(),
                task: Box::new(move |ctx: &Vec<i32>| {
                    if ctx[i] == 1 {
                        WorkItemResult::RunOptimization
                    } else {
                        WorkItemResult::DoNotRunOptimization
                    }
                }),
            });
        }
        let ctx = vec![0, 1, 0];
        let results: Vec<_> = queue.drain(..).map(|item| (item.task)(&ctx)).collect();
        assert_eq!(
            results,
            vec![
                WorkItemResult::DoNotRunOptimization,
                WorkItemResult::RunOptimization,
                WorkItemResult::DoNotRunOptimization
            ]
        );
    }
}
