//! Port of `cartographer/common/internal/testing/thread_pool_for_testing` (.h, .cc).

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};
use std::thread::JoinHandle;

use crate::common::task::Task;
use crate::common::thread_pool::{
    DependenciesCompletedNotifier, ThreadPoolInterface, execute, set_thread_pool,
};
use crate::common::time::from_seconds;

struct PoolState {
    running: bool,
    idle: bool,
    task_queue: VecDeque<Arc<Task>>,
    tasks_not_ready: BTreeMap<usize, Arc<Task>>,
}

struct Shared {
    state: Mutex<PoolState>,
    condition: Condvar,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, PoolState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl DependenciesCompletedNotifier for Shared {
    fn notify_dependencies_completed(&self, task: &Arc<Task>) {
        let mut state = self.lock();
        assert!(state.running, "Check failed: running_");
        let task = state
            .tasks_not_ready
            .remove(&(Arc::as_ptr(task) as usize))
            .expect("Check failed: it != tasks_not_ready_.end()");
        state.task_queue.push_back(task);
        self.condition.notify_all();
    }
}

/// A single-threaded thread pool for tests that allows waiting until all
/// scheduled work is done.
pub struct ThreadPoolForTesting {
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
}

impl Default for ThreadPoolForTesting {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ThreadPoolForTesting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ThreadPoolForTesting")
    }
}

impl ThreadPoolForTesting {
    /// Starts the worker thread.
    pub fn new() -> Self {
        let shared = Arc::new(Shared {
            state: Mutex::new(PoolState {
                running: true,
                idle: true,
                task_queue: VecDeque::new(),
                tasks_not_ready: BTreeMap::new(),
            }),
            condition: Condvar::new(),
        });
        let worker_shared = Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name("thread-pool-for-testing".to_string())
            .spawn(move || do_work(&worker_shared))
            .expect("failed to spawn thread");
        ThreadPoolForTesting {
            shared,
            thread: Some(thread),
        }
    }

    /// Blocks until there is no scheduled or queued work left.
    pub fn wait_until_idle(&self) {
        loop {
            let state = self.shared.lock();
            let (state, _) = self
                .shared
                .condition
                .wait_timeout_while(state, from_seconds(0.1).to_std(), |s| !s.idle)
                .unwrap_or_else(|e| e.into_inner());
            if state.idle {
                return;
            }
        }
    }
}

fn do_work(shared: &Shared) {
    loop {
        let task = {
            let state = shared.lock();
            let (mut state, _) = shared
                .condition
                .wait_timeout_while(state, from_seconds(0.1).to_std(), |s| {
                    s.task_queue.is_empty() && s.running
                })
                .unwrap_or_else(|e| e.into_inner());
            let task = state.task_queue.pop_front();
            if !state.running {
                return;
            }
            if state.tasks_not_ready.is_empty() && state.task_queue.is_empty() && task.is_none() {
                state.idle = true;
                shared.condition.notify_all();
            }
            task
        };
        if let Some(task) = task {
            execute(&task);
        }
    }
}

impl ThreadPoolInterface for ThreadPoolForTesting {
    fn schedule(&self, task: Arc<Task>) -> Weak<Task> {
        {
            let mut state = self.shared.lock();
            state.idle = false;
            assert!(state.running, "Check failed: running_");
            let inserted = state
                .tasks_not_ready
                .insert(Arc::as_ptr(&task) as usize, Arc::clone(&task))
                .is_none();
            assert!(inserted, "ScheduleWhenReady called twice");
        }
        let weak_task = Arc::downgrade(&task);
        set_thread_pool(&task, Arc::downgrade(&self.shared));
        weak_task
    }
}

impl Drop for ThreadPoolForTesting {
    fn drop(&mut self) {
        {
            let mut state = self.shared.lock();
            assert!(state.running, "Check failed: running_");
            state.running = false;
            self.shared.condition.notify_all();
            if !std::thread::panicking() {
                assert_eq!(
                    state.task_queue.len(),
                    0,
                    "Check failed: task_queue_.size() == 0"
                );
                assert_eq!(
                    state.tasks_not_ready.len(),
                    0,
                    "Check failed: tasks_not_ready_.size() == 0"
                );
            }
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn runs_tasks_with_dependencies_and_becomes_idle() {
        let pool = ThreadPoolForTesting::new();
        pool.wait_until_idle();
        let counter = Arc::new(AtomicUsize::new(0));
        let order = Arc::new(Mutex::new(Vec::new()));
        let first = Task::new();
        {
            let counter = Arc::clone(&counter);
            let order = Arc::clone(&order);
            first.set_work_item(move || {
                counter.fetch_add(1, Ordering::SeqCst);
                order.lock().unwrap().push(1);
            });
        }
        let second = Task::new();
        {
            let counter = Arc::clone(&counter);
            let order = Arc::clone(&order);
            second.set_work_item(move || {
                counter.fetch_add(1, Ordering::SeqCst);
                order.lock().unwrap().push(2);
            });
        }
        // Schedule the dependent first; it waits for `first`.
        let first_handle = Arc::downgrade(&first);
        second.add_dependency(first_handle);
        pool.schedule(second);
        pool.schedule(first);
        pool.wait_until_idle();
        assert_eq!(counter.load(Ordering::SeqCst), 2);
        assert_eq!(*order.lock().unwrap(), vec![1, 2]);
    }
}
