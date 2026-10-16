//! Port of `cartographer/common/thread_pool` (.h, .cc).

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};
use std::thread::JoinHandle;

use super::task::{State, Task};

/// Receives the notification that all dependencies of a scheduled task have
/// completed (C++ private virtual
/// `ThreadPoolInterface::NotifyDependenciesCompleted`).
///
/// RUST PORT NOTE: split from [`ThreadPoolInterface`] because tasks keep a
/// `Weak` handle to the (shared part of the) pool that they notify, while
/// thread pools themselves are plain values owned by their users.
pub trait DependenciesCompletedNotifier: Send + Sync {
    /// Moves `task` from the set of not-ready tasks to the execution queue.
    fn notify_dependencies_completed(&self, task: &Arc<Task>);
}

/// Interface of thread pools that execute [`Task`]s.
pub trait ThreadPoolInterface: Send + Sync {
    /// Schedules `task` (which must be in state `New`). The returned handle
    /// expires once the task has completed (as long as the caller does not
    /// keep other `Arc`s to it).
    fn schedule(&self, task: Arc<Task>) -> Weak<Task>;
}

/// Runs a task (C++ protected `ThreadPoolInterface::Execute`).
pub fn execute(task: &Arc<Task>) {
    task.execute();
}

/// Dispatches a task to a thread pool (C++ protected
/// `ThreadPoolInterface::SetThreadPool`).
pub fn set_thread_pool<N: DependenciesCompletedNotifier + 'static>(
    task: &Arc<Task>,
    thread_pool: Weak<N>,
) {
    let thread_pool: Weak<dyn DependenciesCompletedNotifier> = thread_pool;
    task.set_thread_pool(thread_pool);
}

struct PoolState {
    running: bool,
    task_queue: VecDeque<Arc<Task>>,
    tasks_not_ready: HashMap<usize, Arc<Task>>,
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
        let task = state
            .tasks_not_ready
            .remove(&(Arc::as_ptr(task) as usize))
            .expect("Check failed: it != tasks_not_ready_.end()");
        state.task_queue.push_back(task);
        self.condition.notify_one();
    }
}

/// A fixed number of threads working on tasks. Adding a task does not block.
/// Tasks may be added whether or not their dependencies are completed.
/// When all dependencies of a task are completed, it is queued up for
/// execution in a background thread. The queue must be empty before calling
/// the destructor. The thread pool will then wait for the currently
/// executing work items to finish and then destroy the threads.
pub struct ThreadPool {
    shared: Arc<Shared>,
    pool: Vec<JoinHandle<()>>,
}

impl std::fmt::Debug for ThreadPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThreadPool")
            .field("num_threads", &self.pool.len())
            .finish()
    }
}

impl ThreadPool {
    /// Creates a pool with `num_threads` worker threads.
    pub fn new(num_threads: i32) -> Self {
        assert!(
            num_threads > 0,
            "Check failed: num_threads > 0 ({num_threads} vs. 0) ThreadPool requires a positive num_threads!"
        );
        let shared = Arc::new(Shared {
            state: Mutex::new(PoolState {
                running: true,
                task_queue: VecDeque::new(),
                tasks_not_ready: HashMap::new(),
            }),
            condition: Condvar::new(),
        });
        let pool = (0..num_threads)
            .map(|i| {
                let shared = Arc::clone(&shared);
                std::thread::Builder::new()
                    .name(format!("cartographer-thread-pool-{i}"))
                    .spawn(move || do_work(&shared))
                    .expect("failed to spawn thread")
            })
            .collect();
        ThreadPool { shared, pool }
    }
}

// RUST PORT NOTE: the C++ worker threads lower their priority with
// `nice(10)` on Linux; this needs `unsafe` libc calls and is omitted.
//
// RUST PORT NOTE: a failed `CHECK`/`LOG(FATAL)` inside a task aborts the
// whole C++ process. A Rust panic would only unwind the worker thread and
// leave everything waiting on that task (e.g. the pose graph work queue)
// blocked forever, so a panicking task aborts the process as well, after
// the panic message has been printed by the panic hook.
fn do_work(shared: &Shared) {
    loop {
        let task = {
            let state = shared.lock();
            let mut state = shared
                .condition
                .wait_while(state, |s| s.task_queue.is_empty() && s.running)
                .unwrap_or_else(|e| e.into_inner());
            match state.task_queue.pop_front() {
                Some(task) => task,
                None => {
                    debug_assert!(!state.running);
                    return;
                }
            }
        };
        assert_eq!(
            task.get_state(),
            State::DependenciesCompleted,
            "Check failed: task->GetState() == common::Task::DEPENDENCIES_COMPLETED"
        );
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| execute(&task)));
        if result.is_err() {
            log::error!("A task panicked on a ThreadPool worker thread; aborting.");
            eprintln!("A task panicked on a ThreadPool worker thread; aborting.");
            std::process::abort();
        }
    }
}

impl ThreadPoolInterface for ThreadPool {
    fn schedule(&self, task: Arc<Task>) -> Weak<Task> {
        {
            let mut state = self.shared.lock();
            let inserted = state
                .tasks_not_ready
                .insert(Arc::as_ptr(&task) as usize, Arc::clone(&task))
                .is_none();
            assert!(inserted, "Schedule called twice");
        }
        let weak_task = Arc::downgrade(&task);
        set_thread_pool(&task, Arc::downgrade(&self.shared));
        weak_task
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        {
            let mut state = self.shared.lock();
            assert!(state.running, "Check failed: running_");
            state.running = false;
            self.shared.condition.notify_all();
        }
        for thread in self.pool.drain(..) {
            // Workers never unwind (a panicking task aborts the process).
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Receiver {
        received_numbers: Mutex<Vec<i32>>,
        condition: Condvar,
    }

    impl Receiver {
        fn receive(&self, number: i32) {
            self.received_numbers.lock().unwrap().push(number);
            self.condition.notify_all();
        }

        fn wait_for_number_sequence(&self, expected_numbers: &[i32]) {
            let numbers = self.received_numbers.lock().unwrap();
            let numbers = self
                .condition
                .wait_while(numbers, |n| n.len() < expected_numbers.len())
                .unwrap();
            assert_eq!(expected_numbers, &numbers[..]);
        }
    }

    fn task_receiving(receiver: &Arc<Receiver>, number: i32) -> Arc<Task> {
        let task = Task::new();
        let receiver = Arc::clone(receiver);
        task.set_work_item(move || receiver.receive(number));
        task
    }

    mod thread_pool_test {
        use super::*;

        #[test]
        fn run_task() {
            let pool = ThreadPool::new(1);
            let receiver = Arc::new(Receiver::default());
            let task = task_receiving(&receiver, 1);
            pool.schedule(task);
            receiver.wait_for_number_sequence(&[1]);
        }

        #[test]
        fn many_tasks() {
            for _ in 0..5 {
                let pool = ThreadPool::new(3);
                let receiver = Arc::new(Receiver::default());
                let num_tasks = 10;
                for _ in 0..num_tasks {
                    pool.schedule(task_receiving(&receiver, 1));
                }
                receiver.wait_for_number_sequence(&vec![1; num_tasks]);
            }
        }

        #[test]
        fn run_with_dependency() {
            let pool = ThreadPool::new(2);
            let receiver = Arc::new(Receiver::default());
            let task_2 = task_receiving(&receiver, 2);
            let task_1 = task_receiving(&receiver, 1);
            let weak_task_1 = pool.schedule(task_1);
            task_2.add_dependency(weak_task_1);
            pool.schedule(task_2);
            receiver.wait_for_number_sequence(&[1, 2]);
        }

        #[test]
        fn run_with_out_of_scope_dependency() {
            let pool = ThreadPool::new(2);
            let receiver = Arc::new(Receiver::default());
            let task_2 = task_receiving(&receiver, 2);
            {
                let task_1 = task_receiving(&receiver, 1);
                let weak_task_1 = pool.schedule(task_1);
                task_2.add_dependency(weak_task_1);
            }
            pool.schedule(task_2);
            receiver.wait_for_number_sequence(&[1, 2]);
        }

        #[test]
        fn many_dependencies() {
            for _ in 0..5 {
                let pool = ThreadPool::new(5);
                let receiver = Arc::new(Receiver::default());
                let num_dependencies = 10;
                let task = task_receiving(&receiver, 1);
                for _ in 0..num_dependencies {
                    let dependency_task = Task::new();
                    dependency_task.set_work_item(|| {});
                    task.add_dependency(pool.schedule(dependency_task));
                }
                pool.schedule(task);
                receiver.wait_for_number_sequence(&[1]);
            }
        }

        #[test]
        fn many_dependants() {
            for _ in 0..5 {
                let pool = ThreadPool::new(5);
                let receiver = Arc::new(Receiver::default());
                let num_dependants = 10;
                let dependency_task = Task::new();
                dependency_task.set_work_item(|| {});
                let dependency_handle = pool.schedule(dependency_task);
                for _ in 0..num_dependants {
                    let task = Task::new();
                    task.add_dependency(dependency_handle.clone());
                    let receiver = Arc::clone(&receiver);
                    task.set_work_item(move || receiver.receive(1));
                    pool.schedule(task);
                }
                receiver.wait_for_number_sequence(&vec![1; num_dependants]);
            }
        }

        #[test]
        fn run_with_multiple_dependencies() {
            let pool = ThreadPool::new(2);
            let receiver = Arc::new(Receiver::default());
            let task_1 = task_receiving(&receiver, 1);
            let task_2a = task_receiving(&receiver, 2);
            let task_2b = task_receiving(&receiver, 2);
            let task_3 = task_receiving(&receiver, 3);
            //          -> task_2a \
            //  task_1 /-> task_2b --> task_3
            let weak_task_1 = pool.schedule(task_1);
            task_2a.add_dependency(weak_task_1.clone());
            let weak_task_2a = pool.schedule(task_2a);
            task_3.add_dependency(weak_task_1.clone());
            task_3.add_dependency(weak_task_2a);
            task_2b.add_dependency(weak_task_1);
            let weak_task_2b = pool.schedule(task_2b);
            task_3.add_dependency(weak_task_2b);
            pool.schedule(task_3);
            receiver.wait_for_number_sequence(&[1, 2, 2, 3]);
        }

        #[test]
        fn run_with_finished_dependency() {
            let pool = ThreadPool::new(2);
            let receiver = Arc::new(Receiver::default());
            let task_1 = task_receiving(&receiver, 1);
            let task_2 = task_receiving(&receiver, 2);
            let weak_task_1 = pool.schedule(task_1);
            task_2.add_dependency(weak_task_1);
            receiver.wait_for_number_sequence(&[1]);
            pool.schedule(task_2);
            receiver.wait_for_number_sequence(&[1, 2]);
        }
    }

    #[test]
    #[should_panic(expected = "ThreadPool requires a positive num_threads!")]
    fn zero_threads_panics() {
        ThreadPool::new(0);
    }

    #[test]
    fn weak_handle_expires_after_completion() {
        let pool = ThreadPool::new(1);
        let receiver = Arc::new(Receiver::default());
        let weak = pool.schedule(task_receiving(&receiver, 1));
        receiver.wait_for_number_sequence(&[1]);
        // The worker drops its reference right after execution.
        let start = std::time::Instant::now();
        while weak.upgrade().is_some() {
            assert!(start.elapsed() < std::time::Duration::from_secs(10));
            std::thread::yield_now();
        }
    }

    #[test]
    fn drop_drains_the_ready_queue() {
        let receiver = Arc::new(Receiver::default());
        {
            let pool = ThreadPool::new(1);
            for _ in 0..20 {
                pool.schedule(task_receiving(&receiver, 1));
            }
        }
        assert_eq!(receiver.received_numbers.lock().unwrap().len(), 20);
    }
}
