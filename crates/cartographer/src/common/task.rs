//! Port of `cartographer/common/task` (.h, .cc).
//!
//! RUST PORT NOTE: tasks are created as `Arc<Task>` (C++ creates a
//! `unique_ptr<Task>` that is converted to a `shared_ptr` by `Schedule`),
//! because dependents register themselves with their dependencies before
//! being scheduled. Dependencies store `Arc`s of their dependents (C++ raw
//! pointers), which removes the dangling pointer hazard for dependents that
//! are never scheduled. The work item is a `FnOnce` (it runs at most once).

use std::sync::{Arc, Mutex, MutexGuard, Weak};

use super::thread_pool::DependenciesCompletedNotifier;

/// The work executed by a task.
pub type WorkItem = Box<dyn FnOnce() + Send + 'static>;

/// The life cycle of a task.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum State {
    /// Created, not yet scheduled.
    New,
    /// Scheduled, waiting for dependencies.
    Dispatched,
    /// Scheduled, all dependencies completed; queued for execution.
    DependenciesCompleted,
    /// The work item is running.
    Running,
    /// Done.
    Completed,
}

struct TaskInner {
    work_item: Option<WorkItem>,
    thread_pool_to_notify: Option<Weak<dyn DependenciesCompletedNotifier>>,
    state: State,
    uncompleted_dependencies: u32,
    dependent_tasks: Vec<Arc<Task>>,
}

/// A unit of work with dependencies on other tasks, executed by a
/// [`ThreadPoolInterface`](super::thread_pool::ThreadPoolInterface).
pub struct Task {
    inner: Mutex<TaskInner>,
}

impl std::fmt::Debug for Task {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Task")
            .field("state", &self.get_state())
            .finish()
    }
}

impl Task {
    /// Creates a task in state [`State::New`] without a work item.
    pub fn new() -> Arc<Task> {
        Arc::new(Task {
            inner: Mutex::new(TaskInner {
                work_item: None,
                thread_pool_to_notify: None,
                state: State::New,
                uncompleted_dependencies: 0,
                dependent_tasks: Vec::new(),
            }),
        })
    }

    fn lock(&self) -> MutexGuard<'_, TaskInner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Returns the current state.
    pub fn get_state(&self) -> State {
        self.lock().state
    }

    /// Sets the work item. State must be `New`.
    pub fn set_work_item(&self, work_item: impl FnOnce() + Send + 'static) {
        let mut inner = self.lock();
        assert_eq!(inner.state, State::New, "Check failed: state_ == NEW");
        inner.work_item = Some(Box::new(work_item));
    }

    /// Adds a dependency. State must be `New`. An expired `dependency` is
    /// assumed completed.
    pub fn add_dependency(self: &Arc<Self>, dependency: Weak<Task>) {
        let shared_dependency;
        {
            let mut inner = self.lock();
            assert_eq!(inner.state, State::New, "Check failed: state_ == NEW");
            shared_dependency = dependency.upgrade();
            if shared_dependency.is_some() {
                inner.uncompleted_dependencies += 1;
            }
        }
        if let Some(dependency) = shared_dependency {
            dependency.add_dependent_task(self);
        }
    }

    /// Allowed in all states.
    fn add_dependent_task(&self, dependent_task: &Arc<Task>) {
        let mut inner = self.lock();
        if inner.state == State::Completed {
            dependent_task.on_dependency_completed();
            return;
        }
        let inserted = !inner
            .dependent_tasks
            .iter()
            .any(|t| Arc::ptr_eq(t, dependent_task));
        assert!(inserted, "Given dependency is already a dependency.");
        inner.dependent_tasks.push(Arc::clone(dependent_task));
    }

    /// State must be `DependenciesCompleted` and becomes `Completed`.
    pub(crate) fn execute(&self) {
        let work_item = {
            let mut inner = self.lock();
            assert_eq!(
                inner.state,
                State::DependenciesCompleted,
                "Check failed: state_ == DEPENDENCIES_COMPLETED"
            );
            inner.state = State::Running;
            inner.work_item.take()
        };

        // Execute the work item.
        if let Some(work_item) = work_item {
            work_item();
        }

        let mut inner = self.lock();
        inner.state = State::Completed;
        for dependent_task in &inner.dependent_tasks {
            dependent_task.on_dependency_completed();
        }
    }

    /// State must be `New` and becomes `Dispatched` or
    /// `DependenciesCompleted`.
    pub(crate) fn set_thread_pool(
        self: &Arc<Self>,
        thread_pool: Weak<dyn DependenciesCompletedNotifier>,
    ) {
        let mut inner = self.lock();
        assert_eq!(inner.state, State::New, "Check failed: state_ == NEW");
        inner.state = State::Dispatched;
        inner.thread_pool_to_notify = Some(thread_pool);
        if inner.uncompleted_dependencies == 0 {
            inner.state = State::DependenciesCompleted;
            Self::notify(&inner, self);
        }
    }

    /// State must be `New` or `Dispatched`. If `Dispatched`, may become
    /// `DependenciesCompleted`.
    fn on_dependency_completed(self: &Arc<Self>) {
        let mut inner = self.lock();
        assert!(
            inner.state == State::New || inner.state == State::Dispatched,
            "Check failed: state_ == NEW || state_ == DISPATCHED"
        );
        inner.uncompleted_dependencies -= 1;
        if inner.uncompleted_dependencies == 0 && inner.state == State::Dispatched {
            inner.state = State::DependenciesCompleted;
            Self::notify(&inner, self);
        }
    }

    // Called with this task's lock held, like in C++.
    fn notify(inner: &TaskInner, task: &Arc<Task>) {
        let thread_pool = inner
            .thread_pool_to_notify
            .as_ref()
            .expect("Check failed: thread_pool_to_notify_");
        // RUST PORT NOTE: if the thread pool is already gone (UB in C++),
        // the notification is dropped.
        if let Some(thread_pool) = thread_pool.upgrade() {
            thread_pool.notify_dependencies_completed(task);
        }
    }
}

impl Drop for Task {
    fn drop(&mut self) {
        let state = self.inner.get_mut().map_or(State::New, |inner| inner.state);
        if state != State::New && state != State::Completed {
            log::warn!("Delete Task between dispatch and completion.");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::thread_pool::{ThreadPoolInterface, execute, set_thread_pool};
    use std::collections::{BTreeMap, VecDeque};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct FakeState {
        task_queue: VecDeque<Arc<Task>>,
        tasks_not_ready: BTreeMap<usize, Arc<Task>>,
    }

    #[derive(Default)]
    struct FakeShared {
        state: Mutex<FakeState>,
    }

    impl DependenciesCompletedNotifier for FakeShared {
        fn notify_dependencies_completed(&self, task: &Arc<Task>) {
            let mut state = self.state.lock().unwrap();
            let task = state
                .tasks_not_ready
                .remove(&(Arc::as_ptr(task) as usize))
                .expect("task not found");
            state.task_queue.push_back(task);
        }
    }

    /// Synchronous thread pool: tasks run when `run_next` is called.
    #[derive(Default)]
    struct FakeThreadPool {
        shared: Arc<FakeShared>,
    }

    impl ThreadPoolInterface for FakeThreadPool {
        fn schedule(&self, task: Arc<Task>) -> Weak<Task> {
            let inserted = self
                .shared
                .state
                .lock()
                .unwrap()
                .tasks_not_ready
                .insert(Arc::as_ptr(&task) as usize, Arc::clone(&task))
                .is_none();
            assert!(inserted);
            let notifier: Weak<FakeShared> = Arc::downgrade(&self.shared);
            set_thread_pool(&task, notifier);
            Arc::downgrade(&task)
        }
    }

    impl FakeThreadPool {
        fn run_next(&self) {
            let task = {
                let state = self.shared.state.lock().unwrap();
                assert!(!state.task_queue.is_empty());
                Arc::clone(state.task_queue.front().unwrap())
            };
            execute(&task);
            self.shared.state.lock().unwrap().task_queue.pop_front();
        }

        fn is_empty(&self) -> bool {
            self.shared.state.lock().unwrap().task_queue.is_empty()
        }
    }

    /// Counts calls and records the global call order.
    #[derive(Clone)]
    struct MockCallback {
        name: &'static str,
        calls: Arc<AtomicUsize>,
        sequence: Arc<Mutex<Vec<&'static str>>>,
    }

    impl MockCallback {
        fn new(name: &'static str, sequence: &Arc<Mutex<Vec<&'static str>>>) -> Self {
            MockCallback {
                name,
                calls: Arc::default(),
                sequence: Arc::clone(sequence),
            }
        }

        fn run(&self) {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.sequence.lock().unwrap().push(self.name);
        }

        fn work_item(&self) -> impl FnOnce() + Send + 'static {
            let callback = self.clone();
            move || callback.run()
        }

        fn times(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    mod task_test {
        use super::*;

        #[test]
        fn run_task() {
            let thread_pool = FakeThreadPool::default();
            let sequence = Arc::default();
            let a = Task::new();
            let callback = MockCallback::new("a", &sequence);
            a.set_work_item(callback.work_item());
            assert_eq!(a.get_state(), State::New);
            let shared_a = thread_pool.schedule(a).upgrade();
            assert!(shared_a.is_some());
            let shared_a = shared_a.unwrap();
            assert_eq!(shared_a.get_state(), State::DependenciesCompleted);
            thread_pool.run_next();
            assert_eq!(callback.times(), 1);
            assert_eq!(shared_a.get_state(), State::Completed);
            assert!(thread_pool.is_empty());
        }

        #[test]
        fn run_task_with_dependency() {
            let thread_pool = FakeThreadPool::default();
            let sequence = Arc::default();
            let a = Task::new();
            let b = Task::new();
            let callback_a = MockCallback::new("a", &sequence);
            a.set_work_item(callback_a.work_item());
            let callback_b = MockCallback::new("b", &sequence);
            b.set_work_item(callback_b.work_item());
            assert_eq!(a.get_state(), State::New);
            assert_eq!(b.get_state(), State::New);
            let shared_a = thread_pool.schedule(a).upgrade().unwrap();
            b.add_dependency(Arc::downgrade(&shared_a));
            let shared_b = thread_pool.schedule(b).upgrade().unwrap();
            assert_eq!(shared_b.get_state(), State::Dispatched);
            assert_eq!(shared_a.get_state(), State::DependenciesCompleted);
            thread_pool.run_next();
            assert_eq!(shared_b.get_state(), State::DependenciesCompleted);
            thread_pool.run_next();
            assert_eq!(shared_a.get_state(), State::Completed);
            assert_eq!(shared_b.get_state(), State::Completed);
            assert_eq!(callback_a.times(), 1);
            assert_eq!(callback_b.times(), 1);
            assert_eq!(*sequence.lock().unwrap(), vec!["a", "b"]);
        }

        #[test]
        fn run_task_with_two_dependency() {
            //         c \
            //  a -->  b --> d
            let thread_pool = FakeThreadPool::default();
            let sequence = Arc::default();
            let a = Task::new();
            let b = Task::new();
            let c = Task::new();
            let d = Task::new();
            let callbacks: Vec<MockCallback> = ["a", "b", "c", "d"]
                .iter()
                .map(|n| MockCallback::new(n, &sequence))
                .collect();
            a.set_work_item(callbacks[0].work_item());
            b.set_work_item(callbacks[1].work_item());
            c.set_work_item(callbacks[2].work_item());
            d.set_work_item(callbacks[3].work_item());
            let shared_a = thread_pool.schedule(a).upgrade().unwrap();
            b.add_dependency(Arc::downgrade(&shared_a));
            let shared_b = thread_pool.schedule(b).upgrade().unwrap();
            let shared_c = thread_pool.schedule(c).upgrade().unwrap();
            d.add_dependency(Arc::downgrade(&shared_b));
            d.add_dependency(Arc::downgrade(&shared_c));
            let shared_d = thread_pool.schedule(d).upgrade().unwrap();
            assert_eq!(shared_b.get_state(), State::Dispatched);
            assert_eq!(shared_d.get_state(), State::Dispatched);
            thread_pool.run_next();
            assert_eq!(shared_a.get_state(), State::Completed);
            assert_eq!(shared_b.get_state(), State::DependenciesCompleted);
            assert_eq!(shared_c.get_state(), State::DependenciesCompleted);
            thread_pool.run_next();
            thread_pool.run_next();
            assert_eq!(shared_b.get_state(), State::Completed);
            assert_eq!(shared_c.get_state(), State::Completed);
            assert_eq!(shared_d.get_state(), State::DependenciesCompleted);
            thread_pool.run_next();
            assert_eq!(shared_d.get_state(), State::Completed);
            for callback in &callbacks {
                assert_eq!(callback.times(), 1);
            }
        }

        #[test]
        fn run_with_completed_dependency() {
            let thread_pool = FakeThreadPool::default();
            let sequence = Arc::default();
            let a = Task::new();
            let callback_a = MockCallback::new("a", &sequence);
            a.set_work_item(callback_a.work_item());
            let shared_a = thread_pool.schedule(a).upgrade().unwrap();
            assert_eq!(shared_a.get_state(), State::DependenciesCompleted);
            thread_pool.run_next();
            assert_eq!(callback_a.times(), 1);
            assert_eq!(shared_a.get_state(), State::Completed);
            let b = Task::new();
            let callback_b = MockCallback::new("b", &sequence);
            b.set_work_item(callback_b.work_item());
            b.add_dependency(Arc::downgrade(&shared_a));
            assert_eq!(b.get_state(), State::New);
            let shared_b = thread_pool.schedule(b).upgrade().unwrap();
            assert_eq!(shared_b.get_state(), State::DependenciesCompleted);
            thread_pool.run_next();
            assert_eq!(callback_b.times(), 1);
            assert_eq!(shared_b.get_state(), State::Completed);
        }
    }

    #[test]
    fn expired_dependency_is_completed() {
        let b = Task::new();
        {
            let a = Task::new();
            let weak = Arc::downgrade(&a);
            drop(a);
            b.add_dependency(weak);
        }
        let thread_pool = FakeThreadPool::default();
        let b = thread_pool.schedule(b).upgrade().unwrap();
        assert_eq!(b.get_state(), State::DependenciesCompleted);
        thread_pool.run_next();
        assert_eq!(b.get_state(), State::Completed);
    }

    #[test]
    #[should_panic(expected = "Given dependency is already a dependency.")]
    fn adding_a_dependency_twice_panics() {
        let a = Task::new();
        let b = Task::new();
        b.add_dependency(Arc::downgrade(&a));
        b.add_dependency(Arc::downgrade(&a));
    }

    #[test]
    #[should_panic(expected = "state_ == NEW")]
    fn set_work_item_after_schedule_panics() {
        let thread_pool = FakeThreadPool::default();
        let a = thread_pool.schedule(Task::new()).upgrade().unwrap();
        a.set_work_item(|| {});
    }
}
