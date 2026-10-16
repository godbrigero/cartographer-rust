//! Port of `cartographer/common/internal/blocking_queue` (.h).

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex, MutexGuard};

use crate::common::time::Duration;

/// A thread-safe blocking queue that is useful for producer/consumer
/// patterns. `T` must be movable.
///
/// RUST PORT NOTE: `PopWithTimeout` returns `Option<T>` instead of a null
/// pointer, and `Peek`/`PeekWithTimeout` take a closure that gets a reference
/// to the front element (a reference cannot escape the lock in Rust).
#[derive(Debug)]
pub struct BlockingQueue<T> {
    queue_size: usize,
    deque: Mutex<VecDeque<T>>,
    // Notified whenever the deque changes.
    changed: Condvar,
}

impl<T> Default for BlockingQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> BlockingQueue<T> {
    /// Queue size meaning "unbounded".
    pub const INFINITE_QUEUE_SIZE: usize = 0;

    /// Constructs a blocking queue with infinite queue size.
    pub fn new() -> Self {
        Self::with_queue_size(Self::INFINITE_QUEUE_SIZE)
    }

    /// Constructs a blocking queue with a size of `queue_size`.
    pub fn with_queue_size(queue_size: usize) -> Self {
        BlockingQueue {
            queue_size,
            deque: Mutex::new(VecDeque::new()),
            changed: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, VecDeque<T>> {
        self.deque.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn queue_not_full(&self, deque: &VecDeque<T>) -> bool {
        self.queue_size == Self::INFINITE_QUEUE_SIZE || deque.len() < self.queue_size
    }

    /// Pushes a value onto the queue. Blocks if the queue is full.
    pub fn push(&self, t: T) {
        let guard = self.lock();
        let mut guard = self
            .changed
            .wait_while(guard, |d| !self.queue_not_full(d))
            .unwrap_or_else(|e| e.into_inner());
        guard.push_back(t);
        self.changed.notify_all();
    }

    /// Like push, but returns false if `timeout` is reached.
    pub fn push_with_timeout(&self, t: T, timeout: Duration) -> bool {
        let guard = self.lock();
        let (mut guard, result) = self
            .changed
            .wait_timeout_while(guard, timeout.to_std(), |d| !self.queue_not_full(d))
            .unwrap_or_else(|e| e.into_inner());
        if result.timed_out() && !self.queue_not_full(&guard) {
            return false;
        }
        guard.push_back(t);
        self.changed.notify_all();
        true
    }

    /// Pops the next value from the queue. Blocks until a value is available.
    pub fn pop(&self) -> T {
        let guard = self.lock();
        let mut guard = self
            .changed
            .wait_while(guard, |d| d.is_empty())
            .unwrap_or_else(|e| e.into_inner());
        let t = guard.pop_front().expect("queue is not empty");
        self.changed.notify_all();
        t
    }

    /// Like pop, but can timeout. Returns `None` in this case.
    pub fn pop_with_timeout(&self, timeout: Duration) -> Option<T> {
        let guard = self.lock();
        let (mut guard, _) = self
            .changed
            .wait_timeout_while(guard, timeout.to_std(), |d| d.is_empty())
            .unwrap_or_else(|e| e.into_inner());
        let t = guard.pop_front()?;
        self.changed.notify_all();
        Some(t)
    }

    /// Like peek, but can timeout. Returns `None` in this case.
    pub fn peek_with_timeout<R>(&self, timeout: Duration, f: impl FnOnce(&T) -> R) -> Option<R> {
        let guard = self.lock();
        let (guard, _) = self
            .changed
            .wait_timeout_while(guard, timeout.to_std(), |d| d.is_empty())
            .unwrap_or_else(|e| e.into_inner());
        guard.front().map(f)
    }

    /// Calls `f` with the next value in the queue, or returns `None` if the
    /// queue is empty. Maintains ownership.
    pub fn peek<R>(&self, f: impl FnOnce(&T) -> R) -> Option<R> {
        self.lock().front().map(f)
    }

    /// Returns the number of items currently in the queue.
    pub fn size(&self) -> usize {
        self.lock().len()
    }

    /// Blocks until the queue is empty.
    pub fn wait_until_empty(&self) {
        let guard = self.lock();
        let _guard = self
            .changed
            .wait_while(guard, |d| !d.is_empty())
            .unwrap_or_else(|e| e.into_inner());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::time::from_milliseconds;
    use std::sync::Arc;

    mod blocking_queue_test {
        use super::*;

        #[test]
        fn test_push_peek_pop() {
            let blocking_queue: BlockingQueue<Box<i32>> = BlockingQueue::new();
            blocking_queue.push(Box::new(42));
            assert_eq!(1, blocking_queue.size());
            blocking_queue.push(Box::new(24));
            assert_eq!(2, blocking_queue.size());
            assert_eq!(Some(42), blocking_queue.peek(|v| **v));
            assert_eq!(2, blocking_queue.size());
            assert_eq!(42, *blocking_queue.pop());
            assert_eq!(1, blocking_queue.size());
            assert_eq!(24, *blocking_queue.pop());
            assert_eq!(0, blocking_queue.size());
            assert_eq!(None, blocking_queue.peek(|v| **v));
            assert_eq!(0, blocking_queue.size());
        }

        #[test]
        fn test_push_pop_shared_ptr() {
            let blocking_queue: BlockingQueue<Arc<i32>> = BlockingQueue::new();
            blocking_queue.push(Arc::new(42));
            blocking_queue.push(Arc::new(24));
            assert_eq!(42, *blocking_queue.pop());
            assert_eq!(24, *blocking_queue.pop());
        }

        #[test]
        fn test_pop_with_timeout() {
            let blocking_queue: BlockingQueue<Box<i32>> = BlockingQueue::new();
            assert_eq!(
                None,
                blocking_queue.pop_with_timeout(from_milliseconds(150))
            );
        }

        #[test]
        fn test_push_with_timeout() {
            let blocking_queue: BlockingQueue<Box<i32>> = BlockingQueue::with_queue_size(1);
            assert!(blocking_queue.push_with_timeout(Box::new(42), from_milliseconds(150)));
            assert!(!blocking_queue.push_with_timeout(Box::new(15), from_milliseconds(150)));
            assert_eq!(42, *blocking_queue.pop());
            assert_eq!(0, blocking_queue.size());
        }

        #[test]
        fn test_push_with_timeout_infinte_queue() {
            let blocking_queue: BlockingQueue<Box<i32>> = BlockingQueue::new();
            assert!(blocking_queue.push_with_timeout(Box::new(42), from_milliseconds(150)));
            assert!(blocking_queue.push_with_timeout(Box::new(45), from_milliseconds(150)));
            assert_eq!(42, *blocking_queue.pop());
            assert_eq!(45, *blocking_queue.pop());
            assert_eq!(0, blocking_queue.size());
        }

        #[test]
        fn test_blocking_pop() {
            let blocking_queue: BlockingQueue<Box<i32>> = BlockingQueue::new();
            assert_eq!(0, blocking_queue.size());
            let pop = std::thread::scope(|s| {
                let thread = s.spawn(|| *blocking_queue.pop());
                std::thread::sleep(from_milliseconds(100).to_std());
                blocking_queue.push(Box::new(42));
                thread.join().unwrap()
            });
            assert_eq!(0, blocking_queue.size());
            assert_eq!(42, pop);
        }

        #[test]
        fn test_blocking_pop_with_timeout() {
            let blocking_queue: BlockingQueue<Box<i32>> = BlockingQueue::new();
            assert_eq!(0, blocking_queue.size());
            let pop = std::thread::scope(|s| {
                let thread = s.spawn(|| {
                    *blocking_queue
                        .pop_with_timeout(from_milliseconds(2500))
                        .expect("value before timeout")
                });
                std::thread::sleep(from_milliseconds(100).to_std());
                blocking_queue.push(Box::new(42));
                thread.join().unwrap()
            });
            assert_eq!(0, blocking_queue.size());
            assert_eq!(42, pop);
        }
    }

    #[test]
    fn peek_with_timeout_and_wait_until_empty() {
        let queue: Arc<BlockingQueue<i32>> = Arc::new(BlockingQueue::new());
        assert_eq!(None, queue.peek_with_timeout(from_milliseconds(10), |v| *v));
        queue.push(7);
        assert_eq!(
            Some(7),
            queue.peek_with_timeout(from_milliseconds(10), |v| *v)
        );
        let q = Arc::clone(&queue);
        let consumer = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(20));
            q.pop()
        });
        queue.wait_until_empty();
        assert_eq!(consumer.join().unwrap(), 7);
        assert_eq!(queue.size(), 0);
    }

    #[test]
    fn push_blocks_while_full() {
        let queue: Arc<BlockingQueue<i32>> = Arc::new(BlockingQueue::with_queue_size(1));
        queue.push(1);
        let q = Arc::clone(&queue);
        let producer = std::thread::spawn(move || q.push(2));
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert_eq!(queue.pop(), 1);
        producer.join().unwrap();
        assert_eq!(queue.pop(), 2);
    }
}
