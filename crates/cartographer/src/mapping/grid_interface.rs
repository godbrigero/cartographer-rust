//! Port of `cartographer/mapping/grid_interface` (.h).

use std::any::Any;

/// Common base of all grids (`Grid2D` implementors, `HybridGrid`).
///
/// C++ `GridInterface` is an empty polymorphic base; range data inserters
/// receive a `GridInterface*` and `static_cast` it to the concrete grid. In
/// Rust this is a marker trait with `Any` as supertrait so that
/// `dyn GridInterface` can be downcast safely with [`downcast_ref`] /
/// [`downcast_mut`] (a failed downcast corresponds to C++ UB and is reported
/// as `None`).
///
/// [`downcast_ref`]: trait.GridInterface.html#method.downcast_ref
/// [`downcast_mut`]: trait.GridInterface.html#method.downcast_mut
pub trait GridInterface: Any + Send + Sync {}

impl dyn GridInterface {
    /// Returns the concrete grid if it is of type `T` (C++ `static_cast`).
    pub fn downcast_ref<T: GridInterface>(&self) -> Option<&T> {
        (self as &dyn Any).downcast_ref::<T>()
    }

    /// Returns the concrete grid if it is of type `T` (C++ `static_cast`).
    pub fn downcast_mut<T: GridInterface>(&mut self) -> Option<&mut T> {
        (self as &mut dyn Any).downcast_mut::<T>()
    }
}

#[cfg(test)]
mod tests {
    use super::GridInterface;

    struct A;
    impl GridInterface for A {}
    struct B;
    impl GridInterface for B {}

    #[test]
    fn downcast() {
        let mut a: Box<dyn GridInterface> = Box::new(A);
        assert!(a.downcast_ref::<A>().is_some());
        assert!(a.downcast_ref::<B>().is_none());
        assert!(a.downcast_mut::<A>().is_some());
    }
}
