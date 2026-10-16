//! Port of `cartographer/mapping/internal/submap_controller` (.h, .cc).

use std::sync::{Arc, PoisonError, RwLock};

use crate::mapping::id::{MapById, SubmapId};
use crate::mapping::proto;
use crate::mapping::submaps::{SharedSubmap, Submap};
use crate::mapping::three_d::submap_3d::Submap3D;
use crate::mapping::two_d::submap_2d::Submap2D;
use crate::mapping::value_conversion_tables::ValueConversionTables;

/// Submap types that a [`SubmapController`] can create from protos.
///
/// RUST PORT NOTE: replaces the C++ explicit specializations of
/// `SubmapController<SubmapType>::CreateSubmap`.
pub trait ControllableSubmap: Submap + Sized {
    /// Creates a submap from `proto`, or returns `None` if the proto holds a
    /// partially filled submap (C++ `CreateSubmap`).
    fn create_submap(
        proto: &proto::Submap,
        conversion_tables: &Arc<ValueConversionTables>,
    ) -> Option<Self>;
}

impl ControllableSubmap for Submap2D {
    fn create_submap(
        proto: &proto::Submap,
        conversion_tables: &Arc<ValueConversionTables>,
    ) -> Option<Self> {
        let submap_2d = proto.submap_2d.clone().unwrap_or_default();
        if submap_2d.num_range_data != 1 {
            log::warn!(
                "Refusing to create partially filled submap: {}",
                submap_2d.num_range_data
            );
            return None;
        }
        Some(Submap2D::from_proto(&submap_2d, conversion_tables))
    }
}

impl ControllableSubmap for Submap3D {
    fn create_submap(
        proto: &proto::Submap,
        _conversion_tables: &Arc<ValueConversionTables>,
    ) -> Option<Self> {
        let submap_3d = proto.submap_3d.clone().unwrap_or_default();
        if submap_3d.num_range_data != 1 {
            log::info!(
                "Refusing to create partially filled submap: {}",
                submap_3d.num_range_data
            );
            return None;
        }
        Some(Submap3D::from_proto(&submap_3d))
    }
}

/// Keeps track of unfinished submaps received as protos (e.g. from a local
/// SLAM uplink) and updates them incrementally.
pub struct SubmapController<S> {
    unfinished_submaps: MapById<SubmapId, SharedSubmap<S>>,
    conversion_tables: Arc<ValueConversionTables>,
}

impl<S> std::fmt::Debug for SubmapController<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubmapController")
            .field("num_unfinished_submaps", &self.unfinished_submaps.size())
            .finish_non_exhaustive()
    }
}

impl<S: ControllableSubmap> Default for SubmapController<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: ControllableSubmap> SubmapController<S> {
    /// Creates a controller without submaps.
    pub fn new() -> Self {
        Self {
            unfinished_submaps: MapById::new(),
            conversion_tables: Arc::new(ValueConversionTables::new()),
        }
    }

    /// Creates or updates the submap described by `proto`. Returns `None` if
    /// a new, partially filled submap was refused.
    pub fn update_submap(&mut self, proto: &proto::Submap) -> Option<SharedSubmap<S>> {
        let submap_id_proto = proto.submap_id.unwrap_or_default();
        let submap_id = SubmapId::new(submap_id_proto.trajectory_id, submap_id_proto.submap_index);
        let Some(submap_ptr) = self.unfinished_submaps.get(&submap_id).cloned() else {
            let submap_ptr = S::create_submap(proto, &self.conversion_tables)
                .map(|submap| Arc::new(RwLock::new(submap)));
            if let Some(submap_ptr) = &submap_ptr {
                self.unfinished_submaps
                    .insert(&submap_id, Arc::clone(submap_ptr));
            }
            return submap_ptr;
        };
        let mut submap = submap_ptr.write().unwrap_or_else(PoisonError::into_inner);
        submap.update_from_proto(proto);
        // If the submap was just finished by the recent update, remove it from
        // the list of unfinished submaps.
        if submap.insertion_finished() {
            self.unfinished_submaps.trim(&submap_id);
        } else {
            // If the submap is unfinished set the 'num_range_data' to 0 since
            // we haven't changed the HybridGrid.
            submap.set_num_range_data(0);
        }
        drop(submap);
        Some(submap_ptr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn submap_2d_proto(index: i32, num_range_data: i32, finished: bool) -> proto::Submap {
        proto::Submap {
            submap_id: Some(proto::SubmapId {
                trajectory_id: 0,
                submap_index: index,
            }),
            submap_2d: Some(proto::Submap2D {
                num_range_data,
                finished,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn refuses_partially_filled_new_submaps() {
        let mut controller = SubmapController::<Submap2D>::new();
        assert!(
            controller
                .update_submap(&submap_2d_proto(0, 2, false))
                .is_none()
        );
        let mut controller = SubmapController::<Submap3D>::new();
        let proto = proto::Submap {
            submap_3d: Some(proto::Submap3D {
                num_range_data: 5,
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(controller.update_submap(&proto).is_none());
    }

    #[test]
    fn updates_and_trims_submaps() {
        let mut controller = SubmapController::<Submap2D>::new();
        let created = controller
            .update_submap(&submap_2d_proto(0, 1, false))
            .unwrap();
        assert_eq!(created.read().unwrap().num_range_data(), 1);

        let updated = controller
            .update_submap(&submap_2d_proto(0, 7, false))
            .unwrap();
        assert!(Arc::ptr_eq(&created, &updated));
        // Unfinished updates reset 'num_range_data' to 0.
        assert_eq!(updated.read().unwrap().num_range_data(), 0);

        let finished = controller
            .update_submap(&submap_2d_proto(0, 9, true))
            .unwrap();
        assert!(Arc::ptr_eq(&created, &finished));
        assert!(finished.read().unwrap().insertion_finished());
        assert_eq!(finished.read().unwrap().num_range_data(), 9);
        assert_eq!(controller.unfinished_submaps.size(), 0);

        // A finished submap was forgotten, so an update creates a new one.
        assert!(
            controller
                .update_submap(&submap_2d_proto(0, 9, true))
                .is_none()
        );
    }
}
