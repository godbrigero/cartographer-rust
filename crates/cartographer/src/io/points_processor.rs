//! Port of `cartographer/io/points_processor` (.h).

use crate::io::points_batch::PointsBatch;

/// Result of [`PointsProcessor::flush`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlushResult {
    /// The processor needs another pass over the data.
    RestartStream,
    /// The processor is done.
    Finished,
}

/// A processor in a pipeline. It processes a `points_batch` and hands it to
/// the next processor in the pipeline.
///
/// RUST PORT NOTE: in C++ every processor holds a non-owning
/// `PointsProcessor* next_` into a `std::vector<std::unique_ptr<...>>` owned by
/// the caller. Here every processor owns its successor as a
/// `Box<dyn PointsProcessor>`, so a pipeline is a single boxed head (the C++
/// `pipeline.back()`). The driver loop is unchanged:
/// `loop { for batch in source { head.process(batch) } if head.flush() == Finished { break } }`.
pub trait PointsProcessor: Send {
    /// Receive a `points_batch`, process it and pass it on.
    fn process(&mut self, points_batch: Box<PointsBatch>);

    /// Some implementations will perform expensive computations and others
    /// that do multiple passes over the data might ask for restarting the
    /// stream.
    fn flush(&mut self) -> FlushResult;
}
