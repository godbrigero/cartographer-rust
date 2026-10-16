//! Port of `cartographer/io/draw_trajectories` (.h, .cc).

use std::f64::consts::PI;

use nalgebra::Vector2;

use crate::io::cairo::{Context, Surface};
use crate::io::color::FloatColor;
use crate::mapping::proto;
use crate::transform::rigid_transform::Rigid3d;
use crate::transform::transform::to_rigid3;

/// Maps a pose to the (integer) pixel it is drawn at
/// (C++ `std::function<Eigen::Array2i(const transform::Rigid3d&)>`).
pub type PoseToPixelFunction<'a> = dyn Fn(&Rigid3d) -> Vector2<i32> + 'a;

/// Draws the `trajectory` into the `surface`: a 4 pixel wide line in `color`
/// (alpha 0.7) and a green start / red end marker of radius 6.
pub fn draw_trajectory(
    trajectory: &proto::Trajectory,
    color: &FloatColor,
    pose_to_pixel: &PoseToPixelFunction<'_>,
    surface: &mut Surface,
) {
    if trajectory.node.is_empty() {
        return;
    }
    const TRAJECTORY_WIDTH: f64 = 4.;
    const TRAJECTORY_END_MARKERS: f64 = 6.;
    const ALPHA: f64 = 0.7;

    let pixel_of =
        |node: &proto::trajectory::Node| pose_to_pixel(&to_rigid3(&node.pose.unwrap_or_default()));

    let mut cr = Context::new(surface);
    cr.set_source_rgba(
        f64::from(color[0]),
        f64::from(color[1]),
        f64::from(color[2]),
        ALPHA,
    );
    cr.set_line_width(TRAJECTORY_WIDTH);
    for node in &trajectory.node {
        let pixel = pixel_of(node);
        cr.line_to(f64::from(pixel.x), f64::from(pixel.y));
    }
    cr.stroke();

    // Draw beginning and end markers.
    {
        let pixel = pixel_of(&trajectory.node[0]);
        cr.set_source_rgba(0., 1., 0., ALPHA);
        cr.arc(
            f64::from(pixel.x),
            f64::from(pixel.y),
            TRAJECTORY_END_MARKERS,
            0.,
            2. * PI,
        );
        cr.fill();
    }
    {
        let pixel = pixel_of(trajectory.node.last().unwrap());
        cr.set_source_rgba(1., 0., 0., ALPHA);
        cr.arc(
            f64::from(pixel.x),
            f64::from(pixel.y),
            TRAJECTORY_END_MARKERS,
            0.,
            2. * PI,
        );
        cr.fill();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::transform::to_proto;
    use nalgebra::Vector3;

    fn trajectory(points: &[(f64, f64)]) -> proto::Trajectory {
        proto::Trajectory {
            node: points
                .iter()
                .map(|&(x, y)| proto::trajectory::Node {
                    pose: Some(to_proto(&Rigid3d::from_translation(Vector3::new(x, y, 0.)))),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }

    fn to_pixel(pose: &Rigid3d) -> Vector2<i32> {
        Vector2::new(pose.translation().x as i32, pose.translation().y as i32)
    }

    #[test]
    fn empty_trajectory_draws_nothing() {
        let mut surface = Surface::new(10, 10);
        draw_trajectory(&trajectory(&[]), &[1., 1., 1.], &to_pixel, &mut surface);
        assert!(surface.data().iter().all(|&p| p == 0));
    }

    #[test]
    fn draws_line_and_markers() {
        let mut surface = Surface::new(60, 30);
        for p in surface.data_mut() {
            *p = 0xffff_ffff;
        }
        draw_trajectory(
            &trajectory(&[(10., 15.), (30., 15.), (50., 15.)]),
            &[0., 0., 1.],
            &to_pixel,
            &mut surface,
        );
        // On the line, away from the markers: 0.7 blue over white.
        // src = (179, 0, 0, 179); dst = src + white * 76 / 255.
        assert_eq!(surface.pixel(30, 15), 0xff4c_4cff);
        assert_eq!(surface.pixel(30, 14), 0xff4c_4cff);
        // Outside of the 4 pixel wide line.
        assert_eq!(surface.pixel(30, 18), 0xffff_ffff);
        assert_eq!(surface.pixel(30, 11), 0xffff_ffff);
        // Start marker: green over the blue line.
        let start = surface.pixel(10, 15);
        assert!((start >> 8) & 0xff > 150, "{start:08x}");
        // End marker: red over the blue line.
        let end = surface.pixel(50, 15);
        assert!((end >> 16) & 0xff > 150, "{end:08x}");
        // The markers have radius 6.
        assert_eq!(surface.pixel(10, 7), 0xffff_ffff);
        assert_ne!(surface.pixel(10, 11), 0xffff_ffff);
    }
}
