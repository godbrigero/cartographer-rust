"""Regenerates py_imu_lz4.bag / py_imu_bz2.bag with the Python `rosbags` library.

Usage: python make_fixtures.py <output dir>   (needs `pip install rosbags numpy`)

40 sensor_msgs/Imu messages on /imu at 100 Hz starting at t = 1 s, small
chunks (several per bag) so cross-chunk iteration is exercised. These give
independent (non-Rust) coverage of lz4 frames and bz2 streams.
"""

import sys
from pathlib import Path

import numpy as np
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

ts = get_typestore(Stores.ROS1_NOETIC)
T = ts.types
out = Path(sys.argv[1])
for fmt in (Writer.CompressionFormat.LZ4, Writer.CompressionFormat.BZ2):
    p = out / f'py_imu_{fmt.name.lower()}.bag'
    if p.exists():
        p.unlink()
    w0 = Writer(p)
    w0.set_compression(fmt)
    w0.chunk_threshold = 2000
    with w0 as w:
        c = w.add_connection('/imu', 'sensor_msgs/msg/Imu', typestore=ts)
        for i in range(40):
            t = 1_000_000_000 + i * 10_000_000
            hdr = T['std_msgs/msg/Header'](
                seq=i, stamp=T['builtin_interfaces/msg/Time'](sec=t // 10**9, nanosec=t % 10**9),
                frame_id='imu')
            z = np.zeros(9)
            V = T['geometry_msgs/msg/Vector3']
            m = T['sensor_msgs/msg/Imu'](
                header=hdr, orientation=T['geometry_msgs/msg/Quaternion'](x=0, y=0, z=0, w=1),
                orientation_covariance=z, angular_velocity=V(x=0, y=0, z=i * 0.01),
                angular_velocity_covariance=z, linear_acceleration=V(x=0, y=0, z=9.81),
                linear_acceleration_covariance=z)
            w.write(c, t, ts.serialize_ros1(m, 'sensor_msgs/msg/Imu'))
    print(p)
