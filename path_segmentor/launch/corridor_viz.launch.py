import os

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    # On the Jetson, force the CUDA-enabled OpenCV 4.10 core lib to load before
    # cv_bridge's transitive libopencv_core.so.4.5d (ELF symbol resolution is
    # first-wins). On a simulation host /opt/opencv4.10 does not exist, so we
    # skip LD_PRELOAD entirely and the node runs on the CPU backend.
    node_env = {}
    _ocv410 = '/opt/opencv4.10/lib/libopencv_core.so.4.10.0'
    if os.path.exists(_ocv410):
        node_env['LD_PRELOAD'] = _ocv410

    return LaunchDescription([

        # ── Tunable launch arguments ─────────────────────────────────────
        DeclareLaunchArgument('ground_angle_deg', default_value='30.0',
            description='Max angle (deg) from expected ground normal to still count as ground. '
                        '30° works for gravel + moderate slopes. Increase to 40° for very rough terrain.'),

        DeclareLaunchArgument('bilateral_sigma_d', default_value='0.10',
            description='Depth sigma for bilateral filter (metres). '
                        '0.10 smooths gravel (1-3cm noise) while preserving wall edges (>10cm jumps). '
                        'On very rough gravel, try 0.15. On smooth surfaces, 0.05 is enough.'),

        DeclareLaunchArgument('max_corridor_half_m', default_value='1.0',
            description='Maximum corridor half-width in metres (1.0 = ±1m from path centreline).'),

        DeclareLaunchArgument('robot_half_width_m', default_value='0.3',
            description='Minimum corridor half-width = robot body half-width. '
                        'Corridor never shrinks below this.'),

        DeclareLaunchArgument('max_depth_m', default_value='12.0',
            description='Ignore depth beyond this range. ZED X is accurate to ~15m stereo, '
                        'but for corridor viz 8-12m is usually enough.'),

        DeclareLaunchArgument('boundary_lookahead_m', default_value='15.0',
            description='Arc-length (m) of path to publish as corridor fence. '
                        'Limits boundary_cloud to nearby poses; the full global plan would '
                        'flood the costmap with thousands of useless far-away fence points.'),

        DeclareLaunchArgument('use_sim_time', default_value='false',
            description='Set true when running in Gazebo simulation. '
                        'Aligns the node clock with sim time so TF lookups in boundaryCb '
                        'use the same time base as TF publishers.'),

        # ── Node ─────────────────────────────────────────────────────────
        Node(
            package='path_segmentor',
            executable='path_corridor_segmenter',
            name='path_corridor_segmenter',
            output='screen',
            additional_env=node_env,
            parameters=[{
                'use_sim_time':           LaunchConfiguration('use_sim_time'),
                'max_corridor_half_m':    LaunchConfiguration('max_corridor_half_m'),
                'robot_half_width_m':     LaunchConfiguration('robot_half_width_m'),
                'ground_angle_thresh_deg': LaunchConfiguration('ground_angle_deg'),
                'bilateral_sigma_d':      LaunchConfiguration('bilateral_sigma_d'),
                'bilateral_sigma_s':      2.0,
                'max_depth_jump_m':       0.5,
                'min_depth_m':            0.3,
                'max_depth_m':            LaunchConfiguration('max_depth_m'),
                'overlay_alpha':          0.35,
                'shrink_safety_margin_m': 0.10,
                'shrink_samples':         6,
                'base_frame':             'base_link',
                'publish_terrain_debug':  True,
                'boundary_lookahead_m':   LaunchConfiguration('boundary_lookahead_m'),
            }],
            remappings=[
                ('/zed/zed_node/left/image_rect_color', '/zed/zed_node/rgb/color/rect/image'),
                ('/zed/zed_node/left/camera_info',      '/zed/zed_node/rgb/color/rect/camera_info'),
            ],
        ),
    ])
