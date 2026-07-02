from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def get_vehicle_config_path():
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'src' / 'config' / 'vehicle_config.yaml'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/D-Racer/src/config/vehicle_config.yaml'


def get_default_model_path():
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'src' / 'inference' / 'model' / 'test19.h5'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/D-Racer/src/inference/model/test19.h5'


def generate_launch_description():
    vehicle_config_path = get_vehicle_config_path()
    default_model_path = get_default_model_path()
    model_path = LaunchConfiguration('model_path')

    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path',
            default_value=default_model_path,
            description='Path to the H5 model file used by inference_node',
        ),
        Node(
            package='camera',
            executable='camera_node',
            name='camera_node',
            output='screen',
            parameters=[
                {
                    'vehicle_config_file': vehicle_config_path,
                },
            ],
        ),
        Node(
            package='control',
            executable='control_node',
            name='control_node',
            output='screen',
            parameters=[
                {
                    'vehicle_config_file': vehicle_config_path,
                },
            ],
        ),
        Node(
            package='battery',
            executable='battery_node',
            name='battery_node',
            output='screen',
        ),
        Node(
            package='inference',
            executable='inference_node',
            name='inference_node',
            output='screen',
            parameters=[
                {
                    'model_path': model_path,
                    'vehicle_config_file': vehicle_config_path,
                },
            ],
        ),
    ])
