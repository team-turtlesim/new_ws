import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'sim_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mingu',
    maintainer_email='smsdvos@gmail.com',
    description='Gazebo <-> new_ws pipeline bridge nodes (image_bridge, cmd_bridge).',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_bridge = sim_bridge.image_bridge:main',
            'cmd_bridge = sim_bridge.cmd_bridge:main',
            'debug_viewer = sim_bridge.debug_viewer:main',
        ],
    },
)
