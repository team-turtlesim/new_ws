from setuptools import find_packages, setup

package_name = 'aruco'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mingu',
    maintainer_email='smsdvos@gmail.com',
    description='ArUco 동적 장애물 감지/정지 (시뮬·실차 공용). detector + override.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'aruco_detector_node = aruco.aruco_detector_node:main',
            'mission_override = aruco.mission_override:main',
        ],
    },
)
