import os
from glob import glob

from setuptools import find_packages, setup


package_name = 'monitor'


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_data={
        package_name: [
            'templates/*.html',
            'static/css/*.css',
            'static/js/*.js',
        ],
    },
    include_package_data=True,
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'resource'), glob('resource/*')),
    ],
    install_requires=['setuptools', 'flask'],
    zip_safe=False,
    maintainer='topst',
    maintainer_email='sooyong.park@telechips.com',
    description='Flask-based ROS dashboard for vehicle monitoring.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'monitor_node = monitor.monitor_node:main',
        ],
    },
)
