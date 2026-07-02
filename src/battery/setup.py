from setuptools import find_packages, setup

package_name = 'battery'

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
    maintainer='topst',
    maintainer_email='sooyong.park@telechips.com',
    description='Battery status publisher using INA219.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'battery_node = battery.battery_node:main',
        ],
    },
)
