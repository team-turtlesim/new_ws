from setuptools import find_packages, setup

package_name = 'bluesign'

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
    description='Blue-sign trigger node: opencv blue mask -> upper-ROI blue count + debounce '
                '-> /sign/near (Bool) to wake YOLO near the fork (cheap cascade gate).',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bluesign_node = bluesign.bluesign_node:main',
        ],
    },
)
