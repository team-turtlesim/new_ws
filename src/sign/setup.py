import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'sign'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # YOLO onnx 모델을 share/sign/models 로 설치 (노드가 여기서 로드).
        (os.path.join('share', package_name, 'models'),
            glob('models/*.onnx')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mingu',
    maintainer_email='smsdvos@gmail.com',
    description='YOLO 표지판/신호등 인지 노드 (sign_detector).',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'sign_detector_node = sign.sign_detector_node:main',
        ],
    },
)
