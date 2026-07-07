import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'yolo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 모델/라벨을 share 에도 설치(라벨은 항상, .onnx 는 있으면). 단, 기본 model_path
        # 는 src/yolo/models 를 먼저 본다 → --symlink-install 로 소스에 모델을 떨궈도 동작.
        (os.path.join('share', package_name, 'models'),
            glob('models/*.txt') + glob('models/*.onnx') + glob('models/*.md')),
    ],
    install_requires=['setuptools'],  # onnxruntime 는 pip 로 별도 설치(rosdep 키 없음)
    zip_safe=True,
    maintainer='topst',
    maintainer_email='sooyong.park@telechips.com',
    description='YOLO object detection node (onnxruntime CPU): camera image -> '
                'DetectionArray + debug overlay.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_node = yolo.yolo_node:main',
        ],
    },
)
