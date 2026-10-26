from setuptools import find_packages, setup

package_name = 'interbotix_arms_controller'

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
    maintainer='Daniel Sosa',
    maintainer_email='dasc14@gmail.com',
    description='This package will help us in controlling the interbotix robot arms',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'basic_controller = interbotix_arms_controller.basic_controller:main',
        ],
    },
)
