from setuptools import setup, find_packages

setup(
    name='my_robot_utils',  # The name your package will be known by
    version='0.1.0',
    packages=find_packages(),   # Automatically finds all packages
    install_requires=[
        # List any dependencies your package needs, e.g., 'requests>=2.20.0'
    ],
    author='Daniel Sosa',
    author_email='dasc14@gmail.com',
    description='A collection of my most used functions for robotics.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/my_awesome_package', # Optional
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)