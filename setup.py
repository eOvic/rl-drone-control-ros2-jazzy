import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'px4_rl_project'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(), # This automatically finds your 'environment' sub-package
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'gymnasium', 'stable_baselines3[extra]'],
    zip_safe=True,
    maintainer='chahine', # Change to your name
    maintainer_email='chahine@chahine-Pc', # Change to your email
    description='DRL for PX4 Drone Control',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # This creates an executable named 'train_agent' from your script
            'train_agent = px4_rl_project.train_agent:main',
        ],
    },
)
