from setuptools import setup, find_packages

setup(
    name='fast_mc',
    description='A collection of Monte Carlo Samplers',
    author='Sreekumar Thaithara Balan',
    author_email='tbs1980@gmail.com',
    version='0.0.1',
    packages=find_packages(exclude=['doc', 'tests*']),
    install_requires=[
        'numpy'
    ],
    tests_require=[
        'pytest'
    ]
)
