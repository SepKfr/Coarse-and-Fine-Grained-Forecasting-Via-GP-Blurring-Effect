from setuptools import setup
from forecastblurdenoise import __version__
setup(
    name='forecastblurdenoise',
    version=__version__,
    packages=['forecastblurdenoise'],
    install_requires=[
        'python>=3.10',
        'gpytorch>=1.9.0',
        'torch>=2.0.1',
        'pandas>=1.5.2',
        'numpy>=1.23.5',
        'optuna>=3.3.0'
    ],
    license='MIT'
)
