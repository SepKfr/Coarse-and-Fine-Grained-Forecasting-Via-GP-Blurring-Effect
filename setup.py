from setuptools import setup, find_packages

setup(
    name='forecastblurdenoise',
    version='0.1.0',
    packages=['forecastblurdenoise'],
    install_requires=[
        # List your dependencies here
        'torch',
        'numpy',
        'pandas',
        'optuna',
        'gpytorch'
        # Add other dependencies as needed
    ],
    license='MIT'
)
