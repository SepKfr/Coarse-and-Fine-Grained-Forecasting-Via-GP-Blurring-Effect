from setuptools import setup

setup(
    name='forecastblurdenoise',
    version='0.1.1',
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
