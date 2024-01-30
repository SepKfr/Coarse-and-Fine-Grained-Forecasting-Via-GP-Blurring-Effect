from setuptools import setup, find_packages
from forecastblurdenoise import __version__
setup(
    name='forecastblurdenoise',
    version=__version__,
    author='Sepideh Koohfar',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'gpytorch>=1.9.0',
        'torch>=2.0.1',
        'pandas>=1.5.2',
        'numpy>=1.23.5',
        'optuna>=3.3.0'
    ],
    license='MIT',
    author_email='sepideh.koohfar@unh.edu',
    url='https://github.com/SepKfr/Fine_grained_forecasting',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        "Topic :: Scientific/Engineering :: Artificial Intelligence"],
    entry_points={
        'console_scripts': [
             'example_usage=forecastblurdenoise.main:main'
         ],
    }
)
