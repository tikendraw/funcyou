
from setuptools import find_packages, setup

setup(
    name='funcyou',
    version='1.0.7',
    author='Tikendra sahu',
    author_email='tikendraksahu1029@gmail.com',
    description='useful function to save time',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'scikit-learn',
        'pandas',
        'numpy',
        'kaggle'
    ],
)
