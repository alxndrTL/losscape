from setuptools import setup, find_packages

setup(
    name='losscape',
    version='1.0',
    url='https://github.com/Procuste34/losscape',
    author='Author Name',
    author_email='author@gmail.com',
    description='Description of my package',
    packages=find_packages(),  
    install_requires=['torch>=1.8.1', 'numpy', 'scipy', 'matplotlib'],
)