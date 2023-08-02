from setuptools import setup, find_packages

from pathlib import Path
current_dir = Path(__file__).parent
desc = (current_dir / "README.md").read_text()

setup(
    name='losscape',
    version='1.5.10',
    url='https://github.com/Procuste34/losscape',
    author='Alexandre TL',
    author_email='alexandretl3434@gmail.com',
    description='Visualize easily the loss landscape of your neural networks',
    long_description=desc,
    long_description_content_type='text/markdown',
    packages=find_packages(),  
    install_requires=['torch>=1.8.1', 'numpy', 'scipy', 'matplotlib', 'h5py'],
)
