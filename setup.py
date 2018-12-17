from setuptools import setup, find_packages


with open('requirements.txt', encoding='utf-8') as f:
    required = f.read().splitlines()

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='dropblock',
    version='0.3.0',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=required,
    url='https://github.com/miguelvr/dropblock',
    license='MIT',
    author='Miguel Varela Ramos',
    author_email='miguelvramos92@gmail.com',
    description='Implementation of DropBlock: A regularization method for convolutional networks in PyTorch. '
)
