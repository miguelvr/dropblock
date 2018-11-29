from setuptools import setup, find_packages


with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='dropblock',
    version='0.2.1',
    packages=find_packages(),
    install_requires=required,
    url='https://github.com/miguelvr/dropblock',
    license='MIT',
    author='Miguel Varela Ramos',
    author_email='miguelvramos92@gmail.com',
    description='Implementation of DropBlock: A regularization method for convolutional networks in PyTorch. '
)
