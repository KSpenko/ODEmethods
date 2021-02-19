from setuptools import setup, find_packages

setup(
    name='ODEmethods',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='Python package that includes numerical methods for solving ordinary differential equations (initial value problems).',
    long_description=open('README.txt').read(),
    install_requires=['numpy'],
    url='https://github.com/KSpenko/ODEmethods',
    author='Krištof Špenko',
    author_email='kristof.spenko.scrm@gmail.com'
)
