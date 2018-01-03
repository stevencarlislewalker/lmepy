from setuptools import setup


with open('LICENSE') as f:
    package_license = f.read()

setup(
    name='lmepy',
    version='0.1.0',
    description='R-style mixed effects models in Python',
    long_description="Port of R's lme4 package for mixed effects modelling",
    author='Steve Walker',
    author_email='swalker@angoss.com',
    url='https://github.com/stevencarlislewalker/lmepy',
    license=package_license
)
