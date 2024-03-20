##############################################################################
#
# https://github.com/zopefoundation/DateTime/blob/master/setup.py
#
##############################################################################



# import os
from setuptools import find_packages, setup


# abspath = os.path.abspath(os.path.dirname(__file__))
# with open(os.path.join(abspath, 'README.rst'), encoding='utf8') as f:
#     HEADER = f.read()


version = '0.0.1'

setup(
    name='damageability',
    version=version,
    description="""\
Calculates the damageability of an engineering structure \
from the action of variable force factors""",
    author='Vladislav Nagaev',
    author_email='vladislav.nagaew@gmail.com',
    # long_description=HEADER,
    # packages=find_packages('damageability'),
    packages=['damageability'],
    # package_dir={'': 'damageability'},
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.24.0',
        'matplotlib>=3.8.0',
        'pydantic>=2.5.0',
    ], 
    include_package_data=True,

    # setup_requires=['pytest-runner'],
    # tests_require=['pytest==7.4.3'], 
    # test_suite='tests', 
    
)




