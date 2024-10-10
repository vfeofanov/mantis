from setuptools import setup, find_packages

setup(
    name='mantis',
    version='0.1.0',  # Start with an initial version
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    packages=find_packages(),
    # Other metadata like author, description, etc.
)
