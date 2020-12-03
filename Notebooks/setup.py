from setuptools import find_packages
from setuptools import setup

#with open('requirements.txt') as f:
#    content = f.readlines()
#requirements = [x.strip() for x in content if 'git+' not in x]

#setup(name='soccer_pred_487',
      #version="1.0",
      #description="Project Description",
      #packages=find_packages(),
      #test_suite = 'tests',
      # include_package_data: to install data from MANIFEST.in
      #include_package_data=True,
      #scripts=['scripts/soccer_pred_487-run'],
      #zip_safe=False)


REQUIRED_PACKAGES = [
    'Flask==1.1.1',
    'Flask-Cors==3.0.8',
    'gunicorn==20.0.4']

setup(
    name='soccer-pred',
    version='1.0',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='soccer-pred'
)
