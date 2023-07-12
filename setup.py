from setuptools import find_packages, setup

HYPEN_E_DOT='-e .'
def get_requirements():
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open('requirements.txt') as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    author='ardong.s',
    author_email='ardong_suwan@hotmail.com',
    name='ml-template',
    packages=find_packages(),
    install_requires=get_requirements()
)