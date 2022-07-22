from pathlib import Path

import setuptools
from setuptools.command.install import install

install_requires = Path('requirements.txt').read_text().splitlines()

class PostInstall:
    def __init__(self, *args, **kwargs):
        super(PostInstall, self).__init__(*args, **kwargs)
        self.check_data_folder()
        self.check_and_create_database()
        if Path(__file__).parent.stem != 'meowtrix-purrdiction':
            warn(f"Tour repository is not named meowtrix-purrdiction. It is instead named as {Path(__file__).parent.stem}. Please make sure to modify `course` variable in setup.sh accordingly")
    def check_and_create_database(self):
        DB_PATH = Path(__file__).parent / 'cil.db'
        if not DB_PATH.exists():
            DB_PATH.touch()
            print(f"Created {DB_PATH}")

    def check_data_folder(self):
        DATA_PATH = Path(__file__).parent / 'data'
        assert DATA_PATH.exists(), 'Data directory does not exist'
        assert DATA_PATH.joinpath('data_train.csv').exists(), 'Data directory does not contain data_train.csv'
        assert DATA_PATH.joinpath('sampleSubmission.csv').exists(), 'Data directory does not contain sampleSubmission.csv'


setuptools.setup(
    name='Cil-Project-meowtrix-purrdiction',
    version='0.0.1',
    author='Meowtrix Purrdiction',
    url='',
    author_email='galtintas@student.ethz.ch',
    description='ETH Zurich Computational Intelligence Semester Project, Spring 2022',
    long_description=Path('README.md').read_text(),
    keywords=['recommender systems', 'matrix factorization', 'collaborative filtering'],
    packages=setuptools.find_packages(exclude=['tests', 'dist', 'build']),
    python_requires='>=3.8',
    install_requires=install_requires,
)
