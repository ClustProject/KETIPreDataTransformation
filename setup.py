from setuptools import setup, find_packages

setup(
    name             = 'KETIPreDataTransformation',
    version          = '1.0',
    description      = 'Data Transformtion Packages',
    author           = 'Jaewon Moon',
    author_email     = 'jwmoon@keti.re.kr',
    #url              = '',
    #download_url     = '',
    install_requires = [ ],
    packages         = find_packages(exclude = ['docs', 'tests*']),
    keywords         = ['liquibase', 'db migration'],
    python_requires  = '>=3',
    package_data     =  {
        'pyquibase' : [
            'db-connectors/sqlite-jdbc-3.18.0.jar',
            'db-connectors/mysql-connector-java-5.1.42-bin.jar',
            'db-connectors/postgresql-42.1.3.jar',
            'liquibase/liquibase.jar',
            'liquibase/lib/snakeyaml-1.13.jar'

        ]},
    zip_safe=False,
    classifiers      = [
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ]
)