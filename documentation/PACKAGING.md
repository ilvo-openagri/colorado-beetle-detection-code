
# Create a python package
To use the AI model in other projects, you can create a python package from the source code.
The resulting package can be uploaded to the gitlab package manager, to be later used by pip (install).

## Dependencies
Ensure that you have the build dependency in python.
```shell
python3 -m pip install -q build
```

## Generate the python package
The description of the python package is done through two files:
- pyproject.toml: Package descritpion
- MANIFEST.in: Manifest file which includes/excludes specific files and directories


To generate the python package, execute following commands:
```shell
python3 -m build # Create the python package
```

The resulting files can be found under the dist/ directory.

Make sure to update the version number in pyproject.toml on every new build.

Note: Make sure the remove the old packages when generating a new one. Otherwise the next step (upload) will fail on the old ones with error code 400.

## Deploy the python package to the gitlab package registry
First create deploy tokens to read/write the package registry: https://gitlab.ilvo.be/jbauwens/openagri_ai/-/settings/repository
To deploy the package, the write_package_registry scope needs to be granted to the deploy key.
The deploy key should be stored in ~/.pypirc.

```shell
[distutils]
index-servers = 
    ilvo-gitlab-openagri-ai

[ilvo-gitlab-openagri-ai]
repository = https://gitlab.ilvo.be/api/v4/projects/719/packages/pypi
username = <deploy-key-name>
password = <deploy-key-password>
```

To upload the package to the package registry, execute following command/
```shell
python3 -m twine upload --repository ilvo-gitlab-openagri-ai dist/*

# Output:
Uploading distributions to https://gitlab.ilvo.be/api/v4/projects/719/packages/pypi
Uploading OpenAgriAiModule-1.0-py3-none-any.whl
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.4/16.4 kB • 00:00 • 591.0 kB/s
Uploading openagriaimodule-1.0.tar.gz
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 14.8/14.8 kB • 00:00 • ?
```

You can only store the package with the same version once.
To upload a new version, create a package with a different version.
