# Installation

This guide covers installing the OpenAgri AI Python package and preparing the
model files used by the Colorado beetle detector. Hailo-specific system and
hardware setup is described in [HAILO.md](HAILO.md).

## Requirements

- Linux, Windows, or macOS with Python 3.10 or newer. Python 3.10 is the version
  identified in the project installation notes.
- Git, including support for initializing Git submodules.
- A virtual environment. Some optional accelerator dependencies have additional
  platform and Python-version requirements; see below.

## Get the code and models

From the repository root, initialize the trained-model and Hailo Model Zoo
submodules:

```shell
git submodule update --init --recursive
```

The Hailo Model Zoo is only needed for Hailo model workflows. The trained-model
submodule is separate from this source repository.

## Create an environment

Create and activate a virtual environment. For example, with Python 3.10:

```shell
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows, activate it with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Install dependencies

For a cloud or general-purpose environment, install the cloud dependency set:

```shell
python -m pip install -r requirements-cloud.txt
```

For the edge dependency set, use:

```shell
python -m pip install -r requirements-edge.txt
```

The edge requirements currently reference a HailoRT wheel for Linux ARM64 and
CPython 3.11. Use a compatible Python/platform combination for that wheel; for
other Hailo setups, follow the hardware-specific instructions in
[HAILO.md](HAILO.md) and use matching vendor packages. Hailo runtime
installation may also require system packages and permissions that cannot be
installed through pip alone.

Install this repository in editable mode after installing the selected
requirements:

```shell
python -m pip install --no-deps -e .
```

The dependency manifests install the runtime dependencies separately; `--no-deps`
avoids resolving a second, potentially different set from package metadata.

## Run the detector

Evaluate an image and optionally save an annotated copy:

```shell
python -m openagri_ai.main \
  --config default \
  --input ./test_images/sectie32.JPG \
  --output ./test_images/sectie32_annotated.JPG
```

The command accepts `--input`, optional `--output`, and optional `--config`.
`default` selects the standard YOLO configuration, `gpu` selects the GPU YOLO
configuration, and `raspi` selects the Hailo configuration. The latter requires
a working Hailo runtime and a compatible device.

For available training and conversion scripts, see
[SCRIPTS.md](SCRIPTS.md). For package building and distribution, see
[PACKAGING.md](PACKAGING.md).

## Licensing

See [the repository license](../LICENSE) for the license applicable to the
project's own software. Trained models, datasets, Hailo software, submodules,
and dependencies may carry additional terms; review [the notice](../NOTICE) and
the relevant upstream licenses before redistribution.
