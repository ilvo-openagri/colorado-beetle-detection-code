# OpenAgri AI

OpenAgri AI detects Colorado beetles in high-resolution crop images. It includes
YOLO-based detection, a Hailo accelerator backend for supported edge devices,
and image-processing utilities for working with the project's models and data.

## Install

See the [installation guide](documentation/INSTALL.md) for Python setup,
dependencies, model submodules, and Hailo-specific notes.

## Run detection

The command-line entry point accepts an image and an optional path for an
annotated output image:

```shell
python -m openagri_ai.main \
  --config default \
  --input ./test_images/sectie32.JPG \
  --output ./test_images/sectie32_annotated.JPG
```

The `default` and `gpu` configurations use the YOLO backend. Use `--config raspi`
for the Hailo backend on a suitably configured device. See
[`src/openagri_ai/config/`](src/openagri_ai/config/) for configuration values
and [documentation/HAILO.md](documentation/HAILO.md) for Hailo setup.

## Training and utilities

See the documentation:

- [Installation](documentation/INSTALL.md) — environment setup, dependencies,
  and model submodules.
- [Hailo setup](documentation/HAILO.md) — Hailo runtime installation and model
  compilation.
- [Scripts](documentation/SCRIPTS.md) — training-data preparation, model
  training, and TensorFlow Lite conversion.
- [Package distribution](documentation/PACKAGING.md) — building and publishing
  the Python package.

## Models and third-party components

The trained models and Hailo Model Zoo are maintained as Git submodules. See
[the installation guide](documentation/INSTALL.md) to initialize them. Model
weights, datasets, bundled vendor files, and third-party dependencies may have
separate licenses or terms; see [NOTICE](NOTICE) and their upstream
documentation.
