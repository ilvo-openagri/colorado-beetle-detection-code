# Coloradobeetles
This code detects Colorado Beetles in high resolution images.
There are three different models to execute:
<ol>
  <li>**Pipeline:** Runs the model as described in Thesis_Robin.pdf. </li>
  <li>**Default:** Runs a YoloV8n model on 640x640 image pieces. </li>
  <li>**Hailo:** Runs a YoloV8n model on 640x640 image pieces on the Hailo AI accelerator.</li>
</ol>

To filter out the potato plants in the images, following techiques are used:
<ol>
  <li>**Kmeans:** Kmeans clustering on the image colors. </li>
  <li>**Segmentation:** Runs a YoloV8n-seg model on the complete drone images to segment the potato plants. </li>
</ol>

## Install the dependencies
See the documentation [here](documentation/INSTALL.md)

## Evaluate an image:
To evaluate an image, use the main.py script.

```shell
usage: main.py [-h] --input path [--output path] [--config path]

Evaluate image(s) from Colorado Beetles

options:
  -h, --help     show this help message and exit
  --input path   The path to the image
  --output path  The path where an annotated output image can be stored.
  --config path  Configuration option for the evaluation script [default, raspi, pipeline]
```

The enabled options for each conf can be found in src/openagri_ai/config/.

Here is an example of the main script:

```shell
python3 -m openagri_ai.main --config pipeline --input ./test_images/sectie32.JPG --output ./test_images/sectie32_annotated.JPG 
```

Example output
```log
...
[INFO OpenAgriAIModule_openagri_ai.model.base @ 2024-10-28 10:20:37,264] Start evaluation
[INFO OpenAgriAIModule_openagri_ai.model.base @ 2024-10-28 10:20:37,264] Crop image to 5120x7936
[INFO OpenAgriAIModule_openagri_ai.model.base @ 2024-10-28 10:20:37,264] Start filtering to extract the leaves
[INFO OpenAgriAIModule_openagri_ai.model.base @ 2024-10-28 10:20:41,036] Matched 165 out of 620 image pieces for leaves
[INFO OpenAgriAIModule_openagri_ai.model.base @ 2024-10-28 10:20:41,086] Starting pipeline
[INFO OpenAgriAIModule_openagri_ai.model.base @ 2024-10-28 10:20:41,977] Pipeline matches 91 out of 165 image pieces (size 256)
[INFO OpenAgriAIModule_openagri_ai.model.base @ 2024-10-28 10:20:41,978] Starting pipeline
[INFO OpenAgriAIModule_openagri_ai.model.base @ 2024-10-28 10:20:43,399] Pipeline matches 45 out of 364 image pieces (size 128)
[INFO OpenAgriAIModule_openagri_ai.model.base @ 2024-10-28 10:20:43,400] Starting pipeline
[INFO OpenAgriAIModule_openagri_ai.model.base @ 2024-10-28 10:20:43,821] Pipeline matches 50 out of 180 image pieces (size 64)
[INFO OpenAgriAIModule_openagri_ai.model.base @ 2024-10-28 10:20:43,821] Completed
[INFO __main__ @ 2024-10-28 10:20:43,822] Evaluation took 6.56 seconds
[INFO __main__ @ 2024-10-28 10:20:51,955] Written to test_images/sectie32_annotated.JPG
```

## Scripts:
Some [scripts](documentation/SCRIPTS.md) are available to transform data and to train the required models.

## Create a python package
To use the AI model in other projects, you can create a python package from the source code.
See the documentation [here](documentation/PACKAGING.md)