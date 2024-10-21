# ASL3D

This is a thesis work that aims to develop a machine learning model that translates sign language words into 3D hand coordinate sequences, enabling the animation of realistic hand movements for sign language interpretation.

## About Dataset
We use the [WLASL (Word-Level American Sign Language)](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed) dataset, which includes 12K labeled videos for around 2,000 common ASL words. Using [MediaPipe](https://github.com/google-ai-edge/mediapipe), we extract 3D coordinates of hand movements. The resulting dataset contains 10-15 frames per word, with each frame capturing approximately 45 coordinate points for hands and pose (21 for each hand, 3 for pose).

### Generating Dataset

Download the WLASL dataset first and then extract it in a `Dataset` folder of the root directory.

```bash
unzip WLASL_v0.3.zip -d Dataset
```

Create two more empty directories inside `Dataset` directory named `frames` and `coordinates`, where the former stores selected frames from videos and the later stores the annotated images from the frames.

```bash
mkdir Dataset/coordinates Dataset/frames
```

Install depenencies for python script.
```bash
pip install -r requirments.txt
```

Generate dataset by running the python script `generate_dataset.py`. Adjust the `GENERATE_COUNT` variable to limit the number of words the dataset will contain. A `dataset.json` file will be generated at the root directory.
```bash
python generate_dataset.py
```

