# Foliage Generator

This project is a framework to emulate natural foliage from high resolution single leaf images.
The dataset obtained by this framework will be publicly available and can be used for Foliar Disease
classification.

## Running the code

To run the framework, you will have to run `main.py` file with necessary configuations.
For the current version, only soybean foliage is supported. 

### Requirements
1. Single leaf images with transparent background
2. Background images to place the single leaf images
3. Necessary configurations


#### Single Leaf images
Single leaf images are input to the foliage generator. The single leaf images should have a transparent background.
Here are some examples of single leaf images for soybean. 
<p align="center">
  <img src="src/sample_single_leaf/bacterial_blight_1.png" width="200"/>
  <img src="src/sample_single_leaf/potassium_deficiency_2.png" width="200"/>
  <img src="src/sample_single_leaf/downey_mildew_2.png" width="200"/>
  <img src="src/sample_single_leaf/healthy_2.png" width="200"/>
</p>

#### Background images 
Here the background images are to emulate the field that the plants grow on. Keeping only plain 
background will not create a more natural foliage image. The background image are obtained from [Freepic](https://www.freepik.com/)
![Alt text](src/sample_single_leaf/img.png)

#### Configuration
The configuration is plant based, which is a json file containing all the necessary information to create the foliar images from single 
leaf images. A sample of the configuration file is as shown below:
```
{
  "num_leaves": 35,
  "diseases": "frogeye, bacterial_blight, rust, cercospora_leaf_blight, downey_mildew, mosiac_virus, potassium_deficiency, sudden_death_syndrom, target_spot, healthy",
  "foliage_size": "(1024, 1500)",
  "single_plant_size": 512,
  "num_plants": 12,
  "disease_rate": 5,
  "background_image_path": "path_to_background_images",
  "input_path": "path_to_input_dir",
  "output_path": "path_to_output_dir",
  "type": "soybean"
}
```

### Running the foliage generator

Create python environment with `python>=3.9.6` with the necessary packages installed as showin in `requirement.txt` file.

Run the `main.py` with path of config file as argument 
`python main.py -c "path_to_config_file"`

### Experimentation with classifiers
The experiments conducted for this paper can be reproduced using classifiers defined in the paper. 
The directory `classifier` consists of all the code used to obtain results presented in the paper. The results
can be reproduced by running the models in the code. `main.py` is the entry point to the experiments and the models can be changes by commenting and 
uncommenting some parts of the file.