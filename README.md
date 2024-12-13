Pytorch implementation for Coling2025 paper: Modal Feature Optimization Network with Prompt for Multimodal Sentiment Analysis.

![Image text](https://raw.github.com/yourName/repositpry/master/yourprojectName/img-folder/test.jpg)



The training and testing of the MOSI, MOSEI, and SIMS datasets are implemented in three files with the same name.

### Setup the environment

We work with a conda environment.

```
conda env create -f environment.yml
conda activate pytorch
```

### Data Download

- Get datasets from public link:https://github.com/thuiar/Self-MM and  change the raw_data_path  to your local path(In config.py).

### Running the code

Take MOSI for example:
1. cd MOSI
2. python main.py 
