# deep-learning
Learn deep learning with d2l.ai

## Setup

### Install miniconda

https://conda.io/en/latest/miniconda.html

### You should be able to create a new environment as follows

    conda create --name d2l python=3.9 -y

### Now we can activate the d2l environment:

    conda activate d2l

## Installing the Deep Learning Framework and the d2l Package

    pip install torch==1.12.0 torchvision==0.13.0
    pip install d2l==1.0.0b0

## Downloading and Running the Code

    mkdir d2l-en && cd d2l-en
    curl https://d2l.ai/d2l-en.zip -o d2l-en.zip
    unzip d2l-en.zip && rm d2l-en.zip
    cd pytorch

### Now we can start the Jupyter Notebook server by running

    jupyter notebook

At this point, you can open http://localhost:8888 (it may have already opened automatically) in your Web browser. Then we can run the code for each section of the book. Whenever you open a new command line window, you will need to execute conda activate d2l to activate the runtime environment before running the D2L notebooks, or updating your packages (either the deep learning framework or the d2l package). To exit the environment, run conda deactivate.
