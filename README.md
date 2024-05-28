# Extending InGram to Hyper-relational Facts
This is the implementation for the our CS471 Graph Machine Learning project 'Extending InGram to Hyper-relational Facts'. We aim to extend InGram's capability on inductive inference 

### Training
To train from scratch, run `./train.sh [dataset_name]`, to further modify the model parameters, you can edit the shell file and provide argument defined in `--my_parser.py`

### Testing
Assuming the model checkpoints is saved in `./ckpt/exp/` and the checkpoint name is the same as defined in the training regiment, run `./test.sh [dataset_name]` to produce the test results for a given dataset. Our checkpoints can be downloaded from https://1drv.ms/f/s!Alqk-HgcoGHOnq5uYmZ-OtRbGKno8w?e=47T7hg

### Data
We use JF17K data based on the paper [HINGE](https://github.com/eXascaleInfolab/HINGE_code). We compile it using `./data/compile_data.py` to get the full graphs. We then divide it by running the script `./data/gen_data.sh`, by changing the variables for different percentages in the dataset

### Reproducibility and Requirements
We use a conda environenment as described in environment.yml. Furthermore we used a docker container build from the Dockerfile provided on `pytorch 2.3.0` with `cuda 12.1`. To build the docker image run `docker build -t pytorch-gpu . -f Dockerfile`, and to run the container run the bash script `./run_container.sh`. We ran training locally on a nVidia RTX 3060 Mobile GPU

### Acknowledgement
All of our codes and base methods are based on [InGram](https://github.com/bdi-lab/InGram)