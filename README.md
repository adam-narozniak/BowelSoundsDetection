# BowelSoundsDetection
Machine Learning LSTM based NN with MFCC features for Bowel Sounds Detection (Audio Classification) 
## Aims

## Setup
Clone this repository.

Conda Installation.

[ ] Download an installer
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

[ ] Verify the hash
```
sha256sum Miniconda3-py39_4.12.0-Linux-x86_64.sh | grep 78f39f9bae971ec1ae7969f0516017f2413f17796670f7040725dd83fcff5689
```
[ ] Install conda
```bash
bash ./Miniconda3-latest-Linux-x86_64.sh
```
[ ] Reload the bash
```bash
exec $SHELL
```

Now you can remove the installation script.
```bash
rm ./Miniconda3-latest-Linux-x86_64.sh
```


Move to the bowel repository
```bash
cd bowel
```
Create environment with required libraries using conda:

```
conda env create -f environment.yml
```
or for GPU
```
conda env create -f environment-gpu.yml
```
```
conda activate bowel
```

Finally, finish the gpu setup.
```bash
bash ./gpu-setup.sh
```

Data Collection
Some ubuntu systems might be missing the following library (and the system won't be able to load audio files so install this library)
error without this file: OSError: cannot load library 'libsndfile.so': libsndfile.so: cannot open shared object file: No such file or directory
```bash
sudo apt-get install libsndfile1
```
Make sure you have git-lfs for large files downloads.
Firstly check your architecture (to download the appropriate installation file)
```bash
dpkg --print-architecture
```
For amd:
```
wget https://github.com/git-lfs/git-lfs/releases/download/v3.3.0/git-lfs-linux-amd64-v3.3.0.tar.gz
```
Mistake below (arm vs amd)
```
sha256sum git-lfs-linux-amd64-v3.3.0.tar.gz | grep find-sha
```
unpack
```
tar -xf git-lfs-linux-amd64-v3.3.0.tar.gz
```
```
sudo bash ./git-lfs-3.3.0/install.sh
```
Remove the tar.gz file
```bash
rm git-lfs-linux-amd64-v3.3.0.tar.gz
```
```
sudo git lfs pull
```

## Verify GPU working (optional)
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
or in python terminal
```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```
If you see 0. Please reboot your machine first.
```bash
sudo reboot
```

## Data Preprocessing
```
make data
```
## Login to wandb
Go to wandb.ai/authorize and copy api code
```bash
wandb login
```
then paste the code when asked.

To a selected model train model follow the argparse arguments given in bowel.main.
To see them from command line type:
```
python3 -m bowel.main --help
```

Running a model looks like this:
```
python3 -m bowel.main --mode train_test --transform_config_name mfcc_transformation.yaml --train_config_name mfcc_train.yaml --log 
--wandb_log_name 
"train_test Bidirectional LSTM(256), LSTM(256), Convolution(64, kernel=15); #mfcc=40_2,5kHzHamming"
```
Running the experiments to determine the best parameters:

```bash
python3 -m bowel.experiments
```
