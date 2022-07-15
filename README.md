# Satellite Anomaly Detection Using MC dropout method

Source code author: ‪Mohammad Amin Maleki Sadr, Post-doctoral Fellow, University of Waterloo
Email: aminmalekisadr@gmail.com, mohammadamin.malekisadr@uwaterloo.ca

File list:
* main.py contains the code of our proposed Baysian LSTM method.
* the Jupyter notebook file contains the methods we compared with and also our method and results of each Figure separately.
* The Benchmarking for reproducing the results of competative papers by using Orion Package (Benchmarking needs having a patience)
* NASA Satellite Data files


## Papers:
The source code is for the paper titled "An Anomaly Detection Method for Satellites using Monte Carlo Dropout", submitted to IEEE Trans. on Aerospace and Electronic Systems, Jan 2022.

### Citation:

Please cite our article in the following if our work/code is used/referenced:

```
@Article{Malekisadr2022,
  author="Mohammadamin Malekisadr, Yeying Zhu, Peng Hu",
  title="{"An Anomaly Detection Method for Satellites using Monte Carlo Dropout}",
  journal="IEEE Transactions on Aerospace and Electronic Systems ",
  year="2022",
  month="Jan",
  day="13",
}
```
## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Before Installing
To ensure a smooth process, please ensure you have the following requirements.

**Hardware**
- Nvidia GPU with Compute Capability 3.5 or higher


**Software**

The following Softwares and Packages are recommended to be used before installation
```
Python: 3.6.1
Numpy: 1.12.1
Pandas: 0.20.1
Keras: 2.0.6
Scikit-Learn: 0.18.1
Theano: 0.9.0
Tensorflow: 1.2.1
Pydot: 1.0.29
GraphViz: 2.38.0
CUDA: 11.0
```
``
### Installation
Clone this repository, and then install it and its requirements. It should be something similar to this:

```
git clone https://github.com/DreamweaverU/SatAnomalyDetection.git
pip3 install -e SatAnomalyDetection/
pip3 install -r SatAnomalyDetectio/requirements.txt
```

### Dataset
We use the satellite telemetry data from NASA. The dataset comes from two spacecrafts: the Soil Moisture Active Passive satellite (SMAP) and the Curiosity Rover on Mars (MSL).

There are 82 signals available in the NASA dataset. We found that 54 of the 82 signals  to be continuous by inspection, and the remaining signals were discrete. We only consider the time-series sequences from the telemetry signals in our evaluation, where the telemetry values can be discrete or continuous in these signals.

The dataset is available [here](https://s3-us-west-2.amazonaws.com/telemanom/data.zip). If the link is broken or something is not working properly, please contact me through email (aminmalekisadr@gmail.com). By using the following command from root of repo, you can also curl and unzip data:
```
curl -O https://s3-us-west-2.amazonaws.com/telemanom/data.zip && unzip data.zip && rm data.zip
```
## Experiments
These configuration settings at the .py and .ipynb files can reproduce the results of this paper, and the available options are either self-explainatory, or have plenty of comments in the file.
### Configuration
All the results are set with custom random seed (for all the random variable like random numpy arrays or random python initialization or random tensors in torch) To reproduce the results of this paper. By running the codes with this random seed, you will be able to reproduce the same results. 

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
## Acknowledgments:
* **Yeying Zhu** and **Peng Hu**, my research supervisors;
* **University of Waterloo**, who hosted my research;
* **National Research Council Canada**, who funded my Research.
* ** The authors of this paper and repo also appreciate the anonymous reviewers of IEEE TAES and also the Associate Editor that handeling the process.   

