# Social Recommendation with Graph Neural Networks
### Authors: Calvin Eng, Isha Mody, Emily Wang, and Jennifer Wang
Link to Paper-
https://drive.google.com/file/d/1CQrVsUxFxehLlPq2n4wvhvbVS_T1-gGY/view
## Project Details
The following project is a reimplementation in TensorFlow of the GraphRec model from the 2019 paper titled "Graph Neural Networks for Social Recommendation." Analyzing datasets sourced from product review websites that incorporate social elements enables the construction of a product recommendation network through the utilization of Graph Neural Networks (GNNs). The primary goal is to generate precise product recommendations tailored to individual users, drawing insights from both their personal rating histories and the rating patterns of individuals they trust. In this scenario, trust is established through user connections, manifested by the act of following other users. Through the implementation of GNNs with attention mechanisms, the aim is to achieve superior outcomes compared to alternative networks designed for social recommendations, such as PMF, Soec, and SoReg. The underlying challenge involves structured prediction, where the task leverages the interconnections among users within the platform to formulate predictions based on their past interactions and social relationships. In addition to the Epinions dataset from the original paper, the GNN is applied to a new dataset from the website Douban. 
## Data
The two datasets used are [Epinions](https://snap.stanford.edu/data/soc-Epinions1.html) and [Douban](https://www.kaggle.com/datasets/fengzhujoey/douban-datasetratingreviewside-information). Due to memory constraints, the datasets are downsized for training. 
## Getting Started
Necessary packages include: `pickle`, `numpy`, `tensorflow`, `sklearn`, `matplotlib`

To run the program for the Epinions dataset:
```
python3 run_val_epinions.py
```
To run the program for the Douban dataset:
```
python3 run_val_douban.py
```
## Training Process on Brown Department GPU Grid
To train on the Brown Department GPU Grid, first, make the necessary file path changes to run_GRID_GPU.sh, then run from the command line after SSH-ing into the department machine:
```
qsub -l day -l vf=128G -l gpus=1 -N final_douban_val run_GRID_GPU.sh run_val_douban.py
qsub -l day -l vf=128G -l gpus=1 -N final_epinions_val run_GRID_GPU.sh run_val_epinions.py
```
## References
Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin. 2019. Graph Neural Networks for Social Recommendation. In *Proceedings of the 2019 World Wide Web Conference (WWW ’19), May 13–17, 2019, San Francisco, CA, USA*. ACM, New York, NY, USA, 11 pages. https://doi.org/10.1145/3308558.3313488
