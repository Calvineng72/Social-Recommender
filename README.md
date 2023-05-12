# DLFinal

We examine datasets from product review websites  with social components to create a product recommendation network using Graph Neural Networks (GNNs). The objective is to generate accurate product recommendations for different users based on their own rating histories and the rating histories of people they trust, where trust is conveyed through following other users. Using GNNs, we aim to see better results than other networks designed for social recommendations like PMF, Soec, SoReg etc. The problem is one of structured prediction: it uses the connections between users on a platform to create predictions based on their previous interactions and social network.

The two datasets used are [Epinions](https://snap.stanford.edu/data/soc-Epinions1.html) and [Douban](https://www.kaggle.com/datasets/fengzhujoey/douban-datasetratingreviewside-information)

Necessary packages include:
  - pickle
  - numpy
  - tensorflow
  - time
  - sklearn
  - matplotlib
  - argparse

To run the program for Epinions dataset:

  python3 run_val_epinions.py

To run the program for Douban dataset:

  python3 run_val_douban.py


To train on the Brown Department GPU Grid:

  qsub -l day -l vf=128G -l gpus=1 -N final_douban_val run_GRID_GPU.sh run_val_douban.py
  qsub -l day -l vf=128G -l gpus=1 -N final_epinions_val run_GRID_GPU.sh run_val_epinions1.py
