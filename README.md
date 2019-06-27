# Autonomous Driving

![](videos/training_animation_smaller.gif)

Autonomous driving is one of the great technical challenges of our time. There are tons of interesting applications of computer vision to autononmous driving, we'll explore a few of them in this module. 

## Lectures
| Order |   Notebook/Slides  | Required Viewing/Reading |  Notes |
| ----- | ------------------ | ----------------------- | ------------------ |
| 1 | [How to Drive a Car with a Camera [Part 1]](https://github.com/unccv/autonomous_driving/blob/master/notebooks/How%20to%20Drive%20a%20Car%20with%20a%20Camera%20%5BPart%201%5D.ipynb) | [SDCs Part 1](https://www.youtube.com/watch?v=cExJbbwOfcw&t=26s), [VITS Paper](https://sites.cs.ucsb.edu/~mturk/Papers/ALV.pdf) |  |
| 2 | [How to Drive a Car with a Camera [Part 2]](https://github.com/unccv/autonomous_driving/blob/master/notebooks/How%20to%20Drive%20a%20Car%20with%20a%20Camera%20%5BPart%202%5D.ipynb) | [SDCs Part 2](https://www.youtube.com/watch?v=H0igiP6Hg1k), [ALVINN Paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.830.2188&rep=rep1&type=pdf) | The ALVINN implementation in this notebook should be helpful on your programming challenge!|

## Note About Tensorflow
We'll be using tensorflow to build a neural network in one lecture, but tensorflow (or other deep learning libraries) are not permitted in your programming challenge. 

## Setup 

The Python 3 [Anaconda Distribution](https://www.anaconda.com/download) is the easiest way to get going with the notebooks and code presented here. 

(Optional) You may want to create a virtual environment for this repository: 

~~~
conda create -n autonomous_driving python=3 
source activate autonomous_driving
~~~

You'll need to install the jupyter notebook to run the notebooks:

~~~
conda install jupyter

# You may also want to install nb_conda (Enables some nice things like change virtual environments within the notebook)
conda install nb_conda
~~~

This repository requires the installation of a few extra packages, you can install them all at once with:
~~~
pip install -r requirements.txt
~~~

(Optional) [jupyterthemes](https://github.com/dunovank/jupyter-themes) can be nice when presenting notebooks, as it offers some cleaner visual themes than the stock notebook, and makes it easy to adjust the default font size for code, markdown, etc. You can install with pip: 

~~~
pip install jupyterthemes
~~~

Recommend jupyter them for **presenting** these notebook (type into terminal before launching notebook):
~~~
jt -t grade3 -cellw=90% -fs=20 -tfs=20 -ofs=20 -dfs=20
~~~

Recommend jupyter them for **viewing** these notebook (type into terminal before launching notebook):
~~~
jt -t grade3 -cellw=90% -fs=14 -tfs=14 -ofs=14 -dfs=14
~~~


