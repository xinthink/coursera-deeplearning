# Programming assignments of the [Deep Learning Specialization] on [Coursera].

## Directory Structure

- [01-nn](01-nn) Assignments for course 1/5: [Neural Networks and Deep Learning]
- [02-dnn](02-dnn) Assignments for course 2/5: [Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization]
- `*.ipynb` files are Jupyter Notebooks of the assignments
- `*.py` files are standalone code (without documentation) you can run & debug. Some of them are helper files.

## Getting Started

### Setup a virtualenv

- Install [Pipenv]
- Install dependencies with `pipenv install` or specify your python location `pipenv install --python $(which python3)`
- Now you can activate the virtualenv with `pipenv shell`, or run python script directly by `pipenv run python`

### Running Notebooks & python scripts

- `pipenv run jupyter notebook` then open a notebook
- or open a specified notebook: `pipenv run jupyter notebook 01-nn/week02/week02.ipynb`
- or run python scripts directly: `cd 01-nn/week02 && pipenv run python logistic_regression.py`


[Coursera]: https://www.coursera.org/
[Deep Learning Specialization]: https://www.coursera.org/specializations/deep-learning
[Neural Networks and Deep Learning]: https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning
[Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization]: https://www.coursera.org/learn/deep-neural-network/?specialization=deep-learning
[Pipenv]: https://pipenv.kennethreitz.org/
