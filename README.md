# Decision trees with missing values
This project measures the accuracy of the decision tree learning algorithm with multiple missing values in the dataset.
![Screenshot](/sample_output.png?raw=true "Sample Output")

## Prerequisites
- [Python 3](https://www.python.org/downloads/) installed (tested on v3.6).
- [matplotlib](https://matplotlib.org) library installed for 2d plotting.

## How to use
Run `tests.py` to start testing accuracies for three datasets provided (`car.txt`, `phishing.txt`, `nursery.txt`). 

Each dataset is tested multiple times using `test()` function with different probabilities of a missing value (0%, 10%, 20% and 50%).

Accuracies are measured with the _10-fold cross-validation_ technique, averaged over 10 shuffles.

When all tests are done, a 2d graph will be plotted with `matplotlib` to compare the results, using `plot_accuracies()` function. The resulting plot is shown below.

<p align="center">
  <img src="/plot.jpg?raw=true">
</p>

## Code
Missing values are handled by assigning the most common value for that attribute among all training examples, as seen in section 3.7.4 of the book [_Machine Learning_](http://www.cs.cmu.edu/~tom/mlbook.html) by T. Mitchell (1997). The code to remove and handle missing values is located in `missing_values.py`.

`decision_fork.py`, `decision_leaf.py`, `dt_learner.py`, `utils.py`, `dataset.py` and `cross_validation.py` combine the data structures and functions required to run the project. Most of this code is adaptated from the [aima-python repository](https://github.com/aimacode/aima-python).

More information about the datasets used in this project can be found [here](/datasets/README.txt).

## Test your own dataset
To test your own dataset, copy the file to `/datasets` (both `txt` and `csv` file formats are supported). Then in `tests.py` you can run the code below:
```
file = parse_file("datasets/YOUR_DATASET_HERE.csv")
dataset = DataSet(name="YOUR_DATASET_NAME", examples=file, target=x)
test(dataset)
```
NOTE: `x` is the index of the attribute target used for classification.

If you also want to plot the accuracy (requires matplotlib):
```
accuracy = test(dataset)
plot_accuracies([accuracy], ["YOUR_DATASET_NAME"])
```

## References
- [UCI](https://archive.ics.uci.edu/ml/datasets/) - offers multiple free datasets. 
- [aima-python](https://github.com/aimacode/aima-python) - provides most of the code for decision trees and cross validation used in this project.
- [Machine Learning, Tom Mitchell, McGraw Hill, 1997](http://www.cs.cmu.edu/~tom/mlbook.html)
- [Artificial Intelligence - A Modern Approach](http://aima.cs.berkeley.edu)
- [Stack Overflow](https://stackoverflow.com/)
