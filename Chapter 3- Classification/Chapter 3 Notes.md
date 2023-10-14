# Classification

### sklearn.datasets
The sklearn.datasets package contains three types of functions:
- fetch_* functions which download real-life datasets
- load_* functions which load small toy datasets bundled with sklearn,
- make_* functions which generate fake datasets, useful for testing. These datasets are in the form of (X,y) where each is a NumPy array. Other datasets are sklearn Bunch objects, which are dictionaries whose objects can also be accessed as attributes. These are "DESCR" (a description of the dataset), "data" (the input data, 2D NumPy array), and "target" (labels, 1D NumPy array)

### The importance of shuffling
Some learning algorithms are sensitive to the order of data, and perform poorly if they get many similar instances in a row.

### The importance of model clones
It is often a good idea to use clones of models at each iteration in cross-validation because after each training iteration, internal model parameters are adjusted. Using clones at each iteration ensures that a fresh instance of the model is used. This stops there being unwanted dependencies between the folds (being trained on one set influences how it is trained on the next set) and stops results being biased.

## Binary Classifiers
Binary classifiers distinguish between two classes. For example, 5 and not 5.

**SGD**
The Stochastic Gradient Descent classifier is capable of handling large datasets efficiently. This is because SGD deals with training instances one at a time. This also makes it suited for online learning.

### Evaluating a Classifier

**Accuracy**
Accuracy tends to not be a great performance measure for classifiers, especially for skewed datasets (where some classes are more frequent than others). For example, assigning every element to the most popular group increases accuracy especially in an "is-x/is-not-x" problem 
Instead, the confusion matrix is preffered.

**Confusion Matrix**
The general idea of a confusion matrix is to count the number of times instances of class A are classified as class B, for all A/B pairs.

To compute the confusion matrix you need to have a set of predictions to compare to the targets.

cross_val_predict performs k-fold cross validation, but instead of calculating evaluation scores, it returns the predictions made on each test fold. Therefore there is an 'out-of-sample' prediction for each instance. Then you can use sklearn's confusion matrix function.

In a confusion matrix each row represents an actual class, and each column represents a predicted class. A perfect classifier only has values on its main diagonal.

**Precision and Recall**
The precision of the classifier is the accuracy of the positive predictions. 

*Precision* = (true positive) / (false positive + true positive).

Precision can be gamed - you could essentially create a classifier that always makes negative predictions except for one positive prediction it is confident about. Thus, precision is often considered in conjunction with the *recall*, also called *sensitivity* or the *true positive rate*. This is the ratio of positive instances correctly identified by the classifier.

*Recall* = (true positive) / (false negative + true positive).

*The F1 score*

The precision and recall can be combined into a single metric called the F1 score. It is the harmonic mean of precision and recall - more weight is given to low values, so a classifier only gets a high F1 score if both recall and precision is high.

*F1* = 2 / (1/precision + 1/recall) 

The F1 score favours classifiers with similar precision and recall. In some contexts you actually prefer one over the other, e.g. a low recall (when it is X, does not often predict it as X) and high precision (when it predicts X, it is often X) filter for when designing a system that marks videos as safe for kids - since you would rather reject loads of normal videos and keep loads of safe ones, compared to one where there is a higher recall and a few explicit videos bypass the filter.
There is a precision/recall trade-off.

**Precision/Recall Trade-off**

### The SGD Classifier
For each instance, it computes a score based on a decision function. If that score is greater than a threshold, it assigns the instance to the positive class; otherwise it assigns it to the negative class.
Increasing this threshold can result in false positives becoming true negatives (increasing the precision) and true positives becoming false negatives (decreasing the recall). Vice versa for decreasing the threshold.
Sklearn gives you the decision scores in the decision_function() method which can be used to make manual predictions to classify an instance.
