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
The SGD Classifier is a strictly binary classifier.

For each instance, it computes a score based on a decision function. If that score is greater than a threshold, it assigns the instance to the positive class; otherwise it assigns it to the negative class.
Increasing this threshold can result in false positives becoming true negatives (increasing the precision) and true positives becoming false negatives (decreasing the recall). Vice versa for decreasing the threshold.

Sklearn gives you the decision scores in the decision_function() method which can be used to make manual predictions to classify an instance.

The SGD classifier uses a threshold of 0. You can choose your own threshold, compare the output to the decision score to get a manual prediction.

### Deciding the threshold

To decide the threshold to use, obtain the scores of all instances in the training set. Graph a curve to compute the precision and recall for all possible thresholds (an infinite threshold gives a recall of 1 and precision of 0). You could also graph precision against recall.

**ROC curves**

The receiver operating characteristic (ROC) curve another tool used with binary classifiers.
The ROC curve plots true positive rate (TPR, same as recall) against the false positive rate (FPR, also called 'fall-out'). The FPR is the ratio of negative instances that are incorrectly classified as positive. It is 1 - TNR, the true negative rate (also called specificity).
Hence, the ROC curve plots sensitivity (recall, TPR) vs (1 - specificity).

On an ROC curve a straight x=y line represents the ROC curve of a purely random classifier. Good classifiers stay as far away from the line as possible, towards the top left corner.

**Comparing classifiers: Area under the curve (AUC)**
One way to compare classifiers is to measure the area under the curve (AUC). 
Perfect classifiers have an ROC AUC equal to 1, whereas a purely random classifier has an ROC AUC equal to 0.5.

**When to use the ROC curve vs precision/recall curve**
Prefer the PR curve when the positive class is rate or the false positives matter than the false negatives.
Otherwise, use the ROC curve.

## Multiclass Classification

Multiclass (multinomial) classifiers can distinguish between two or more classes.

### Using Binary Classifiers for Multiclass Classification

**One vs The Rest (OvR) / One vs All(OvA)**
You could create a binary classifier for each class and assign the instance to the class with the highest decision score/probability.

**One vs One (OvO)**
You could also train a binary classifier on each possible pair of classes. If there are N classes then you have to train N(N-1)/2 classifiers. To classify an instance you have to run the instance through each classifier and see which class wins the most duels.
The advantage of OvO is that each classifier only needs to be trained on the instances of the relevant classes it is assigned to distinguish between.

**OvO or OvR?**
Some algorithms scale poorly with the size of the training set, so for these algorithms OvO is preferred since it is faster to train many classifiers on small training sets compared to fewer classifiers on large training sets. For most binary algorithms, OvR is preferred instead.

### Error Analysis

How do we go about improving a promising model? Analyse the types of errors.

First look at confusion matrix. Since there will be more than two types it might be hard to read. Instead create a coloured version of the confusion matrix, e.g. using the ConfusionMatrixDisplay.from_predictions() function.

**Interpreting results**

If results are below what is wanted,

- Probably gather more training data that will allow the classifier to distinguish between true data and data it is commonly mistaken for.
- Or engineer new features that would help the classifier
- Preprocess the images to make some patterns stand out more

*Analyze individual errors*
You can plot instances on a confusion matrix to visualise what the classifier is doing and potentially see what it is doing wrong.

### How does SGD classify MNIST data?

SGDClassifier, which is just a linear model, assigns a weight per class to each pixel, and when it sees a new image it just sums up the weighted pixel intensities to get a score for each class. 

Since 3s and 5s differ by only a few pixels, the model easily confuses them.

*Data Augmentation*

One way to reduce the 3/5 confusion is to preprocess the images to ensure that they are well centered and not too rotated. However, this requires predicting the correct rotation of each image. 

A much simpler approach consists of augmenting the training set with slightly shifted and rotated variants of the training images. This will force the model to learn to be more tolerant to such variations. This is called data augmentation.

## Multilabel Classification

SGD classifiers can support multilabel classification.
Multilabel classifications systems output multiple binary tags (one for the presence of each class).
Each instance in the training input data should have multiple tags.

**Chain Classifiers**

If you are using a classifier that does not natively support multilabel classification, such as SVC, you can train one model per label. However, the strategy will have a hard time capturing the dependencies between labels. To deal with this, models can be organized in a chain - when a model makes a prediction, it uses the input features plus all the predictions of the models that come before it in the chain. Sklearn has a 'ChainClassifier' class that does this. It uses true labels for training, feeding each model the appropriate labels depending on their position in the chain. However, if the cv hyperparameter is set, it uses cross-validation to get out-of-sample predictions from each trained model for every instance in the training set, and these predictions are used to train all the models later in the chain.

### Evaluating a multilabel classifier

You could measure the F1 score for each individual label (or any other binary classifier metric), then compute the average score.

You might want to give more weight to some classes especially if they are more common than others. In this case you can give each label a weight equal to its support (the number of instances with that target label).

## Multioutput Classification
(also called multiclass classification)

Multioutput classification is a generalization of multilabel classification where each label can be multiclass (i.e. can have more than 2 possible values). 

The line between multioutput classification and regression is often blurry. Multioutput systems are not limited to classification tasks. You could even have a system that outputs multiple labels per instance, including both class and value labels.
