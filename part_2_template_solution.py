# Add your imports here.
# Note: only sklearn, numpy, utils and new_utils are allowed.

import numpy as np
from numpy.typing import NDArray
from typing import Any
import utils as u
import new_utils as nu


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    ShuffleSplit,
    cross_validate,
    KFold,
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# ======================================================================

# I could make Section 2 a subclass of Section 1, which would facilitate code reuse.
# However, both classes have the same function names. Better to pass Section 1 instance
# as an argument to Section 2 class constructor.


class Section2:
    def __init__(
        self,
        normalize: bool = True,
        seed: int | None = None,
        frac_train: float = 0.2,
    ):
        """
        Initializes an instance of MyClass.

        Args:
            normalize (bool, optional): Indicates whether to normalize the data. Defaults to True.
            seed (int, optional): The seed value for randomization. If None, each call will be randomized.
                If an integer is provided, calls will be repeatable.

        Returns:
            None
        """
        self.normalize = normalize
        self.seed = seed
        self.frac_train = frac_train

    # ---------------------------------------------------------

    """
    A. Repeat part 1.B but make sure that your data matrix (and labels) consists of
        all 10 classes by also printing out the number of elements in each class y and 
        print out the number of classes for both training and testing datasets. 
    """

    def partA(
        self,
    ) -> tuple[
        dict[str, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        X, y, Xtest, ytest = u.prepare_data()
        Xtrain, ytrain = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)
        Xtrain = nu.scale_data(Xtrain)
        Xtest = nu.scale_data(Xtest)
        # According to filter_out_7_9s, y is the labels
        # Also check that the labels are integers
        ytrain = nu.enforce_matrix_type(y, np.int32, int)
        ytest = nu.enforce_matrix_type(ytest, np.int32, int)

        unique_classes_y_train, counts_y_train = np.unique(ytrain, return_counts=True)
        unique_classes_y_test, counts_y_test = np.unique(ytest, return_counts=True)

        unique_classes_x_train, counts_x_train = np.unique(Xtrain, return_counts=True)
        unique_classes_x_test, counts_x_test = np.unique(Xtest, return_counts=True)

        # Enter your code and fill the `answer`` dictionary

        # `answer` is a dictionary with the following keys:
        # - nb_classes_train: number of classes in the training set
        # - nb_classes_test: number of classes in the testing set
        # - class_count_train: number of elements in each class in the training set
        # - class_count_test: number of elements in each class in the testing set
        # - length_Xtrain: number of elements in the training set
        # - length_Xtest: number of elements in the testing set
        # - length_ytrain: number of labels in the training set
        # - length_ytest: number of labels in the testing set
        # - max_Xtrain: maximum value in the training set
        # - max_Xtest: maximum value in the testing set

        # return values:
        # Xtrain, ytrain, Xtest, ytest: the data used to fill the `answer`` dictionary

        answer = {}

        # Enter your code and fill the `answer` dictionary

        answer["nb_classes_train"] = len(unique_classes_y_train) + len(unique_classes_x_train)
        answer["nb_classes_test"] = len(unique_classes_y_test) + len(unique_classes_x_test)
        answer["class_count_train"] = counts_y_train
        answer["class_count_test"] = counts_y_test
        answer["length_Xtrain"] = len(Xtrain)  # Number of samples
        answer["length_Xtest"] = len(Xtest)
        answer["length_ytrain"] = len(ytrain)
        answer["length_ytest"] = len(ytest)
        answer["max_Xtrain"] = np.max(Xtrain)
        answer["max_Xtest"] = np.max(Xtest)
        print("Part A Answer:", answer)
        Xtrain = Xtest = np.zeros([1, 1], dtype="float")
        ytrain = ytest = np.zeros([1], dtype="int")

        return answer, Xtrain, ytrain, Xtest, ytest

    """
    B.  Repeat part 1.C, 1.D, and 1.F, for the multiclass problem. 
        Use the Logistic Regression for part F with 300 iterations. 
        Explain how multi-class logistic regression works (inherent, 
        one-vs-one, one-vs-the-rest, etc.).
        Repeat the experiment for ntrain=1000, 5000, 10000, ntest = 200, 1000, 2000.
        Comment on the results. Is the accuracy higher for the training or testing set?
        What is the scores as a function of ntrain.

        Given X, y from mnist, use:
        Xtrain = X[0:ntrain, :]
        ytrain = y[0:ntrain]
        Xtest = X[ntrain:ntrain+test]
        ytest = y[ntrain:ntrain+test]
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
        ntrain_list: list[int] = [],
        ntest_list: list[int] = [],
    ) -> dict[int, dict[str, Any]]:
        """ """
        # Enter your code and fill the `answer`` dictionary
        answer = {}

        """
        `answer` is a dictionary with the following keys:
           - 1000, 5000, 10000: each key is the number of training samples

           answer[k] is itself a dictionary with the following keys
            - "partC": dictionary returned by partC section 1
            - "partD": dictionary returned by partD section 1
            - "partF": dictionary returned by partF section 1
            - "ntrain": number of training samples
            - "ntest": number of test samples
            - "class_count_train": number of elements in each class in
                               the training set (a list, not a numpy array)
            - "class_count_test": number of elements in each class in
                               the training set (a list, not a numpy array)
        """
        answer = {}
        answer["partC"] = {}
        answer["partD"] = {}
        answer["partF"] = {}
        ntest = { 1000: 200, 5000: 1000, 10000: 2000 }
        for k in [1000, 5000, 10000]:
            ntrain = k
            Xtrain = X[0:ntrain, :]
            ytrain = y[0:ntrain]
            Xtest = X[ntrain:ntrain+ntest[k]]
            ytest = y[ntrain:ntrain+ntest[k]]

            classifier = DecisionTreeClassifier(random_state=42)
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_validate(estimator=classifier, X=Xtrain, y=ytrain, cv=cv)
            
            answer["partC"]["clf"] = classifier 
            answer["partC"]["cv"] = cv
            answer["partC"]["scores"] = {}
            answer["partC"]["scores"]["mean_fit_time"] = np.mean(cv_scores['fit_time'])
            answer["partC"]["scores"]["std_fit_time"] =  np.std(cv_scores['fit_time'])
            answer["partC"]["scores"]["mean_accuracy"] = np.mean(cv_scores['test_score'])
            answer["partC"]["scores"]["std_accuracy"] = np.std(cv_scores['test_score'])


            cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
            cv_scores = cross_validate(estimator=classifier, X=Xtrain, y=ytrain, cv=cv)
            
            answer["partD"]["clf"] = classifier
            answer["partD"]["cv"] = cv 
            answer["partD"]["scores"] = {}
            answer["partD"]["scores"]["mean_fit_time"] = np.mean(cv_scores['fit_time'])
            answer["partD"]["scores"]["std_fit_time"] =  np.std(cv_scores['fit_time'])
            answer["partD"]["scores"]["mean_accuracy"] = np.mean(cv_scores['test_score'])
            answer["partD"]["scores"]["std_accuracy"] = np.std(cv_scores['test_score'])
            answer["partD"]["explain_kfold_vs_shuffle_split"] = "Shuffle-Split offers more flexibility and can be more efficient for large datasets, but it may introduce more variance in the performance estimates due to the random permutations. While k-fold cross-validation provides a more stable and thorough assessment at the cost of computational efficiency, it is particularly beneficial for smaller datasets or when every data point's utilization is crucial. Both seem to perform relatively similarly on the tested data set."

            classifier_RF = RandomForestClassifier(random_state=42)
            classifier_DT = DecisionTreeClassifier(random_state=42)

            scores_RF = cross_validate(estimator=classifier_RF, X=X, y=y, cv=cv)
            scores_DT = cross_validate(estimator=classifier_DT, X=X, y=y, cv=cv)

            answer["partF"]["clf_RF"] = classifier_RF
            answer["partF"]["clf_DT"] = classifier_DT
            answer["partF"]["cv"] = cv
            
            answer["partF"]["scores_RF"] = {}
            mean_fit_time_RF = np.mean(scores_RF['fit_time'])
            answer["partF"]["scores_RF"]["mean_fit_time"] = mean_fit_time_RF
            answer["partF"]["scores_RF"]["std_fit_time"] =  np.std(scores_RF['fit_time'])
            
            mean_accuracy_RF = np.mean(scores_RF['test_score'])
            answer["partF"]["scores_RF"]["mean_accuracy"] = mean_accuracy_RF
            
            std_accuracy_RF = np.std(scores_RF['test_score'])
            answer["partF"]["scores_RF"]["std_accuracy"] = std_accuracy_RF
            
            answer["partF"]["scores_DT"] = {}
            mean_fit_time_DT = np.mean(scores_DT['fit_time'])
            answer["partF"]["scores_DT"]["mean_fit_time"] = mean_fit_time_DT
            answer["partF"]["scores_DT"]["std_fit_time"] =  np.std(scores_DT['fit_time'])
            
            mean_accuracy_DT = np.mean(scores_DT['test_score'])
            answer["partF"]["scores_DT"]["mean_accuracy"] = mean_accuracy_DT
            
            std_accuracy_DT = np.std(scores_DT['test_score'])
            answer["partF"]["scores_DT"]["std_accuracy"] = std_accuracy_DT

            answer["partF"]["model_highest_accuracy"] = "random-forest" if mean_accuracy_RF > mean_accuracy_DT else "decision-tree" 
            answer["partF"]["model_lowest_variance"] = "random-forest" if std_accuracy_RF < std_accuracy_DT else "decision-tree" 
            answer["partF"]["model_fastest"] = "random-forest" if mean_fit_time_RF > mean_fit_time_DT else "decision-tree" 

            answer["ntrain"] = ntrain
            answer["ntest"] = ntest[k]
            answer["class_count_train"] = np.unique(ytrain)
            answer["class_count_test"] = np.unique(Xtrain)

        print("Part B answer: ", answer)
        return answer
