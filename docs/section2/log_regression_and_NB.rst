Logistic Regression and Naive Bayes
===================================

In this section we introduce the Logistic Regression and Naive Bayes classifiers. We discuss general
guidelines for when to use each. Finally, we show how to implement Logistic Regression and Naive
Bayes using SciKit-Learn. By the end of this section, you should be able to:

* Describe how the logistic regression and Naive Bayes classifier models work at a high level
* Describe when each model is most appropriate to use (and why)
* Implement each using the SciKit-Learn package


Logistic Regression
-------------------

In this section, we introduce the Logistic Regression model. As with the other methods in this 
unit, we will not cover all details but instead will give just a basic sense of the ideas 
involved. 

The basic idea with Logistic Regression is to build upon the Linear Regression model with the 
goal of learning a *probability distribution function* that can be used for classification 
problems. Despite "regression" appearing in the name, logistic regression models are used 
for **classification** problems.

In Logistic Regression, we build a linear regression model and then pass the result through a 
"logistic" function. The logistic function has the form:

.. math:: 

    p(x) = \frac{1}{1 + e^{-k(x-x_0)}}

where :math:`k, x_0` are constants/parameters with :math:`k>0`. 

Note the following attributes of this function: 

* As :math:`x\to -\infty`, :math:`e^{-k(x-x_0)} \to \infty` and thus :math:`p(x) \to 0`
* As :math:`x\to\infty`, :math:`e^{-k(x-x_0)} \to 0` and thus :math:`p(x) \to 1`
* For :math:`x:= x_0`, :math:`e^{-k(x-x_0)} = 1` and thus :math:`p(x_0) = 0.5`

As a result, the logistic function can be thought of as mapping an arbitrary real number 
to a probability, i.e., a value between 0 and 1. 


Example: Diabetes vs Glucose  
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For simplicity, let's assume we have a binary classification problem with just one independent 
variable. For example, suppose we are trying to predict whether a person has diabetes based 
only on their glucose level. We know the higher a person's glucose, the more likely they are 
to have diabetes. 

The idea is that, in this case, we can model the probability that the individual has 
diabetes as a logistic function of their glucose level. It might look similar to the following:

.. figure:: ./images/Log_Regression_diabetes_vs_glucose.png
    :width: 4000px
    :align: center
    :alt: Example logistic regression plot for diabetes vs glucose

    Example logistic regression plot for diabetes vs glucose

All logistic functions have an "S shaped curve", similar to the shape to the curve above. 
In logistic regression, the model learns a set of linear coefficients corresponding to each 
of the independent variables, just as in the case of linear regression.

As in the case of linear regression, we can define a loss function (or error function) 
and use it to define a cost function which we can then minimize using an algorithm such 
as gradient descent. 


Logistic Regression in SciKit-Learn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The SciKit-Learn package provides the ``LogisticRegression`` class from the 
``sklearn.linear_model`` module. 

Let's use this session to develop a logistic regression model for the cancer dataset we looked at in
the hands-on lab. 

We'll begin by importing the required libraries, as usual: 

.. code-block:: python

   >>> import numpy as np
   >>> import pandas as pd
   >>> from sklearn.model_selection import train_test_split 
   >>> from sklearn.datasets import load_breast_cancer

And then load the data and create our train/test split: 

.. code-block:: python

   >>> data = load_breast_cancer()
   >>> X = data.data
   >>> y = data.target

   >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)

We can now use the ``LogisticRegression`` model. We proceed in a similar way as with other models. 
We pass the following arguments to the ``LogisticRegression`` constructor:

* ``max_iter=1000`` -- This is the maximum number of iterations the solver will use for converging. 
  The default is 100, but here we pass 1000 to give it more time to converge. 
* ``random_state=1`` -- This is used to shuffle the data. (Same as with the SGD Linear Classifier) 

We also introduce the convenience function ``classification_report`` from the ``sklearn.metrics``
module. This function produces a nice report of several measures we have looked at, including
accuracy, recall, precision, and F1-score. 

Keep in mind when reading the output of ``classification_report`` that the values for precision, 
recall, F1-score, and support are provided for **all target class labels.** This could cause 
confusion. We have defined these metrics essentially for the target class equal to ``1``. 
For simplicity, you can just ignore the class 0 values. In this context, "support" refers to how 
many samples are in each class.

.. code-block:: python
   :emphasize-lines: 16, 18, 27, 29

   >>> from sklearn.linear_model import LogisticRegression
   >>> from sklearn.metrics import classification_report 

   >>> # fit the LG model -- random_state is used to shuffle the data; max_iter is max # of iterations for solver to converge (default is 100)
   >>> model = LogisticRegression(random_state=1, max_iter=1000).fit(X_train, y_train)

   >>> # print the report
   >>> print(f"Performance on TEST\n*******************\n{classification_report(y_test, model.predict(X_test))}")
   >>> print(f"Performance on TRAIN\n********************\n{classification_report(y_train, model.predict(X_train))}")

   Performance on TEST
   *******************
                 precision    recall  f1-score   support

             0       0.95      0.92      0.94        64
             1       0.95      0.97      0.96       107

       accuracy                           0.95       171
     macro avg       0.95      0.95      0.95       171
   weighted avg       0.95      0.95      0.95       171

   Performance on TRAIN
   ********************
                 precision    recall  f1-score   support

             0       0.96      0.94      0.95       148
             1       0.96      0.98      0.97       250

       accuracy                           0.96       398
     macro avg       0.96      0.96      0.96       398
   weighted avg       0.96      0.96      0.96       398

The performance we see on the cancer dataset is quite good, with:

* Precision: 95% on test; 96% on train.
* Recall: 97% on test; 98% on train.
* F1-score: 96% on test; 97% on train.
* Accuracy: 95% on test; 96% on train.


Additional Attributes of the ``LogisticRegression`` Model 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``LogisticRegression`` model has properties that correspond to those in the 
``LinearRegression`` model. For example: 

* ``coef_`` -- These are the coefficients of the linear model, one for each independent variable. 
* ``intercept_`` -- This is the y-intercept of the linear model. 
* ``decision_function()`` -- This function computes the linear combination of the coefficients and 
  intercept on the input value.

Examples: 

.. code-block:: python

   >>> model.coef_
   array([[1.799e+01, 1.038e+01, 1.228e+02, ..., 2.654e-01, 4.601e-01,
           1.189e-01],
          [2.057e+01, 1.777e+01, 1.329e+02, ..., 1.860e-01, 2.750e-01,
           8.902e-02],
          [1.969e+01, 2.125e+01, 1.300e+02, ..., 2.430e-01, 3.613e-01,
           8.758e-02],
          ...,
          [1.660e+01, 2.808e+01, 1.083e+02, ..., 1.418e-01, 2.218e-01,
           7.820e-02],
          [2.060e+01, 2.933e+01, 1.401e+02, ..., 2.650e-01, 4.087e-01,
           1.240e-01],
          [7.760e+00, 2.454e+01, 4.792e+01, ..., 0.000e+00, 2.871e-01,
           7.039e-02]], shape=(569, 30))

   >>> model.intercept_
   array([18.82533262])

   >>> model.decision_function(X[0:30])
   array([-37.50155217, -14.75203776, -17.05788567,   0.57214522,
          -11.22699773,  -1.45778081, -13.02507044,  -4.68089914,
           -2.62129355,  -6.41207649,  -6.23024417, -11.46582777,
          -12.27306369,   1.03957452,  -2.37747547,  -7.81454901,
           -7.76769613, -12.48215485, -34.0382094 ,   4.16659662,
            5.32451617,  11.01281779,  -4.67487841, -32.61813604,
          -34.72687955, -19.24160889,  -4.98111518, -11.77709786,
          -15.73240047,  -4.85708105])

   >>> # Compute the dot product and add the intercept "by hand"
   >>> # Note: output agrees with first output from decision function above. 
   >>> np.sum(model.coef_*X[0:1]) + model.intercept_
   array([-37.50155217])

   >>> # Predict the first 30 samples; note that the prediction agrees with the sign 
   >>> # of the decision function. 
   >>> model.predict(X[0:30])
   array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1,
          0, 0, 0, 0, 0, 0, 0, 0])


Logistic Regression: Strengths and Weaknesses 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we try to summarize the strengths and weaknesses of the Logistic Regression model. Keep in 
mind, these are general statement that *tend to apply* to most datasets. 


Logistic Regression Strengths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* *Easy to understand and interpret:* Logistic Regression models tend to be relatively easy to
  understand and interpret, as they produce probabilities that are foundational in statistics. 

* *Overfitting is usually avoidable:* A number of techniques, such as regularization, enable 
  logistic regression models to avoid overfitting. 


Logistic Regression Weaknesses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* *Cannot learn non-linear decision boundaries:* Like all linear models, the big disadvantage here 
  is that logistic regression models cannot learn non-linear decision boundaries. For many
  real-world datasets, this is a show stopper. 
* *Required data preparation:* Logistic regression requires significant data preparation to perform
  well, even in the best cases. Examples include normalization and scaling. These can be time
  consuming and computationally expensive on large datasets. We will look at some of these 
  techniques in the next module. 


Naive Bayes 
-----------

The next type of ML model we will discuss is the Naive Bayes model. 

This model is based on a simple (i.e., "naive") assumption that that feature variables 
in the dataset are *pair-wise conditionally independent*, meaning that, given two variables, 
knowing the value of one variable does not provide any information about the value of the other.
For example, the following pairs of variables could be considered independent:

 * student height and course grade
 * car color and car fuel efficiency
 * petal length and petal color

On the other hand, the following variables are unlikely to be independent:

 * petal length and stem length
 * student height and weight
 * car model and car fuel efficiency

.. note:: 

   The above notion of conditional independence can be made into a mathematically
   precise definition, but we will not go into those details here.


Note that Naive Bayes may still be of some use even in cases where the assumption of independence 
may not hold. 

The assumption of Naive Bayes allows us to write down a simple equation: 

.. math::

  P(y| x_1, ..., x_n) \sim P(y) \prod_{i=1}^n P(x_i | y)

where the notation :math:`P(y| x)` can be read as "the probability of *y* given *x*". For 
a supervised learning classification problem, the :math:`y` here represents some 
possible target class label. 

Note that the left hand side of the equation is the thing we are trying to model in 
any machine learning problem. We usually don't have an easy formula for it. 

But this equation says that the probability of the thing we care about --- i.e., the conditional 
probability of our dependent variable, :math:`y`, given the independent variables
:math:`x_1, ..., x_n` --- is proportional to the the product of the individual conditional
probabilities, :math:`P(x_i| y)`, and the probability of y itself. Those are much simpler objects to
work with. 

For example, thinking of y as some target class label, :math:`P(y)` is then just the frequency of 
occurrences of that label in the training set, which is trivial to compute (just count up the 
number of occurrences and divide by the total size of the dataset). 

Similarly, :math:`P(x_i|y)` is just the frequency of occurrences of :math:`x_i` when restricting 
to the subset of records with target label :math:`y`. When :math:`x_i` is a categorical 
feature, this is straight-forward: it is literally just the fraction of occurrences in the 
subset of the rows of the dataset that have target class :math:`y`. 

When :math:`x` is a continuous variable, something more is needed --- essentially we require a way
of computing likelihoods for a continuous feature. That in turn requires some additional
assumptions, for instance, that the continuous feature variables are sampled from a Gaussian (i.e.,
"normal") distribution. With an assumption like that in place (and a little bit of Calculus), we can
compute the probabilities. 

Deriving all the equations is actually fairly involved and would take much more time than we 
want to spend on it, but hopefully this gives you a general sense of the ideas involved. 


Types of Naive Bayes Models 
^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are several types of Naive Bayes Models. Here we mention just a few:

* Gaussian Naive Bayes: Can be used for classification problems involving datasets with 
  continuous variables. In addition to the "naive" assumption of conditional independence, 
  the model makes the assumption that the continuous features are sampled from a Gaussian 
  (i.e., normal) distribution. 

* Multinomial Naive Bayes: This model is good for discrete feature variables. It has found 
  good use in text classification problems, where the goal is to classify an article by type 
  (e.g., "Biology", "Computer Science", "Mathematics")
  or sentiment analysis (e.g.,
  classifying social media responses to advertisement campaigns as either "liking" or "not 
  liking" the ad). In this case, the independent variables consist of word count vectors, i.e., 
  the number of times a specific word occurs in the text. 

* Bernoulli Naive Bayes: This model assumes each feature is binary-valued (i.e., 0 or 1).
  Like Multinomial Naive Bayes, this model can be used on text classification problems. 
  Instead of using word count vectors, word occurrence vectors are used, 

All of these types and others are supported by the
`SciKit-Learn Naive Bayes classifier <https://scikit-learn.org/stable/modules/naive_bayes.html>`_.


Naive Bayes in SciKit-Learn
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's return to our Iris dataset and use Naive Bayes. As with the previous models, the 
pattern will be similar. For expediency, we do not discuss in detail the data analysis 
and pre-processing. For details, see our original discussion of the Iris dataset in the 
linear classification `section <linear_classification.html#linear-classification-with-scikit-learn>`_. 

To begin, we import libraries, load and split the dataset: 

.. code-block:: python

   >>> from sklearn.datasets import load_iris
   >>> from sklearn.model_selection import train_test_split

   >>> X, y = load_iris(return_X_y=True)
   >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)    

We'll use the ``GaussianNB`` class from the ``sklearn.naive_bayes`` module. This class 
implements a Gaussian Naive Bayes algorithm, as described above. We can instantiate the 
constructor without passing any arguments: 

.. code-block:: python

   >>> from sklearn.naive_bayes import GaussianNB
   >>> gnb = GaussianNB()
   >>> y_pred = gnb.fit(X_train, y_train).predict(X_test)

As before, we'll use ``classification_report`` to report the performance:

.. code-block:: python3 

   >>> from sklearn.metrics import classification_report
   >>> print(classification_report(y_test, y_pred))

               precision    recall  f1-score   support

           0       1.00      1.00      1.00        11
           1       0.93      1.00      0.96        13
           2       1.00      0.83      0.91         6

   accuracy                            0.97        30
   macro avg       0.98      0.94      0.96        30
   weighted avg    0.97      0.97      0.97        30


Naive Bayes: Strengths and Weaknesses 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we try to summarize the strengths and weaknesses of the Naive Bayes model. Keep in 
mind, these are general statement that *tend to apply* to most datasets. 


Naive Bayes Strengths
~~~~~~~~~~~~~~~~~~~~~

* *Conceptually easy:* Like, Logistic Regression, the Naive Bayes model is conceptually 
  relatively easy to understand and implement. 
* *Good scaling:* Naive Bayes tends to be faster and more efficient to implement than 
  Logistic Regression, and requires less storage.
* *Good in high dimensions:* Naive Bayes can work better with high dimensional data (e.g., 
  text classification) than other classifiers. 


Naive Bayes Weaknesses
~~~~~~~~~~~~~~~~~~~~~~

* *Poor accuracy when assumptions fail:* When the pair-wise conditional independence assumption
  fails, the performance of Naive Bayes classifiers can suffer. 
* *Zero frequency issue:* Given that the probabilities are multiplied together in the equation 
  above, Naive Bayes suffers from the "zero frequency issue" where, if some class value 
  does not appear in the training set, its probability formally is 0, which causes the entire 
  expression to be 0. In practice, there do exist techniques to handle this issue, but they
  add complexity. 


Additional Resources
--------------------

* Adapted from: 
  `COE 379L: Software Design For Responsible Intelligent Systems <https://coe-379l-sp24.readthedocs.io/en/latest/index.html>`_
* `SciKit-Learn: Logistic Regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_
* `SciKit-Learn: Naive Bayes <https://scikit-learn.org/stable/modules/naive_bayes.html>`_
* `SciKit-Learn: Classification Report <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html>`_
