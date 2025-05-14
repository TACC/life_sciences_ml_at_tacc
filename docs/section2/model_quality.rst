Measuring and Validating Classification Model Quality 
=====================================================

So far, we have looked at model *accuracy* as the measure for assessing the extent to which a
classification model is valid, but is model accuracy on the test set enough? In this section, we
look at additional measures and techniques for ensuring the quality of a classification model. By
the end of this section, you should be able to:

* Explain the difference between accuracy, recall, precision and F1-score, and when to use each
* Compute these scores, both "by hand" and using sklearn for models that they develop

Note that in this section we focus on quality measures for classification, as regression requires 
different measures (for example, Mean Squared Error, r2-score; see the
`SciKit-Learn Documentation <https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics>`_
if you are interested in learning about metrics for regression models). 


Limitations of Accuracy
-----------------------

With accuracy, we are simply measuring the proportion of data points we classified correctly. 

.. math:: 

    Accuracy = \frac{Nbr. Correct}{Total}

This has the effect of treating all data points the same, which may not be true. For example, some
data points might be more important to get right than others. 

Furthermore, accuracy can be a poor measure of a model in cases where the dataset is imbalanced. 


Example: Breast Cancer
^^^^^^^^^^^^^^^^^^^^^^

For example, we looked at tumor malignancy detection in the last section. Our model tried to
determine if an tumor was benign or malignant. There are four cases, two cases where our model
predicts correctly and two cases where it does not: 

Correct: 

1. The biopsy was malignant, and our model predicted it was malignant. 
2. The biopsy was not malignant, and our model predicted it was not malignant. 

Incorrect:

3. The biopsy was not malignant, but our model predicted it was. 
4. The biopsy was malignant, but our model predicted it was not. 

*Discussion:* Of the two incorrect cases, would you say that the one type of error is "worse" than
the other or would you say they are both equally bad? If this were your biopsy, which would be more
troubling for you? 

In many cases, as with virtually all other features and behaviors, this issue comes down to
*requirements* for the software. It is possible that, given a choice of case 3 or 4, a majority of
people would prefer case 3 errors to case 4 errors. If an biopsy was actually benign but it was
flagged as malignant, the worst part would be the distress of dealing with that diagnosis. This is
the case 3 error. On the other hand, with a case 4 error, the biopsy was marked as benign but it was
actually malignant, that could delay potentially life saving treatment.


Example: Diabetes Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The medical field provides another set of examples. Usually (though not always), it is more
important to avoid missing an illness diagnosis than it is to diagnose an illness that is not there. 

For example, consider a preliminary screening based on an ML model that analyzes patient history to
determine if they might have diabetes and referes them for follow-up tests. Again, there are two
possible ways the model can fail:

1. The patient has diabetes but the model predicts that they do not. 
2. The patient does not have diabetes but the model predicts that they do. 

*Discussion:* Which of these failure modes is worse? 

Probably case 1 is worse, since the patient could be missing out on life-saving treatment. In case
2, the worst thing that might happen is the patient receives additional tests that come back
negative, which could be inconvenient and monetarily costly but does not risk human life.


Example: Predicting a Rare Event
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Suppose we wanted to build a ML model to predict whether a high school athlete will go on to win 
an olympic gold medal. Since winning a gold medal at the olympics is an extremely rare event, our 
model would probably score a pretty high accuracy by simply predicting every student to *not* win 
a medal. Even though this would result in a high accuracy, it would typically not achieve the 
desired results. In particular, our model would not be utilizing any learned pattern from prior 
data. 


Recall and Precision
--------------------

Recall, Precision and F1-score are three measures designed to help with the above issues. 
To introduce them, we first need to introduce some additional terminology:

* True Positives (TP) -- The number of times the model *correctly* predicted that a sample was 
  in a class. 
* True Negatives (TN) -- The number of times the model *correctly* predicted that a sample was not 
  in a class. 
* False Positives (FP) -- The number of times the model *incorrectly* predicted that a sample was 
  in a class. 
* False Negatives (FN) -- The number of times the model *incorrectly* predicted that a sample was 
  not in a class. 

For example, in the case of tumor detection: 

* True Positives: Total number of samples the model correctly predicted were malignant. 
* True Negatives: Total number of sample the model correctly predicted were not malignant. 
* False Positives: Total number of samples the model incorrectly predicted were malignant (i.e., 
  the model predicted they were malignant but in fact they were not).  
* False Negatives: Total number of samples the model incorrectly predicted were not malignant.
  (i.e., the model predicted they were benign but in fact they were malignant).

Conceptually, FPs and FNs represent the two kinds of ways our model can be wrong. For certain
problems, FPs might be worse than FNs and vice versa. For example, with tumor detection, we already
observed that FNs (falsely detecting an sample was benign) would likely be considered worse than
FPs. 

.. note:: 

   The set of values TPs, TNs, FPs, and FNs are what we displayed using a confusion matrix at the
   end of the previous section. 

With these terms defined, *recall* is defined as:

.. math:: 

    Recall = \frac{TP}{TP+FN}

We can see from the definition that recall gets worse as the number of false negatives increases, 
but it is not impacted by false positives.

Similarly, *precision* is defined as:

.. math:: 

    Precision = \frac{TP}{TP+FP}

We can see from the definition that precision gets worse as the number of false positives increases, 
but it is not impacted by false negatives.

.. note:: 

    Observe that :math:`0 \leq precision, recall \leq 1` and that both precision and recall 
    are optimal when they have a value of 1. 

*Discussion:* Given these definitions, which do you think is more important in the following cases:

* Cancer detection?
* Spam email detection? 

For tumor malignancy, it would likely be more important to improve recall (i.e., reduce FNs) because 
not diagnosing someone with a malignant tumor is likely more detrimental than diagnosing someone as
having malignancy when they do not. 

For spam email, it would likely be more important to improve precision (i.e. reduce FPs) because
falsely labeling an email as spam is worse than falsely labeling an email as not spam.

Note also that, without improving the overall accuracy of a model, if a model's recall improves then
its precision necessarily gets worse and vice versa. 


:math:`F1`-score
----------------

The :math:`F_1`-score (or just, *F*-score for short) is the *harmonic mean* of the precision and
recall, that is, a certain kind of average, and is thus given by the following formula: 

.. math:: 

    F_1 = \frac{2}{precision^{-1} + recall^{-1}}

Note that since precision and recall are both fractions less than 1, their inverses are bigger than
1. The worse the precision or recall (i.e., the smaller the value), the larger their inverses and
therefore the worse the :math:`F_1` score. 

When would it be appropriate to use *F*-score for a model? Since *F*-score averages precision and 
recall, it can be a good choice in cases where accuracy would be misleading -- e.g., with an
imbalanced data set -- but there is no preference for precision or recall. Predicting a "rare"
event, such as which student athlete will go onto win the olympic gold medal might be one such
example. 


Computing Recall, Precision, and :math:`F_1` with SciKit-Learn
--------------------------------------------------------------

The ``sklearn`` package has convenience functions for computing recall, precision and :math:`F_1`
score within the ``sklearn.metrics`` module. Each of these functions provides the same, simple API
taking two arguments: the actual values and the predicted values. 

Let's compute these for the breast cancer malignancy linear classifier we created last time. 

.. code-block:: python3 

    >>> from sklearn.metrics import recall_score, precision_score, f1_score

    >>> recall_test = recall_score(y_test, clf.predict(X_test))
    >>> recall_train = recall_score(y_train, clf.predict(X_train))

    >>> precision_test = precision_score(y_test, clf.predict(X_test))
    >>> precision_train = precision_score(y_train, clf.predict(X_train))

    >>> f1_test = f1_score(y_test, clf.predict(X_test))
    >>> f1_train = f1_score(y_train, clf.predict(X_train))

    >>> print(f"recall score on test: {recall_test}, recall score on train: {recall_train}")
    >>> print(f"precision score on test: {precision_test}, precision score on train: {precision_train}")
    >>> print(f"f1_score on test: {f1_test}, f1 score on train: {f1_train}")

    recall score on test: 0.9906542056074766, recall score on train: 0.988
    precision score on test: 0.8907563025210085, precision score on train: 0.8790035587188612
    f1_score on test: 0.9380530973451328, f1 score on train: 0.9303201506591338

We see that precision is worse than accuracy recall for our malignancy detector, at 89% on the 
test dataset as compared to 99%. This agrees with our confusion matrix where we saw that 
there were more false positives than false negatives.


Additional Resources
--------------------

* `SciKit-Learn metrics documentation <https://scikit-learn.org/stable/modules/model_evaluation.html>`_
