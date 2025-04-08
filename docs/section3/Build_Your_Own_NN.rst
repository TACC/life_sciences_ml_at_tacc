Build Your Own Neural Network
============================

In this section we will build a simple neural network, train it and validate it on a sample test data.
For this exercise, we will use the `Mushroom dataset from the Audobon Society Field Guide <https://archive.ics.uci.edu/dataset/73/mushroom>`_.
This dataset includes 22 physical characteristics of ~8,000 mushrooms spanning 23 species of gilled mushrooms in the Agaricus and Lepiota Family.
Our task is to predict whether a mushroom is edible or poisonous based on its physical characteristics.

By the end of this excercise participants will be able to:

1. Import the Mushroom dataset from the UCI Machine Learning Repository.
2. Examine and preprocess the data to be fed to the neural network.
3. Build a sequential model neural network using TensorFlow Keras.
4. Evaluate the model's performance on test data.

============================
Step 1: Importing required libraries and data
============================

The Mushroom dataset is available in the University of California, Irvine Machine Learning Repository, which is a popular repository for machine learning datasets.
Conveniently, the ``ucimlrepo`` Python package provides a simple interface to download and load datasets directly from this repository.

First, we will install the ``ucimlrepo`` package if it is not already installed:

.. code-block:: python3

    pip install -U ucimlrepo

Next, we will import the Mushroom dataset using the ``ucimlrepo`` package:

.. code-block:: python3

    import pandas as pd
    from ucimlrepo import fetch_ucirepo 

    # fetch dataset 
    mushroom = fetch_ucirepo(id=73) 

Let's take a look at the metadata:

.. code-block:: python3

    print("Dataset Overview:", mushroom.metadata.abstract)
    print("Number of Instances:", mushroom.metadata.num_instances)
    print("Number of Features:", mushroom.metadata.num_features)
    print("Has Missing Values:", mushroom.metadata.has_missing_values)

    # Dataset Overview: From Audobon Society Field Guide; mushrooms described in terms of physical characteristics; classification: poisonous or edible
    # Number of Instances: 8124
    # Number of Features: 22
    # Has Missing Values: yes

We know that the Mushroom dataset has 8124 instances (samples) and 22 features (physical characteristics), and there are missing values in the dataset.
Before we go any further, let's split the dataset so that `X` contains the features and `y` contains the target variable and take a closer look at the data.

.. code-block:: python3

    >>> X = mushroom.data.features
    >>> y = mushroom.data.targets 

    >>> print(X.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8124 entries, 0 to 8123
    Data columns (total 22 columns):
     #   Column                    Non-Null Count  Dtype 
    ---  ------                    --------------  ----- 
     0   cap-shape                 8124 non-null   object
     1   cap-surface               8124 non-null   object
     2   cap-color                 8124 non-null   object
     3   bruises                   8124 non-null   object
     4   odor                      8124 non-null   object
     5   gill-attachment           8124 non-null   object
     6   gill-spacing              8124 non-null   object
     7   gill-size                 8124 non-null   object
     8   gill-color                8124 non-null   object
     9   stalk-shape               8124 non-null   object
     10  stalk-root                5644 non-null   object
     11  stalk-surface-above-ring  8124 non-null   object
     12  stalk-surface-below-ring  8124 non-null   object
     13  stalk-color-above-ring    8124 non-null   object
     14  stalk-color-below-ring    8124 non-null   object
     15  veil-type                 8124 non-null   object
     16  veil-color                8124 non-null   object
     17  ring-number               8124 non-null   object
     18  ring-type                 8124 non-null   object
     19  spore-print-color         8124 non-null   object
     20  population                8124 non-null   object
     21  habitat                   8124 non-null   object
    dtypes: object(22)
    memory usage: 1.4+ MB
    None
    >>> print(y.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8124 entries, 0 to 8123
    Data columns (total 1 columns):
     #   Column     Non-Null Count  Dtype 
    ---  ------     --------------  ----- 
     0   poisonous  8124 non-null   object
    dtypes: object(1)
    memory usage: 63.6+ KB
    None

In pandas, a Dtype of ``object`` typically means the column contains strings or mixed types of values. Let's examine further:

.. code-block:: python3

    >>> print(X.head(3))
      cap-shape cap-surface cap-color bruises odor gill-attachment gill-spacing  \
    0         x           s         n       t    p               f            c   
    1         x           s         y       t    a               f            c   
    2         b           s         w       t    l               f            c   

      gill-size gill-color stalk-shape  ... stalk-surface-below-ring  \
    0         n          k           e  ...                        s   
    1         b          k           e  ...                        s   
    2         b          n           e  ...                        s   

      stalk-color-above-ring stalk-color-below-ring veil-type veil-color  \
    0                      w                      w         p          w   
    1                      w                      w         p          w   
    2                      w                      w         p          w   

      ring-number ring-type spore-print-color population habitat  
    0           o         p                 k          s       u  
    1           o         p                 n          n       g  
    2           o         p                 n          n       m  

    [3 rows x 22 columns] 

In this dataset, the features are categorical variables stored as strings (which pandas represents as ``object`` dtype).
For example, we have values like:
 * cap-shape: 'x' (convex), 'b' (bell)
 * cap-color: 'n' (brown), 'y' (yellow), 'w' (white)

Next, let's take a look at the target variable:

.. code-block:: python3

    >>> print(y.head())
      poisonous
    0         p
    1         e
    2         e
    3         p
    4         e

We see that the target variable is a categorical variable that holds labels ``p`` (poisonous) and ``e`` (edible). 
Now that we have a better understanding of the data, let's preprocess the data to be fed to the neural network.

**Thought Challenge:** What are some things that you have noticed about the data that you think we will need to fix before feeding it to the neural network? Pause here and write down your thoughts before continuing.

============================
Step 2: Data Pre-processing
============================

As we have discovered, our Mushroom dataset is a pandas DataFrame that contains 22 features and 1 target variable for 8124 samples.
However, our data examination revealed that this dataset isn't quite ready for training a neural network yet. 

Specifically, we have noticed that:
 1. There are missing values in the dataset. 
 2. The features are categorical variables stored as strings (which pandas represents as ``object`` dtype).
 3. The target variable is a categorical variable (also stored as ``object`` dtype) that holds labels ``p`` (poisonous) and ``e`` (edible).

First, let's handle the missing values. Let's see how many missing values are in the dataset, and where they are located:

.. code-block:: python3

    >>> missing_values = X.isnull().sum()
    >>> print("Columns with missing values:")
    >>> print(missing_values[missing_values > 0])
    Columns with missing values:
    stalk-root    2480
    dtype: int64

We see that the ``stalk-root`` feature is the only one with missing values, and there are 2480 missing values.
Let's remove this column from the dataset:

.. code-block:: python3

    >>> X_clean = X.drop(columns='stalk-root')
    
Next, let's handle the categorical variables. We will use the ``pd.get_dummies`` method to convert the ``object`` dtype string categories into binary columns, which is what we need for our neural network.
Each unique value in a column will become its own binary column:

.. code-block:: python3

    >>> X_encoded = pd.get_dummies(X_clean)
    >>> print(X_encoded.head(2))
       cap-shape_b  cap-shape_c  cap-shape_f  cap-shape_k  cap-shape_s  \
    0        False        False        False        False        False   
    1        False        False        False        False        False   

       cap-shape_x  cap-surface_f  cap-surface_g  cap-surface_s  cap-surface_y  \
    0         True          False          False           True          False   
    1         True          False          False           True          False   

       ...  population_s  population_v  population_y  habitat_d  habitat_g  \
    0  ...          True         False         False      False      False   
    1  ...         False         False         False      False       True   

       habitat_l  habitat_m  habitat_p  habitat_u  habitat_w  
    0      False      False      False       True      False  
    1      False      False      False      False      False  

    [2 rows x 112 columns]

Now, instead of having 22 features, we have 112 features, each representing a binary True/False value for each categorical value in the original features.

Finally, let's encode the target variable. We will simply convert the string labels ``p`` and ``e`` into binary numeric values of 1 and 0, respectively.
In this case, 1 will represent a poisonous mushroom and 0 will represent an edible mushroom.

.. code-block:: python3

    >>> y_encoded = y['poisonous'].map({'p': 1, 'e': 0})

Now would be a good time to check the class distribution of our dataset:

.. code-block:: python3

    >>> print("\nClass Distribution:")
    >>> print(y_encoded.value_counts())
    >>> print("\nPercentage:")
    >>> print(y_encoded.value_counts(normalize=True) * 100)

We have a roughly balanced dataset with 51.8% of the samples being edible and 48.2% being poisonous.
We can now split the dataset into training and test sets:

.. code-block:: python3

    from sklearn.model_selection import train_test_split

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y_encoded,
        test_size=0.3,
        stratify=y_encoded,
        random_state=1
    )

    # Examine the shape of the training and testing sets
    print("Training set shape:", X_train.shape, y_train.shape)
    print("Testing set shape:", X_test.shape, y_test.shape)

In the code above, we are collecting four new objects: ``X_train``, ``X_test``, ``y_train``, and ``y_test`` representing a splitting of the preprocessed ``X`` and ``y`` data.
The ``test_size=0.3`` parameter specifies that 30% of the data will be used for testing, and 70% will be used for training. 
Feel free to experiment with different values for the ``test_size`` parameter to see how it affects the performance of the model.

Next, we specify ``stratify=y_encoded``. This is a very important parameter–it instructs sklearn to split the data in a way that ensures that the class distribution in the training and testing sets is the same as in the original dataset.
In our case, we have a roughly equal number of samples for each target class, so a random splitting is likely fine. 
In general, however, using a stratified split will ensure a proportional splitting even when the samples are not balanced.

Finally, we set ``random_state=1``. This ensures that the split is reproducible–the same random seed will always produce the same split.

============================
Step 3: Building a sequential model neural network 
============================

Let's now create a neural network!
In the example provided, we will create a neural network with one input layer, one hidden layer, and one output layer.
We will then check its prediction accuracy on the test data.
Feel free to experiment with different model architectures and compare your results to ours!

First, we import required libraries from Keras:

.. code-block:: python3

    # Importing libraries needed for creating neural network,
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Input, Dense

    # Create model
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(10, activation='relu'),
        Dense(1, activation='tanh')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Print model summary
    model.summary()

**Thought Challenge**: Here, we used ``Input(shape=(X_train.shape[1],))`` to specify the input layer. What does this do?

**Thought Challenge**: How many parameters does the model have? Can you calculate this manually and get the same result?

Finally, we fit the model to the training data:

.. code-block:: python3

    model.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=20, verbose=2)

Here, we set ``validation_split=0.2``, which means that 20% of the training data will be used for validation during training, and the remaining 80% will be used for actual training.
We also set ``epochs=5``, which specifies the number of epochs (iterations over the entire training dataset) to train the model.
In our case, the model will iterate over the entire training dataset 5 times.
We also set ``batch_size=20``, which specifies the number of samples that will be propagated through the network before the model weights are updated.

**Thought Challenge**: How does the choice of ``batch_size`` affect the training process?

**Thought Challenge**: How does the choice of ``epochs`` affect the training process?

.. code-block:: python3

    Epoch 1/5
    228/228 - 1s - 2ms/step - accuracy: 0.8511 - loss: 0.3924 - precision: 0.9355 - recall: 0.7419 - val_accuracy: 0.9649 - val_loss: 0.1651 - val_precision: 0.9621 - val_recall: 0.9656
    Epoch 2/5
    228/228 - 0s - 521us/step - accuracy: 0.9793 - loss: 0.0909 - precision: 0.9812 - recall: 0.9758 - val_accuracy: 0.9868 - val_loss: 0.0570 - val_precision: 0.9891 - val_recall: 0.9837
    Epoch 3/5
    228/228 - 0s - 534us/step - accuracy: 0.9914 - loss: 0.0375 - precision: 0.9949 - recall: 0.9872 - val_accuracy: 0.9938 - val_loss: 0.0297 - val_precision: 0.9946 - val_recall: 0.9928
    Epoch 4/5
    228/228 - 0s - 531us/step - accuracy: 0.9982 - loss: 0.0207 - precision: 0.9991 - recall: 0.9973 - val_accuracy: 0.9965 - val_loss: 0.0180 - val_precision: 0.9982 - val_recall: 0.9946
    Epoch 5/5
    228/228 - 0s - 507us/step - accuracy: 0.9989 - loss: 0.0129 - precision: 0.9995 - recall: 0.9982 - val_accuracy: 0.9982 - val_loss: 0.0122 - val_precision: 1.0000 - val_recall: 0.9964


1. **Progress metrics**:
  - ``228/228``: Shows progress through the training batches; 228 batches were completed out of 228, and each batch contains 20 samples (as specified by ``batch_size=20``)
  - ``1s``: Indicates the time taken for each epoch; here, the first epoch took 1 second to complete.
  - ``2ms/step``: This indicates the average time taken per training step (one forward and backward pass through a single batch) during training.

2. **Training metrics**:
  - ``accuracy: 0.8511``: Represents the accuracy of the model on the training dataset. The accuracy value of approximately 0.8511 indicates that the model correctly predicted 85.11% of the training samples.
  - ``loss: 0.3924``: Represents the training loss value (using binary cross-entropy loss function) on the training dataset. Higher loss values indicate that the model's predictions are further from the true labels.
  - ``precision: 0.9355``: Represents the precision of the model on the training dataset. Precision is the ratio of true positive predictions to all positive predictions.
  - ``recall: 0.7419``: Represents the recall of the model on the training dataset. Recall is the ratio of true positive predictions to all actual positive samples.

3. **Validation metrics**:
  - ``val_accuracy: 0.9649``: Represents the accuracy of the model on the validation dataset. The accuracy value of approximately 0.9649 indicates that the model correctly predicted 96.49% of the validation samples.
  - ``val_loss: 0.1651``: Represents the validation loss value (using binary cross-entropy loss function) on the validation dataset. Lower loss values indicate that the model's predictions are closer to the true labels.
  - ``val_precision: 0.9621``: Represents the precision of the model on the validation dataset.
  - ``val_recall: 0.9656``: Represents the recall of the model on the validation dataset.

**Thought Challenge**: What do you think the ``precision`` and ``recall`` metrics tell us about the model's performance? In what scenarios would you use precision and recall? (Hint: Think about the cost of false positives and false negatives.)

**Optional:**
In order to see the bias and weights at each epoch we can use the helper function below

.. code-block:: python3

    from tensorflow.keras.callbacks import LambdaCallback
     # Define a callback function to print weights and biases at the end of each epoch
    def print_weights_and_biases(epoch, logs):
        if epoch % 1 == 0:  
            print(f"\nWeights and Biases at the end of Epoch {epoch}:")
            for layer in model.layers:
                print(f"Layer: {layer.name}")
                weights, biases = layer.get_weights()
                print(f"Weights:\n{weights}")
                print(f"Biases:\n{biases}")

    # Create a LambdaCallback to call the print_weights_and_biases function
    print_weights_callback = LambdaCallback(on_epoch_end=print_weights_and_biases)

When we fit the model, we will specify the ``callback parameter``

.. code-block:: python3

    model.fit(X_train_normalized, y_train_cat, validation_split=0.2, epochs=5, batch_size=128, verbose=2,callbacks=[print_weights_callback])

This will print all the weights and biases in each epoch. 

Once we fit the model, next important step is predicting on the test data.

============================
Step 4: Evaluate model's performance on test data
============================

Now, we can use the trained model to make predictions on the test data.

.. code-block:: python3

    # Make predictions on the test data
    y_pred=model.predict(X_test_normalized)

For a binary classification problem like our (poisonous vs edible), the model outputs probabilities between 0 and 1 for each sample:

.. code-block:: python3

    # Show the first sample's prediction
    y_pred[0]

    #Output:
    #array([0.00026373], dtype=float32)

This shows the probability for the first mushroom sample in the test set.
The output is a single value between 0 and 1, where:
 - Values closer to 1 indicate the model is more confident that the sample is poisonous.
 - Values closer to 0 indicate the model is more confident that the sample is edible.

For example, our output value is 0.00026, which means that the model is 99.974% confident that the sample is edible.

To convert these probabilities into a binary prediction (0 for edible, 1 for poisonous), we can use a threshold of 0.5:

.. code-block:: python3

    import numpy as np
    y_pred_final = (y_pred > 0.5).astype(int)

Now, let's visualize the model's prediction accuracy with a **confusion matrix**. 
This will allow us to see how many correct vs incorrect predictions were made using the model above.

You may have to first install the ``seaborn`` library if you haven't already:

.. code-block:: bash

    pip install seaborn

Then, we can use the following code to create a confusion matrix:

.. code-block:: python3

    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot as plt
    import seaborn as sns

    # Create confusion matrix
    cm=confusion_matrix(y_test,y_pred_final)

    # Create visualization
    plt.figure(figsize=(10,7))          # Set figure size to 10x7 inches
    sns.heatmap(cm,annot=True,fmt='d')  # Create heatmap with annotations and display counts as integers
    plt.xlabel('Predicted')             # Label x-axis as 'Predicted'
    plt.ylabel('Truth')                 # Label y-axis as 'Truth'
    plt.show()                          # Display the plot

Output of the above confusion matrix is as follows:

.. figure:: ./images/nn-confusion-matrix.png
    :width: 600px
    :align: center
    :alt: 

The confusion matrix visualization shows how well our model classifies mushrooms as edible or poisonous. The matrix is a 2x2 grid where:

* The y-axis (Truth) shows the actual class of the mushrooms
* The x-axis (Predicted) shows what our model predicted
* Each cell contains the count of predictions falling into that category
* The heatmap coloring provides visual intensity, where lighter colors indicate higher counts

Reading the matrix:

* Top-left: True Negatives - Correctly identified edible mushrooms (0,0)
* Top-right: False Positives - Edible mushrooms incorrectly classified as poisonous (0,1)
* Bottom-left: False Negatives - Poisonous mushrooms incorrectly classified as edible (1,0)
* Bottom-right: True Positives - Correctly identified poisonous mushrooms (1,1)

**Thought Challenge**: Which prediction metric (e.g., accuracy, precision, recall) is most important for this model? Why? 

.. toggle:: Click to see the answer

    For mushroom classification, false negatives (bottom-left) are particularly concerning as they represent poisonous mushrooms that were incorrectly classified as edible.
    Recall measures a model's ability to correctly identify all true positives within a dataset, minimizing false negatives. 
    Therefore, **recall** is the most important metric for this model.

Let's also print the accuracy of this model using code below

.. code-block:: python3

    from sklearn.metrics import classification_report
    print(classification_report(y_test,y_pred_final, digits=4))

    # Output:
    #               precision    recall  f1-score   support
    #            0        0.998    0.999    0.998      1263
    #            1        0.999    0.998    0.998      1175
    # 
    #     accuracy                          0.998      2438
    #    macro avg        0.998    0.998    0.998      2438
    # weighted avg        0.998    0.998    0.998      2438

As you can see, the accuracy of the above model is 99.8%.
99.8% of the time, this model predicted the correct label on the test data.

**Thought Challenge**: Did we build a successful model? Why or why not? Is there anything we can do to improve the model?
