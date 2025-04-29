TensorFlow
=============

=============
Introduction to TensorFlow
=============

**What is TensorFlow?**

.. image:: ./images/TensorFlow-Icon.png
    :width: 150px
    :align: right

`TensorFlow <https://www.tensorflow.org/>`_ is one of the most powerful open-source machine learning libraries available today. 
Developed by Google, TensorFlow offers a wide range of tools and resources to help you build, train, and deploy neural networks, making it accessible to both beginners and experts.

At its core, TensorFlow is a library for programming with linear algebra and statistics, using multi-dimensional arrays called *tensors* to represent data. 

**What is a Tensor?**

Everything in TensorFlow is build around tensors. 
To understand TensorFlow, we first need to understand what a tensor is.

A **tensor** is a multi-dimensional array, similar to NumPy arrays. Unlike traditional lists or NumPy arrays, however, *TensorFlow tensors are optimized for parallel computing* and can be processed across mutiple CPU/GPU/TPU cores simultaneously, significantly speeding up computations. 

.. list-table:: 

    * - **Tensor Type**
      - **Example**
      - **Shape**
    * - **Scalar (Rank-0)**
      - ``5``
      - ``()``
    * - **Vector (Rank-1)**
      - ``[1, 2, 3]``
      - ``(3,)``
    * - **Matrix (Rank-2)**
      - ``[[1, 2, 3], [4, 5, 6]]``
      - ``(2, 3)``

Neural networks use tensors to represent:

 * Input data (e.g., images, text, audio, etc.)
 * Weights (parameters the model learns)
 * Outputs (predictions from the model)

Every layer in a neural network takes tensors as input, applies mathematical operations, and produces tensors as output.

-----------
Getting Started with TensorFlow
-----------

First, install and import TensorFlow:

.. code-block:: python3

    pip install tensorflow
    import tensorflow as tf

Now, let's create some tensors!

**Creating Tensors**
-----------

**Scalar or Rank-0 Tensor:** 

Let's start by creating a simple scalar (Rank-0) tensor. Recall that a scalar is just a single number.

.. code-block:: python3
    
    # Create a scalar (Rank-0 Tensor)
    rank_0_tensor = tf.constant(4)
    print(rank_0_tensor)
    # Output: tf.Tensor(4, shape=(), dtype=int32)

What does this output tell us?

  1) The value of the tensor is ``4``,
  2) It has no shape (since it's just a single number), and 
  3) The data type is integer (int32). 

**Vector or Rank-1 Tensor:** 

A vector is just a list of numbers (a 1D array). In TensorFlow, we create it like this:
   
.. code-block:: python3

    # Create a vector (Rank-1 tensor) of floats.
    rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
    print(rank_1_tensor)
    # Output: tf.Tensor([2. 3. 4.], shape=(3,), dtype=float32)

What does this output tell us?

.. toggle:: Click to see the answer

    1) The tensor is a vector of numbers: [2.0, 3.0, 4.0]
    2) The shape of the tensor is (3,), meaning it has 3 elements in one row.
    3) The data type is float32 (since we used decimals). 

**Matrix or Rank-2 Tensor:** 

**Code Challenge**: Trying making a tensor with 2 rows and 3 columns (a 2x3  matrix) using ``tf.constant()``. 
 
Write down your answer first. Then click below to see our answer:

.. toggle:: Click to see the answer

    .. code-block:: python3

        rank_2_tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
        print(rank_2_tensor)

        # Output:
        tf.Tensor(
            [[1 2 3]
            [4 5 6]], shape=(2, 3), dtype=int32)

**Tensor Operations**
-----------

TensorFlow also provides built-in functions for mathematical operations, including common activation functions:

.. code-block:: python3

    z = tf.constant([-2.0, 0.0, 2.0])

    # Sigmoid function
    print(tf.math.sigmoid(z))
    # tf.Tensor([0.11920292 0.5        0.8807971 ], shape=(3,), dtype=float32)

    # Tanh function
    print(tf.math.tanh(z))
    # tf.Tensor([-0.9640276  0.         0.9640276], shape=(3,), dtype=float32)

    # ReLU function
    print(tf.nn.relu(z))
    # tf.Tensor([0. 0. 2.], shape=(3,), dtype=float32)

Perhaps you noticed that the last one is taken from the neural networks API (i.e., the ``nn`` module) of TensorFlow.
You can also get similar APIs from **TensorFlow Keras**, which we are also going to use for building neural networks. 

At this point, we are ready to build our first neural network using Keras!

=============
Building a First Neural Network with TensorFlow Keras
=============

----------------
What is Keras?
----------------

*Keras* is the high-level API of the TensorFlow platform. 
It provides a simple and intuitive way to define neural network architectures, and it's designed to be easy to use and understand.

Keras simplifies every step of the machine learning workflow, including data preprocessing, model building, training, and deployment.
Unless you're developing custom tools on top of TensorFlow, you should use Keras as your default API for deep learning tasks. 

**Core Concepts: Models and Layers**

Keras is built around two key concepts: ``Layers`` and ``Models``. 

**1. Layers**

The ``tf.keras.layers.Layer`` class is the fundamental abstraction in Keras.
A ``Layer`` is a fundamental building block of a neural network. It takes input tensors, applies some transformation, and produces output tensors.
Weights created by layers can be trainable or non-trainable. 
You can also use layers to handle data preprocessing tasks like normalization and text vectorization. 

**2. Models**

A ``Model`` is an object that groups layers together and that can be trained on data.
There are three types of models in Keras:
* **Sequential Model**: The simplest type of model, where layers are stacked linearly (one after another). 
* **Functional API**: Allows for more complex model architectures, including multi-input and multi-output models. 
* **Model Subclassing**: Provides full flexibility for custom model development by subclassing the ``tf.keras.Model`` class. 

In the example below, you will see how easy it is to build a simple neural network with Keras. 
We will build a **Sequential Model** to classify plants using the Iris dataset.

----------------
Step 1: Loading the Data
----------------

Before we get started building the model, let's import the dataset and look at its basic characteristics:

.. code-block:: python3

    >>> from sklearn import datasets
    >>> iris = datasets.load_iris()

    # The independent variables
    >>> iris.feature_names  # This tells us the column names
    >>> iris.data.shape     # This tells us the shape of the data
    >>> iris.data           # This shows us the data


    # The dependent variables
    >>> iris.target_names
    >>> iris.target.shape 
    >>> iris.target

**Thought Challenge**: Describe the data in the Iris dataset. Specifically:
 1. How many independent variables (features) are present in the dataset? What are they?
 2. How many samples are there in the dataset?
 3. How many dependent variables (classes) are in the dataset? What are they?
 4. What is the shape and format of the features? The classes?

Type your answer first. Then click below to see the answer:

.. toggle:: Click to show the answer

    1. There are 4 independent variables (features) in the dataset: sepal length (cm), sepal width (cm), petal length (cm), and petal width (cm).
    2. There are 150 samples in the dataset.
    3. There are 3 dependent variables (classes) in the dataset: setosa, versicolor, and virginica.
    4. The features are encoded as floats in a 2D array with shape (150, 4). The classes are encoded as integers in a 1D array with shape (150,).

Let's split the data into train and test sets and *one-hot encode* the target variable. 
One-hot encoding refers to converting categorical data (like the iris species: setosa, versicolor, and virginica) into a binary vector format:

.. code-block:: python3

    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical

    X = iris.data       # Contains all the flower measurements (features)
    y = iris.target     # Contains the flower species (classes)

    # Split the data into train and test sets using the train_test_split function from sklearn.model_selection
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

    # One-hot encode the target variable using the to_categorical function from Keras
    y_train_encoded = to_categorical(y_train)
    y_test_encoded = to_categorical(y_test)

    # Examine the one-hot encoded target variable:
    print(y_train_encoded)
    # Setosa is represented as [1. 0. 0.]
    # Versicolor is represented as [0. 1. 0.]
    # Virginica is represented as [0. 0. 1.]

Before we  continue, let's think about the architecture of the neural network we want to build. 

**Thought Challenge**: What is the fundamental architecture of the neural network we want to build? Take 5-10 minutes to examine the data and draw a rough sketch of the neural network. HINT: What does the input layer look like? What about the output layer? The hidden layers?

----------------
Step 2: Import Modules from Keras and Build the Model
----------------

We import Sequential from ``Keras.models``: Sequential is the main model class we'll use to build our neural network layer by layer in a linear stack. 
We also import Input from ``Keras.layers``: Input defines the shape of the input tensor and serves as an entry point into the model.
Finally, we import Dense from ``Keras.layers``: Dense represents a fully connected neural network layer where each neuron connects to all neurons in the previous layer.

.. code-block:: python3

    from keras.models import Sequential
    from keras.layers import Input, Dense

Now, let's build a sequential model with 3 layers: an input layer, a hidden layer, and an output layer:

* **Input Layer**:

   - Uses the Input function to explicitly define the shape of the input tensor
   - The input dimension *must* match the number of features in the input data (4 in this case)
   - Typically, there is no activation function for the input layer
  
* **Second Layer (Hidden Layer)**: 

   - We can use any number of perceptrons we want
   - We can use any activation function we want
   - We do not need to specify an input dimension because Keras can infer the input dimension from the output dimension of the previous layer.
   - **QUESTION**: What should the input dimension be?
  
* **Third Layer (Output Layer)**: 

  - The number of perceptrons *must* match the number of dependent variables in the dataset (3 in this case)
  - We can use any activation function we want (but think about why we might want to use a different activation function for the output layer than the hidden layer)
  - We do not need to specify an input dimension because Keras can infer the dimension from the output dimension of the previous layer.

.. code-block:: python3
    
    # Create a sequential model with 3 layers:
    model = Sequential([                # Create a sequential model
        Input(shape=(4,)),              # Input layer with 4 features
        Dense(128, activation='relu'),  # Hidden layer with 128 perceptrons and ReLU activation
        Dense(3, activation='softmax')  # Output layer with 3 perceptrons and softmax activation
    ])

----------------
Step 3: Compile the Model and Check Model Summary
----------------

Before the model is ready for training, it needs a few more settings. 
These are added during the model's ``compile`` step. Here are a few important parameters to consider:

* **Optimizer**: This parameter specifies the optimizer (algorithm used update the weights) to use during training. Options include: ``'rmsprop'``, ``'adam'``, ``'sgd'``, etc.
  
  - **Learning rate** is a crucial hyperparameter that determines how quickly the model learns. A higher learning rate can lead to faster convergence but may also lead to overshooting the optimal solution.
  - We set the learning rate within the optimizer parameter (e.g., model.compile(optimizer="adam", learning_rate=0.001)).
  
* **Loss**: This parameter specifies the loss function to use during training. The loss function measures how well the model performs on the training data and guides the optimizer in adjusting the model's parameters. Options include: 

  - Binary Classification: ``'binary_crossentropy'``
  - Multi-Class Classification: ``'categorical_crossentropy'``, ``'sparse_categorical_crossentropy'``
  - Regression: ``'mean_squared_error'``, ``'mean_absolute_error'``

* **Metrics**: This parameter defines the metrics used to monitor the training and testing steps. Options include: ``'accuracy'``, ``'precision'``, ``'recall'``, etc.

You need to provide appropriate values for these parameters based on your specific task and model architecture.

In the Iris example when we compile the model, we specify the optimizer (``'Adam'``), the loss function (``'categorical_crossentropy'``, suitable for multi-label classification problems), and metrics to evaluate during training (``'accuracy'``). 

Time permitting we will look at different types of optimizers.

.. code-block:: python3

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

Let's now print and explore the model summary:

.. code-block:: python3

    model.summary()

The output should look similar to the following:

.. code-block:: python3

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    dense (Dense)                (None, 128)                640
    dense_1 (Dense)              (None, 3)                  387
    =================================================================
    Total params: 1,047 (4.01 KB)
    Trainable params: 1,047 (4.01 KB)
    Non-trainable params: 0 (0.00 Byte)

Let's break down the summary:

**Model**: The type of model being used. In this case, it's a sequential model.

**Layer (type)**: Each layer in the model is listed along with its type.
For example, "dense" indicates a fully connected layer. Recall that we had 3 total layers: one input layer (not shown), one dense "hidden" layer with 128 perceptrons, and one dense output layer with 3 perceptrons.

**Output Shape**: The output shape of each layer. For example, ``(None, )`` indicates that the batch size is not specified (i.e., the model can handle any number of training samples), and ``( ,128)`` refers to the output dimension of the layer. Note that the output dimension is the same as the number of perceptrons in the layer, which is what we would expect for a fully connected network (i.e., dense layers).

**Param #**: The number of trainable parameters (weights and biases) in each layer. In the first dense layer there are 128 perceptrons, the input dimension was 4, and there is 1 bias term associated with each perceptron. 
Therefore, the first layer has a total of:

.. math::
    
    (4\ input\ features * 128\ perceptrons) + 128\ bias\ terms = 640\ parameters

**Thought Challenge**: Why are there 387 parameters in the output layer?

----------------
Step 4: Train the model
----------------

Once we have our model constructed we are ready for training!
We use the ``model.fit()`` method to train our model. This method takes several arguments:

* ``x`` and ``y``: The input and target data, respectively. A number of valid types can be passed here, including numpy arrays, TensorFlow tensors, Pandas DataFrames, and others.
* ``epochs``: The number of complete passes over the entire training dataset that will be performed during training.
* ``batch_size``: The number of samples per gradient update during training. Can be  an integer or ``None``. If ``None``, the batch size will be set to the size of the training dataset.

.. note::

    The choice of batch_size can affect the memory usage while fitting the model. 
    Bigger batch sizes can sometimes cause out of memory issues.

* ``validation_split``: The percentage, as a float, of the dataset to hold out for validation. Keras will compute the validation score at the end of each epoch. 
* ``verbose``: (0, 1, or 2). An integer controlling how much debug information is printed during training. A value of 0 suppresses all messages. 

.. code-block:: python3

    >>> model.fit(X_train, y_train_encoded, validation_split=0.1, epochs=20, verbose=2)

    Epoch 1/20
    4/4 - 0s - 14ms/step - accuracy: 0.8704 - loss: 0.4889 - val_accuracy: 0.9167 - val_loss: 0.4317
    Epoch 2/20
    4/4 - 0s - 10ms/step - accuracy: 0.8611 - loss: 0.4798 - val_accuracy: 0.9167 - val_loss: 0.4198
    Epoch 3/20
    4/4 - 0s - 10ms/step - accuracy: 0.8704 - loss: 0.4708 - val_accuracy: 0.9167 - val_loss: 0.4094
    Epoch 4/20
    4/4 - 0s - 10ms/step - accuracy: 0.8704 - loss: 0.4616 - val_accuracy: 0.9167 - val_loss: 0.4014
    Epoch 5/20
    4/4 - 0s - 10ms/step - accuracy: 0.9352 - loss: 0.4517 - val_accuracy: 1.0000 - val_loss: 0.3937
    Epoch 6/20
    4/4 - 0s - 10ms/step - accuracy: 0.9537 - loss: 0.4421 - val_accuracy: 0.9167 - val_loss: 0.3853
    Epoch 7/20
    4/4 - 0s - 10ms/step - accuracy: 0.9537 - loss: 0.4338 - val_accuracy: 0.9167 - val_loss: 0.3784
    Epoch 8/20
    4/4 - 0s - 9ms/step - accuracy: 0.9537 - loss: 0.4254 - val_accuracy: 1.0000 - val_loss: 0.3743
    Epoch 9/20
    4/4 - 0s - 10ms/step - accuracy: 0.9630 - loss: 0.4180 - val_accuracy: 1.0000 - val_loss: 0.3668
    Epoch 10/20
    4/4 - 0s - 9ms/step - accuracy: 0.9630 - loss: 0.4121 - val_accuracy: 1.0000 - val_loss: 0.3569
    Epoch 11/20
    4/4- 0s - 9ms/step - accuracy: 0.9722 - loss: 0.4057 - val_accuracy: 1.0000 - val_loss: 0.3521
    Epoch 12/20
    4/4 - 0s - 9ms/step - accuracy: 0.9630 - loss: 0.3980 - val_accuracy: 1.0000 - val_loss: 0.3420
    Epoch 13/20
    4/4 - 0s - 10ms/step - accuracy: 0.9537 - loss: 0.3891 - val_accuracy: 1.0000 - val_loss: 0.3348
    Epoch 14/20
    4/4 - 0s - 9ms/step - accuracy: 0.9537 - loss: 0.3831 - val_accuracy: 1.0000 - val_loss: 0.3295
    Epoch 15/20
    4/4 - 0s - 9ms/step - accuracy: 0.9630 - loss: 0.3771 - val_accuracy: 1.0000 - val_loss: 0.3273
    Epoch 16/20
    4/4 - 0s - 9ms/step - accuracy: 0.9630 - loss: 0.3757 - val_accuracy: 1.0000 - val_loss: 0.3221
    Epoch 17/20
    4/4 - 0s - 9ms/step - accuracy: 0.9815 - loss: 0.3626 - val_accuracy: 0.9167 - val_loss: 0.3109
    Epoch 18/20
    4/4 - 0s - 10ms/step - accuracy: 0.9259 - loss: 0.3639 - val_accuracy: 0.9167 - val_loss: 0.3096
    Epoch 19/20
    4/4 - 0s - 10ms/step - accuracy: 0.8981 - loss: 0.3630 - val_accuracy: 0.9167 - val_loss: 0.3004
    Epoch 20/20
    4/4 - 0s - 9ms/step - accuracy: 0.9537 - loss: 0.3492 - val_accuracy: 1.0000 - val_loss: 0.2959
    <keras.src.callbacks.history.History object at 0x1476b3350>     

You can read more about the parameters available to the ``fit()`` function in the documentation [1]_.

----------------
Step 5: Test the Model
----------------

We evaluate the model's performance on a test dataset using the ``model.evaluate()`` method. 

.. code-block:: python3

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded, verbose=0)
    print(f"Test Loss:", test_loss)
    print(f"Test Accuracy:", test_accuracy)

How well did your neural network perform?

With these steps we were able to set up a simple feedforward neural network using Keras with three layers (input, hidden, output) and specify the model's architecture, compilation parameters, and make preditions on some input data!

**Exercise**: Can you walk through this code and explain what's happening?
Come up with a hypothetical scenario where this model might be useful.

.. code-block:: python3

    from keras.models import Sequential
    from keras.layers import Input, Dense

    model = Sequential([                   
        Input(shape=(28,)),              
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(2, activation='sigmoid')     
    ])

    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy', 'precision'])

    model.summary()

Write down your answer first. Then click below to see our answer:

.. toggle:: Click to see the answer

    .. code-block:: python3

        from keras.models import Sequential
        from keras.layers import Input, Dense

        model = Sequential([                   
            Input(shape=(28,)),              # Input layer expecting 28 features
            Dense(64, activation='relu'),    # First hidden layer with 64 neurons using ReLU activation
            Dense(32, activation='relu'),    # Second hidden layer with 32 neurons using ReLU activation
            Dense(2, activation='sigmoid')   # Output layer with 2 neurons using sigmoid activation
        ])

        # Compile the model:
        # - optimizer: Stochastic Gradient Descent (SGD)
        # - loss: Binary Crossentropy (for binary classification)
        # - metrics: Accuracy and Precision
        model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy', 'precision'])

        # Print a summary of the model's architecture
        model.summary()

    This neural network has the following key characteristics:

    - The dataset has 28 input variables (features)
    - There are two hidden layers with 64 and 32 neurons, respectively
    - Both hidden layers use the ReLU activation function
    - The output layer has 2 neurons with the sigmoid activation function, indicative of a binary classification problem



**Reference List**
 * The material in this module is based on `COE 379L: Software Design for Responsible Intelligent Systems <https://coe-379l-sp24.readthedocs.io/en/latest/unit03/neural_networks.html>`_
.. [1] Keras Documentation: Model fit. https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit