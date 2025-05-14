Python Refresher
================

In this workshop we will primarily be using Python to prepare our data, and build, train, and test
our models. After going through this section, you should be familiar with basic Python concepts
including:

* Data types and variables (ints, floats, bools, strings, type(), print())
* Arithmetic operations (+, -, \*, /, \*\*, %, //)
* Lists and dictionaries (creating, interpreting, appending)
* Conditionals and control loops (comparison operators, if/elif/else, while, for, break, continue,
  pass)
* Functions (defining, passing arguments, returning values)
* File handling (open, with, read(), readline(), strip(), write())
* Importing libraries (import, random, names, pip)


Start the Python Interpreter
----------------------------

The exercises below can be done in a Jupyter notebook, or on the command line in the Python
interactive interpreter. To start a notebook environment, follow the steps on the
`TACC Analysis Portal guide <./tap_and_jupyter.html>`_.

.. tip::

   Access a free, but resource-limited, Jupyter notebook environment in your browser at the 
   `Jupyter website <https://jupyter.org/try>`_.

To start an interactive Python interpreter on the command line, log in to Frontera (or any other
system with Python3 installed), and type the following:

.. code-block:: console

   [frontera]$ python3
   Python 3.7.0 (default, Jun  4 2019, 10:47:24)
   [GCC Intel(R) C++ gcc 8.3 mode] on linux
   Type "help", "copyright", "credits" or "license" for more information.
   >>>

.. tip::

   To exit the Python3 interactive interpreter, type ``quit()``.


Data Types and Variables
------------------------

The most common data types in Python are similar to other programming languages.
For this class, we’ll focus on:

- **int** – integers like `3`, `42`
- **float** – decimal numbers like `3.14`, `0.01`
- **bool** – `True` or `False`
- **str** – strings like `"hello"`

Assign values to variables:

.. code-block:: python3

   >>> x = 5
   >>> y = 3.14
   >>> name = "Alice"
   >>> flag = True

In Python, you don't have to declare type. Python figures out the type automatically.
You can check the type using the ``type()`` function:

.. code-block:: python3

   >>> print(type(x))     # int
   >>> print(type(y))     # float
   >>> print(type(name))  # str
   >>> print(type(flag))  # bool
   
The ``print()`` function is used to show output on the screen. You can print numbers, 
text, or the result of expressions.

Notice what happens when we print with and without single quotes?

.. code-block:: python3

   >>> print(x)        # prints the number 5
   >>> print("x")      # prints the letter x

**Converting between types**

You can convert between types using a few different functions. For example, when
you read in data from a file, numbers are often read as strings. Thus, you may
want to convert the string to integer or float as appropriate:

.. code-block:: python3

   >>> a = 10
   >>> b = str(a)         # convert int to string
   >>> c = "20"
   >>> d = int(c)         # convert string to int

   >>> print(b)           # '10'
   >>> print(d)           # 20
   >>> print(type(b))     # str
   >>> print(type(d))     # int

.. tip::

   Not all strings can be turned into numbers. Try ``int("hello")`` and see what happens!


Arithmetic Operations
---------------------

Next, we will look at some basic arithmetic. You are probably familiar with the
standard operations from other languages:

.. code-block:: text
   :emphasize-lines: 1

   Operator   Function          Example   Result
   +          Addition          1+1       2
   -          Subtraction       9-5       4
   *          Multiplication    2*2       4
   /          Division          8/4       2
   **         Exponentiation    3**2      9
   %          Modulus           5%2       1
   //         Floor division    5//2      2

Try a few things to see how they work:

.. code-block:: python3

   >>> print(2+2)
   >>> print(355/113)
   >>> print(10%9)
   >>> print(3+5*2)
   >>> print('hello' + 'world')
   >>> print('hello' + 5)
   >>> print('hello' * 5)

Also, carefully consider how arithmetic options may affect type:

.. code-block:: python3

   >>> number1 = 5.0/2
   >>> type(number1)
   <class 'float'>
   >>> print(number1)
   2.5
   >>> number2 = 5/2
   >>> type(number2)
   <class 'float'>
   >>> print(number2)
   2.5
   >>> print(int(number2))
   2


Lists and Dictionaries
----------------------

**Lists** are a data structure in Python that can contain multiple elements.
They are ordered, they can contain duplicate values, and they can be modified.
Declare a list with square brackets as follows:

.. code-block:: python3

   >>> my_list = ['thing1', 'thing2', 'thing3', 'thing4', 'thing5']
   >>> type(my_list)
   <class 'list'>
   >>> print(my_list)
   ['thing1', 'thing2', 'thing3', 'thing4', 'thing5']

Access individual list elements:

.. code-block:: python3

   >>> print(my_list[0])
   thing1
   >>> type(my_list[0])
   <class 'str'>
   >>> print(my_list[2])
   thing3

Create an empty list and add things to it:

.. code-block:: python3

   >>> my_number_list = []
   >>> my_number_list.append(5)     # 'append()' is a method of the list class
   >>> my_number_list.append(6)
   >>> my_number_list.append(2)
   >>> my_number_list.append(2**2)
   >>> print(my_number_list)
   [5, 6, 2, 4]
   >>> type(my_number_list)
   <class 'list'>
   >>> type(my_number_list[1])
   <class 'int'>

Lists are not restricted to containing one data type. Combine the lists together
to demonstrate:

.. code-block:: python3

   >>> my_big_list = my_list + my_number_list
   >>> print(my_big_list)
   ['thing1', 'thing2', 'thing3', 'thing4', 'thing5', 5, 6, 2, 4]

Another way to access the contents of lists is by slicing. Slicing supports a
start index, stop index, and step taking the form: ``my_list[start:stop:step]``.
Only the first colon is required. If you omit the start, stop, or :step, it is
assumed you mean the beginning, end, and a step of 1, respectively. Here are
some examples of slicing:

.. code-block:: python3

   >>> my_list = ['thing1', 'thing2', 'thing3', 'thing4', 'thing5']
   >>> print(my_list[0:2])     # returns the first two things
   ['thing1', 'thing2']
   >>> print(my_list[:2])      # if you omit the start index, it assumes the beginning
   ['thing1', 'thing2']
   >>> print(my_list[-2:])     # returns the last two things (omit the stop index and it assumes the end)
   ['thing4', 'thing5']
   >>> print(my_list[:])       # returns the entire list
   ['thing1', 'thing2', 'thing3', 'thing4', 'thing5']
   >>> print(my_list[::2])     # return every other thing (step = 2)
   ['thing1', 'thing3', 'thing5']

.. note::

   If you slice from a list, it returns an object of type list. If you access a
   list element by its index, it returns an object of whatever type that element
   is. The choice of whether to slice from a list, or iterate over a list by
   index, will depend on what you want to do with the data.


**Dictionaries** in Python store data as **key:value** pairs. They are:

- Unordered
- Mutable (you can change values)
- Keys must be unique

Let’s create a simple English-to-Spanish dictionary:

.. code-block:: python3

   >>> eng2spa = {
   ...   'one': 'uno',
   ...   'two': 'dos',
   ...   'three': 'tres'
   ... }
   >>> type(eng2spa)
   <class 'dict'>
   >>> print(eng2spa)
   {'one': 'uno', 'two': 'dos', 'three': 'tres'}

You can retrieve values using the **key**:

.. code-block:: python3

   >>> print(eng2spa['two'])
   dos

But you **cannot** use the value to look up the key directly:

.. code-block:: python3

   >>> eng2spa['dos']
   Traceback (most recent call last):
     ...
   KeyError: 'dos'

This shows that dictionaries are **one-directional**: they map from keys to values 
(not the other way around).

You can **change the value** associated with a key. For example, let’s say we decide 
to update the translation for `'one'`:

.. code-block:: python3

   >>> eng2spa['one'] = 'UNO'
   >>> print(eng2spa['one'])
   UNO

You can also **add a new key:value pair**:

.. code-block:: python3

   >>> eng2spa['four'] = 'cuatro'
   >>> print(eng2spa['four'])
   cuatro

Now the dictionary contains:

.. code-block:: python3

   >>> print(eng2spa)
   {'one': 'UNO', 'two': 'dos', 'three': 'tres', 'four': 'cuatro'}


Conditionals and Control Loops
------------------------------

Python **comparison operators** allow you to add conditions into your code in
the form of ``if`` / ``elif`` / ``else`` statements. Valid comparison operators
include:

.. code-block:: text
   :emphasize-lines: 1

   Operator   Comparison                 Example   Result
   ==         Equal                      1==2       False
   !=         Not equal                  1!=2       True
   >          Greater than               1>2        False
   <          Less than                  1<2        True
   >=         Greater than or equal to   1>=2       False
   <=         Less Than or equal to      1<=2       True

A valid conditional statement might look like:

.. code-block:: python3

   >>> num1 = 10
   >>> num2 = 20
   >>>
   >>> if (num1 > num2):                  # notice the colon
   ...     print('num1 is larger')        # notice the indent
   ... elif (num2 > num1):
   ...     print('num2 is larger')
   ... else:
   ...     print('num1 and num2 are equal')

In addition, conditional statements can be combined with **logical operators**.
Valid logical operators include:

.. code-block:: text
   :emphasize-lines: 1

   Operator   Description                           Example
   and        Returns True if both are True         a < b and c < d
   or         Returns True if at least one is True  a < b or c < d
   not        Negate the result                     not( a < b )

For example, consider the following code:

.. code-block:: python3

   >>> num1 = 10
   >>> num2 = 20
   >>>
   >>> if (num1 < 100 and num2 < 100):
   ...     print('both are less than 100')
   ... else:
   ...     print('at least one of them is not less than 100')


**While loops** also execute according to conditionals. They will continue to
execute as long as a condition is True. For example:

.. code-block:: python3

   >>> i = 0
   >>>
   >>> while (i < 10):
   ...     print( f'i = {i}' )       # literal string interpolation
   ...     i = i + 1

The ``break`` statement can also be used to escape loops:

.. code-block:: python3

   >>> i = 0
   >>>
   >>> while (i < 10):
   ...     if (i==5):
   ...         break
   ...     print( f'i = {i}' )
   ...     i = i + 1

.. note::

   Try replacing ``break`` with ``continue`` and observe how the behavior changes.  
   While ``break`` exits the loop entirely when the condition is met, ``continue`` skips the current iteration and moves on to the next one.


**For loops** in Python are useful when you need to execute the same set of
instructions over and over again. They are especially great for iterating over
lists:

.. code-block:: python3

   >>> my_shape_list = ['circle', 'heart', 'triangle', 'square']
   >>>
   >>> for shape in my_shape_list:
   ...     print(shape)
   >>>
   >>> for shape in my_shape_list:
   ...     if (shape == 'circle'):
   ...         pass                    # do nothing
   ...     else:
   ...         print(shape)

You can also use the ``range()`` function to iterate over a range of numbers:

.. code-block:: python3

   >>> for x in range(10):
   ...     print(x)
   >>>
   >>> for x in range(10, 100, 5):
   ...     print(x)
   >>>
   >>> for a in range(3):
   ...     for b in range(3):
   ...         for c in range(3):
   ...             print( f'{a} + {b} + {c} = {a+b+c}' )

.. note::

   The code is getting a little bit more complicated now. If you are using the Python3 interactive
   interpreter, it will be easier to start writing the code in Python scripts, as seen below. If you
   are using a Jupyter notebook, make sure to add all code to a cell before executing it.


Functions
---------

**Functions** are blocks of codes that are run only when we call them. We can
pass data into functions, and have functions return data to us. Functions are
absolutely essential to keeping code clean and organized.

On the command line, use a text editor to start writing a Python script:

.. code-block:: console

   [frontera]$ vim function_test.py

Enter the following text into the script:

.. code-block:: python3
   :linenos:

   def hello_world():
       print('Hello, world!')
   
   hello_world()

After saving and quitting the file, execute the script (Python code is not
compiled - just run the raw script with the ``python3`` executable):

.. code-block:: console

   [frontera]$ python3 function_test.py
   Hello, world!

.. note::

   Future examples from this point on will assume familiarity with using the
   text editor and executing the script. We will just be showing the contents of
   the script and console output.

More advanced functions can take parameters and return results:

.. code-block:: python3
   :linenos:

   def add5(value):
       return(value + 5)
   
   final_number = add5(10)
   print(final_number)

.. code-block:: console

   15

Pass multiple parameters to a function:

.. code-block:: python3
   :linenos:

   def add5_after_multiplying(value1, value2):
       return( (value1 * value2) + 5)
   
   final_number = add5_after_multiplying(10, 2)
   print(final_number)

.. code-block:: console

   25

It is a good idea to put your list operations into a function in case you plan
to iterate over multiple lists:

.. code-block:: python3
   :linenos:

   def print_ts(mylist):
       for x in mylist:
           if (x[0] == 't'):      # a string (x) can be interpreted as a list of chars!
               print(x)
   
   list1 = ['circle', 'heart', 'triangle', 'square']
   list2 = ['one', 'two', 'three', 'four']

   print_ts(list1)
   print_ts(list2)

.. code-block:: console

   triangle
   two
   three

There are many more ways to call functions, including handing an arbitrary
number of arguments, passing keyword / unordered arguments, assigning default
values to arguments, and more.


File Handling
-------------

The ``open()`` function does all of the file handling in Python. It takes two
arguments - the *filename* and the *mode*. The possible modes are read (``r``),
write (``w``), append (``a``), or create (``x``).

When writing output to a new file on the file system, make sure you are attempting to
write somwhere where you have permissions to write:

.. code-block:: python3
   :linenos:

   my_shapes = ['circle', 'heart', 'triangle', 'square']
   
   with open('my_shapes.txt', 'w') as f:
       for shape in my_shapes:
           f.write(shape)

.. code-block:: console

   (in my_shapes.txt)
   circlehearttrianglesquare

.. tip::

   By opening the file with the ``with`` statement above, you get built in
   exception handling, and it automatically will close the file handle for you.
   It is generally recommended as the best practice for file handling.

You may notice the output file is lacking in newlines this time. Try adding
newline characters to your output:

.. code-block:: python3
   :linenos:

   my_shapes = ['circle', 'heart', 'triangle', 'square']
   
   with open('my_shapes.txt', 'w') as f:
       for shape in my_shapes:
           f.write( f'{shape}\n' )

.. code-block:: console

   (in my_shapes.txt)
   circle
   heart
   triangle
   square

Now notice that the original line in the output file is gone - it has been
overwritten. Be careful if you are using write (``w``) vs. append (``a``).

To read a file in, do the following:

.. code-block:: python3
   :linenos:

   with open('my_shapes.txt', 'r') as f:
       for x in range(4):
           print(f.readline())

.. code-block:: text

   circle
   
   heart
   
   triangle
   
   square

You may have noticed in the above that there seems to be an extra space between
each word. What is actually happening is that the file being read has newline
characters on the end of each line (``\n``). When read into the Python script,
the original new line is being printed, followed by another newline added by the
``print()`` function. Stripping the newline character from the original string
is the easiest way to solve this problem:

.. code-block:: python3
   :linenos:

   with open('my_shapes.txt', 'r') as f:
       for x in range(4):
           print(f.readline().strip('\n'))

.. code-block:: text

   circle
   heart
   triangle
   square

Read the whole file and store it as a list:

.. code-block:: python3
   :linenos:

   words = []

   with open('my_shapes.txt', 'r') as f:
       words = f.read().splitlines()                # careful of memory usage

   for x in range(4):
       print(words[x])

.. code-block:: text

   circle
   heart
   triangle
   square


Importing Libraries
-------------------

The Python built-in functions, some of which we have seen above, are useful but
limited. Part of what makes Python so powerful is the huge number and variety
of libraries that can be *imported*. For example, if you want to work with
random numbers, you have to import the ``random`` library into your code, which
has a method for generating random numbers called 'random'.

.. code-block:: python3
   :linenos:

   import random

   for i in range(5):
       print(random.random())

.. code-block:: bash

   0.47115888799541383
   0.5202615354150987
   0.8892412583071456
   0.7467080997595558
   0.025668541754695906

More information about using the ``random`` library can be found in the
`Python docs <https://docs.python.org/3/library/random.html>`_

Some libraries that you might want to use are not included in the official
Python distribution - called the *Python Standard Library*. Libraries written
by the user community can often be found on `PyPI.org <https://pypi.org/>`_ and
downloaded to your local environment using a tool called ``pip3``.

For example, if you wanted to download the
`names <https://pypi.org/project/names/>`_ library and use it in your Python
code, you would do the following:

.. code-block:: bash

   [frontera]$ pip3 install --user names
   Collecting names
     Downloading https://files.pythonhosted.org/packages/44/4e/f9cb7ef2df0250f4ba3334fbdabaa94f9c88097089763d8e85ada8092f84/names-0.3.0.tar.gz (789kB)
       100% |████████████████████████████████| 798kB 1.1MB/s
   Installing collected packages: names
     Running setup.py install for names ... done
   Successfully installed names-0.3.0

Notice the library is installed above with the ``--user`` flag. The class server
is a shared system and non-privileged users can not download or install packages
in root locations. The ``--user`` flag instructs ``pip3`` to install the library
in your own home directory.

.. tip::

   If you are using a Jupyter notebook, you can install packages directly from
   the notebook by using the ``!`` operator. For example, ``! pip3 install --user names``.

.. code-block:: python3
   :linenos:

   import names
   
   for i in range(5):
       print(names.get_full_name())

.. code-block:: bash

   Johnny Campbell
   Lawrence Webb
   Johnathan Holmes
   Mary Wang
   Jonathan Henry


Additional Resources
--------------------

* `The Python Standard Library <https://docs.python.org/3/library/>`_
* `PEP 8 Python Style Guide <https://www.python.org/dev/peps/pep-0008/>`_
* `Jupyter notebooks in a browser <https://jupyter.org/try>`_
* `Jupyter notebooks on TACC systems <https://tap.tacc.utexas.edu/>`_
