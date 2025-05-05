Exploratory Data Analysis
=========================

We will begin by giving a high-level overview of the kinds of data analysis 
tasks we will be performing and why they are important. We'll also introduce 
the ``numpy`` python package and work through some examples in a Jupyter notebook.

By the end of this module, you should be able to: 

* Understand what data analysis is and why (at a high level) it is important. 
* Have a basic understanding of the different kinds of tasks we will perform and what libraries 
  we will use for each kind of task. 
* (Numpy) Understand the primary differences between the ``ndarray`` object from ``numpy`` and basic Python 
  lists, and when to use each.
* (Numpy) Utilize ``ndarray`` objects to perform various computations, including linear algebra calculations 
  and statistical operations. 





Step 1: Data Loading and Organization
-------------------------------------

..
 In this step, we load all coral images from the dataset directory and organize them into a DataFrame. 
 Each image is assigned a label based on the name of the directory it's stored in (i.e., 'ACER' - *Acropora cervicornis*, 'CNAT' - *Colpophyllia natans*, 'MCAV' - * Montastraea cavernosa*). 
..
 This DataFrame will serve as the foundation for splitting our data into training, validation, and test sets later in the tutorial.

1.1 List Dataset Directory Contents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 Before loading the images, we first want to inspect the directory structure to make sure everything is in the right place. 
 The code below lists the contents of the ``coral-species`` data directory to verity that the subdirectories for each coral species are present and correctly named:

 .. code-block:: python

    from pathlib import Path

    # Define the path to the dataset directory
    # NOTE: Replace the path below with the full path to your scratch directory
    dataset_dir = Path('/full/path/to/your/scratch/directory/coral-species-CNN-tutorial/data/coral-species')

    # List the contents of the data directory
    print(list(dataset_dir.iterder()))

    # You should see something like this:
    # [PosixPath('../data/coral-species/MCAV'), PosixPath('../data/coral-species/ACER'), PosixPath('../data/coral-species/CNAT')]
    
1.2 Check File Extensions
~~~~~~~~~~~~~~~~~~~~~~~~~

 Next, we scan the dataset directory and all its subdirectories to find out what types of image files are present. 
 This helps us catch unexpected or unsupported file types (e.g., GIFs, txt files, etc.), which could cause problems later when loading images. 

 This also allows us to see if the images are all in the same format or not.

 .. code-block:: python

    # Recursively list all files under the dataset directory
    image_files = list(dataset_dir.rglob("*"))

    # Extract and print the unique file extensions
    # This helps us confirm that only valid image files are present
    extensions = set(p.suffix.lower() for p in image_files if p.is_file())
    print("File extensions found:", extensions)

 **Question**: What file extensions are present in the dataset? Write down your answer.

1.3 Explore Image Dimensions and Color Modes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 Before feeding images into a CNN, it's important to understand the basic properties of the dataset.
 In this step, we examine the **dimensions** (width x height) as well as the **color mode** (e.g., RGB, RGBA, grayscale) of each image.
 This helps us decide if we need to resize or convert images before we begin training our CNN. 

 The script below prints a summary and gives recommendations if inconsistencies are found.

 .. code-block:: python

    from PIL import Image
    from pathlib import Path
    from collections import Counter

    def explore_image_dataset(data_root):
        """
        Explore basic properties of images: size and color mode.
        """
        print("Starting image dataset exploration...\n")
        
        # Gather all .jpg files in the dataset
        image_files = list(Path(data_root).rglob('*.jpg'))
        print(f"Found {len(image_files)} image files\n")
        
        # Track sizes and color modes
        image_sizes = []
        color_modes = []

        print("Checking image dimensions and color modes...\n")
        for img_path in image_files:
            with Image.open(img_path) as img:
                image_sizes.append(img.size)   
                color_modes.append(img.mode)  

        # Summarize image sizes
        size_counts = Counter(image_sizes)
        print("=== Image Sizes ===")
        print(f"Found {len(size_counts)} unique image sizes:")
        for size, count in size_counts.most_common():
            print(f"- {size}: {count} images")

        # Summarize color modes
        mode_counts = Counter(color_modes)
        print("\n=== Color Modes ===")
        print(f"Found {len(mode_counts)} unique color modes:")
        for mode, count in mode_counts.most_common():
            print(f"- {mode}: {count} images")

        # Simple recommendations
        print("\n=== Recommendations ===")
        if len(size_counts) > 1:
            print(f"Images have different sizes. Consider resizing.")
        else:
            print("All images are the same size.")
        
        if len(mode_counts) > 1:
            print("Images have different color modes. Consider converting to RGB.")
        else:
            print("All images share the same color mode.")

    # Run the function
    data_root = Path('../data/coral-species')
    explore_image_dataset(data_root)
    
 Our dataset analysis reveals some important characteristics that we'll need to keep in mind as we proceed with the tutorial:

 1. **Image Size Variation**: We have 500 total images in out dataset, with 132 different image sizes (dimensions). Also notice that some images are in portrait orientation (height > width) while others are landscape (width > height). CNNs expect all images to have the same dimensions, so we'll need to resize them to a standard size before training our model.

 2. **Color Mode**: Not all images have the same color mode. CNNs also expect all images to have the same color mode, so we'll need to convert any images with non-RGB color modes to RGB.

 We will address these issues in Step 5 when we prepare our data for input into the CNN. 

1.4 Check for Corrupted Images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 Before continuing, we want to make sure that all images files are readable. 
 Corrupted files can break your model training or cause unexpected errors during preprocessing. 

 In this step, we:

  1. Attempt to open each '.jpg' file using PIL 
  2. Discard any files that fail to load 

 This ensures we only keep clean, valid images for training.

 .. code-block:: python

    from PIL import Image
    from tqdm import tqdm

    # Find all .jpg files in the dataset
    # NOTE: add the correct file extension(s) for your image dataset in the space indicated below
    # TIP: see Step 1.2
    image_paths = list(dataset_dir.rglob('*.___'))

    # Create lists to store valid and corrupted files
    valid_images = []
    bad_images = []

    print("Checking for corrupted images...\n")

    # tqdm adds a progress bar to show how long the process will take
    for path in tqdm(image_paths):
        try:
            # Try to open and verify the image
            with Image.open(path) as img:
                img.verify()
            # If the image is valid, add it to valid_images
            valid_images.append(path)

        except Exception:
            # If any error occurs while opening/verifying the image, add it to bad_images
            bad_images.append(path)

    print(f"Valid images: {len(valid_images)}")
    print(f"Corrupted images removed: {len(bad_images)}")
 
 If there are any corrupted images, in your dataset, this code will automatically remove them. 

1.5 Create a DataFrame of Image Paths and Labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 Now that we have a good idea of what our data looks like and have removed any corrupted images, we can start setting up our data for training.
 In this step, we build a ``pandas.DataFrame`` that organizes all the image data into two columns:

  1. **filepath**: The full path to each image file
  2. **label**: The class label for each image, taken from the directory name

 This structured DataFrame is essential for training with Keras' ``flow_from_dataframe`` method that we'll use later in the tutorial.

 .. code-block:: python

    import pandas as pd

    # Build (filepath, label) pairs from valid image paths
    data = []
    for path in valid_images:
        label = path.parent.name # Extract label from directory name
        data.append((str(path), label))

    # Create a DataFrame with columns for filepath and label
    df = pd.DataFrame(data, columns=["filepath", "label"])

    # (Optional) Shuffle the DataFrame to randomize order of images
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Show a preview of the DataFrame
    df.head()
    
 
Step 2: Visualize the Class Distribution
----------------------------------------

 Before training our CNN, it's important to understand how many images we have for each class (i.e., coral species in this case).

 In this step we:

  1. Count how many images belong to each class
  2. Plot the class distribution as a pie chart and bar graph

 If the dataset is imbalanced (i.e., some classes have far more images than others), we may need to account for this later using **class weights** or **data augmentation**.

 .. code-block:: python

    import matplotlib.pyplot as plt

    # Count class distribution
    counts = df['label'].value_counts()

    # Create a 1-row, 2-column subplot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Define a color palette for consistency
    colors = ['#8158ff', '#ff9423', '#7fcdbb'] 

    # Pie chart
    axes[0].pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
    axes[0].axis('equal')
    axes[0].set_title('Class Distribution (Percentage)')

    # Bar chart
    axes[1].bar(counts.index, counts.values, color=colors)
    axes[1].set_title('Class Distribution (Values)')
    axes[1].set_ylabel('Number of Images')
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')

    # Layout adjustment
    plt.tight_layout()
    plt.show()

    # Print label counts and percentages
    for label, count in counts.items():
        print(f"{label}: {count} images ({count/len(df)*100:1f}%)")

 **Thought Challenge**: Describe the class distribution in your own words. How much of the dataset is made up by the largest class? The smallest class? Is there anything that we need to address before continuing?


Step 3: Visualizing Images from the Dataset
-------------------------------------------

 It's helpful to look at a few images from each class to get a better understanding of the dataset.
 This will give us a better sense of:

 - What each coral species looks like
 - How much visual variation exists within each class (e.g., different angles, lighting, etc.)
 - Whether the dataset includes noise, blur, or other artifacts

 We'll display a grid of randomly selected images, grouped by class.

 .. code-block:: python

    import matplotlib.pyplot as plt
    from tensorflow.keras.preprocessing.image import load_img
    import random

    # Set seed for reproducibility
    random.seed(123)

    # Set the number of images to display per class
    samples_per_class = 3

    # Get list of unique coral species names (classes)
    classes = df['label'].unique()

    # Create a figure with appropriate size
    # The height (2.5 * len(classes)) ensures enough space for all images
    plt.figure(figsize=(12, len(classes) * 2.5))

    # Loop through each class to create a grid of images
    for i, label in enumerate(sorted(classes)):
        # Filter DataFrame to get only images from the current class
        class_df = df[df['label'] == label]

        # Randomly select 3 images from the current class 
        sample_paths = random.sample(list(class_df['filepath']), samples_per_class)

        # Create subplot for each image
        for j, img_path in enumerate(sample_paths):

            # Calculate position in grid: (row * width) + column + 1
            plt.subplot(len(classes), samples_per_class, i * samples_per_class + j + 1)

            # Load and display the image
            img = load_img(img_path)        # Load the image
            plt.imshow(img)                 # Display the image
            plt.title(label)                # Add species name as title
            plt.axis('off') 

    plt.tight_layout()
    plt.show()

 .. image:: ./images/coral_species_images.png
   :width: 800px
   :align: center

 **Thought Challenge**: Try changing the ``random.seed`` value a few times to view different images from our dataset. What do you notice? Take a moment to write down your observations.

 *Remember: the quality of a machine learning model is decided largely by the quality of the dataset it was trained on!*


Step 4: Split the Dataset and Handle Class Imbalance
----------------------------------------------------

4.1 Split the Dataset into Training, Validation, and Test Sets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 We are now ready to split our labeled image dataset into three parts:

  1. **Training Set**: Used to train the model
  2. **Validation Set**: Used to tune hyperparameters and monitor model performance during training
  3. **Test Set**: Used to evaluate the final model's performance after training is complete

 We will use the ``train_test_split`` function from scikit-learn in two stages:

  1. First, we split the original dataset into **training + test** sets
  2. Then, we split the training set again into **training + validation** 

 This approach ensures that our CNN *never sees the test set* during training, which is important for obtaining an unbiased estimate of the model's performance.

 To preserve the class distribution across splits, we use ``stratify=df["label"]`` to ensure each split has the same proportion of each class as in the original dataset.
 This is called **stratified sampling**. 

 .. code-block:: python

    # NOTE: Replace the spaces indicated below with your code
    from sklearn.model_selection import ____

    # First, split the original dataset into training + test sets
    train_df, test_df = train_test_split(
        df,                            # This is our DataFrame from step 1.5
        test_size=____,                # How much of the data should be in the test set?
        stratify=____,                 # Ensure each split maintains original class distribution
        random_state=123               # Set the random seed for reproducibility
    )

    # Then, split the training set into training + validation sets
    ____, ____ = train_test_split(
        ____,                          # What goes here?
        test_size=____,                # How much of the data should be in the validation set?
        stratify=____,                 # Ensure each split maintains original class distribution
        random_state=123               # Set the random seed for reproducibility
    )

    # Print split sizes
    total = len(df)
    print(f"\nDataset splits:")
    print(f"Train: {len(train_df)} images ({len(train_df)/total:.2%})")
    print(f"Validation: {len(val_df)} images ({len(val_df)/total:.2%})")
    print(f"Test: {len(test_df)} images ({len(test_df)/total:.2%})")

 **Thought Challenge**: Will changing the ``random_state`` value in the ``train_test_split`` function change your model's performance? Why or why not?

 .. toggle:: Click to show

    **Answer**: Yes – even though stratification preserves class balance, changing ``random_state`` changes *which individual images* go into the training set. For example:

    - With ``random_state=123``, the model might learn from images A, B, and C
    - With ``random_state=456``, the model might learn from images D, E, and F 
 
    Since each image has unique properties (lighting, orientation, scale, background, etc.), the model will learn slightly different features depending on the exact training set.
    As a result, its internal weights and final accuracy may vary. 

    Try running the full training pipeline multiple times with different ``random_state`` values. Do your metrics stay stable? What might that tell you about the robustness of your model?

4.2 Compute Class Weights
~~~~~~~~~~~~~~~~~~~~~~~~~

 If our dataset is imbalanced (i.e., some classes have many more images than others), the model may learn to favor those majority classes. 
 To address this, we can compute **class weights** based on the training data using the ``compute_class_weight`` function from scikit-learn.

 These weights:
 - Assign higher importance to underrepresented classes
 - Are passed into ``model.fit()`` using the ``class_weight`` argument
 