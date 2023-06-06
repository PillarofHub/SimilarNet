class OmniglotDataset:
    def __init__(self):
        import tensorflow as tf
        import tensorflow_datasets as tfds
        import numpy as np
        
        train_ds = tfds.load("omniglot", split="train", as_supervised=True, shuffle_files=False)
        test_ds = tfds.load("omniglot", split="test", as_supervised=True, shuffle_files=False)
        
        imageList_train = []
        labelList_train = []
        imageList_test = []
        labelList_test = []
        
        def extraction(image, label):
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.rgb_to_grayscale(image)
            image = tf.image.resize(image, [28, 28])
            return image, label
        
        for image, label in train_ds.map(extraction):
            image = image.numpy()
            label = label.numpy()

            imageList_train.append(image)
            labelList_train.append(label)
            
        for image, label in test_ds.map(extraction):
            image = image.numpy()
            label = label.numpy()

            imageList_test.append(image)
            labelList_test.append(label)
            
        self.image_train = np.array(imageList_train)
        self.label_train = np.array(labelList_train)
        self.image_test = np.array(imageList_test)
        self.label_test = np.array(labelList_test)