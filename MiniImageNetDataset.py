class MiniImageNetDataset:
    def __init__(self):
        import tensorflow as tf
        import numpy as np
        
        train_ds = tf.keras.preprocessing.image_dataset_from_directory("./data/miniimagenet/train/", batch_size=None, image_size=(84,84), interpolation='nearest', shuffle=False)
        valid_ds = tf.keras.preprocessing.image_dataset_from_directory("./data/miniimagenet/val/", batch_size=None, image_size=(84,84), interpolation='nearest', shuffle=False)
        test_ds = tf.keras.preprocessing.image_dataset_from_directory("./data/miniimagenet/test/", batch_size=None, image_size=(84,84), interpolation='nearest', shuffle=False)
        
        imageList_train = []
        labelList_train = []
        imageList_valid = []
        labelList_valid = []
        imageList_test = []
        labelList_test = []
        
        def extraction(image, label):
            image = tf.image.convert_image_dtype(image, tf.float32)
            return image, label
        
        for image, label in train_ds.map(extraction):
            image = image.numpy()
            label = label.numpy()

            imageList_train.append(image)
            labelList_train.append(label)
            
        for image, label in valid_ds.map(extraction):
            image = image.numpy()
            label = label.numpy()

            imageList_valid.append(image)
            labelList_valid.append(label)
            
        for image, label in test_ds.map(extraction):
            image = image.numpy()
            label = label.numpy()

            imageList_test.append(image)
            labelList_test.append(label)
            
        self.image_train = np.array(imageList_train)
        self.label_train = np.array(labelList_train)
        self.image_valid = np.array(imageList_valid)
        self.label_valid = np.array(labelList_valid)
        self.image_test = np.array(imageList_test)
        self.label_test = np.array(labelList_test)