class PairGen2:
    def __init__(self, X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size=32, step=256, positive_label=1.0, negative_label=0.0):
        import numpy as np
        import tensorflow as tf
        
        input_shape = np.shape(X_train)
        
        # create indices of classes
        from collections import defaultdict
        train_idx = defaultdict(list)
        for y_train_idx, y in enumerate(y_train):
            train_idx[y].append(y_train_idx)
            
        valid_idx = defaultdict(list)
        for y_valid_idx, y in enumerate(y_valid):
            valid_idx[y].append(y_valid_idx)

        y_test = y_test - np.min(y_test)
        test_idx = defaultdict(list)
        for y_test_idx, y in enumerate(y_test):
            test_idx[y].append(y_test_idx)
                
        def generator_train():
            step_cnt = 0
            while step_cnt < step:
                step_cnt += 1
                batch_cnt = 0
                image_left = []
                image_right = []
                label = []
                
                while batch_cnt < batch_size:
                    batch_cnt += 1
                    current_label = np.random.randint(2)
                    if current_label == 1:
                        classNum = np.random.randint(len(train_idx))
                        indexArr = np.random.choice(len(train_idx[classNum]), 2, replace=False)
                        image_left.append(X_train[train_idx[classNum][indexArr[0]]])
                        image_right.append(X_train[train_idx[classNum][indexArr[1]]])
                        label.append([positive_label])
                    else:
                        classNum = np.random.choice(len(train_idx), 2, replace=False)
                        indexArr = [np.random.randint(len(train_idx[classNum[0]])), np.random.randint(len(train_idx[classNum[1]]))]
                        image_left.append(X_train[train_idx[classNum[0]][indexArr[0]]])
                        image_right.append(X_train[train_idx[classNum[1]][indexArr[1]]])
                        label.append([negative_label])
                        
                yield np.array([image_left, image_right]), np.array(label)
                
        def generator_valid():
            step_cnt = 0
            while step_cnt < step:
                step_cnt += 1
                batch_cnt = 0
                image_left = []
                image_right = []
                label = []
                
                while batch_cnt < batch_size:
                    batch_cnt += 1
                    current_label = np.random.randint(2)
                    if current_label == 1:
                        classNum = np.random.randint(len(valid_idx))
                        indexArr = np.random.choice(len(valid_idx[classNum]), 2, replace=False)
                        image_left.append(X_valid[valid_idx[classNum][indexArr[0]]])
                        image_right.append(X_valid[valid_idx[classNum][indexArr[1]]])
                        label.append([positive_label])
                    else:
                        classNum = np.random.choice(len(valid_idx), 2, replace=False)
                        indexArr = [np.random.randint(len(valid_idx[classNum[0]])), np.random.randint(len(valid_idx[classNum[1]]))]
                        image_left.append(X_valid[valid_idx[classNum[0]][indexArr[0]]])
                        image_right.append(X_valid[valid_idx[classNum[1]][indexArr[1]]])
                        label.append([negative_label])
                        
                yield np.array([image_left, image_right]), np.array(label)
                
        def generator_test():
            step_cnt = 0
            while step_cnt < step:
                step_cnt += 1
                batch_cnt = 0
                image_left = []
                image_right = []
                label = []
                
                while batch_cnt < batch_size:
                    batch_cnt += 1
                    current_label = np.random.randint(2)
                    if current_label == 1:
                        classNum = np.random.randint(len(test_idx))
                        indexArr = np.random.choice(len(test_idx[classNum]), 2, replace=False)
                        image_left.append(X_test[test_idx[classNum][indexArr[0]]])
                        image_right.append(X_test[test_idx[classNum][indexArr[1]]])
                        label.append([positive_label])
                    else:
                        classNum = np.random.choice(len(test_idx), 2, replace=False)
                        indexArr = [np.random.randint(len(test_idx[classNum[0]])), np.random.randint(len(test_idx[classNum[1]]))]
                        image_left.append(X_test[test_idx[classNum[0]][indexArr[0]]])
                        image_right.append(X_test[test_idx[classNum[1]][indexArr[1]]])
                        label.append([negative_label])
                        
                yield np.array([image_left, image_right]), np.array(label)
                
                
        self.ds_train = tf.data.Dataset.from_generator(generator_train, output_signature=(
            tf.TensorSpec(shape=(2, None, input_shape[1], input_shape[2], input_shape[3]), dtype=tf.float32), 
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
        ))
        self.ds_valid = tf.data.Dataset.from_generator(generator_valid, output_signature=(
            tf.TensorSpec(shape=(2, None, input_shape[1], input_shape[2], input_shape[3]), dtype=tf.float32), 
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
        ))
        self.ds_test = tf.data.Dataset.from_generator(generator_test, output_signature=(
            tf.TensorSpec(shape=(2, None, input_shape[1], input_shape[2], input_shape[3]), dtype=tf.float32), 
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
        ))