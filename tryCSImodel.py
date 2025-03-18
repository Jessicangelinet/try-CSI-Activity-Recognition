"""
The Codes in this file are used to classify Human Activity using Channel State Information. 
The deep learning architecture used here is Bidirectional LSTM stacked with One Attention Layer.
Author: https://github.com/ludlows
2019-12
"""
import numpy as np 
import tensorflow as tf
import glob
import os
import csv

from tensorflow.python.client import device_lib

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(device_lib.list_local_devices())

# Set TensorFlow to use the GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def merge_csi_label(csifile, win_len=100, thrshd=0.6, step=20):
    """
    Merge CSV files into a Numpy Array  X,  csi amplitude feature
    Returns Numpy Array X, Shape(Num, Win_Len, 166)
    Args:
        csifile  :  str, csv file containing CSI data
        win_len  :  integer, window length
        thrshd   :  float,  determine if an activity is strong enough inside a window
        step     :  integer, sliding window by step
    """
    csi = []
    with open(csifile, 'r') as csif:
        reader = csv.reader(csif, delimiter=',')
        for line in reader:
            try:
                line_array = np.array([float(v) for v in line])
                csi.append(line_array[np.newaxis,...])
            except ValueError:
                print(f"Skipping line due to conversion error: {line}")
                continue
    csi = np.concatenate(csi, axis=0)

    print('💔CSI Shape:', csi.shape, 'CSI', csi)

    # Pad the sequences to ensure they all have the same length
    if csi.shape[0] < win_len:
        padding = np.zeros((win_len - csi.shape[0], csi.shape[1]))
        csi = np.vstack((csi, padding))

    # screen the data with a window
    index = 0
    feature = []
    while index + win_len <= csi.shape[0]:
        print('=== goes into the loop ===')
        cur_feature = np.zeros((1, win_len, 166))
        cur_feature[0] = csi[index:index+win_len, :]
        feature.append(cur_feature)
        index += step
    print('😭',feature)

    return np.concatenate(feature, axis=0)

def extract_csi_by_label(raw_folder, label, labels, save=False, win_len=100, thrshd=0.6, step=20):
    """
    Returns all the samples (X,y) of "label" in the entire dataset
    Args:
        raw_folder: The path of Dataset folder
        label    : str, could be one of labels
        labels   : list of str, ['waving', 'twist', 'standing', 'squatting', 'rubhand', 'pushpull', 'punching', 'nopeople', 'jump', 'clap']
        save     : boolean, choose whether save the numpy array 
        win_len  :  integer, window length
        thrshd   :  float,  determine if an activity is strong enough inside a window
        step     :  integer, sliding window by step
    """
    print('Starting Extract CSI for Label {}'.format(label))
    label = label.lower()
    if not label in labels:
        raise ValueError("The label {} should be among 'waving', 'twist', 'standing', 'squatting', 'rubhand', 'pushpull', 'punching', 'nopeople', 'jump', 'clap'".format(labels))

    data_path_pattern = os.path.join(raw_folder, label, '*.csv')
    input_csv_files = sorted(glob.glob(data_path_pattern))
    feature = []
    index = 0

    for csi_file in input_csv_files:
        print(csi_file)
        index += 1
        csi_data = merge_csi_label(csi_file, win_len=win_len, thrshd=thrshd, step=step)
        if csi_data is not None:
            feature.append(csi_data)
        print('Finished {:.2f}% for Label {}'.format(index / len(input_csv_files) * 100,label))

    feat_arr = np.concatenate(feature, axis=0)
    if save:
        np.savez_compressed("X_{}_win_{}_thrshd_{}percent_step_{}.npz".format(
            label, win_len, int(thrshd*100), step), feat_arr)
    # one hot
    feat_label = np.zeros((feat_arr.shape[0], len(labels)))
    feat_label[:, labels.index(label)] = 1
    return feat_arr, feat_label



def train_valid_split(numpy_tuple, train_portion=0.9, seed=379):
    """
    Returns Train and Valid Datset with the format of (x_train, y_train, x_valid, y_valid),
    where x_train and y_train are shuffled randomly.

    Args:
        numpy_tuple  : tuple of numpy array: (x_waving, x_twist, x_standing, x_squatting, x_rubhand, x_pushpull, x_punching, x_nopeople, x_jump, x_clap)
        train_portion: float, range (0,1)
        seed         : random seed
    """
    np.random.seed(seed=seed)
    x_train = []
    x_valid = []
    y_valid = []
    y_train = []

    for i, x_arr in enumerate(numpy_tuple):
        index = np.random.permutation([i for i in range(x_arr.shape[0])])
        split_len = int(train_portion * x_arr.shape[0])
        x_train.append(x_arr[index[:split_len], ...])
        tmpy = np.zeros((split_len, len(numpy_tuple)))
        tmpy[:, i] = 1
        y_train.append(tmpy)
        x_valid.append(x_arr[index[split_len:],...])
        tmpy = np.zeros((x_arr.shape[0]-split_len, len(numpy_tuple)))
        tmpy[:, i] = 1
        y_valid.append(tmpy)
    
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    x_valid = np.concatenate(x_valid, axis=0)
    y_valid = np.concatenate(y_valid, axis=0)

    index = np.random.permutation([i for i in range(x_train.shape[0])])
    x_train = x_train[index, ...]
    y_train = y_train[index, ...]
    return x_train, y_train, x_valid, y_valid
    
    

def extract_csi(raw_folder, labels, save=False, win_len=1000, thrshd=0.6, step=200):
    """
    Return List of Array in the format of [X_label1, y_label1, X_label2, y_label2, .... X_Label10, y_label10]
    Args:
        raw_folder: the folder path of raw CSI csv files, input_* annotation_*
        labels    : all the labels existing in the folder
        save      : boolean, choose whether save the numpy array 
        win_len   :  integer, window length
        thrshd    :  float,  determine if an activity is strong enough inside a window
        step      :  integer, sliding window by step
    """
    ans = []
    for label in labels:
        feature_arr, label_arr = extract_csi_by_label(raw_folder, label, labels, save, win_len, thrshd, step)
        ans.append(feature_arr)
        ans.append(label_arr)
    return tuple(ans)


class AttenLayer(tf.keras.layers.Layer):
    """
    Attention Layers used to Compute Weighted Features along Time axis
    Args:
        num_state :  number of hidden Attention state
    
    2019-12, https://github.com/ludlows
    """
    def __init__(self, num_state, **kw):
        super(AttenLayer, self).__init__(**kw)
        self.num_state = num_state
    
    def build(self, input_shape):
        self.kernel = self.add_weight('kernel', shape=[input_shape[-1], self.num_state])
        self.bias = self.add_weight('bias', shape=[self.num_state])
        self.prob_kernel = self.add_weight('prob_kernel', shape=[self.num_state])

    def call(self, input_tensor):
        atten_state = tf.tanh(tf.tensordot(input_tensor, self.kernel, axes=1) + self.bias)
        logits = tf.tensordot(atten_state, self.prob_kernel, axes=1)
        prob = tf.nn.softmax(logits)
        weighted_feature = tf.reduce_sum(tf.multiply(input_tensor, tf.expand_dims(prob, -1)), axis=1)
        return weighted_feature
    
    # for saving the model
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_state': self.num_state,})
        return config


class CSIModelConfig:
    """
    class for Human Activity Recognition ("waving", "twist", "standing", "squatting", "rubhand", "pushpull", "punching", "nopeople", "jump", "clap")
    Using CSI (Channel State Information)
    Specifically, the author here wants to classify Human Activity using Channel State Information. 
    The deep learning architecture used here is Bidirectional LSTM stacked with One Attention Layer.
       2019-12, https://github.com/ludlows
    Args:
        win_len   :  integer (1000 default) window length for batching sequence
        step      :  integer (200  default) sliding window by this step
        thrshd    :  float   (0.6  default) used to check if the activity is intensive inside a window
        downsample:  integer >=1 (2 default) downsample along the time axis
    """
    def __init__(self, dataset_path, win_len=100, step=20, thrshd=0.6, downsample=2):
        self._win_len = win_len
        self._step = step
        self._thrshd = thrshd
        # self._labels = tuple(sorted(os.listdir(dataset_path)))
        self._labels = ('clap', 'jump', 'nopeople', 'punching', 'pushpull', 'rubhand', 'squatting', 'standing', 'twist', 'waving')
        self._downsample = downsample

    def preprocessing(self, raw_folder, save=False):
        """
        Returns the Numpy Array for training within the format of (X_lable1, y_label1, ...., X_label10, y_label10)
        Args:
            raw_folder: the folder containing raw CSI 
            save      : choose if save the numpy array
        """
        numpy_tuple = extract_csi(raw_folder, self._labels, save, self._win_len, self._thrshd, self._step)
        if self._downsample > 1:
            return tuple([v[:, ::self._downsample,...] if i%2 ==0 else v for i, v in enumerate(numpy_tuple)])
        return numpy_tuple
    
    def load_csi_data_from_files(self, np_files):
        """
        Returns the Numpy Array for training within the format of (X_lable1, y_label1, ...., X_label10, y_label10)
        Args:
            np_files: ('x_waving.npz', 'x_twist.npz', 'x_standing.npz', 'x_squatting.npz', 'x_rubhand.npz', 'x_pushpull.npz', 'x_punching.npz', 'x_nopeople.npz', 'x_jump.npz', 'x_clap.npz')
        """
        if len(np_files) != 10:
            raise ValueError('There should be 10 numpy files for waving, twist, standing, squatting, rubhand, pushpull, punching, nopeople, jump, clap.')
        x = [np.load(f)['arr_0'] for f in np_files]
        if self._downsample > 1:
            x = [arr[:,::self._downsample, :] for arr in x]
        y = [np.zeros((arr.shape[0], len(self._labels))) for arr in x]
        numpy_list = []
        for i in range(len(self._labels)):
            y[i][:,i] = 1
            numpy_list.append(x[i])
            numpy_list.append(y[i])
        return tuple(numpy_list)


    
    def build_model(self, n_unit_lstm=200, n_unit_atten=400):
        """
        Returns the Tensorflow Model which uses AttenLayer
        """
        if self._downsample > 1:
            length = len(np.ones((self._win_len,))[::self._downsample])
            x_in = tf.keras.Input(shape=(length, 166))
        else:
            x_in = tf.keras.Input(shape=(self._win_len, 166))
        x_tensor = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=n_unit_lstm, return_sequences=True))(x_in)
        x_tensor = AttenLayer(n_unit_atten)(x_tensor)
        pred = tf.keras.layers.Dense(len(self._labels), activation='softmax')(x_tensor)
        model = tf.keras.Model(inputs=x_in, outputs=pred)
        return model

    @staticmethod
    def load_model(hdf5path):
        """
        Returns the Tensorflow Model for AttenLayer
        Args:
            hdf5path: str, the model file path
        """
        model = tf.keras.models.load_model(hdf5path, custom_objects={'AttenLayer':AttenLayer})
        return model
    

if __name__ == "__main__":
    import timeit
    start_time = timeit.default_timer()

    # import sys
    # if len(sys.argv) != 2:
    #     print("Error! Correct Command: python3 csimodel.py Dataset_folder_path")
    # raw_data_folder = sys.argv[1]

    raw_data_folder = "FYP_Data\\front"

    # preprocessing
    cfg = CSIModelConfig(dataset_path=raw_data_folder, win_len=1000, step=200, thrshd=0.6, downsample=2)
    activities = list(cfg._labels)
    print(activities)
    
    numpy_tuple = cfg.preprocessing(raw_data_folder, save=True)
    '''
    
    # load previous saved numpy files, ignore this if you haven't saved numpy array to files before
    # numpy_tuple = cfg.load_csi_data_from_files(('x_waving.npz', 'x_twist.npz', 'x_standing.npz', 'x_squatting.npz', 'x_rubhand.npz', 'x_pushpull.npz', 'x_punching.npz', 'x_nopeople.npz', 'x_jump.npz', 'x_clap.npz'))
    x_waving, y_waving, x_twist, y_twist, x_standing, y_standing, x_squatting, y_squatting, x_rubhand, y_rubhand, x_pushpull, y_pushpull, x_punching, y_punching, x_nopeople, y_nopeople, x_jump, y_jump, x_clap, y_clap = numpy_tuple
    x_train, y_train, x_valid, y_valid = train_valid_split(
        (x_waving, x_twist, x_standing, x_squatting, x_rubhand, x_pushpull, x_punching, x_nopeople, x_jump, x_clap),
        train_portion=0.75, seed=379)

    ''' 
        # Assign the numpy_tuple to variables in the specified order
    x_clap, y_clap, x_jump, y_jump, x_nopeople, y_nopeople, x_punching, y_punching, x_pushpull, y_pushpull, x_rubhand, y_rubhand, x_squatting, y_squatting, x_standing, y_standing, x_twist, y_twist, x_waving, y_waving = numpy_tuple
    
    x_train, y_train, x_valid, y_valid = train_valid_split(
        (x_clap, x_jump, x_nopeople, x_punching, x_pushpull, x_rubhand, x_squatting, x_standing, x_twist, x_waving),
        train_portion=0.8, seed=379)   
    '''
    # Create dictionaries to store the data
    x_data = {}
    y_data = {}
    
    # Populate the dictionaries with the data
    for i, activity in enumerate(activities):
        x_data[activity] = numpy_tuple[2 * i]
        y_data[activity] = numpy_tuple[2 * i + 1]
   
    # Combine the data for training and validation
    x_train_list = []
    y_train_list = []
    x_valid_list = []
    y_valid_list = []
    
    for activity in activities:
        x_train, y_train, x_valid, y_valid = train_valid_split(
            (x_data[activity],), train_portion=0.75, seed=379)
        x_train_list.append(x_train)
        y_train_list.append(y_train)
        x_valid_list.append(x_valid)
        y_valid_list.append(y_valid)
    
    x_train = np.concatenate(x_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    x_valid = np.concatenate(x_valid_list, axis=0)
    y_valid = np.concatenate(y_valid_list, axis=0)

    '''

    print('👍', x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)
    # parameters for Deep Learning Model
    model = cfg.build_model(n_unit_lstm=200, n_unit_atten=400)
    # train
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy', 
        metrics=['accuracy'])
    model.summary()
    model.fit(
        x_train,
        y_train,
        batch_size=128, epochs=30,
        validation_data=(x_valid, y_valid),
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint('best_atten.hdf5',
                                                monitor='val_accuracy',
                                                save_best_only=True,
                                                save_weights_only=False)
            ])
    # load the best model
    model = cfg.load_model('best_atten.hdf5')
    y_pred = model.predict(x_valid)

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    
    activities = list(cfg._labels)
    print(activities)
    print(y_pred.shape, np.argmax(y_pred, axis=1))
    print(y_valid.shape, np.argmax(y_valid, axis=1))

    cm = confusion_matrix(np.argmax(y_valid, axis=1), np.argmax(y_pred, axis=1))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=activities)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("./confusion_matrix_test.png", bbox_inches="tight")
    plt.show()

# =======================================================================

end_time = timeit.default_timer()
print(f"time taken to run: {end_time - start_time} seconds")
