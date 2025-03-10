import numpy as np 
import tensorflow as tf
import glob
import os
import csv
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def merge_csi_label(csifile, win_len=100, thrshd=0.6, step=20):
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
    if len(csi) == 0:
        print(f"No valid CSI data found in file: {csifile}")
        return None

    csi = np.concatenate(csi, axis=0)
    print('💔CSI Shape:', csi.shape, 'CSI', csi)

    if csi.shape[0] < win_len:
        padding = np.zeros((win_len - csi.shape[0], csi.shape[1]))
        csi = np.vstack((csi, padding))

    index = 0
    feature = []
    while index + win_len <= csi.shape[0]:
        print('=== goes into the loop ===')
        cur_feature = np.zeros((1, win_len, 166))
        cur_feature[0] = csi[index:index+win_len, :]
        feature.append(cur_feature)
        index += step
    print('😭', feature)

    return np.concatenate(feature, axis=0) if feature else None

def extract_csi_by_label(raw_folder, label, labels, save=False, win_len=100, thrshd=0.6, step=20):
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
        print('Finished {:.2f}% for Label {}'.format(index / len(input_csv_files) * 100, label))

    if not feature:
        print(f"No valid data found for label: {label}")
        return None, None

    feat_arr = np.concatenate(feature, axis=0)
    if save:
        np.savez_compressed("X_{}_win_{}_thrshd_{}percent_step_{}.npz".format(
            label, win_len, int(thrshd*100), step), feat_arr)
    feat_label = np.zeros((feat_arr.shape[0], len(labels)))
    feat_label[:, labels.index(label)] = 1
    return feat_arr, feat_label

def extract_csi(raw_folder, labels, save=False, win_len=1000, thrshd=0.6, step=200):
    ans = []
    for label in labels:
        feature_arr, label_arr = extract_csi_by_label(raw_folder, label, labels, save, win_len, thrshd, step)
        if feature_arr is not None and label_arr is not None:
            ans.append(feature_arr)
            ans.append(label_arr)
    if not ans:
        raise ValueError("No valid data found for any label.")
    return tuple(ans)

def train_valid_split(numpy_tuple, train_portion=0.9, seed=379):
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

class AttenLayer(tf.keras.layers.Layer):
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
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_state': self.num_state,})
        return config

class CSIModelConfig:
    def __init__(self, labels, win_len=100, step=20, thrshd=0.6, downsample=2):
        self._win_len = win_len
        self._step = step
        self._thrshd = thrshd
        self._labels = labels
        self._downsample = downsample

    def preprocessing(self, raw_folder, save=False):
        numpy_tuple = extract_csi(raw_folder, self._labels, save, self._win_len, self._thrshd, self._step)
        if self._downsample > 1:
            return tuple([v[:, ::self._downsample,...] if i%2 ==0 else v for i, v in enumerate(numpy_tuple)])
        return numpy_tuple
    
    def load_csi_data_from_files(self, np_files):
        if len(np_files) != len(self._labels):
            raise ValueError(f'There should be {len(self._labels)} numpy files for the specified labels.')
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
        model = tf.keras.models.load_model(hdf5path, custom_objects={'AttenLayer':AttenLayer})
        return model

def plot_training_history(history_path, plot_path):
    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = history["epoch"]
    train_loss = history["train_loss"]
    train_accuracy = history["train_accuracy"]
    test_loss = history["test_loss"]
    test_accuracy = history["test_accuracy"]

    plt.figure(figsize=(12, 12))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_loss, label="Training Loss", color="blue", marker=",")
    plt.plot(epochs, test_loss, label="Testing Loss", color="orange", marker=",")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_accuracy, label="Training Accuracy", color="green", marker=",")
    plt.plot(epochs, test_accuracy, label="Testing Accuracy", color="red", marker=",")
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.show()
    print(f"Curve Plot saved at {plot_path}")

def plot_confusion_matrix(model, x_valid, y_valid, labels, save_path):
    y_true = np.argmax(y_valid, axis=1)
    y_pred = np.argmax(model.predict(x_valid), axis=1)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    print(f"Confusion matrix saved at {save_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Error! Correct Command: python3 csimodel.py Dataset_folder_path")
    raw_data_folder = sys.argv[1]

    # Define the three configurations
    configs = [
        # ("all", ["waving", "twist", "standing", "squatting", "rubhand", "pushpull", "punching", "nopeople", "jump", "clap"]),
        ("macro", ["standing", "squatting", "jump", "nopeople"]),
        ("micro", ["waving", "twist", "rubhand", "pushpull", "punching", "clap"])
    ]

    for config_name, labels in configs:
        print(f"Running configuration: {config_name}")
        cfg = CSIModelConfig(labels=labels, win_len=1000, step=200, thrshd=0.6, downsample=2)
        numpy_tuple = cfg.preprocessing(raw_data_folder, save=True)
        if numpy_tuple is None:
            print("No valid data found. Exiting.")
            continue
        
        x_train, y_train, x_valid, y_valid = train_valid_split(numpy_tuple, train_portion=0.9, seed=379)
        
        model = cfg.build_model(n_unit_lstm=200, n_unit_atten=400)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy', 
            metrics=['accuracy'])
        model.summary()
        history = model.fit(
            x_train,
            y_train,
            batch_size=128, epochs=60,
            validation_data=(x_valid, y_valid),
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(f'best_atten_{config_name}.hdf5',
                                                    monitor='val_accuracy',
                                                    save_best_only=True,
                                                    save_weights_only=False)
                ])
        
        history_dict = {
            "epoch": list(range(1, 61)),
            "train_loss": history.history['loss'],
            "train_accuracy": history.history['accuracy'],
            "test_loss": history.history['val_loss'],
            "test_accuracy": history.history['val_accuracy']
        }
        with open(f"training_history_{config_name}.json", "w") as f:
            json.dump(history_dict, f)

        plot_training_history(f"training_history_{config_name}.json", f"training_plot_{config_name}.png")

        model = cfg.load_model(f'best_atten_{config_name}.hdf5')
        y_pred = model.predict(x_valid)

        plot_confusion_matrix(model, x_valid, y_valid, labels, f"confusion_matrix_{config_name}.png")

        print(confusion_matrix(np.argmax(y_valid, axis=1), np.argmax(y_pred, axis=1)))