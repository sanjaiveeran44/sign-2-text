import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

DATA_PATH = os.path.join('MP_Data')

def load_data(actions, num_sequences=30, sequence_length=30):
    sequences, labels = [], []
    label_map = {label: num for num, label in enumerate(actions)}

    for action in actions:
        for seq in range(num_sequences):
            window = []
            for frame in range(sequence_length):
                path = os.path.join(DATA_PATH, action, str(seq), f"{frame}.npy")
                res = np.load(path)
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    return np.array(sequences), to_categorical(labels).astype(int)

def train_model(actions, num_sequences=30, sequence_length=30, model_path='model/sign_language_model.h5'):
    X, y = load_data(actions, num_sequences, sequence_length)

    # Add channel dimension
    X = np.expand_dims(X, axis=-1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    model = Sequential()
    model.add(TimeDistributed(Conv1D(64, 3, activation='relu', padding='same'), input_shape=(sequence_length, X.shape[2], 1)))
    model.add(TimeDistributed(MaxPooling1D(2)))
    model.add(TimeDistributed(Conv1D(128, 3, activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling1D(2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(128))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(actions), activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=5, factor=0.5)

    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), callbacks=[checkpoint, early_stop, reduce_lr])

    print(f"✅ Training complete. Model saved at: {model_path}")
