import pickle
import string
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras import layers

import argparse
from models import inference

parser = argparse.ArgumentParser()
parser.add_argument('--mode')
parser.add_argument('--dataset')
parser.add_argument('--model')
parser.add_argument('--input')
parser.add_argument('--output')
args = parser.parse_args()
if args.mode == 'train':
    # нейросеть
    data = pd.read_csv(args.dataset)

    y_train = data['label']
    y_train = to_categorical(y_train.values, num_classes=10)  # вектор класса -> матрицу классов. Всего 10 классов
    X_train = data.drop(labels=['label'], axis=1)  # только пиксели
    X_train = X_train.to_numpy()  # CNN требует массив numpy
    X_train = X_train / 255  # нормализация данных

    # представление изображения 28x28 из 784
    X_train = X_train.reshape(-1, 28, 28, 1)
    # инициализация модели
    model = Sequential()
    # слой свертки с активацией relu
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))

    model.compile('RMSprop', 'CategoricalCrossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train,
              batch_size=64,
              epochs=3, verbose=1,
              validation_split=0.2)

    with open(args.model + "\model_neural_" + args.dataset + ".pkl", "wb") as q:
        pickle.dump(model, q)

    # дерево решений
    data = pd.read_csv(args.dataset)
    y_train = data['label']
    X_train = data.drop(labels=['label'], axis=1)
    # Дерево решений
    ed = DecisionTreeClassifier()
    model = ed.fit(X_train, y_train)
    with open(args.model + "\model_tree_" + args.dataset + ".pkl", "wb") as q:
        pickle.dump(model, q)

elif args.mode == 'inference':
    inference(args.model, args.input, args.output)

# fashion
# python mnist.py --mode train --dataset fashion-mnist_train.csv --model "C:\Users\Oleg\PycharmProjects\pythonProject"
# python mnist.py --mode inference --model model_neural_fashion-mnist_train.csv.pkl --input fashion-mnist_test.csv --output test_output.csv
# python mnist.py --mode inference --model model_tree_fashion-mnist_train.csv.pkl --input fashion-mnist_test.csv --output test_output.csv
# python evaluate.py --ground-truth "fashion-mnist_test.csv" --predictions "test_output.csv"

# mnist
# python mnist.py --mode train --dataset mnist_train.csv --model "C:\Users\Oleg\PycharmProjects\pythonProject"
# python mnist.py --mode inference --model model_neural_mnist_train.csv.pkl --input mnist_test.csv --output test_output.csv
# python mnist.py --mode inference --model model_tree_mnist_train.csv.pkl --input mnist_test.csv --output test_output.csv
# python evaluate.py --ground-truth "mnist_test.csv" --predictions "test_output.csv"
