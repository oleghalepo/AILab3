import pickle
import string
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics

def inference(model_path, test, output):
    with open(model_path, "rb") as q : model = pickle.load(q)
    data = pd.read_csv(test)
    data = data.drop("label", axis = 1)
    if model_path.startswith('model_neural'):
        data = data.to_numpy()
        data = data / 255
        data = data.reshape(-1, 28, 28, 1)

        prediction1 = model.predict(data)
        prediction2 = np.argmax(prediction1, axis = 1)

        res_s = pd.Series(prediction2, name = "Label")
        res_s.to_csv(output, index = False)
    else:
        res_s = pd.Series(model.predict(data), name = "Label")
        res_s.to_csv(output, index = False)

def evaluate(truth, output):
    data_t = pd.read_csv(truth)
    data_o = pd.read_csv(output)
    elements = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    confMatrux = metrics.confusion_matrix(data_t['label'], data_o, labels = elements)
    sb.heatmap(confMatrux, annot = True, fmt = ".0f", xticklabels = elements, yticklabels = elements)
    plt.title("Accuracy:" +  str(metrics.accuracy_score(data_t['label'], data_o)))
    plt.show()