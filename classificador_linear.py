from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from imutils import paths

import numpy as np

import imutils
import cv2
import os

path = "dataset/animals"

print("[INFO] carregando as imagens...")
image_paths = list(paths.list_images(path))

data = []
labels = []

def extrair_histograma_cores(image, bins=(8, 8, 8)):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])

    if imutils.is_cv2():
        hist = cv2.normalize(hist)

    else:
        cv2.normalize(hist, hist)

    return hist.flatten()

for (i, image_path) in enumerate(image_paths):

    image = cv2.imread(image_path)
    label = image_path.split(os.path.sep)[-2]

    hist = extrair_histograma_cores(image)
    data.append(hist)
    labels.append(label)

    if i > 0 and i % 1000 == 0:
        print("[INFO] processado {}/{}".format(i, len(image_paths)))

le = LabelEncoder()
labels = le.fit_transform(labels)

print("[INFO] separando os dados de treino e teste...")
(dados_treino, dados_teste, labels_treino, labels_teste) = train_test_split(np.array(data), labels, test_size=0.25, random_state=42)

print("[INFO] treinando o classificador Linear SVM...")
model = LinearSVC()
model.fit(dados_treino, labels_treino)

print("[INFO] avaliando o classificador Linear SVM...")
predicoes = model.predict(dados_teste)
print(classification_report(labels_teste, predicoes, target_names=le.classes_))