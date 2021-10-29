from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from pre_processing import SimplePreprocessor
from load_dataset import DatasetLoader

from imutils import paths

path = "dataset/animals"
k = 3
jobs = -1

print("[INFO] carregando as imagens...")
image_paths = list(paths.list_images(path))

# inicializando o pre-processador e o dataset loader
preprocessador = SimplePreprocessor(32, 32)
dataset = DatasetLoader(preprocessors=preprocessador)

# carregando o dataset
(data, labels) = dataset.load(image_paths, verbose=500)

# o array data tem o shape (3000, 32, 32, 3), o que indica que 
# existem 3000 imagens com 32x32 pixels e 3 canais de cores
# para o k-NN, as imagens precisam ser reduzidas de uma 
# representacao 3D para uma unica lista de intensidades de pixels

# o novo array data tem o shape (3000, 3072), que eh uma lista com
# 3000 entradas e 3072 colunas (32x32x3)
data = data.reshape((data.shape[0], 3072))

# mostrando informacoes de consumo de memoria das imagens
print("[INFO] matriz de features: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))

# codificando as labels como valores inteiros
le = LabelEncoder()
labels = le.fit_transform(labels)

# dividindo os dados entre treinamento e teste
print("[INFO] separando os dados de treino e teste...")
(treino_x, teste_x, treino_y, teste_y) = train_test_split(data, labels, test_size=0.25, random_state=42)

print("[INFO] avaliando o classificador k-NN...")

modelo = KNeighborsClassifier(n_neighbors=k, n_jobs=jobs)
modelo.fit(treino_x, treino_y)

print(classification_report(teste_y, modelo.predict(teste_x), target_names=le.classes_))