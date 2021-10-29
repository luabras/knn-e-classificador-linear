# import the necessary packages
import numpy as np
import cv2
import os

class DatasetLoader:

	def __init__(self, preprocessors=None):

		# guardando os preprocessadores numa lista
		self.preprocessors = []
		self.preprocessors.append(preprocessors)

	def load(self, image_path, verbose=-1):

		# inicializando as listas dos dados e das labels
		data = []
		labels = []

		for (i, image_path) in enumerate(image_path):

			# salvando as labels de acordo com a pasta de cada imagem
			# /path/{class}/{image}.jpg
			image = cv2.imread(image_path)
			label = image_path.split(os.path.sep)[-2]

			# se existir algum preprocessador, aplicar ele em cada uma das imagens
			if self.preprocessors is not None:

				for p in self.preprocessors:
					image = p.preprocess(image)

			# adicionando as imagens e as labels nas listas
			data.append(image)
			labels.append(label)

			if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
				print("[INFO] processado {}/{}".format(i + 1, len(image_path)))

		# retornando uma tupla com as imagens e suas labels
		return (np.array(data), np.array(labels))