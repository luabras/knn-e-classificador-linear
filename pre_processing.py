import cv2

class SimplePreprocessor:

    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # recebe o tamanho que a imagem sera redimensionada 
        # e o tipo de interpolacao a ser utilizada
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # redimensiona a imagem para o tamanho especificado ignorando o aspect ratio
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)