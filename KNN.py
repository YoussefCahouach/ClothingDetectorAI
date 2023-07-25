__authors__ = ['1636129, 1636546, 1638618']
__group__ = 'DL.12'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        P = train_data.shape[0]
        self.train_data = train_data.reshape(P, -1)
        self.train_data = self.train_data.astype(float)
        return self.train_data

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        # 1. Canvia de dimensions de les imatges de la mateixa manera que ho hem fet amb el conjunt d’entrenament.
        N = test_data.shape[0]
        test_data = test_data.reshape(N, -1)
        test_data = test_data.astype(float)

        # 2. Calcula la distància entre les característiques del test_data amb les del train_data.
        dis = cdist(test_data, self.train_data)

        # 3. Guarda a la variable de classe self.neighbors les K etiquetes de les imatges més pròximes per a cada mostra del test.

        index = np.argsort(dis, axis=1)[:, :k]
        self.neighbors = self.labels[index]

        return self.neighbors

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """
        most_voted = []
        for row in self.neighbors:
            unique_labels, rep_labels = np.unique(row, return_counts=True)
            if len(unique_labels) == len(row):  # Si todos son diferentes, cogemos el primero.
                most_voted.append(row[0])
            else:
                num_max = np.max(rep_labels)  # Guardamos el numero maximo de veces que aparece un label.
                for label in row:
                    if list(row).count(label) == num_max:  # Si ese label se repite igual al maximo, lo guardamos.
                        most_voted.append(label)
                        break

        return np.array(most_voted)

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the class 2nd the  % of votes it got
        """
        self.get_k_neighbours(test_data, k)  # Busca els seus veïns.
        return self.get_class()  # Retorna la classe més representativa.