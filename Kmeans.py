__authors__ = ['1636129, 1636546, 1638618']
__group__ = 'DL.12'

import numpy as np
import utils

class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """

        X = np.array(X, dtype=float)  # a)

        if X.ndim == 3 and X.shape[2] == 3:  # b) Si son 3 dimensions i si F x C x 3
            X = X.reshape(-1, 3)  # N x 3
        elif X.ndim > 2:    # c) Converteix les dimensions a 2 --> N x D.
            X = X.reshape(X.shape[0], -1)

        self.X = X

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """

        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'last' not in options:
            options['last'] = 'last'
        if 'random' not in options:
            options['random'] = 'random'
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):

        self.centroids = np.zeros((self.K, self.X.shape[1]))
        self.old_centroids = np.zeros((self.K, self.X.shape[1]))

        if self.options['km_init'].lower() == 'first':
            # Asignar la primera fila de self.X a la primera fila de centroids
            self.centroids[0] = self.X[0]

            # Inicializar contador
            contK = 0

            # Iterar sobre las filas de self.X
            for fila in self.X[1:]:
                # Verificar si la fila actual está en centroids
                esta_en_centroids = np.any(np.all(fila == self.centroids[:contK + 1], axis=1))

                if not esta_en_centroids:
                    # Si la fila no está en centroids, aumentar el contador contK y asignar la fila a centroids
                    contK += 1
                    self.centroids[contK] = fila

                # Verificar si se han alcanzado K filas únicas en centroids
                if contK == self.K - 1:
                    break

        elif self.options['random'].lower() == 'random':
            indices_aleatorios = np.random.choice(self.X.shape[0], size=self.K, replace=False)
            self.centroids = self.X[indices_aleatorios]
        elif self.options['last'].lower() == 'last':
            # Asignar la última fila de self.X a la primera fila de centroids
            self.centroids[0] = self.X[-1]

            # Inicializar contador
            contK = 0

            # Iterar sobre las filas de self.X en orden inverso
            for fila in self.X[-2::-1]:
                # Verificar si la fila actual está en centroids
                esta_en_centroids = np.any(np.all(fila == self.centroids[:contK + 1], axis=1))

                if not esta_en_centroids:
                    # Si la fila no está en centroids, aumentar el contador contK y asignar la fila a centroids
                    contK += 1
                    self.centroids[contK] = fila

                # Verificar si se han alcanzado K filas únicas en centroids
                if contK == self.K - 1:
                    break


    def get_labels(self):
        """        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        res = distance(self.X, self.centroids)  # Calcula la distància entre tots els punts y els centroids.
        self.labels = np.argmin(res, axis=1)  # Asigna cada punto al centroide más cercano.

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        self.old_centroids = np.copy(self.centroids)
        for fila in range(self.centroids.shape[0]):
            #de cada fila calculem la mitja de nomes els punts que siguin de la mateixa clase
            self.centroids[fila] = np.mean(self.X[self.labels == fila], axis=0)

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #comprobem si tenim els mateixos array
        if np.array_equal(self.old_centroids, self.centroids):
            return True
        else:
            return False

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """

        self._init_centroids()
        while not self.converges():
        # 1. Per a cada punt de la imatge, troba quin és el centroide més proper.
            self.get_labels()
        # 2. Calcula nous centroides utilitzant la funció get_centroids
            self.get_centroids()
        # 3. Augmenta en 1 el nombre d’iteracions
            self.num_iter += 1

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """
        wcd = 0
        N = len(self.X)
        for i in range(self.K):
            punts_cluster = self.X[self.labels == i]
            if len(punts_cluster) > 0:
                Cx = self.centroids[i]
                dist = np.linalg.norm(punts_cluster - Cx, axis=1)  # euclidiana
                wcd += np.sum(dist ** 2)
        wcd = wcd / N
        return wcd

    def interClassDistance(self):
        dist = 0
        N = len(self.X)
        for i in range(len(self.centroids)):
            points_cluster = self.X[self.labels == i]
            if len(points_cluster) > 0:
                other_centroids = self.centroids[np.arange(len(self.centroids)) != i]
                for c in other_centroids:
                    dist += np.sum((points_cluster - c) ** 2)
        inter_class_dist = dist / N
        return inter_class_dist
    def coeficientfisher(self):
        return self.withinClassDistance() / self.interClassDistance()

    def find_bestK(self, max_K):
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """

        WCD_list = []
        dec_list = []
        optim = 0
        for K in range(2, max_K + 1):
            self.K = K
            self.fit()
            WCD_list.append(self.withinClassDistance())
            if len(WCD_list) > 1:
                dec_list.append(100 * (WCD_list[-1]) / WCD_list[-2])
                if len(dec_list) > 1:
                    if (100 - dec_list[-1]) < 20:
                        self.K = K-1
                        optim = 1
                        break
        if optim == 0:
            self.K = max_K

    def find_bestK_update(self, max_K, heuristica, llindar):
        self.fit()  # Fem el fit.
        if heuristica == "Fisher":
            auxiliar = self.coeficientfisher()  # Fisher
        elif heuristica == "Inter":
            auxiliar = self.interClassDistance()    # Inter-class
        elif heuristica == "Intra":
            auxiliar = self.withinClassDistance()   # Intra-class
        else:
            return -1

        x = False
        self.K += 1
        while self.K <= max_K and not x:
            self.fit()
            if heuristica == "Fisher":
                w = self.coeficientfisher()  # Fisher
                percentatgeW = (w / auxiliar) * 100
            elif heuristica == "Inter":
                w = self.interClassDistance()  # Inter-class
                percentatgeW = (auxiliar / w) * 100
            else:
                w = self.withinClassDistance()  # Intra-class
                percentatgeW = (w / auxiliar) * 100

            if (100 - percentatgeW) < llindar:
                x = True
                break
            else:
                auxiliar = w
                self.K += 1  # Provar un altre valor de K

        if not x:
            self.K = max_K

        self.fit()

def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)
    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """

    dist = np.empty((X.shape[0], C.shape[0]))  # Creem matriu dist amb dimensions N x K.

    for i in range(len(C)):  # Per a cada centre d'inercia (o Centroid).
        dist[:, i] = np.sqrt(np.sum((X - C[i]) ** 2, axis=1))  # Per a cada fila de dist (:) en columna de centroid,
        # calculem la distància euclidiana entre cada punt de la imatge amb cada centroide. Axis indica que les
        # operacions es fan en files.

    return dist

def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)
    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    centroids = np.array(centroids)
    color_prob = utils.get_color_prob(centroids)  # retorn prob colors
    color_max = np.argmax(color_prob, axis=1)  # max cada fila
    labels = [utils.colors[i] for i in color_max]  # assignar etiqueta

    return (labels)