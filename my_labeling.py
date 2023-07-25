__authors__ = ['']
__group__ = ''

from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_retrieval, visualize_k_means, \
    Plot3DCloud
import Kmeans as km
from Kmeans import *
import KNN as knn
from KNN import *
import time
import matplotlib.pyplot as plt
import math


# --------------------------------------------- FUNCIONS QUALITATIVES -------------------------------------------------
def retrieval_by_color(images, etiq, color):
    colorIm = []
    for i, valores in enumerate(etiq):
        if any(valor in color for valor in valores):
            colorIm.append(images[i])
    return colorIm


def retrieval_by_shape(images, etiq, forma):
    formaIm = []
    for i, valor in enumerate(etiq):
        if valor in forma:
            formaIm.append(images[i])
    return formaIm


def retrieval_combined(images, etiqC, etiqF, forma, color, infoF, infoC):
    info = []
    formaColorIm = []
    ok = []
    for i, (valoresC, valoresF) in enumerate(zip(etiqC, etiqF)):
        if (valoresF in forma) and any(valor in color for valor in valoresC):
            formaColorIm.append(images[i])
            valoresC[:] = list(set(valoresC))
            if infoF[i] == valoresF and any(valorC in color for valorC in infoC[i]):
                ok.append(1)
            else:
                ok.append(0)
            info.append([infoC[i], infoF[i]])
    formaColorIm = np.array(formaColorIm)
    return formaColorIm, ok, info


# --------------------------------------------- FUNCIONS  QUANTITATIVES--------------------------------------------------
def Kmean_statistics(km, Kmax):
    WCDs = []
    num_iterations = []
    total_times = []
    for K_iteration in range(2, Kmax + 1):
        km.K = K_iteration
        start_time = time.time()
        km.fit()
        end_time = time.time()
        WCDs.append(km.withinClassDistance())
        num_iterations.append(km.num_iter)
        total_times.append(end_time - start_time)

    return WCDs, num_iterations, total_times


def get_shape_accuracy(knn_labels, ground_t):
    correct_shape = 0
    for label, gtruth in zip(knn_labels, ground_t):
        if label == gtruth:
            correct_shape += 1

    accuracy = (correct_shape / len(knn_labels)) * 100
    return accuracy


def get_color_accuracy(kmeans_labels, ground_t):
    correct_colors = 0
    total_colors = 0
    for km_label, gt_label in zip(kmeans_labels, ground_t):
        union_labels = set(km_label).union(set(gt_label))  # Unió entre kmlabels y groundtruth
        intersection_labels = set(km_label).intersection(set(gt_label))  # Intersecció entre kmlabels i groundtruth
        correct_colors += len(intersection_labels) / len(union_labels)  # Divisió entre intersecció i unió
        total_colors += 1
    percent_accuracy = (correct_colors / total_colors) * 100

    return percent_accuracy


# --------------------------------------------- EXPERIMENTS ------------------------------------------------------------

if __name__ == '__main__':
    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here

    # -------------------------- EXPERIMENTS D'ANÀLISI QUALITATIU --------------------
    '''
    # 1r EXPERIMENT
    listaColor = []
    listaForma = []
    imatgesColors = []

    cantidad = 850
    imagenes = test_imgs

    for i in range(0, cantidad):
        km = KMeans(imagenes[i], 4)
        km.find_bestK_update(10, 'Inter', 20)  # <-- Escollir Heuristica 
        listaColor.append(get_colors(km.centroids))

    knn = KNN(imagenes[:cantidad], test_class_labels[:cantidad])
    listaForma = (knn.predict(imagenes[:cantidad], 2))

    clothing = ['Flip Flops']  # <-- Insertar ropa a mostrar
    color = ['Black']        # <-- Insertar color a mostrar
    title = f'Busqueda: {color[0]} {clothing[0]}'
    imatgesCombinades, ok, info = retrieval_combined(imagenes[:cantidad], listaColor, listaForma, clothing, color,test_class_labels,test_color_labels)
    visualize_retrieval(imatgesCombinades, 10, info, ok, title, None)
    '''

    # -------------------------- EXPERIMENTS D'ANÀLISI QUANTITATIU --------------------
    '''
    # 1r EXPERIMENT
    for i in range(0, 150):
        km = KMeans(imgs[i])

    Kmax = 10
    WCDs, num_iterations, total_times = Kmean_statistics(km, Kmax)
    plt.plot(range(2, Kmax + 1), WCDs)
    plt.xlabel('K')
    plt.ylabel('WCD')
    plt.show()
    plt.plot(range(2, Kmax + 1), num_iterations)
    plt.xlabel('K')
    plt.ylabel('Iterations')
    plt.show()
    plt.plot(range(2, Kmax + 1), total_times)
    plt.xlabel('K')
    plt.ylabel('Time (seconds)')
    plt.show()
    '''

    '''
    # 2n EXPERIMENT
    accuracy_color_results = []
    K_values = range(2, 10)

    for k_actual in K_values:
        listaColorP = []
        for i in range(0, 150):
            km = KMeans(imgs[i], k_actual)
            km.K = k_actual
            km.fit()
            listaColorP.append(get_colors(km.centroids))

        percent_Color = get_color_accuracy(listaColorP, color_labels[:150])
        accuracy_color_results.append(percent_Color)

    plt.plot(K_values, accuracy_color_results)
    plt.xlabel('K in Kmeans')
    plt.ylabel('Accuracy')
    plt.yticks(range(0, 101, 20))
    plt.show()
    '''

    '''
    # 3r EXPERIMENT
    accuracy_shape_results = []
    listaFormaNew = []
    K_values = range(2, 11)

    for k_actual in K_values:
        knn = KNN(test_imgs[:500], test_class_labels[:500])
        listaFormaNew = (knn.predict(test_imgs[:500], k_actual))
        percentOfShape = get_shape_accuracy(listaFormaNew, test_class_labels[:500])
        accuracy_shape_results.append(percentOfShape)

    plt.plot(K_values, accuracy_shape_results)
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.yticks(range(0, 101, 20))
    plt.show()
    '''

    # -------------------------- EXPERIMENTS SOBRE ELS MÈTODES DE CLASSIFICACIÓ --------------------

    '''
    # 1r EXPERIMENT
    for i in range(0, 150):
        km = KMeans(imgs[i], 1, options={'last': 'last'}) # <-- Indicar opciones (km_init : first, random : random, last : last)

    Kmax = 10
    WCDs, num_iterations, total_times = Kmean_statistics(km, Kmax)
    plt.plot(range(2, Kmax + 1), total_times)
    plt.xlabel('K')
    plt.ylabel('Time (seconds)')
    plt.show()
    '''

    '''
    # 2n Experiment 

    imagenes = test_imgs
    llistaColor = []
    for i in range(0, 150):
        km = KMeans(imgs[i], 2)
        km.find_bestK_update(10, 'Intra', 30) #modifiquem el segon valor segons l'heurística d'estudi
        km.fit()
        llistaColor.append(get_colors(km.centroids))
        Plot3DCloud(km)
        visualize_k_means(km, [80, 60, 3])
    '''
    '''
    # 3r Experiment

    llistaColor = []
    start_time = time.time()

    for i in range(0, 150):
        km = KMeans(imgs[i], 2)
        km.find_bestK_update(10, 'Fisher', 30) # modifiquem el tercer valor segons el llindar a estudiar
        llistaColor.append(get_colors(km.centroids))

    percent_Color = get_color_accuracy(llistaColor, color_labels[:150])
    Plot3DCloud(km) #comprovem qualitat última imatge generada
    visualize_k_means(km, [80, 60, 3])
    end_time = time.time()
    print(end_time - start_time, "seconds") #temps en generar les imatges
    print(percent_Color, "%")
    '''
