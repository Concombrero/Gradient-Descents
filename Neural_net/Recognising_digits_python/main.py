import mnist_loader
import network
import matplotlib.pyplot as plt
import numpy as np

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 25, 20, 10])
net.SGD(15,training_data,10,3.0, test_data)
nb = np.reshape(training_data[2][0],(28,28))

def afficher_graphique(net, hauteur, largeur, test_data):
    f, graphique = plt.subplots(hauteur,largeur)
    for i in range(hauteur):
        for j in range(largeur):
            pred  = np.argmax(net.feedforward(test_data[i*11+j][0]))
            graphique[i][j].set_title('Prediction :'+ str(pred))
            graphique[i][j].imshow(np.reshape(test_data[i*11+j][0],(28,28)))
    plt.show()
    
afficher_graphique(net,7,7,test_data)