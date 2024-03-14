import numpy as np

class Network:
    def __init__(self, sizes) -> None:
        self.nb_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(i,1) for i in sizes[1:]]
        self.weights = [np.random.randn(y,x) for (x,y) in zip(sizes[:-1], sizes[1:])]
        
    def feedforward(self, a):
        c = a
        for i in range(self.nb_layers-1):
            c = sigmoid(np.dot(self.weights[i], c) + self.biases[i])
        return c
    
    def SGD(self, epochs, train_set, mini_batch_size, learning_rate, test_set = None):
        
        if test_set:
            n_test = len(test_set)
            print("before : {0} / {1}".format(self.evaluate(test_set), n_test))
        n = len(train_set)
        for i in range(epochs):
            np.random.shuffle(train_set)
            mini_batches = [train_set[k:k+mini_batch_size] for k in range(0,n-mini_batch_size+1,mini_batch_size)]
            for batch in mini_batches:
                self.update_mini_batch(batch, learning_rate)
            if test_set:
                p = round((self.evaluate(test_set)/ n_test) * 100,1)
                print("epoque {0} : {1} %".format(i, p), " de réussite sur le test set")
            else:
                print("epoch {0} complete".format(i))

              
    def update_mini_batch(self, batch, learning_rate):
        """fonction qui modifie les poids et les biais grâce à la descente de gradient

        Args:
            batch (np.array): lot d'exemples
            learning_rate (int): facteur
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb + dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw + dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
        self.biases = [b - nb*(learning_rate/len(batch)) for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - nw*(learning_rate/len(batch)) for w, nw in zip(self.weights, nabla_w)]
        
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.nb_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def evaluate(self, test_set):
        """Evalue la performance de l'ia

        Args:
            test_set (_type_): test set
        """
        test_results = [(np.argmax(self.feedforward(x)), y) for (x,y) in test_set]
        return sum(int(x==y) for (x,y) in test_results)
        
    
        
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
            
        
    
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1.0 - sigmoid(z))