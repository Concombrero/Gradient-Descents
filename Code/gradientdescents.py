import numpy as np
   
class ClassicBatchGD:
    def __init__(self, learning_rate=0.001):
        """
        Constructeur pour la classe GradientDescent.

        Paramètres
        ----------
        learning_rate : float
            Taux d'apprentissage pour l'optimiseur.

        Renvoie
        -------
        Aucun.
        """
        self.learning_rate = learning_rate

    def update_params(self, params, grads):
        """
        Met à jour les paramètres du modèle à l'aide de la descente de gradient classique.

        Paramètres
        ----------
        params : dict
            Dictionnaire contenant les paramètres du modèle.
        grads : dict
            Dictionnaire contenant les dégradés pour chaque paramètre.

        Renvoie
        -------
        updated_params : dict
            Dictionnaire contenant les paramètres du modèle mis à jour.
        """
        updated_params = {}

        for key in params.keys():
            updated_params[key] = params[key] - self.learning_rate * grads[key]

        return updated_params


class  AdamGD : 
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Constructor for the AdamOptimizer class.

        Parameters
        ----------
        learning_rate : float
            Learning rate for the optimizer.
        beta1 : float
            Exponential decay rate for the first moment estimates.
        beta2 : float
            Exponential decay rate for the second moment estimates.
        epsilon : float
            Small value to prevent division by zero.

        Returns
        -------
        None.
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def initialize_moments(self, params):
        """
        Initializes the first and second moment estimates.

        Parameters
        ----------
        params : dict
            Dictionary containing the model parameters.
        
        Returns
        -------
        None.
        """
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def update_params(self, params, grads):
        """
        Updates the model parameters using the Adam optimizer.

        Parameters
        ----------
        params : dict
            Dictionary containing the model parameters.
        grads : dict
            Dictionary containing the gradients for each parameter.
        
        Returns
        -------
        updated_params : dict
            Dictionary containing the updated model parameters.
        """
        if self.m is None or self.v is None:
            self.initialize_moments(params)

        self.t += 1
        updated_params = {}

        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * np.square(grads[key])

            m_corrected = self.m[key] / (1 - self.beta1 ** self.t)
            v_corrected = self.v[key] / (1 - self.beta2 ** self.t)

            updated_params[key] = params[key] - self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)

        return updated_params
 
 
class HeavyBallGD:
    def __init__(self, learning_rate=0.001, momentum=0.9):
        """
        Constructeur pour la classe HeavyBallGD.

        Paramètres
        ----------
        learning_rate : float
            Taux d'apprentissage pour l'optimiseur.
        momentum : float
            Coefficient de momentum pour la descente de gradient HeavyBall.

        Renvoie
        -------
        Aucun.
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = None

    def initialize_moment(self, params):
        """
        Initialise le moment pour la descente de gradient HeavyBall.

        Paramètres
        ----------
        params : dict
            Dictionnaire contenant les paramètres du modèle.

        Renvoie
        -------
        Aucun.
        """
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def update_params(self, params, grads):
        """
        Met à jour les paramètres du modèle à l'aide de la descente de gradient HeavyBall.

        Paramètres
        ----------
        params : dict
            Dictionnaire contenant les paramètres du modèle.
        grads : dict
            Dictionnaire contenant les dégradés pour chaque paramètre.

        Renvoie
        -------
        updated_params : dict
            Dictionnaire contenant les paramètres du modèle mis à jour.
        """
        if self.v is None:
            self.initialize_moment(params)

        updated_params = {}

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.learning_rate * grads[key]
            updated_params[key] = params[key] + self.v[key]

        return updated_params


class NesterovGD:
    def __init__(self, learning_rate=0.001, momentum=0.9):
        """
        Constructeur pour la classe NesterovGD.

        Paramètres
        ----------
        learning_rate : float
            Taux d'apprentissage pour l'optimiseur.
        momentum : float
            Coefficient de momentum pour la descente de gradient Nesterov.

        Renvoie
        -------
        Aucun.
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = None

    def initialize_moment(self, params):
        """
        Initialise le moment pour la descente de gradient Nesterov.

        Paramètres
        ----------
        params : dict
            Dictionnaire contenant les paramètres du modèle.

        Renvoie
        -------
        Aucun.
        """
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def update_params(self, params, grads):
        """
        Met à jour les paramètres du modèle à l'aide de la descente de gradient Nesterov.

        Paramètres
        ----------
        params : dict
            Dictionnaire contenant les paramètres du modèle.
        grads : dict
            Dictionnaire contenant les dégradés pour chaque paramètre.

        Renvoie
        -------
        updated_params : dict
            Dictionnaire contenant les paramètres du modèle mis à jour.
        """
        if self.v is None:
            self.initialize_moment(params)

        updated_params = {}

        for key in params.keys():
            v_prev = self.v[key]
            self.v[key] = self.momentum * self.v[key] - self.learning_rate * grads[key]
            updated_params[key] = params[key] - self.learning_rate * ((1 + self.momentum) * self.v[key] - self.momentum * v_prev)

        return updated_params
  
  
class RMSpropGD:
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        """
        Constructeur pour la classe RMSpropGD.

        Paramètres
        ----------
        learning_rate : float
            Taux d'apprentissage pour l'optimiseur.
        beta : float
            Facteur de décroissance exponentielle pour l'estimation du deuxième moment.
        epsilon : float
            Petite valeur pour empêcher la division par zéro.

        Renvoie
        -------
        Aucun.
        """
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.v = None

    def initialize_moment(self, params):
        """
        Initialise le moment pour RMSprop.

        Paramètres
        ----------
        params : dict
            Dictionnaire contenant les paramètres du modèle.

        Renvoie
        -------
        Aucun.
        """
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def update_params(self, params, grads):
        """
        Met à jour les paramètres du modèle à l'aide de RMSprop.

        Paramètres
        ----------
        params : dict
            Dictionnaire contenant les paramètres du modèle.
        grads : dict
            Dictionnaire contenant les dégradés pour chaque paramètre.

        Renvoie
        -------
        updated_params : dict
            Dictionnaire contenant les paramètres du modèle mis à jour.
        """
        if self.v is None:
            self.initialize_moment(params)

        updated_params = {}

        for key in params.keys():
            self.v[key] = self.beta * self.v[key] + (1 - self.beta) * np.square(grads[key])
            updated_params[key] = params[key] - self.learning_rate * grads[key] / (np.sqrt(self.v[key]) + self.epsilon)

        return updated_params


class AdagradGD:
    def __init__(self, learning_rate=0.001, epsilon=1e-8):
        """
        Constructeur pour la classe AdagradGD.

        Paramètres
        ----------
        learning_rate : float
            Taux d'apprentissage pour l'optimiseur.
        epsilon : float
            Petite valeur pour empêcher la division par zéro.

        Renvoie
        -------
        Aucun.
        """
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.v = None

    def initialize_moment(self, params):
        """
        Initialise le moment pour Adagrad.

        Paramètres
        ----------
        params : dict
            Dictionnaire contenant les paramètres du modèle.

        Renvoie
        -------
        Aucun.
        """
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def update_params(self, params, grads):
        """
        Met à jour les paramètres du modèle à l'aide de Adagrad.

        Paramètres
        ----------
        params : dict
            Dictionnaire contenant les paramètres du modèle.
        grads : dict
            Dictionnaire contenant les dégradés pour chaque paramètre.

        Renvoie
        -------
        updated_params : dict
            Dictionnaire contenant les paramètres du modèle mis à jour.
        """
        if self.v is None:
            self.initialize_moment(params)

        updated_params = {}

        for key in params.keys():
            self.v[key] += np.square(grads[key])
            updated_params[key] = params[key] - self.learning_rate * grads[key] / (np.sqrt(self.v[key]) + self.epsilon)

        return updated_params


class AdadeltaGD:
    def __init__(self, rho=0.95, epsilon=1e-6):
        """
        Constructeur pour la classe AdadeltaGD.

        Paramètres
        ----------
        rho : float
            Coefficient de décroissance pour l'estimation du deuxième moment.
        epsilon : float
            Petite valeur pour empêcher la division par zéro.

        Renvoie
        -------
        Aucun.
        """
        self.rho = rho
        self.epsilon = epsilon
        self.v = None
        self.s = None

    def initialize_moment(self, params):
        """
        Initialise les moments pour Adadelta.

        Paramètres
        ----------
        params : dict
            Dictionnaire contenant les paramètres du modèle.

        Renvoie
        -------
        Aucun.
        """
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.s = {k: np.zeros_like(v) for k, v in params.items()}

    def update_params(self, params, grads):
        """
        Met à jour les paramètres du modèle à l'aide de Adadelta.

        Paramètres
        ----------
        params : dict
            Dictionnaire contenant les paramètres du modèle.
        grads : dict
            Dictionnaire contenant les dégradés pour chaque paramètre.

        Renvoie
        -------
        updated_params : dict
            Dictionnaire contenant les paramètres du modèle mis à jour.
        """
        if self.v is None or self.s is None:
            self.initialize_moment(params)

        updated_params = {}

        for key in params.keys():
            self.v[key] = self.rho * self.v[key] + (1 - self.rho) * np.square(grads[key])
            delta_params = -np.sqrt(self.s[key] + self.epsilon) * grads[key] / np.sqrt(self.v[key] + self.epsilon)
            updated_params[key] = params[key] + delta_params
            self.s[key] = self.rho * self.s[key] + (1 - self.rho) * np.square(delta_params)

        return updated_params  
  
  
    
class ModelTrainer:
    def __init__(self, model, optimizer, n_epochs):
        """
        Constructor for the ModelTrainer class.

        Parameters
        ----------
        model : object
            Model to be trained.
        optimizer : object
            Optimizer to be used for training.
        n_epochs : int
            Number of training epochs.

        Returns
        -------
        None.
        """
        self.model = model
        self.optimizer = optimizer
        self.n_epochs = n_epochs

    def compute_gradients(self, X, y):
        """
        Computes the gradients of the mean squared error loss function
        with respect to the model parameters.

        Parameters
        ----------
        X : numpy array
            Input data.
        y : numpy array
            Target variable.
        
        Returns
        -------
        dict
            Dictionary containing the gradients for each parameter.
        """
        predictions = self.model.predict(X)
        errors = predictions - y
        dW = 2 * np.dot(X.T, errors) / len(y)
        db = 2 * np.mean(errors)
        return {'weights': dW, 'bias': db}

    def train(self, X, y, verbose=False):
        """
        Runs the training loop, updating the model parameters and optionally printing the loss.

        Parameters
        ----------
        X : numpy array
            Input data.
        y : numpy array
            Target variable.
        
        Returns
        -------
        None.
        """
        for epoch in range(self.n_epochs):
            grads = self.compute_gradients(X, y)
            params = {'weights': self.model.weights, 'bias': self.model.bias}
            updated_params = self.optimizer.update_params(params, grads)

            self.model.weights = updated_params['weights']
            self.model.bias = updated_params['bias']
    def  __init__ ( self, learning_rate= 0.001 , beta1= 0.9 , beta2= 0.999 , epsilon= 1e-8 ): 
        """ 
        Constructeur pour la classe AdamGD. 

        Paramètres 
        - --------- 
        learning_rate : float 
            Taux d'apprentissage pour l'optimiseur. 
        beta1 : float 
            Taux de décroissance exponentielle pour les premières estimations de moment. 
        beta2 : float 
            Taux de décroissance exponentielle pour les deuxièmes estimations de moment. 
        epsilon : float 
            Petite valeur pour empêcher la division par zéro. 

        Renvoie 
        ------- 
        Aucun. 
        """
        self.learning_rate = learning_rate 
        self.beta1 = beta1 
        self.beta2 = beta2 
        self.epsilon = epsilon 
        self.m = None
        self.v = None
        self.t = 0 

    def  initialize_moments ( self, params ): 
        """ 
        Initialise les première et deuxième estimations de moment. 

        Paramètres 
        ---------- 
        params : dict 
            Dictionnaire contenant les paramètres du modèle. 
        
        Renvoie 
        ------- 
        Aucun. 
        """
        self.m = {k : np.zeros_like(v) for k, v in params.items()} 
        self.v = {k : np.zeros_like(v) for k, v in params.items() } 

    def  update_params ( self, params, grads ): 
        """ 
        Met à jour les paramètres du modèle à l'aide de la GD Adam. 

        Paramètres 
        ---------- 
        params : dict 
            Dictionnaire contenant les paramètres du modèle. 
        grads : dict 
            Dictionnaire contenant les dégradés pour chaque paramètre. 
        
        Renvoie 
        ------- 
        update_params : dict 
            Dictionnaire contenant les paramètres du modèle mis à jour. 
        """ 
        if self.m is None or self.v is None : 
            self.initialize_moments(params) 

        self.t += 1
        update_params = {} 

        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * np.square(grads[key])

            m_corrected = self.m[key] / (1 - self.beta1 ** self.t)
            v_corrected = self.v[key] / (1 - self.beta2 ** self.t)

            updated_params[key] = params[key] - self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)

        return updated_params