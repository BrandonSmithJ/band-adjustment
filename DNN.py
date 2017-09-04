import tensorflow as tf 
import numpy as np
import tqdm
import os 


def weight_variable(name, shape):
    init = tf.truncated_normal_initializer(stddev=np.sqrt(2. / shape[0]))
    var  = tf.get_variable(name+'_W', shape, initializer=init)
    return var

def bias_variable(name, shape):
    init = tf.truncated_normal_initializer(stddev=0)
    var  = tf.get_variable(name+'_b', shape, initializer=init)
    return var


class Network(object):

    def __init__(self, in_size, out_size, name, hidden_sizes, 
                    hidden_activation = tf.nn.relu,
                    output_activation = tf.nn.relu,
                    dropout = None):

        self.in_size  = in_size
        self.out_size = out_size
        self.dropout  = dropout
        self.weights  = []
        self.biases   = []

        self.hidden_act = hidden_activation
        self.output_act = output_activation
        self.name = name 

        name  = name + '_layer%s'
        prior = in_size
        for i, hs in enumerate(hidden_sizes):
            weight = weight_variable(name % i, [prior, hs])
            bias   = bias_variable(name % i, [hs])
            prior  = hs
            self.weights.append(weight)
            self.biases.append(bias)
        self.weights.append( weight_variable(name % 'Out', [prior, out_size]) )
        self.biases.append( bias_variable(name % 'Out', [out_size]) )


    def forward(self, x):
        value = x 
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            value = tf.nn.xw_plus_b(value, W, b)
            
            if i < (len(self.weights)-1):
                value = self.hidden_act(value)
            
            if self.dropout is not None:
                value = tf.layers.dropout(value, self.dropout)

        out = self.output_act(value)
        return tf.identity(out, name=self.name + '_output')



def create_gradients(error, learning_rate, step):
    ''' Manually apply gradient calculations for clipping and summaries '''
    opt   = tf.train.AdamOptimizer(learning_rate)
    grads = opt.compute_gradients(error)

    for i, (grad, var) in enumerate(grads):
        tf.summary.histogram(var.name, var)

        if grad is not None:
            grads[i] = (tf.clip_by_norm(grad, 5), var)
            tf.summary.histogram(var.name + '/gradient', grad)

    return opt.apply_gradients(grads, global_step=step)


def batch_generator(data, batch_size, shuffle=True):
    """
    Credit: https://github.com/pumpikano/tf-dann/blob/master/utils.py
    Generate batches of data.
    
    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    def shuffle_aligned_list(data):
        """Shuffle arrays in a list by shuffling each array identically."""
        num = data[0].shape[0]
        p = np.random.permutation(num)
        return [d[p] for d in data]
        
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]


class DNN(object):
    def __init__(self,  learning_rate = 0.0005, 
                        hidden_layers = [100]*6, 
                        maximum_iter  = 1000, 
                        batch_size    = 128,
                        dropout_rate  = 0.1, 
                        l2_weight     = 1e-4, 
                        save_path     = None, 
                        log_path      = None):

        self.learning_rate = learning_rate
        self.hidden_layers = np.atleast_1d(hidden_layers)
        self.maximum_iter  = maximum_iter
        self.batch_size    = batch_size
        self.dropout_rate  = dropout_rate
        self.l2_weight     = l2_weight
        self.save_path     = save_path
        self.log_path      = log_path


    def initialize(self):
        X_ph = tf.placeholder(tf.float32, [None, self.n_features], name='Input')
        Y_ph = tf.placeholder(tf.float32, [None, self.n_outputs])

        learning_rate = tf.placeholder(tf.float32, [])
        dropout_rate  = tf.placeholder(tf.float32, [], name='Dropout')
        global_step   = tf.Variable(0., name='global_step', trainable=False)

        self.predict_network = Network( in_size  = self.n_features, 
                                        out_size = self.n_outputs, 
                                        name     ='PredictNetwork', 
                                        dropout  = dropout_rate,
                                        hidden_sizes = self.hidden_layers, 
                                        output_activation = lambda x:x)

        prediction = self.predict_network.forward(X_ph)
        l2_loss    = tf.add_n([tf.nn.l2_loss(v) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'PredictNetwork')]) * self.l2_weight
        pred_loss  = tf.reduce_mean( (prediction - Y_ph) ** 2 ) ** 0.5
        objective  = pred_loss + l2_loss
        optimize   = create_gradients(objective, learning_rate, global_step)
        tf.summary.scalar('Objective', objective)

        session    = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        session.run( tf.global_variables_initializer() )

        def run_iteration(X, y, lr, dr):
            return session.run([optimize, pred_loss],
                    feed_dict={ X_ph: X, Y_ph: y,
                                learning_rate: lr,
                                dropout_rate:  dr})

        def run_prediction(X):
            return session.run(prediction, 
                    feed_dict={X_ph: X, dropout_rate: 0.})
            
        self.run_iteration  = run_iteration
        self.run_prediction = run_prediction
        self.session = session


    def fit(self, X, Y):
        tf.reset_default_graph()

        # Same as numpy method, but adds dimension on last axis
        atleast_2d = lambda x: x if len(x.shape) >= 2 else x[..., None]
        X = atleast_2d(X)
        Y = atleast_2d(Y)

        self.n_features = X.shape[1]
        self.n_outputs  = Y.shape[1]
        self.initialize()

        if self.save_path is not None and not os.path.exists('Build/%s/' % self.save_path):
            os.makedirs('Build/%s/' % self.save_path)

        generator = batch_generator([X, Y], self.batch_size)
        iterator  = tqdm.trange(1, self.maximum_iter+1)
        last_valid= np.inf
        for i in iterator:

            # Annealed learning rate
            p  = float(i) / self.maximum_iter
            lr = self.learning_rate / (1. + 10 * p)**0.75

            # Training step
            x, y = next(generator)
            rets = self.run_iteration(x, y, lr, self.dropout_rate)

            if i % 100 == 0:
                metrics = { 'Train': '%.1f%%' % self.loss(X,Y) }
                iterator.set_postfix(**metrics)

        if self.save_path is not None:
            self.saver.save(self.session, 'Build/%s/'%self.save_path)


    def loss(self, X, Y, n_sample=10000):
        ''' Mean absolute percentage error '''
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)

        X = X.copy()[idx[:n_sample]]
        Y = Y.copy()[idx[:n_sample]]
        Z = self.predict(X)
        return 100 * np.mean(np.abs(Y - Z) / np.clip(np.abs(Y), 1e-5, np.inf))

    ''' Wrappers to allow overloading '''
    def predict(self, X):  
        return self.run_prediction(X)
    
    def get_params(self, deep=False): 
        return self.__dict__
    
    def set_params(self, deep=False, **kwargs):
        for k in kwargs: self.__dict__[k] = kwargs[k] 