from base import *
from util import *
from dataloader import *

class BaseClassifier(object):
    def __init__(self):
        self.layers = []

    def forward(self, x, mode):
        for layer in self.layers:
            x = layer.forward(x, mode)
        return x

    def backward(self, delta):
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

    def fit(self, data_train, label_train, data_val=None, label_val=None, learning_rate=1e-5, loss_function='MSE', batch_size=1, epochs=1, log_frequency=None):
        train_dataloader = DataLoader(data=data_train, labels=label_train, batch_size=batch_size, shuffle=True)

        loss_hist = []
        for k in range(epochs):
            epoch_loss = np.zeros(len(train_dataloader))
            if log_frequency:
                print('epoch {}'.format(k))
            for i in range(len(train_dataloader)):
                batch_item = train_dataloader[i]

                batch_labels = np.zeros((batch_item['label'].shape[0], 10))
                batch_labels[np.arange(batch_item['label'].shape[0]),np.squeeze(batch_item['label'])] = 1

                batch_outputs = self.forward(batch_item['data'], 'train')

                if loss_function == 'MSE':
                    loss, delta = MSELoss(batch_labels, batch_outputs)
                elif loss_function == 'cross_entropy':
                    loss, delta = SoftmaxCrossEntropyLoss(batch_labels, batch_outputs)

                epoch_loss[i] = loss

                self.backward(delta)

                self.update(learning_rate)

                if log_frequency and (i+1) % log_frequency == 0:
                    val_loss, val_acc = self.validate(data_val, label_val)
                    print('\tIteration {} - train_loss: {} - val_loss: {} - val_accuracy: {}'.format(i+1, loss, val_loss, val_acc))
            loss_hist.append(np.mean(epoch_loss))
        return loss_hist

    def validate(self, data_val, label_val):
        val_dataloader = DataLoader(data=data_val, labels=label_val, batch_size=1, shuffle=False)

        val_loss = np.zeros(len(val_dataloader))
        output = np.zeros(len(val_dataloader))

        for i in range(len(val_dataloader)):
            batch_item = val_dataloader[i]

            batch_labels = np.zeros((batch_item['label'].shape[0], 10))
            batch_labels[np.arange(batch_item['label'].shape[0]),np.squeeze(batch_item['label'])] = 1

            batch_outputs = self.forward(batch_item['data'], 'test')
            output[i] = np.argmax(batch_outputs)

            loss, delta = MSELoss(batch_labels, batch_outputs)
            val_loss[i] = loss
        return np.mean(val_loss), accuracy_score(label_val,output)

    def predict(self, data_test):
        output = np.zeros(data_test.shape[0])
        for i in np.arange(data_test.shape[0]):
            output[i] = np.argmax(self.forward(data_test[i,:], 'test'))
        return output

class Linear(object):
    def __init__(self, n_in, n_out, last_activ, activ, weight_decay=None, momentum=None, drop_rate=None):
        self.n_in = n_in
        self.n_out = n_out
        self.activ = activ
        self.activation_deriv = Activation(last_activ).f_deriv
        self.activation = Activation(activ).f
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.drop_rate = drop_rate
        self.previous_update = np.zeros((n_in,n_out))

        self.input = None
        self.output = None

        self.weights = np.random.uniform(low = -np.sqrt(6. / (self.n_in + self.n_out)),
                                         high = np.sqrt(6. / (self.n_in + self.n_out)),
                                         size = (self.n_in, self.n_out))

        self.bias = np.zeros(self.n_out,)

        self.grad_w = np.zeros(self.weights.shape)
        self.grad_b = np.zeros(self.bias.shape)
        self.prev_grad_w = np.zeros(self.weights.shape)
        self.prev_grad_b = np.zeros(self.bias.shape)

    def forward(self, x, mode):
        self.input = x
        self.output = np.dot(self.input, self.weights) + self.bias
        if self.drop_rate and mode=='train':
            drop = np.random.rand(*self.output.shape)
            self.output[drop < (1 - self.drop_rate)] = 0.0
        if self.activation and (self.activ != 'softmax' or mode == 'test'):
            self.output = self.activation(self.output)
        return self.output

    def backward(self, delta):
        self.grad_w = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
        self.grad_b = np.mean(delta, axis=0)
        if self.activation_deriv:
            delta = delta.dot(self.weights.T) * self.activation_deriv(self.input)
        return delta

    def update(self, lr):
        if self.momentum:
            self.previous_update = (self.momentum * self.previous_update) + (lr * self.grad_w)
            self.weights -= self.previous_update
            self.bias -= (lr * self.grad_b)
        else:
            self.weights -= (lr * self.grad_w)
            self.bias -= (lr * self.grad_b)
        if self.weight_decay:
            self.weights = self.weights - (self.weight_decay * self.weights)

class BatchNorm(object):
    def __init__(self, n_features, epsilon):
        self.n_features = n_features
        self.epsilon = epsilon

        self.gamma = np.ones(n_features,)
        self.beta = np.zeros(n_features,)

        self.layer_mean = np.zeros(n_features,)
        self.layer_var = np.zeros(n_features,)

    def forward(self, x, mode, momentum=0.9):
        if mode == 'train':
            self.x_input = x
            self.x_mean = np.mean(self.x_input, axis=0)
            self.x_var = np.var(self.x_input, axis=0)

            self.layer_mean += self.x_mean
            self.layer_var += self.x_var

            self.x_norm = (self.x_input - self.x_mean) / np.sqrt(self.x_var + self.epsilon)

            output = self.gamma * self.x_norm + self.beta
        elif mode == 'test':
            norm = (x - self.layer_mean) / np.sqrt(self.layer_var + self.epsilon)
            output = self.gamma * norm + self.beta
        return output

    def backward(self, delta):
        self.grad_beta = np.sum(delta, axis=0)
        self.grad_gamma = np.sum(delta*self.x_norm, axis=0)

        # implementation based off https://zaffnet.github.io/batch-normalization#bprop
        t = 1./np.sqrt(self.x_var + self.epsilon)
        m = self.x_input.shape[0]

        delta = (self.gamma*t/m) * (m * delta - np.sum(delta, axis=0) - t**2 * (self.x_input - self.x_mean) * np.sum(delta*(self.x_input - self.x_mean), axis=0))

        return delta

    def update(self, lr):
        self.gamma -= lr * self.grad_gamma
        self.beta -= lr * self.grad_beta