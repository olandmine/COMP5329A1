from base import *
np.random.seed(0)

train_data = np.load(os.path.join('Assignment1-Dataset','Assignment1-Dataset','train_data.npy'))
train_label = np.load(os.path.join('Assignment1-Dataset','Assignment1-Dataset','train_label.npy'))
test_data = np.load(os.path.join('Assignment1-Dataset','Assignment1-Dataset','test_data.npy'))
test_label = np.load(os.path.join('Assignment1-Dataset','Assignment1-Dataset','test_label.npy'))

x_train_data = np.array(train_data)
x_test_data = np.array(test_data)

x_test = x_test_data
y_test = test_label

train_len = x_train_data.shape[0]
x_train = x_train_data[:((9*train_len)//10)]
y_train = train_label[:((9*train_len)//10)]

x_val = x_train_data[-((1*train_len)//10):]
y_val = train_label[-((1*train_len)//10):]

idxs = np.array([i for i in range(x_train.shape[0])])
np.random.shuffle(idxs)


print(len(x_train))
print(len(x_val))
print(len(x_test))

class BaseModel(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = None
        self.momentum = None
        self.drop_rate = None
        self.epsilon = None

        self.layers.append(Linear(128, 64, last_activ=None, activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 32, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(32, 16, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(16, 10, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

# base_model_classifier = BaseModel()
# base_model_hist = base_model_classifier.fit(x_train, y_train, learning_rate=1e-5, loss_function='MSE', batch_size=100, epochs=10)
# val_output = base_model_classifier.predict(x_val)
# print('Base model:')
# print(accuracy_score(y_val,val_output))

def test_base_model(learning_rate):
    print('\nBase model test:')
    base_classifier = BaseModel()
    base_loss_hist = base_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=100, epochs=10)
    val_output = base_classifier.predict(x_val)
    print('Base model:')
    print(accuracy_score(y_val,val_output))

    plt.figure(figsize=(15,4))
    plt.plot(base_loss_hist, label='Base model')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Mean training loss per epoch')
    plt.legend()
    plt.grid()
    plt.savefig('test_base_model.png')

class OneLayer(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = None
        self.momentum = None
        self.drop_rate = None
        self.epsilon = None

        self.layers.append(Linear(128, 10, last_activ=None, activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

class TwoLayer(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = None
        self.momentum = None
        self.drop_rate = None
        self.epsilon = None

        self.layers.append(Linear(128, 64, last_activ=None, activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 10, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

class ThreeLayer(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = None
        self.momentum = None
        self.drop_rate = None
        self.epsilon = None

        self.layers.append(Linear(128, 64, last_activ=None, activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 32, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(32, 10, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

class FourLayer(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = None
        self.momentum = None
        self.drop_rate = None
        self.epsilon = None

        self.layers.append(Linear(128, 64, last_activ=None, activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 32, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(32, 16, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(16, 10, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

def test_multi_layer(learning_rate):
    print('\nMulti layer test:')
    one_layer_classifier = OneLayer()
    one_layer_loss_hist = one_layer_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=100, epochs=10)
    val_output = one_layer_classifier.predict(x_val)
    print('One layer:')
    print(accuracy_score(y_val,val_output))

    two_layer_classifier = TwoLayer()
    two_layer_loss_hist = two_layer_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=100, epochs=10)
    val_output = two_layer_classifier.predict(x_val)
    print('Two layer:')
    print(accuracy_score(y_val,val_output))

    three_layer_classifier = ThreeLayer()
    three_layer_loss_hist = three_layer_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=100, epochs=10)
    val_output = three_layer_classifier.predict(x_val)
    print('Three layer:')
    print(accuracy_score(y_val,val_output))

    four_layer_classifier = FourLayer()
    four_layer_loss_hist = four_layer_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=100, epochs=10)
    val_output = four_layer_classifier.predict(x_val)
    print('Four layer:')
    print(accuracy_score(y_val,val_output))

    # Base model:
    # 0.0992
    # One layer:
    # 0.104
    # Two layer:
    # 0.1024
    # Three layer:
    # 0.1002
    # Four layer:
    # 0.0976

    plt.figure(figsize=(15,4))
    plt.plot(one_layer_loss_hist, label='One layer')
    plt.plot(two_layer_loss_hist, label='Two layers')
    plt.plot(three_layer_loss_hist, label='Three layers')
    plt.plot(four_layer_loss_hist, label='Four layers')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Mean training loss per epoch')
    plt.legend()
    plt.grid()
    plt.savefig('test_mutli_layers.png')

class TanhActivation(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = None
        self.momentum = None
        self.drop_rate = None
        self.epsilon = None

        self.layers.append(Linear(128, 64, last_activ=None, activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 32, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(32, 16, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(16, 10, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

class ReluActivation(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = None
        self.momentum = None
        self.drop_rate = None
        self.epsilon = None

        self.layers.append(Linear(128, 64, last_activ=None, activ='relu', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 32, last_activ='relu', activ='relu', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(32, 16, last_activ='relu', activ='relu', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(16, 10, last_activ='relu', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

def test_relu(learning_rate):
    print('\nRelu test:')
    tanh_classifier = TanhActivation()
    tanh_loss_hist = tanh_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=100, epochs=10)
    val_output = tanh_classifier.predict(x_val)
    print('Tanh:')
    print(accuracy_score(y_val,val_output))

    relu_classifier = ReluActivation()
    relu_loss_hist = relu_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=100, epochs=10)
    val_output = relu_classifier.predict(x_val)
    print('Relu:')
    print(accuracy_score(y_val,val_output))

    # Base model:
    # 0.0992
    # Tanh:
    # 0.0996
    # Relu:
    # 0.098

    plt.figure(figsize=(15,4))
    plt.plot(tanh_loss_hist, label='Tanh')
    plt.plot(relu_loss_hist, label='ReLU')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Mean training loss per epoch')
    plt.legend()
    plt.grid()
    plt.savefig('test_relu.png')

class WeightDecay1(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = 1e-1
        self.momentum = None
        self.drop_rate = None
        self.epsilon = None

        self.layers.append(Linear(128, 64, last_activ=None, activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 32, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(32, 16, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(16, 10, last_activ='tanh', activ='tanhtanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

class WeightDecay2(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = 1e-2
        self.momentum = None
        self.drop_rate = None
        self.epsilon = None

        self.layers.append(Linear(128, 64, last_activ=None, activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 32, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(32, 16, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(16, 10, last_activ='tanh', activ='tanhtanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

class WeightDecay3(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = 1e-3
        self.momentum = None
        self.drop_rate = None
        self.epsilon = None

        self.layers.append(Linear(128, 64, last_activ=None, activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 32, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(32, 16, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(16, 10, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

class WeightDecay4(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = 1e-4
        self.momentum = None
        self.drop_rate = None
        self.epsilon = None

        self.layers.append(Linear(128, 64, last_activ=None, activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 32, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(32, 16, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(16, 10, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

class WeightDecay5(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = 1e-5
        self.momentum = None
        self.drop_rate = None
        self.epsilon = None

        self.layers.append(Linear(128, 64, last_activ=None, activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 32, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(32, 16, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(16, 10, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

class WeightDecay6(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = 1e-6
        self.momentum = None
        self.drop_rate = None
        self.epsilon = None

        self.layers.append(Linear(128, 64, last_activ=None, activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 32, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(32, 16, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(16, 10, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

def test_weight_decay(learning_rate):
    print('\nWeight decay test:')
    wd_1_classifier = WeightDecay1()
    wd_1_loss_hist = wd_1_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=100, epochs=10)
    val_output = wd_1_classifier.predict(x_val)
    print('1e-1:')
    print(accuracy_score(y_val,val_output))

    wd_2_classifier = WeightDecay2()
    wd_2_loss_hist = wd_2_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=100, epochs=10)
    val_output = wd_2_classifier.predict(x_val)
    print('1e-2:')
    print(accuracy_score(y_val,val_output))

    wd_3_classifier = WeightDecay3()
    wd_3_loss_hist = wd_3_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=100, epochs=10)
    val_output = wd_3_classifier.predict(x_val)
    print('1e-3:')
    print(accuracy_score(y_val,val_output))

    wd_4_classifier = WeightDecay4()
    wd_4_loss_hist = wd_4_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=100, epochs=10)
    val_output = wd_4_classifier.predict(x_val)
    print('1e-4:')
    print(accuracy_score(y_val,val_output))

    wd_5_classifier = WeightDecay5()
    wd_5_loss_hist = wd_5_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=100, epochs=10)
    val_output = wd_5_classifier.predict(x_val)
    print('1e-5:')
    print(accuracy_score(y_val,val_output))

    wd_6_classifier = WeightDecay6()
    wd_6_loss_hist = wd_6_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=100, epochs=10)
    val_output = wd_6_classifier.predict(x_val)
    print('1e-6:')
    print(accuracy_score(y_val,val_output))

    # Base model:
    # 0.0992
    # 1e-1:
    # 0.097
    # 1e-2:
    # 0.1058
    # 1e-3:
    # 0.0976
    # 1e-4:
    # 0.0948
    # 1e-5:
    # 0.101
    # 1e-6:
    # 0.1026

    plt.figure(figsize=(15,4))
    plt.plot(wd_1_loss_hist, label='1e-1')
    plt.plot(wd_2_loss_hist, label='1e-2')
    plt.plot(wd_3_loss_hist, label='1e-3')
    plt.plot(wd_4_loss_hist, label='1e-4')
    plt.plot(wd_5_loss_hist, label='1e-5')
    plt.plot(wd_6_loss_hist, label='1e-6')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Mean training loss per epoch')
    plt.legend()
    plt.grid()
    plt.savefig('test_weight_decay.png')

class Momentum1(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = None
        self.momentum = 0.9
        self.drop_rate = None
        self.epsilon = None

        self.layers.append(Linear(128, 64, last_activ=None, activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 32, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(32, 16, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(16, 10, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

class Momentum2(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = None
        self.momentum = 0.8
        self.drop_rate = None
        self.epsilon = None

        self.layers.append(Linear(128, 64, last_activ=None, activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 32, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(32, 16, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(16, 10, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

class Momentum3(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = None
        self.momentum = 0.7
        self.drop_rate = None
        self.epsilon = None

        self.layers.append(Linear(128, 64, last_activ=None, activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 32, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(32, 16, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(16, 10, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

class Momentum4(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = None
        self.momentum = 0.6
        self.drop_rate = None
        self.epsilon = None

        self.layers.append(Linear(128, 64, last_activ=None, activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 32, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(32, 16, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(16, 10, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

def test_momentum(learning_rate):
    print('\nMomentum test:')
    momentum_1_classifier = Momentum1()
    momentum_1_loss_hist = momentum_1_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=100, epochs=10)
    val_output = momentum_1_classifier.predict(x_val)
    print('0.9:')
    print(accuracy_score(y_val,val_output))

    momentum_2_classifier = Momentum2()
    momentum_2_loss_hist = momentum_2_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=100, epochs=10)
    val_output = momentum_2_classifier.predict(x_val)
    print('0.8:')
    print(accuracy_score(y_val,val_output))

    momentum_3_classifier = Momentum3()
    momentum_3_loss_hist = momentum_3_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=100, epochs=10)
    val_output = momentum_3_classifier.predict(x_val)
    print('0.7:')
    print(accuracy_score(y_val,val_output))

    momentum_4_classifier = Momentum4()
    momentum_4_loss_hist = momentum_4_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=100, epochs=10)
    val_output = momentum_4_classifier.predict(x_val)
    print('0.6:')
    print(accuracy_score(y_val,val_output))

    # Base model:
    # 0.0992
    # 0.9:
    # 0.0966
    # 0.8:
    # 0.106
    # 0.7:
    # 0.0972
    # 0.6:
    # 0.0902

    plt.figure(figsize=(15,4))
    plt.plot(momentum_1_loss_hist, label='0.9')
    plt.plot(momentum_2_loss_hist, label='0.8')
    plt.plot(momentum_3_loss_hist, label='0.7')
    plt.plot(momentum_4_loss_hist, label='0.6')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Mean training loss per epoch')
    plt.legend()
    plt.grid()
    plt.savefig('test_momentum.png')

class Dropout1(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = None
        self.momentum = None
        self.drop_rate = 0.3
        self.epsilon = None

        self.layers.append(Linear(128, 64, last_activ=None, activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 32, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(32, 16, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(16, 10, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

class Dropout2(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = None
        self.momentum = None
        self.drop_rate = 0.4
        self.epsilon = None

        self.layers.append(Linear(128, 64, last_activ=None, activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 32, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(32, 16, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(16, 10, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

class Dropout3(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = None
        self.momentum = None
        self.drop_rate = 0.5
        self.epsilon = None

        self.layers.append(Linear(128, 64, last_activ=None, activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 32, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(32, 16, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(16, 10, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

class Dropout4(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = None
        self.momentum = None
        self.drop_rate = 0.6
        self.epsilon = None

        self.layers.append(Linear(128, 64, last_activ=None, activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 32, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(32, 16, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(16, 10, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

class Dropout5(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = None
        self.momentum = None
        self.drop_rate = 0.7
        self.epsilon = None

        self.layers.append(Linear(128, 64, last_activ=None, activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 32, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(32, 16, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(16, 10, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

def test_dropout(learning_rate):
    print('\nDropout test:')
    dropout_1_classifier = Dropout1()
    dropout_1_loss_hist = dropout_1_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=100, epochs=10)
    val_output = dropout_1_classifier.predict(x_val)
    print('0.3:')
    print(accuracy_score(y_val,val_output))

    dropout_2_classifier = Dropout2()
    dropout_2_loss_hist = dropout_2_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=100, epochs=10)
    val_output = dropout_2_classifier.predict(x_val)
    print('0.4:')
    print(accuracy_score(y_val,val_output))

    dropout_3_classifier = Dropout3()
    dropout_3_loss_hist = dropout_3_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=100, epochs=10)
    val_output = dropout_3_classifier.predict(x_val)
    print('0.5:')
    print(accuracy_score(y_val,val_output))

    dropout_4_classifier = Dropout4()
    dropout_4_loss_hist = dropout_4_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=100, epochs=10)
    val_output = dropout_4_classifier.predict(x_val)
    print('0.6:')
    print(accuracy_score(y_val,val_output))

    dropout_5_classifier = Dropout5()
    dropout_5_loss_hist = dropout_5_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=100, epochs=10)
    val_output = dropout_5_classifier.predict(x_val)
    print('0.7:')
    print(accuracy_score(y_val,val_output))

    # Base model:
    # 0.0992
    # 0.3:
    # 0.1018
    # 0.4:
    # 0.1098
    # 0.5:
    # 0.0946
    # 0.6:
    # 0.0956
    # 0.7:
    # 0.0978

    plt.figure(figsize=(15,4))
    plt.plot(dropout_1_loss_hist, label='0.3')
    plt.plot(dropout_2_loss_hist, label='0.4')
    plt.plot(dropout_3_loss_hist, label='0.5')
    plt.plot(dropout_4_loss_hist, label='0.6')
    plt.plot(dropout_5_loss_hist, label='0.7')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Mean training loss per epoch')
    plt.legend()
    plt.grid()
    plt.savefig('test_dropout.png')

class MeanSquaredError(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = None
        self.momentum = None
        self.drop_rate = None
        self.epsilon = None

        self.layers.append(Linear(128, 64, last_activ=None, activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 32, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(32, 16, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(16, 10, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

class SoftmaxCrossEntropy(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = None
        self.momentum = None
        self.drop_rate = None
        self.epsilon = None

        self.layers.append(Linear(128, 64, last_activ=None, activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 32, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(32, 16, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(16, 10, last_activ='tanh', activ='softmax', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

def test_loss(learning_rate):
    print('\nLoss test:')
    mse_classifier = MeanSquaredError()
    mse_loss_hist = mse_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=100, epochs=10)
    val_output = mse_classifier.predict(x_val)
    print('MSE:')
    print(accuracy_score(y_val,val_output))

    softmax_ce_classifier = Dropout2()
    softmax_ce_loss_hist = softmax_ce_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='cross_entropy', batch_size=100, epochs=10)
    val_output = softmax_ce_classifier.predict(x_val)
    print('Softmax CE:')
    print(accuracy_score(y_val,val_output))

    # Base model:
    # 0.0992
    # MSE:
    # 0.0996
    # Softmax CE:
    # 0.1018

    plt.figure(figsize=(15,4))
    # plt.plot(mse_loss_hist, label='MSE')
    plt.plot(softmax_ce_loss_hist, label='Softmax cross-entropy')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Mean training loss per epoch')
    plt.legend()
    plt.grid()
    plt.savefig('test_loss.png')

class MiniBatch1(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = None
        self.momentum = None
        self.drop_rate = None
        self.epsilon = None

        self.layers.append(Linear(128, 64, last_activ=None, activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 32, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(32, 16, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(16, 10, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

class MiniBatch2(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = None
        self.momentum = None
        self.drop_rate = None
        self.epsilon = None

        self.layers.append(Linear(128, 64, last_activ=None, activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 32, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(32, 16, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(16, 10, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

class MiniBatch3(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = None
        self.momentum = None
        self.drop_rate = None
        self.epsilon = None

        self.layers.append(Linear(128, 64, last_activ=None, activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 32, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(32, 16, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(16, 10, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

class MiniBatch4(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = None
        self.momentum = None
        self.drop_rate = None
        self.epsilon = None

        self.layers.append(Linear(128, 64, last_activ=None, activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 32, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(32, 16, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(16, 10, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

class MiniBatch5(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = None
        self.momentum = None
        self.drop_rate = None
        self.epsilon = None

        self.layers.append(Linear(128, 64, last_activ=None, activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 32, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(32, 16, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(16, 10, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

def test_mini_batch(learning_rate):
    print('\nMini-batch test:')
    mini_batch_1_classifier = MiniBatch1()
    start = time.time()
    mini_batch_1_loss_hist = mini_batch_1_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=1, epochs=10)
    end = time.time()
    val_output = mini_batch_1_classifier.predict(x_val)
    print('1:')
    print(accuracy_score(y_val,val_output))
    print(end - start)

    mini_batch_2_classifier = MiniBatch2()
    start = time.time()
    mini_batch_2_loss_hist = mini_batch_2_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=10, epochs=10)
    end = time.time()
    val_output = mini_batch_2_classifier.predict(x_val)
    print('10:')
    print(accuracy_score(y_val,val_output))
    print(end - start)

    mini_batch_3_classifier = MiniBatch3()
    start = time.time()
    mini_batch_3_loss_hist = mini_batch_3_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=100, epochs=10)
    end = time.time()
    val_output = mini_batch_3_classifier.predict(x_val)
    print('100:')
    print(accuracy_score(y_val,val_output))
    print(end - start)

    mini_batch_4_classifier = MiniBatch4()
    start = time.time()
    mini_batch_4_loss_hist = mini_batch_4_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=1000, epochs=10)
    end = time.time()
    val_output = mini_batch_4_classifier.predict(x_val)
    print('1000:')
    print(accuracy_score(y_val,val_output))
    print(end - start)

    mini_batch_5_classifier = MiniBatch5()
    start = time.time()
    mini_batch_5_loss_hist = mini_batch_5_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=10000, epochs=10)
    end = time.time()
    val_output = mini_batch_5_classifier.predict(x_val)
    print('10000:')
    print(accuracy_score(y_val,val_output))
    print(end - start)

    # Base model:
    # 0.0992
    # 1:
    # 0.098
    # 10:
    # 0.1044
    # 100:
    # 0.0982
    # 1000:
    # 0.0904
    # 10000:
    # 0.103

    plt.figure(figsize=(15,4))
    plt.plot(mini_batch_1_loss_hist, label='batch_size = 1')
    plt.plot(mini_batch_2_loss_hist, label='batch_size = 10')
    plt.plot(mini_batch_3_loss_hist, label='batch_size = 100')
    plt.plot(mini_batch_4_loss_hist, label='batch_size = 1000')
    # plt.plot(mini_batch_5_loss_hist, label='batch_size = 10000')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Mean training loss per epoch')
    plt.legend()
    plt.grid()
    plt.savefig('test_mini_batch.png')

class BatchNormalisation(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = None
        self.momentum = None
        self.drop_rate = None
        self.epsilon = 1e-5

        self.layers.append(Linear(128, 64, last_activ=None, activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(BatchNorm(64, epsilon=self.epsilon))
        self.layers.append(Linear(64, 32, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(BatchNorm(32, epsilon=self.epsilon))
        self.layers.append(Linear(32, 16, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(BatchNorm(16, epsilon=self.epsilon))
        self.layers.append(Linear(16, 10, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

def test_batch_norm(learning_rate):
    print('\nBatch norm test:')
    batch_norm_classifier = BatchNormalisation()
    batch_norm_loss_hist = batch_norm_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=100, epochs=10)
    val_output = batch_norm_classifier.predict(x_val)
    print('With BN:')
    print(accuracy_score(y_val,val_output))

    base_classifier = OneLayer()
    base_loss_hist = base_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=100, epochs=10)
    val_output = base_classifier.predict(x_val)
    print('Base model:')
    print(accuracy_score(y_val,val_output))

    # Base model:
    # 0.1048
    # With BN:
    # 0.1024

    plt.figure(figsize=(15,4))
    plt.plot(batch_norm_loss_hist, label='With BN')
    plt.plot(base_loss_hist, label='Without BN')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Mean training loss per epoch')
    plt.legend()
    plt.grid()
    plt.savefig('test_batch_norm.png')

learning_rate=1e-6
# test_base_model(learning_rate)
# test_multi_layer(learning_rate)
# test_relu(learning_rate)
# test_weight_decay(learning_rate)
# test_momentum(learning_rate)
# test_dropout(learning_rate)
test_loss(learning_rate)
# test_mini_batch(learning_rate)
# test_batch_norm(learning_rate)

import warnings

class BestModel1(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = None
        self.momentum = 0.9
        self.drop_rate = None
        self.epsilon = 1e-5

        self.layers.append(Linear(128, 64, last_activ=None, activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 32, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(32, 16, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(16, 10, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

class BestModel2(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = 1e-5
        self.momentum = 0.9
        self.drop_rate = None
        self.epsilon = 1e-5

        self.layers.append(Linear(128, 64, last_activ=None, activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 32, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(32, 16, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(16, 10, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

class BestModel3(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = 1e-5
        self.momentum = 0.9
        self.drop_rate = None
        self.epsilon = 1e-5

        self.layers.append(Linear(128, 64, last_activ=None, activ='relu', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 32, last_activ='relu', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(32, 16, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(16, 10, last_activ='tanh', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

class BestModel4(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = 1e-5
        self.momentum = 0.9
        self.drop_rate = None
        self.epsilon = 1e-5

        self.layers.append(Linear(128, 64, last_activ=None, activ='relu', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(64, 32, last_activ='relu', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(32, 16, last_activ='tanh', activ='relu', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(Linear(16, 10, last_activ='relu', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

class BestModel5(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.weight_decay = 1e-5
        self.momentum = 0.9
        self.drop_rate = None
        self.epsilon = 1e-5

        self.layers.append(Linear(128, 64, last_activ=None, activ='relu', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(BatchNorm(64, epsilon=self.epsilon))
        self.layers.append(Linear(64, 32, last_activ='relu', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(BatchNorm(32, epsilon=self.epsilon))
        self.layers.append(Linear(32, 16, last_activ='tanh', activ='relu', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))
        self.layers.append(BatchNorm(16, epsilon=self.epsilon))
        self.layers.append(Linear(16, 10, last_activ='relu', activ='tanh', weight_decay=self.weight_decay, momentum=self.momentum, drop_rate=self.drop_rate))

def test_best1(learning_rate):
    print('\nBest test:')
    best_1_classifier = BestModel1()
    best_1_loss_hist = best_1_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=10, epochs=100)
    val_output = best_1_classifier.predict(x_val)
    print('Best:')
    print(accuracy_score(y_val,val_output))

    print('\nBest test:')
    best_2_classifier = BestModel2()
    best_2_loss_hist = best_2_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=10, epochs=100)
    val_output = best_2_classifier.predict(x_val)
    print('Best:')
    print(accuracy_score(y_val,val_output))

    plt.figure(figsize=(15,4))
    plt.plot(best_1_loss_hist, label='Best model 1')
    plt.plot(best_2_loss_hist, label='Best model 2')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Mean training loss per epoch')
    plt.legend()
    plt.grid()
    plt.savefig('test_best1.png')


# with warnings.catch_warnings():
#     warnings.simplefilter('error')
#     # function_raising_warning()
# test_best1(learning_rate)

# learning_rate=1e-4
# print('\nBest test:')
# best_classifier = BestModel1()
# best_loss_hist = best_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=10, epochs=100)
# val_output = best_classifier.predict(x_val)
# print('Best:')
# print(accuracy_score(y_val,val_output))
# Best:
# 0.4524

# learning_rate=1e-4
# print('\nBest test:')
# best_classifier = BestModel2()
# best_loss_hist = best_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=10, epochs=100)
# val_output = best_classifier.predict(x_val)
# print('Best:')
# print(accuracy_score(y_val,val_output))
# Best:
# 0.467

# learning_rate=1e-4
# print('\nBest test:')
# best_classifier = BestModel3()
# best_loss_hist = best_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=10, epochs=100)
# val_output = best_classifier.predict(x_val)
# print('Best:')
# print(accuracy_score(y_val,val_output))
# Best:
# 0.5234

# learning_rate=1e-4
# print('\nBest test:')
# best_classifier = BestModel4()
# best_loss_hist = best_classifier.fit(x_train, y_train, learning_rate=learning_rate, loss_function='MSE', batch_size=10, epochs=100)
# val_output = best_classifier.predict(x_val)
# print('Best:')
# print(accuracy_score(y_val,val_output))
# Best:
# 0.524

# output = best_classifier.predict(x_test)
# print(y_test.shape)
# print(output.shape)
# print(accuracy_score(y_test,output))
# # print(classification_report(y_test,output))
# cm = confusion_matrix(y_test,output)
# cmd = ConfusionMatrixDisplay(cm)
# cmd.plot()
# plt.savefig('confusion_matrix.png')