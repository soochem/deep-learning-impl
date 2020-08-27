import numpy as np

###======= activation functions =======###
def sigmoid(x):
    return 1 / (1+np.exp(-x))

def tanh(x):
    return np.tanh(x)

def softmax(x):
    exp_x = np.exp(x)
    sum_exp = np.sum(exp_x)
    return exp_x / sum_exp

###======= layers ======================###
def dense_layer(input_x
                , output_dim=None
                , weight=None
                , bias=None
                , seed=1
               ):
    
    input_dim = input_x.shape[-1]
    
    # weight init
    if weight is None:
        np.random.seed(seed)
        weight = np.random.random((input_dim, output_dim)).round(2)
        bias = np.random.random((output_dim,)).round(2)
    
    output = (input_x @ weight) + bias
    
    return output, weight, bias