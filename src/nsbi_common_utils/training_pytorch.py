import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DensityRatioModel(nn.Module):
    """
    PyTorch equivalent of the TensorFlow build_model().

    Arguments:
        input_shape: list or tuple of input feature dimensions
        n_hidden: number of hidden layers
        n_neurons: number of neurons per hidden layer
        activation: 'swish', 'mish', or any torch-supported activation
        use_log_loss: whether to use BCEWithLogitsLoss (log p_A/p_B regression)
    """
    def __init__(self, 
                 input_shape=(11,), 
                 n_hidden=4, 
                 n_neurons=1000, 
                 activation='swish',
                 use_log_loss=False):
        super().__init__()
        self.use_log_loss = use_log_loss

        # --- Activation ---
        if activation == 'swish':
            act_fn = nn.SiLU()             # Swish = SiLU in PyTorch
        elif activation == 'mish':
            act_fn = nn.Mish()
        elif hasattr(F, activation):
            act_fn = getattr(F, activation)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # --- Build fully connected layers ---
        layers = []
        in_features = input_shape[0]

        for _ in range(n_hidden):
            layers.append(nn.Linear(in_features, n_neurons))
            layers.append(act_fn)
            in_features = n_neurons

        # Output layer
        out_activation = nn.Identity() if use_log_loss else nn.Sigmoid()
        layers.append(nn.Linear(in_features, 1))
        layers.append(out_activation)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# replacing tensorflow with pytorch
class TrainEvaluate_Pytorch_NN:
    '''
    A class for training the density ratio neural networks for SBI analysis
    '''
    def __init__(self, dataset, 
                      weights, 
                      training_labels, 
                      features, 
                      features_scaling, 
                      sample_name, 
                      output_dir, 
                      output_name, 
                      path_to_figures='',
                      path_to_models='', 
                      path_to_ratios='',
                      use_log_loss=False, 
                      split_using_fold=False,
                      delete_existing_models=False):
        '''
        dataset: the main dataframe containing two classes p_A, p_B for density ratio p_A/p_B estimation
        weights: the weight vector, normalized independently for each class A & B
        training_labels: array of 1s for p_A hypothesis and 0s for p_B hypothesis
        features: training features x in p_A(x)/p_B(x)
        features_scaling: training features to standardize before training
        sample_name: set with strings containing names of A and B
        '''
        self.dataset = dataset
        self.weights = weights
        self.training_labels = training_labels
        self.features = features
        self.features_scaling = features_scaling
        self.sample_name = sample_name
        self.output_dir = output_dir
        self.output_name = output_name
        self.use_log_loss = use_log_loss
        self.split_using_fold = split_using_fold

        # Initialize a list of models to train - if no ensemble, this is a 1 member list
        self.model_NN = [None]
        self.scaler = [None]
        
        self.path_to_figures = path_to_figures

        if delete_existing_models:
            if os.path.exists(path_to_figures):
                shutil.rmtree(path_to_figures)
            
            if os.path.exists(path_to_models):
                shutil.rmtree(path_to_models)
                
            if os.path.exists(path_to_ratios):
                shutil.rmtree(path_to_ratios)


        if not os.path.exists(path_to_figures):
                os.makedirs(path_to_figures)
        
        self.path_to_models = path_to_models
        if not os.path.exists(path_to_models):
                os.makedirs(path_to_models)

        self.path_to_ratios=path_to_ratios
        if not os.path.exists(path_to_ratios):
                os.makedirs(path_to_ratios)
        
    
    def train(self):
        pass
    
    def predict_with_model(self):
        pass
    
    def make_overfit_plots(self):
        pass
    
    
def build_model_pytorch(n_hidden=4, 
                n_neurons=1000, 
                learning_rate=0.1, 
                input_shape=[11], 
                use_log_loss=False, 
                optimizer_choice='Adam', 
                activation='swish'):
    '''
    Method that builds the NN model used in density ratio training

    activation: string with any activation function supported by keras. Option to use 'mish' too
    optimizer_choice: Two options to choose from - 'Nadam' or 'Adam'
    use_log_loss: option to use modified BCE loss function that regresses to log p_A/p_B
    '''
    model = DensityRatioModel(
        input_shape=input_shape,
        n_hidden=n_hidden,
        n_neurons=n_neurons,
        activation=activation,
        use_log_loss=use_log_loss
    )
    
    # --- Optimizer ---
    if optimizer_choice == 'Nadam':
        optimizer = optim.NAdam(model.parameters(), lr=learning_rate)
    else:  # Default to Adam
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Loss function ---
    if use_log_loss:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.BCELoss()

    return model, optimizer, loss_fn

    