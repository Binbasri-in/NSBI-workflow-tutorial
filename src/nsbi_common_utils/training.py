#import libraries
import os, importlib, sys, shutil
import numpy as np
import pandas as pd
import math
pd.options.mode.chained_assignment = None 

import pickle 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from joblib import dump, load


import pickle 

from nsbi_common_utils.plotting import plot_loss, plot_all_features
from nsbi_common_utils.calibration import HistogramCalibrator

importlib.reload(sys.modules['nsbi_common_utils.plotting'])
from nsbi_common_utils.plotting import plot_all_features

from joblib import dump, load

from tensorflow.keras.models import model_from_json

class TrainEvaluatePreselNN:
    '''
    A class for training the multi-class classification neural network for preselecting phase space for SBI
    '''
    def __init__(self, dataset, num_classes, features, features_scaling):
        '''
        dataset: dataframe with the multiple classes for training
        num_classes: number of classes corresponding to the number of output nodes of softmax layer
        features: input features to use for training
        features_scaling: subset of input features to standardize before training
        '''
        self.dataset = dataset
        self.data_features_training = dataset[features].copy()
        self.features = features
        self.features_scaling = features_scaling
        self.num_classes = num_classes

    # Defining a simple NN training for preselection - no need for "flexibility" here
    def train(self, test_size=0.15, 
                    random_state=42, 
                    path_to_save='', 
                    epochs=20, 
                    batch_size=1024, 
                    verbose=2, 
                    learning_rate=0.1):

        '''
        The function will train the preselection NN, assign it to self.model variable, and save the model to user-provided path_to_save directory.

        test_size: the fraction of dataset to set aside for diagnostics, not used in training and validation of the loss vs epoch curves
        random_state: random state to use for splitting the train/test dataset before training NN
        epochs: the number of epochs to train the NNs
        batch_size: the size of each batch used during gradient optimization
        learning_rate: the initial learning rate to pass to the optimizer
        '''

        # Split data into training and validation sets (including weights)
        X_train, X_val, y_train, y_val, weight_train, weight_val = train_test_split(self.data_features_training, 
                                                                                    self.dataset['train_labels'], 
                                                                                    self.dataset['weights_normed'], 
                                                                                    test_size=test_size, 
                                                                                    random_state=random_state, 
                                                                                    stratify=self.dataset['train_labels'])

        # Standardize the input features
        self.scaler = ColumnTransformer([("scaler", StandardScaler(), self.features_scaling)],remainder='passthrough')
        X_train = self.scaler.fit_transform(X_train)  # Fit & transform training data
        X_val = self.scaler.transform(X_val)
        
        # Define the neural network model
        self.model = keras.Sequential([
            layers.Input(shape=(self.data_features_training.shape[1],)),  # Input layer
            layers.Dense(1000, activation='swish'),
            layers.Dense(1000, activation='swish'),
            layers.Dense(self.num_classes, activation='softmax')  # Output layer for num_class classes
        ])

        # Using the Nadam optimizer by default
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)

        # Compile the model
        self.model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      weighted_metrics=["accuracy"])
        
        # setup the callbacks
        callback_factor = 0.01
        callback_patience = 30
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=callback_factor,
                                        patience=callback_patience, min_lr=0.000000001)
        
        # Train the model with sample weights
        self.model.fit(X_train, y_train, sample_weight=weight_train, 
                  validation_data=(X_val, y_val, weight_val), callbacks=[reduce_lr], epochs=epochs, batch_size=batch_size, verbose=verbose)

        K.clear_session()

        # Save the trained model if user provides with a path
        if path_to_save!='':

            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)
    
            model_json = self.model.to_json()
            with open(path_to_save+"model_arch_presel.json", "w") as json_file:
                json_file.write(model_json)
    
            # serialize weights to HDF5
            self.model.save_weights(path_to_save+"model_weights_presel.weights.h5")

            # Save the standard scaling
            saved_scaler = path_to_save+"model_scaler_presel.bin"
            dump(self.scaler, saved_scaler, compress=True)


    def get_trained_model(self, path_to_models):
        '''
        Method to load the trained model

        path_to_models: path to the directory with saved model config files
        '''
        json_file = open(path_to_models+'/model_arch_presel.json', "r")

        loaded_model_json = json_file.read()

        json_file.close()

        self.model = model_from_json(loaded_model_json)

        self.model.load_weights(path_to_models+'/model_weights_presel.weights.h5')

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')

        self.scaler = load(path_to_models+'/model_scaler_presel.bin')


    def predict(self, dataset):
        '''
        Method that evaluates density ratios on provided dataset, using self.model

        dataset: the dataset to evaluate trained model on
        '''
        features_scaled = self.scaler.transform(dataset[self.features])
        pred_NN = self.model.predict(features_scaled, batch_size=10000)
        K.clear_session()
        
        return pred_NN

        

class TrainEvaluate_NN:
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
        
    def train_ensemble(self, hidden_layers, 
                            neurons, 
                            number_of_epochs, 
                            batch_size,
                            learning_rate, 
                            scalerType, 
                            calibration=False, 
                            num_bins_cal = 40, 
                            callback = True, 
                            callback_patience=30, 
                            callback_factor=0.01,
                            activation='swish', 
                            verbose=2, 
                            validation_split=0.1, 
                            holdout_split=0.3, 
                            plot_scaled_features=False, 
                            load_trained_models = False,
                            recalibrate_output=False,
                            num_ensemble_members=1):
        '''
        Train an ensemble of NNs
        '''
        print(f"starting ensemble training")
        self.num_ensemble_members = num_ensemble_members
        
        # Define an array with random integers for boostrap training
        random_state_arr = np.random.randint(0, 2**32 -1, size=num_ensemble_members)

        self.model_NN           = [None for i in range(num_ensemble_members)]
        self.histogram_calibrator = [None for i in range(num_ensemble_members)]
        self.scaler             = [None for i in range(num_ensemble_members)]
        self.full_data_prediction     = [None for i in range(num_ensemble_members)]

        self.full_data_prediction = np.zeros((num_ensemble_members, len(self.weights)))
        self.train_idx          = [None for i in range(num_ensemble_members)]
        self.holdout_idx        = [None for i in range(num_ensemble_members)]

        # Train ensemble of NNs in series
        for ensemble_index in range(num_ensemble_members):

            if load_trained_models:
                if os.path.exists(f"{self.path_to_models}/model_weights{ensemble_index}.weights.h5"):
                    print(f"Loading existing model for ensemble member {ensemble_index}")
                    load_trained_models_ensemble_member = True
                else:
                    load_trained_models_ensemble_member = False

            else:
                load_trained_models_ensemble_member = False
            
            # Train ensemble NNs with different train/test split each time (bootstrapping without replacement)
            self.train(hidden_layers, 
                        neurons, 
                        number_of_epochs, 
                        batch_size,
                        learning_rate, 
                        scalerType, 
                        calibration, 
                        num_bins_cal, 
                        callback, 
                        callback_patience, 
                        callback_factor,
                        activation, 
                        verbose                 = verbose,
                        rnd_seed                = random_state_arr[ensemble_index], 
                        ensemble_index          = ensemble_index, 
                        validation_split        = validation_split, 
                        holdout_split           = holdout_split, 
                        plot_scaled_features    = plot_scaled_features, 
                        load_trained_models     = load_trained_models_ensemble_member,
                        recalibrate_output      = recalibrate_output)
        
    def train(self, hidden_layers, 
                    neurons, 
                    number_of_epochs, 
                    batch_size,
                    learning_rate, 
                    scalerType, 
                    calibration=False, 
                    num_bins_cal = 40, 
                    callback = True, 
                    callback_patience=30, 
                    callback_factor=0.01,
                    activation='swish', 
                    verbose=2, 
                    rnd_seed=2,
                    ensemble_index='', 
                    validation_split=0.1, 
                    holdout_split=0.3, 
                    plot_scaled_features=False, 
                    load_trained_models = False,
                    recalibrate_output=False):
        '''
        Method that trains the density ratio NNs

        batch_size: the size of each batch used during gradient optimization
        learning_rate: the initial learning rate to pass to the optimizer
        scalerType: option to one of three standardizing options: ['MinMax', 'StandardScaler', 'PowerTransform_Yeo'] 
        holdout_split: the fraction of dataset to set aside for diagnostics, not used in training and validation of the loss vs epoch curves
        epochs: the number of epochs to train the NNs

        calibration: boolean to do Histogram-based calibration of the NN ourput
        num_bins_cal: number of bins used for calibration histogram
        '''

        self.calibration = calibration
        self.calibration_switch = False # Set the switch to false for first evaluation for calibration

        self.verbose = verbose

        if ensemble_index=='':
            self.model_NN              = [None]
            self.scaler                = [None]
            self.histogram_calibrator  = [None]

            self.full_data_prediction  = np.zeros((1, len(self.weights)))
            self.train_idx             = [None]
            self.holdout_idx           = [None]
            self.num_ensemble_members  = 1
            ensemble_index             = 0

        
        if load_trained_models:
            # Load the number of holdout events and random state used for train/test split when using saved models
            holdout_num, rnd_seed = np.load(f"{self.path_to_models}num_events_random_state_train_holdout_split{ensemble_index}.npy")
        else:
            # Get the number of holdout events from the holdout_split fraction
            holdout_num = math.floor(self.dataset.shape[0] * holdout_split)

        
        # HyperParameters for the NN training
        validation_split = validation_split
        self.batch_size = batch_size

        idx_incl = np.arange(len(self.weights))

        # Get the indicies
        self.train_idx[ensemble_index], self.holdout_idx[ensemble_index] = train_test_split(idx_incl, 
                                                                                            test_size=holdout_num, 
                                                                                            random_state=rnd_seed,                                                        
                                                                                            stratify=self.training_labels)

        # split the original dataset into training and holdout
        data_train_full, data_holdout_full = self.dataset.iloc[self.train_idx[ensemble_index]].copy(), self.dataset.iloc[self.holdout_idx[ensemble_index]].copy()
        label_train, label_holdout = self.training_labels[self.train_idx[ensemble_index]].copy(), self.training_labels[self.holdout_idx[ensemble_index]].copy()
        weight_train, weight_holdout = self.weights[self.train_idx[ensemble_index]].copy(), self.weights[self.holdout_idx[ensemble_index]].copy()

        # dataset to be used for training
        data_train, data_holdout = data_train_full[self.features], data_holdout_full[self.features]

        # Setup callbacks
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=callback_factor,
                                        patience=callback_patience, min_lr=0.000000001)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=300)

        # Load pre-trained models and scaling
        if load_trained_models:

            print(f"Reading saved models from {self.path_to_models}")
            self.scaler[ensemble_index], self.model_NN[ensemble_index] = self.get_trained_model(self.path_to_models, 
                                                                                                ensemble_index = ensemble_index)
        # Else setup a new scaler
        else:

            if (scalerType == 'MinMax'):
                self.scaler[ensemble_index] = ColumnTransformer([("scaler",MinMaxScaler(feature_range=(-1.5,1.5)), self.features_scaling)],remainder='passthrough')
                
            if (scalerType == 'StandardScaler'):
                self.scaler[ensemble_index] = ColumnTransformer([("scaler",StandardScaler(), self.features_scaling)],remainder='passthrough')
                
            if (scalerType == 'PowerTransform_Yeo'):
                self.scaler[ensemble_index] = ColumnTransformer([("scaler",PowerTransformer(method='yeo-johnson', standardize=True), self.features_scaling)],remainder='passthrough')


        scaled_data_train = self.scaler[ensemble_index].fit_transform(data_train)
        scaled_data_train= pd.DataFrame(scaled_data_train, columns=self.features)

        if plot_scaled_features:
            plot_all_features(scaled_data_train, weight_train, label_train)

        scaled_data_holdout = self.scaler[ensemble_index].transform(data_holdout)
        scaled_data_holdout = pd.DataFrame(scaled_data_holdout, columns=self.features)

        # Train the model if not loaded
        if not load_trained_models:

            # Check if the datasets are normalized
            print(f"Sum of weights of class 0: {np.sum(weight_train[label_train==0])}")
            print(f"Sum of weights of class 1: {np.sum(weight_train[label_train==1])}")
    
            print(f"Using {activation} activation function")
    
            self.model_NN[ensemble_index] = build_model(n_hidden=hidden_layers, n_neurons=neurons, 
                                        learning_rate=learning_rate, 
                                        input_shape=[len(self.features)], 
                                        use_log_loss=self.use_log_loss,
                                        activation=activation)
            if self.verbose>0:
                self.model_NN[ensemble_index].summary()
    
            if callback:
    
                print("Using Callbacks")
    
                self.history = self.model_NN[ensemble_index].fit(scaled_data_train, label_train, callbacks=[reduce_lr, es], 
                                                                epochs=number_of_epochs, batch_size=batch_size, 
                                                                validation_split=validation_split, sample_weight=weight_train, 
                                                                verbose=self.verbose)
    
            else:
                print("Not Using Callbacks")
    
                self.history = self.model_NN[ensemble_index].fit(scaled_data_train, label_train, 
                                                                epochs=number_of_epochs, batch_size=batch_size, 
                                                                validation_split=validation_split, sample_weight=weight_train, 
                                                                verbose=self.verbose)
            
            K.clear_session()
        
            print("Finished Training")
    
            saved_scaler = f"{self.path_to_models}model_scaler{ensemble_index}.bin"
            print(saved_scaler)
    
            model_json = self.model_NN[ensemble_index].to_json()
            with open(f"{self.path_to_models}model_arch{ensemble_index}.json", "w") as json_file:
                json_file.write(model_json)
    
            # serialize weights to HDF5
            self.model_NN[ensemble_index].save_weights(f"{self.path_to_models}model_weights{ensemble_index}.weights.h5")
    
            np.save(f"{self.path_to_models}num_events_random_state_train_holdout_split{ensemble_index}.npy", 
                    np.array([holdout_num, rnd_seed]))
    
            dump(self.scaler[ensemble_index], saved_scaler, compress=True)
    
            plot_loss(self.history, path_to_figures=self.path_to_figures)

        
        # Do a first prediction without calibration layers
        train_data_prediction = self.predict_with_model(data_train_full, ensemble_index = ensemble_index, 
                                                        use_log_loss = self.use_log_loss)

        # If calibrating, use the train_data_prediction for building histogram
        if self.calibration:

            self.calibration_switch = True
            path_to_calibrated_object = f"{self.path_to_models}model_calibrated_hist{ensemble_index}.obj"

            calibration_data_num = train_data_prediction[label_train==1]
            calibration_data_den = train_data_prediction[label_train==0]

            w_num = weight_train[label_train==1]
            w_den = weight_train[label_train==0]

            if not load_trained_models:
            
                self.histogram_calibrator[ensemble_index] =  HistogramCalibrator(calibration_data_num, calibration_data_den, w_num, w_den, 
                                                                 nbins=num_bins_cal, method='direct', mode='dynamic')
    
                file_calib = open(path_to_calibrated_object, 'wb') 
    
                pickle.dump(self.histogram_calibrator, file_calib)
    
            else:
                if not os.path.exists(path_to_calibrated_object) or recalibrate_output:
                    
                    print(f"Calibrating the saved model with {num_bins_cal} bins")
                    
                    self.histogram_calibrator =  HistogramCalibrator(calibration_data_num, calibration_data_den, w_num, w_den, 
                                                                 nbins=num_bins_cal, method='direct', mode='dynamic')
    
                    file_calib = open(path_to_calibrated_object, 'wb') 
        
                    pickle.dump(self.histogram_calibrator, file_calib)
                else:
                
                    file_calib = open(path_to_calibrated_object, 'rb') 
                    self.histogram_calibrator = pickle.load(file_calib)
            
            self.full_data_prediction[ensemble_index] = self.predict_with_model(self.dataset, 
                                                                                ensemble_index = ensemble_index, 
                                                                                use_log_loss=self.use_log_loss)

        # Else, continue evaluating using the base model
        else:
            self.full_data_prediction[ensemble_index] = self.predict_with_model(self.dataset, 
                                                                                ensemble_index = ensemble_index, 
                                                                                use_log_loss=self.use_log_loss)

        
        # Some diagnostics to ensure numerical stability - min/max must not be exactly 0 or 1
        min_max_values = [
            (self.sample_name[1], "training", np.amin(self.full_data_prediction[ensemble_index][self.train_idx[ensemble_index]][self.training_labels[self.train_idx[ensemble_index]]==0]), 
                                              np.amax(self.full_data_prediction[ensemble_index][self.train_idx[ensemble_index]][self.training_labels[self.train_idx[ensemble_index]]==0])),
            (self.sample_name[0], "training", np.amin(self.full_data_prediction[ensemble_index][self.train_idx[ensemble_index]][self.training_labels[self.train_idx[ensemble_index]]==1]), 
                                              np.amax(self.full_data_prediction[ensemble_index][self.train_idx[ensemble_index]][self.training_labels[self.train_idx[ensemble_index]]==1])),
            (self.sample_name[1], "holdout", np.amin(self.full_data_prediction[ensemble_index][self.holdout_idx[ensemble_index]][self.training_labels[self.holdout_idx[ensemble_index]]==0]), 
                                              np.amax(self.full_data_prediction[ensemble_index][self.holdout_idx[ensemble_index]][self.training_labels[self.holdout_idx[ensemble_index]]==0])),
            (self.sample_name[0], "holdout", np.amin(self.full_data_prediction[ensemble_index][self.holdout_idx[ensemble_index]][self.training_labels[self.holdout_idx[ensemble_index]]==1]), 
                                              np.amax(self.full_data_prediction[ensemble_index][self.holdout_idx[ensemble_index]][self.training_labels[self.holdout_idx[ensemble_index]]==1]))
        ]
        
        for name, training_holdout_label, min_val, max_val in min_max_values:
            
            if min_val == 0:
                raise Warning(f"WARNING: {name} {training_holdout_label} data has min score = 0 for ensemble member {ensemble_index}, which may indicate numerical instability!")
            
            if max_val == 1:
                raise Watning(f"WARNING: {name} {training_holdout_label} data has max score = 1 for ensemble member {ensemble_index}, which may indicate numerical instability!")            



    def get_trained_model(self, path_to_models, ensemble_index=''):
        '''
        Method to load the trained model

        path_to_models: path to the directory with saved model config files
        '''
        json_file = open(path_to_models+f'/model_arch{ensemble_index}.json', "r")

        loaded_model_json = json_file.read()

        json_file.close()

        model = model_from_json(loaded_model_json)

        model.load_weights(path_to_models+f'/model_weights{ensemble_index}.weights.h5')

        opt = tf.keras.optimizers.Nadam(learning_rate=0.1)

        model.compile(loss='binary_crossentropy', optimizer=opt)

        scaler = load(path_to_models+f'/model_scaler{ensemble_index}.bin')

        return scaler, model

    
    def predict_with_model(self, data, ensemble_index = 0, use_log_loss=False):
        '''
        Method that evaluates density ratios on provided dataset, using self.model

        data: the dataset to evaluate trained model on
        '''
        scaled_data = self.scaler[ensemble_index].transform(data[self.features])
        pred = self.model_NN[ensemble_index].predict(scaled_data, 
                                                     verbose=self.verbose, 
                                                     batch_size=10_000)
        pred = pred.reshape(pred.shape[0],)

        if use_log_loss:

            pred = convert_to_score(pred)

        if (self.calibration) & (self.calibration_switch):

            pred = self.histogram_calibrator[ensemble_index].cali_pred(pred)
            pred = pred.reshape(pred.shape[0],)
            pred = np.clip(pred, 1e-25, 0.999999)

        K.clear_session()
            
        return pred
        

    def make_overfit_plots(self, ensemble_index=0):
        '''
        Plot predictions for training and holdout to test compatibility
        '''

        importlib.reload(sys.modules['nsbi_common_utils.plotting'])
        from nsbi_common_utils.plotting import plot_overfit

        plot_overfit(self.full_data_prediction[ensemble_index][self.train_idx[ensemble_index]][self.training_labels[self.train_idx[ensemble_index]]==0], 
                     self.full_data_prediction[ensemble_index][self.holdout_idx[ensemble_index]][self.training_labels[self.holdout_idx[ensemble_index]]==0], 
                     self.weights[self.train_idx[ensemble_index]][self.training_labels[self.train_idx[ensemble_index]]==0], 
                     self.weights[self.holdout_idx[ensemble_index]][self.training_labels[self.holdout_idx[ensemble_index]]==0], 
                    nbins=30, 
                    plotRange=[0.0,1.0], 
                    holdout_index=0, 
                    label=f'{self.sample_name[1]}', 
                    path_to_figures=self.path_to_figures)
        
        plot_overfit(self.full_data_prediction[ensemble_index][self.train_idx[ensemble_index]][self.training_labels[self.train_idx[ensemble_index]]==1], 
                     self.full_data_prediction[ensemble_index][self.holdout_idx[ensemble_index]][self.training_labels[self.holdout_idx[ensemble_index]]==1], 
                     self.weights[self.train_idx[ensemble_index]][self.training_labels[self.train_idx[ensemble_index]]==1], 
                     self.weights[self.holdout_idx[ensemble_index]][self.training_labels[self.holdout_idx[ensemble_index]]==1], 
                    nbins=30, 
                    plotRange=[0.0,1.0], 
                    holdout_index=0, 
                    label=f'{self.sample_name[0]}', 
                    path_to_figures=self.path_to_figures)


    def make_calib_plots(self, observable='score', nbins=10, ensemble_index=0):
        '''
        Test the probability calibration of NN output

        observable: choose between 'score' for relative probability p_A/p_A+p_B, and 'llr' for log-likelihood ratio log p_A/p_B
        '''
        importlib.reload(sys.modules['nsbi_common_utils.plotting'])
        from nsbi_common_utils.plotting import plot_calibration_curve, plot_calibration_curve_ratio

        if observable=='score':
            # Plot Calibration curves - score function
            plot_calibration_curve(self.full_data_prediction[ensemble_index][self.train_idx[ensemble_index]][self.training_labels[self.train_idx[ensemble_index]]==0], 
                                   self.weights[self.train_idx[ensemble_index]][self.training_labels[self.train_idx[ensemble_index]]==0], 
                                   self.full_data_prediction[ensemble_index][self.train_idx[ensemble_index]][self.training_labels[self.train_idx[ensemble_index]]==1], 
                                   self.weights[self.train_idx[ensemble_index]][self.training_labels[self.train_idx[ensemble_index]]==1], 
                                   self.full_data_prediction[ensemble_index][self.holdout_idx[ensemble_index]][self.training_labels[self.holdout_idx[ensemble_index]]==0], 
                                   self.weights[self.holdout_idx[ensemble_index]][self.training_labels[self.holdout_idx[ensemble_index]]==0], 
                                   self.full_data_prediction[ensemble_index][self.holdout_idx[ensemble_index]][self.training_labels[self.holdout_idx[ensemble_index]]==1], 
                                   self.weights[self.holdout_idx[ensemble_index]][self.training_labels[self.holdout_idx[ensemble_index]]==1], 
                                   self.path_to_figures, 
                                   nbins=nbins, 
                                   label="Calibration Curve - "+str(self.sample_name[0]))

        elif observable=='llr':
            # Plot Calibration curves - nll function
            plot_calibration_curve_ratio(self.full_data_prediction[ensemble_index][self.train_idx[ensemble_index]][self.training_labels[self.train_idx[ensemble_index]]==0], 
                                        self.weights[self.train_idx[ensemble_index]][self.training_labels[self.train_idx[ensemble_index]]==0], 
                                        self.full_data_prediction[ensemble_index][self.train_idx[ensemble_index]][self.training_labels[self.train_idx[ensemble_index]]==1], 
                                        self.weights[self.train_idx[ensemble_index]][self.training_labels[self.train_idx[ensemble_index]]==1], 
                                        self.full_data_prediction[ensemble_index][self.holdout_idx[ensemble_index]][self.training_labels[self.holdout_idx[ensemble_index]]==0], 
                                        self.weights[self.holdout_idx[ensemble_index]][self.training_labels[self.holdout_idx[ensemble_index]]==0], 
                                        self.full_data_prediction[ensemble_index][self.holdout_idx[ensemble_index]][self.training_labels[self.holdout_idx[ensemble_index]]==1], 
                                        self.weights[self.holdout_idx[ensemble_index]][self.training_labels[self.holdout_idx[ensemble_index]]==1], 
                                        self.path_to_figures, 
                                        nbins=nbins, 
                                        label="Calibration Curve - "+str(self.sample_name[0]))

        else:
            raise Exception("observable not recognized - choose between score and llr options")


    def make_reweighted_plots(self, variables, scale, num_bins, ensemble_index = 0):
        '''
        Test the quality of the NN predicted density ratios using a reweighting check p_A/p_B * p_B ~ p_A

        variables: list of variables to plot
        scale: linear or log y-axis scales
        num_bins: number of bins in the reweighting diagnostic plot
        '''
        importlib.reload(sys.modules['nsbi_common_utils.plotting'])
        from nsbi_common_utils.plotting import plot_reweighted

        plot_reweighted(self.dataset.iloc[self.train_idx[ensemble_index]].copy(), 
                        self.full_data_prediction[ensemble_index][self.train_idx[ensemble_index]][self.training_labels[self.train_idx[ensemble_index]]==0], 
                        self.weights[self.train_idx[ensemble_index]][self.training_labels[self.train_idx[ensemble_index]]==0], 
                        self.full_data_prediction[ensemble_index][self.train_idx[ensemble_index]][self.training_labels[self.train_idx[ensemble_index]]==1], 
                        self.weights[self.train_idx[ensemble_index]][self.training_labels[self.train_idx[ensemble_index]]==1],
                        variables=variables, 
                        num=num_bins,
                        sample_name=self.sample_name, scale=scale,  
                        path_to_figures=self.path_to_figures, label='Training Data Diagnostic')

        plot_reweighted(self.dataset.iloc[self.holdout_idx[ensemble_index]].copy(), 
                        self.full_data_prediction[ensemble_index][self.holdout_idx[ensemble_index]][self.training_labels[self.holdout_idx[ensemble_index]]==0], 
                        self.weights[self.holdout_idx[ensemble_index]][self.training_labels[self.holdout_idx[ensemble_index]]==0],
                        self.full_data_prediction[ensemble_index][self.holdout_idx[ensemble_index]][self.training_labels[self.holdout_idx[ensemble_index]]==1], 
                        self.weights[self.holdout_idx[ensemble_index]][self.training_labels[self.holdout_idx[ensemble_index]]==1],
                        variables=variables, 
                        num=num_bins,
                        sample_name=self.sample_name, scale=scale, 
                        path_to_figures=self.path_to_figures, label='Holdout Data Diagnostic')

    def test_normalization(self):
        '''
        Test if \int p_A/p_B x p_B ~ 1
        '''
        # Normalized reference (denominator) hypothesis
        weight_ref = self.weights[self.training_labels==0].copy()

        ratio_rwt = np.zeros((self.num_ensemble_members, weight_ref.shape[0]))

        for ensemble_index in range(self.num_ensemble_members):
            
            # Calculate p_A/p_B for B hypothesis events
            score_rwt = self.predict_with_model(self.dataset[self.features], ensemble_index=ensemble_index, use_log_loss=self.use_log_loss)[self.training_labels==0]
            ratio_rwt[ensemble_index] = score_rwt / ( 1.0 - score_rwt )
    
            # Calculate \sum p_A/p_B x p_B
            print(f"\n\n\nThe sum of PDFs in ensemble member {ensemble_index} is {np.sum(ratio_rwt[ensemble_index] * weight_ref)}\n\n")

        ratio_rwt_aggregate = np.mean(ratio_rwt, axis=0)
        
        print(f"The sum of PDFs using the whole ensemble is {np.sum(ratio_rwt_aggregate * weight_ref)}\n\n\n")
        

    def evaluate_and_save_ratios(self, dataset, aggregation_type = 'mean_ratio'):
        '''
        Evaluate with self.model on the input dataset, and save to self.path_to_ratios

        aggregation_type: choose an option on how to aggregate the ensemble models - 'median_ratio', 'mean_ratio', 'median_score', 'mean_score'
        '''

        print(f"Evaluating density ratios")
        score_pred = np.ones((self.num_ensemble_members, dataset.shape[0]))
        ratio_pred = np.ones((self.num_ensemble_members, dataset.shape[0]))
        log_ratio_pred = np.ones((self.num_ensemble_members, dataset.shape[0]))

        for ensemble_index in range(self.num_ensemble_members):
            score_pred[ensemble_index] = self.predict_with_model(dataset[self.features], 
                                                                 use_log_loss=self.use_log_loss, 
                                                                 ensemble_index=ensemble_index)
            
            ratio_pred[ensemble_index] = score_pred[ensemble_index] / (1.0 - score_pred[ensemble_index])
            log_ratio_pred[ensemble_index] = np.log(score_pred[ensemble_index] / (1.0 - score_pred[ensemble_index]))

        if aggregation_type == 'median_ratio':
            ratio_ensemble = np.median(ratio_pred, axis=0)
            
        elif aggregation_type == 'mean_ratio':
            ratio_ensemble = np.mean(ratio_pred, axis=0)
            
        elif aggregation_type == 'median_score':
            score_aggregate = np.median(score_pred, axis=0)
            ratio_ensemble = score_aggregate / (1.0 - score_aggregate)
            
        elif aggregation_type == 'mean_score':
            score_aggregate = np.mean(score_pred, axis=0)
            ratio_ensemble = score_aggregate / (1.0 - score_aggregate)

        else:
            raise Exception("aggregation_type not recognized, please choose between median_ratio, mean_ratio, median_score or mean_score")
        
        np.save(f"{self.path_to_ratios}ratio_{self.sample_name[0]}.npy", ratio_ensemble)
        print(f"Density ratios saved")
        

def build_model(n_hidden=4, 
                n_neurons=1000, 
                learning_rate=0.1, 
                input_shape=[11], 
                use_log_loss=False, 
                optimizer_choice='Nadam', 
                activation='swish'):
    '''
    Method that builds the NN model used in density ratio training

    activation: string with any activation function supported by keras. Option to use 'mish' too
    optimizer_choice: Two options to choose from - 'Nadam' or 'Adam'
    use_log_loss: option to use modified BCE loss function that regresses to log p_A/p_B
    '''
    model = tf.keras.models.Sequential()
    options = {"input_shape":input_shape}
    for layer in range(n_hidden):

        if activation=='mish':
            def mish(inputs):
                x = tf.nn.softplus(inputs)
                x = tf.nn.tanh(x)
                x = tf.multiply(x, inputs)
                return x

            model.add(Dense(n_neurons, 
                            activation=mish, 
                            **options))
        else:
            model.add(Dense(n_neurons, 
                            activation=activation, 
                            **options))
        options={}

    if not use_log_loss:
        model.add(Dense(1,activation='sigmoid',**options))
    else:
        model.add(Dense(1,activation='linear',**options))

    if optimizer_choice=='Nadam':
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate) 
    elif optimizer_choice=='Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) 
    else:
        raise Exception("Optimizer choice not recognized - please choose between 'Nadam' or 'Adam'")

    if use_log_loss:
        # Use the modified BCE loss that regresses to the log p_A/p_B instead of p_A/p_A+p_B
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
                      optimizer=optimizer, 
                      weighted_metrics=['binary_accuracy'])
    else:
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(), 
                      optimizer=optimizer, 
                      weighted_metrics=['binary_accuracy'])
    return model


def convert_to_score(logLR):
    '''
    Convert regressed logLR into relative probabilities for compatibility with other methods
    '''
    return 1.0/(1.0+np.exp(-logLR))
