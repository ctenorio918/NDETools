import numpy as np
import sklearn
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import Perceptron

##################################################################################################################
# Utilities - Validation Tools, Statistics, Goodness of Fit Analysis                                             #
##################################################################################################################

def nde_test_train(observation_array: np.ndarray,
                   prediction_array: np.ndarray,
                   test_split=0.3,
                   fix_rng=False):
    """
    **Function:** nde_test_train()

    **Author:** Nate Tenorio

    **Date:** 4/14/2023

    **Purpose:** Utility function with more documentation to create train_test splits for machine learning.

    A suggested Train/Test split is 70/30. One thing to be particularly careful about

    :param observation_array: An M x N array with M samples of N features you **observe**
    :param prediction_array: An M x P array with M samples of P features you **wish to predict**
    :param test_split: What percentage of your data would you like to use as test data? Should be <1.
    :param fix_rng: bool, whether you would like to fix your RNG or not (useful for recreating plots).
    :return: observation_train, observation_test, prediction_train, prediction_test
    """

    if fix_rng:
        rng = 9181999
    else:
        rng = None
    observation_train, observation_test, prediction_train, prediction_test = train_test_split(
        observation_array, prediction_array, test_size=test_split, random_state=rng
    )
    return observation_train, observation_test, prediction_train, prediction_test

def nde_cross_validate_kfold(fit_object,
                             observation_train: np.ndarray,
                             prediction_train: np.ndarray,
                             n_folds=5,
                             scoring='neg_mean_squared_error',
                             fix_rng=False):
    """
    **Function:** nde_cross_validate_kfold

    **Date:** 4/14/2023

    **Purpose:** This method uses K-Fold cross-validation to estimate goodness of fit. In general, this is really important
    to ensure that your machine learning model is not prone to overfit. If the standard deviation varies significantly
    between results, your model likely needs some reworking/hyperparameter tuning.

    Typically, the data that you use in k-fold cross validation is training data. This is done before applying test
    data in order to ensure that our model is accepting the test data completely blindly.

    :param fit_object: The sk-learn fit object that you have trained
    :param observation_train: The data to use in your fit
    :param prediction_train: The data you would like to use in your prediction
    :param n_folds: The number of folds to use in cross-validation
    :param scoring: The scoring method you are choosing to use. (https://scikit-learn.org/stable/modules/model_evaluation.html)
    :param fix_rng: Bool, whether you want fixed RNG for reproducing results
    :return: scores: the cross-validation scores calculated using k-folds
    """
    if fix_rng:
        rng = 9181999
    else:
        rng = None
    cv = KFold(n_folds, random_state=rng)
    scores = cross_val_score(fit_object,
                                 observation_train,
                                 prediction_train,
                                 cv=cv,
                                 scoring=scoring)
    scores = -1 * scores  # For some reason the function always outputs negative MSE. I flip the sign back.
    return scores

def auto_warn_cv_overfit(scores: np.ndarray,
                         z_score_warn=2):
    """
    Mini Function: auto_warn_cv_overfit

    Purpose: prints automatic warnings if significant overfit is predicted based on your scores. This is done
    by calculating the z-score of each of your folds. If any is over the given warning threshold, a warning message is
    printed.

    :param scores: The cross-validation scores you want to detect overfit from
    :param z_score_warn: The z-score threshold
    :return: Printed warning if z-score threshold is exceeded.
    """
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    z_scores = (scores-mean_score)/std_score
    pass_fail = abs(z_scores) - z_score_warn < 0
    if True in pass_fail:
        print(f'Warning: Cross Validation Yields Outlier with Z-Score > {z_score_warn}')

def nde_Accuracy_Score(model_prediction: np.ndarray,
                       prediction_test: np.ndarray):
    accuracy = accuracy_score(prediction_test, model_prediction)
    return accuracy

################################################################################################################
# KNN Classification Implementation                                                                                #
################################################################################################################

def nde_KNN_Classifier(numNeighbors: int,
                       observation_train: np.ndarray,
                       prediction_train: np.ndarray) -> type(KNeighborsClassifier):
    knn = KNeighborsClassifier(numNeighbors)
    knn.fit(observation_train, prediction_train)
    return knn

def nde_KNN_Apply_Classifier(knn: type(KNeighborsClassifier),
                             observation_test: np.ndarray):
    prediction_knn = knn.predict(observation_test)
    return prediction_knn

def nde_score_KNN(prediction_knn: np.ndarray,
                  prediction_test: np.ndarray):
    knn_accuracy = accuracy_score(prediction_test, prediction_knn)
    return knn_accuracy

#################################################################################################################
# Support Vector Machines                                                                                       #
#################################################################################################################

def nde_Linear_SVM(observation_train: np.ndarray,
                   prediction_train: np.ndarray,
                   C: int):
    lSVM = SVC(kernel='linear', C=C)
    lSVM.fit(observation_train, prediction_train)
    return lSVM

def nde_apply_Linear_SVM(lSVM,
                         observation_test: np.ndarray) -> np.ndarray:
    prediction_lSVM = lSVM.predict(observation_test)
    return prediction_lSVM

def nde_RBF_SVM(observation_train: np.ndarray,
                prediction_train: np.ndarray,
                C: int,
                gamma: int) -> type(SVC):
    rbf_SVM = SVC(kernel='rbf', C=C, gamma=gamma)
    rbf_SVM.fit(observation_train, prediction_train)
    return rbf_SVM

def nde_apply_RBF_SVM(rbf_SVM,
                      observation_test: np.ndarray) -> np.ndarray:
    prediction_rbf_SVM = rbf_SVM.predict(observation_test)
    return prediction_rbf_SVM

###############################################################################################################
# Gaussian Process Classification
###############################################################################################################

def nde_create_GPC_Model(kernel = 1.0 * RBF(50.0),
                         optimizer='fmin_l_bfgs_b',
                         n_restarts = 5,
                         max_iter = 1000):
    GPC = GaussianProcessClassifier(kernel=kernel,
                                    optimizer=optimizer,
                                    n_restarts_optimizer=n_restarts,
                                    max_iter = max_iter)
    return GPC

def nde_fit_GPC(GPC: GaussianProcessClassifier,
                observation_train: np.ndarray,
                prediction_train: np.ndarray) -> GaussianProcessClassifier:
    GPC.fit(observation_train, prediction_train)
    return GPC

def nde_predict_GPC(GPC: GaussianProcessClassifier,
                    observation_test: np.ndarray) -> np.ndarray:
    GPC_Prediction = GPC.predict(observation_test)
    return GPC_Prediction

#################################################################################################################
# Perceptron Implementation
#################################################################################################################

def Config_Multilayer_Perceptron(hidden_layer_sizes=(40, 100, 30),
                                     max_iter=500,
                                     activation='relu',
                                     solver='adam',
                                     alpha=0.0008):
    """
    Note that default values represent ideal hyperparameters obtained after tuning.

    :param hidden_layer_sizes: Sizes of hidden layers.
    :param max_iter: Number of iterations of the solver.
    :param activation: Activation function for hidden layer. Relu preferred.
    :param solver: The weight optimization solver. Defaults to Adam algorithm.
    :param alpha: L2 regularization term.
    :return:
    """
    nn = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                       max_iter=max_iter,
                       activation=activation,
                       solver=solver,
                       alpha=alpha)
    return nn

def nde_fit_Multilayer_Perceptron(nn: MLPClassifier,
                                  observation_train: np.ndarray,
                                  prediction_train: np.ndarray) -> MLPClassifier:
    nn.fit(observation_train, prediction_train)
    return nn

def nde_predict_Multilayer_Perceptron(nn: MLPClassifier,
                                      observation_test: np.ndarray) -> np.ndarray:
    MLP_prediction = nn.predict(observation_test)
    return MLP_prediction

##################################################################################################################
# Single Layer Perceptron
##################################################################################################################

def nde_Config_Perceptron(penalty='l2',
                          alpha=0.0005) -> Perceptron:
    perceptron = Perceptron(penalty=penalty,
                            alpha=alpha)
    return perceptron

def nde_fit_Perceptron(perceptron: Perceptron,
                       observation_train: np.ndarray,
                       prediction_train: np.ndarray) -> Perceptron:
    perceptron.fit(observation_train, prediction_train)
    return perceptron

def nde_apply_Perceptron(perceptron: Perceptron,
                         observation_test: np.ndarray) -> np.ndarray:
    perceptron_prediction = perceptron.predict(observation_test)
    return perceptron_prediction


###############################################################################################################
# Computer Vision Implementation
###############################################################################################################
def non_maximum_suppression(
        fft_data: float,
        data_file: str,
        sim_path: str,
        kernel: int = 21,
        gradient: list = None,
        x_lim: list = None,
        y_lim: list = None,
        clip_tr: float = 1.0,
        plot_flag: bool = False,
        save_flag: bool = False
) -> tuple:
    """
        do non-maximum-suppression using Pytorch

        returns:
            - fft_data - clipped and removed all data below median of
                input fft_data,
            - x - x-coords of NMS maxima,
            - y - y-coords of NMW maxima

        args:
            - fft_data - 2D-FFT data matrix
            - json_info_file - simulation information dictionary
            - sim_path - path to simulation files
            - kernel=21 - size of kernel for max pooling, needs to be odd,
                61 and 11 look good too
            - gradient = [gl, gu] (list): specify gradient boundaries (lower
                and upper gradient) between which the NMS data should be.
                Ignore data outside of it.
            - x_lim (list): lower and upper limit for extracted max pool
                coordinates on x axis
            - y_lim (list): lower and upper limit for extracted max pool
                coordinates on y axis
            - clip_tr=1 - threshold for clipping, a high clipping value
                (e.g. 1 = clipping off)
            - plot_flag=False - specify if the extration cones should be
                visualized
            - save_flag=False - specify if NMS coords should be stored
    """
    if x_lim is None:
        x_lim = [50, 300]
    if y_lim is None:
        y_lim = [250, 1800]
    if gradient is None:
        gradient = [2.8, 9]


    # remove all entries smaller than median
    fft_fltr = np.multiply(fft_data.copy(), fft_data > np.median(fft_data))
    fft_fltr = my_unsqueeze_2_torch(fft_fltr)

    mp = nn.MaxPool2d(kernel_size=kernel, stride=1, padding=int(kernel // 2))

    fft_mp = mp(fft_fltr)
    fft_bin = torch.eq(fft_mp, fft_fltr)  # check elementwise if elements equal
    fft_out = np.multiply(my_squeeze_2_np(fft_bin), fft_data)  # extract only MAXs from fft_data

    fft_p = fft_out

    # -- get coordinates of maxima
    b = np.where(fft_p != 0)
    c = np.zeros((len(b[0])))  # intensity values at b locations
    # k = 26000 #250
    for i, (x, y) in enumerate(zip(b[0], b[1])):
        c[i] = fft_p[x, y]
    idx = np.argsort(c)
    # 0.2 looks like it gets the job done, but maybe needs to be tuned later on
    idx_reduced = [i for i in idx if c[i] > 0.2*np.mean(c)]

    # import pdb; pdb.set_trace()

    # -- define outputs - remember x and y are swapped for images in python
    x = np.flip(b[1][idx_reduced])
    y = np.flip(b[0][idx_reduced])

    # remove too low and too high values first
    x_new, y_new = [], []
    for (x_elem, y_elem) in zip(x, y):
        if x_lim[0] < x_elem < x_lim[1] and y_lim[0] < y_elem < y_lim[1]:
            x_new.append(x_elem)
            y_new.append(y_elem)
    x = x_new
    y = y_new

    # remove values which are outside a cone
    x_new, y_new = [], []
    gl, gu = gradient[0], gradient[1]
    for (x_elem, y_elem) in zip(x, y):
        if gl * x_elem < y_elem < gu * x_elem:
            x_new.append(x_elem)
            y_new.append(y_elem)
    x = x_new
    y = y_new

    print(f'number of maxima detected: {len(x_new)}')

    x, y = np.array(x), np.array(y)

    if plot_flag:
        plt.figure(1, dpi=600)
        plt.contourf(fft_out, 200, cmap='Spectral')
        plt.scatter(x, y, color='lime', marker='2', alpha=0.7, s=0.5)
        xl_plot = np.linspace(0, 700)
        xu_plot = np.linspace(0, 300)
        plt.plot(xl_plot, gl * xl_plot)
        plt.plot(xu_plot, gu * xu_plot)
        plt.show()

    out = np.concatenate((x, y))
    if save_flag:
        xy_file_name = data_file[0:-9] + 'xy.txt'
        try:
            with open(sim_path / xy_file_name, 'w') as f:
                np.savetxt(f, out, delimiter=',')
        except Exception:
            with open(xy_file_name, 'w') as f:
                np.savetxt(f, out, delimiter=',')

    # -- clip values of 2dfft
    fft_data = np.clip(fft_data, 0, clip_tr)
    fft_data = np.multiply(fft_data.copy(), fft_data > np.median(fft_data))

    return fft_data, x, y

def my_unsqueeze_2_torch(image) -> torch.Tensor:
    """
    Converts np image array of size (n,m) into torch.Tensor of size (1,1,n,m)
    """
    return torch.Tensor(np.expand_dims(np.expand_dims(image, axis=0), axis=0))

def my_squeeze_2_np(image) -> np.array:
    """
    converts image tensor of size (1,1,n,m) into np.array of size (n,m)
    """
    return np.squeeze(np.squeeze(np.array(image)))