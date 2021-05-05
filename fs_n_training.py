#%%
import prepocessing as pp
import utils
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools


SVC_CACHED = 'SVC_model.pickle'
SOFTMAX_CACHED = 'softmax_model.pickle'

# param grid of softmax_regression
softmax_param = {   'name':   'softmax',
        'model':  LogisticRegression(multi_class="multinomial",solver="lbfgs",  random_state=42),
        'param_grid': {
            'softmax__C': [0.1, 1, 10, 100,],
            'softmax__max_iter':[5000,6000,8000],
            'selector__k': [500, 1000, 2000],
            'selector__score_func': [mutual_info_classif, chi2] 
        } 
    }
# param grid of SVM classifier
SVC_param = {   'name':   'SVC',
        'model':   SVC(gamma='auto'),
        'param_grid': {
            'SVC__C': [0.1, 1, 10, 100,],
            'SVC__kernel':['linear','rbf'],
            'selector__k': [500, 1000, 2000],
            'selector__score_func': [mutual_info_classif, chi2] 
        } 
    }  

# 5-fold cross-validation setup
cv = RepeatedStratifiedKFold(n_splits=5, random_state=1)

"""

Grid Search with cross-validation find to train model and find the best

"""

def tune_models_hyperparams(X, y, model):
    print('{:-^70}'.format(' [' + model['name'] + '] '))
    pipe = Pipeline([
                ("min_max_scaler", MinMaxScaler()),
                ("selector", SelectKBest(score_func=chi2)),
                (model['name'], model['model'])   ])
    grid = (GridSearchCV(pipe, # model
                                        param_grid= model['param_grid'], # grid parameters
                                        cv= cv, # cross validation scheme
                                        scoring='accuracy', # scoring function
                                        n_jobs=-1, # compuational resource allocation
                                        )
                                .fit(X, y))
    # saving single trained model
    utils.serialisation('{}_model.pickle'.format(model['name']),grid)
    return grid

def grid_cv_train(X_train,y_train,param):
    grid = tune_models_hyperparams(X_train, y_train, param)
    print_grid_results(grid)
    return grid

"""

Print classification report and confusion matrix

"""
def print_test_report(name,topics,grid,X_test,y_test):
    print('Model:',grid.best_params_)
    y_pred = grid.predict(X_test)
    print('{:-^70}'.format(' <REPORT> '))
    print(classification_report(y_test,y_pred))
    cm = confusion_matrix(y_test,y_pred,labels=pp.topic_label_list)
    plot_confusion_matrix(cm,topics,title=name+" Confusion Matrix")


def print_grid_results(grids): 
    print('Score:\t\t{:.2%}'.format(grids.best_score_))
    print('Parameters:\t{}'.format(grids.best_params_))
    print('*' * 70)

"""

Loading grid searched models, return from cache if exist

"""
def get_grid_instance(X_train=None,y_train=None,svc_cache=SVC_CACHED,softmax_cache=SOFTMAX_CACHED):
    svc_grid = None
    softmax_grid = None
    if not utils.does_file_exist(SVC_CACHED):
        grid_cv_train(X_train,y_train,SVC_param)
    if not utils.does_file_exist(SOFTMAX_CACHED):
        grid_cv_train(X_train,y_train,softmax_param)
    svc_grid = utils.load(SVC_CACHED)
    softmax_grid = utils.load(SOFTMAX_CACHED)
    return svc_grid,softmax_grid

"""
confusion matrix plotting
"""
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()