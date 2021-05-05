#%%
import prepocessing as pp
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import chi2,mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import classification_report,accuracy_score
import utils

# Loading dataset
X_ori,y_ori = pp.processing(True)


# 1. Set Spliting
X_train, X_test, y_train, y_test = train_test_split(
        X_ori, y_ori, test_size=0.2, random_state=42)

X_test_new, X_dev, y_test_new, y_dev = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42)

X_test = X_test_new
y_test = y_test_new

print(len(X_train),len(X_test),len(X_dev))
#%%
""" 

Logistics Regression Test

"""
print("{:*^70}".format("<One-shot test on Softmax Regression>"))
softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10, random_state=42)
softmax_reg_clf = Pipeline([
        ("min_max_scaler", MinMaxScaler()),
        ("selector",SelectKBest(chi2,k=500)),
        ("softmax_reg", softmax_reg),
        ])    
softmax_reg_clf.fit(X_train, y_train)

print("{:-^70}".format("<TRAINING>"),'\n')
# checking underfitting or not
print('Training Fitting Score:\t\t{:.2%}'.format(accuracy_score(softmax_reg_clf.predict(X_train),y_train)))


print("{:-^70}".format("<VALIDATION>"),'\n')
# checking overfitting or not
y_dev_pred_reg = softmax_reg_clf.predict(X_dev)
print('Validation Score:\t\t{:.2%}'.format(accuracy_score(softmax_reg_clf.predict(X_dev),y_dev)))

print("{:-^70}".format("<TEST>"),'\n')
# evaluation on test set
y_test_pred_reg = softmax_reg_clf.predict(X_test)
print(classification_report(y_test,y_test_pred_reg))

#%%
""" 

SVM Test

"""
# 3. Training
print('\n',"{:*^70}".format("<One-shot test on Linear SVM>"))
svm_clf = Pipeline([
        ("min_max_scaler", MinMaxScaler()),
        ("selector",SelectKBest(chi2,k=500)),
        ("svm_classifier", SVC(kernel='linear',gamma='auto',C=10)),
        ])
svm_clf.fit(X_train,y_train)

print('\n',"{:-^70}".format("<TRAINING>"))
# checking underfitting or not
print('Training Fitting Score:\t\t{:.2%}'.format(accuracy_score(svm_clf.predict(X_train),y_train)))


print('\n',"{:-^70}".format("<VALIDATION>"))
# checking overfitting or not
y_dev_pred_svm = svm_clf.predict(X_dev)
print('Validation Score:\t\t{:.2%}'.format(accuracy_score(svm_clf.predict(X_dev),y_dev)))

print('\n',"{:-^70}".format("<TEST>"))
y_test_pred = svm_clf.predict(X_test)
print(classification_report(y_test,y_test_pred))
# %%
