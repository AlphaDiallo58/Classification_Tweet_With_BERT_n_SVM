from sklearn.metrics import accuracy_score

ytrain_pred_mod_SVM = mod_SVM.predict(X_train)
ytest_pred_mod_SVM = mod_SVM.predict(X_test)


print(accuracy_score(y_train, ytrain_pred_mod_SVM))
print(accuracy_score(y_test, ytest_pred_mod_SVM))