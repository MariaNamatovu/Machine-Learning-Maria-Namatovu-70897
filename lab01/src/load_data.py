import sklearn
import joblib
import os
import time
import numpy as np
from memory_profiler import memory_usage

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


#to import breast cancer dataset
data = load_breast_cancer(return_X_y=True)
#view the data
print(data)

#split the data 70/30 stratified with random state 42
x_train, x_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.3, stratify=data[1], random_state=42)

#train Logistric regression model and tree ensemble and record test accuracy to 4 decimals
Linear_Model = LogisticRegression(max_iter=1000, random_state=42)
TreeEnsemble_Model = RandomForestClassifier(n_estimators=100, random_state=42)

#train the models
linear_model = Linear_Model.fit(x_train, y_train)
tree_ensemble_model = TreeEnsemble_Model.fit(x_train, y_train)

#evaluate the models
linear_model_accuracy = linear_model.score(x_test, y_test)
tree_ensemble_model_accuracy = tree_ensemble_model.score(x_test, y_test)

print(f"Logistic Regression Test Accuracy: {linear_model_accuracy:.4f}")
print(f"Random Forest Test Accuracy: {tree_ensemble_model_accuracy:.4f}")


# measure system cost for both models
#measure median of training time for 5 runs for each model

#training time for logistic regression
training_times_linear = []

for i in range(5):
    start_time = time.time()
    linear_model_train = linear_model.fit(x_train, y_train)
    end_time = time.time()
    training_times_linear.append(end_time - start_time)

#training time for random forest
training_times_tree = []

for i in range(5):
    start_time = time.time()
    tree_ensemble_model_train = tree_ensemble_model.fit(x_train, y_train)
    end_time = time.time()
    training_times_tree.append(end_time - start_time)


#measure median of training time for 5 runs for each model

median_training_time_linear = np.median(training_times_linear)
median_training_time_tree = np.median(training_times_tree)

print("Median training time for logistic regression: {:.4f} seconds".format(median_training_time_linear))
print("Median training time for random forest: {:.4f} seconds".format(median_training_time_tree))

#single sample inference: repeat 100 times and record median for each model
inference_times_linear = []
for i in range (100):
    start_time = time.time()
    linear_model_inference = linear_model.predict(x_test[0].reshape(1, -1))
    end_time = time.time()
    inference_times_linear.append(end_time - start_time)


inference_times_tree = []
for i in range (100):
    start_time = time.time()
    tree_ensemble_model_inference = tree_ensemble_model.predict(x_test[0].reshape(1, -1))
    end_time = time.time()
    inference_times_tree.append(end_time - start_time)

#median of inference time for each model
median_inference_time_linear = np.median(inference_times_linear)
median_inference_time_tree = np.median(inference_times_tree)

print("Median linear inference time: {:.4f} seconds".format(median_inference_time_linear))
print("Median tree ensemble inference time: {:.4f} seconds".format(median_inference_time_tree))

#model size: save with joblib.dump to results/model.joblib and record size in bytes and KB for each model


joblib.dump(linear_model, 'lab01/results/linear_model.joblib')
joblib.dump(tree_ensemble_model, 'lab01/results/tree_ensemble_model.joblib')

#record the size of each model in bytes and KB
linear_model_size_bytes = os.path.getsize('lab01/results/linear_model.joblib')
tree_ensemble_model_size_bytes = os.path.getsize('lab01/results/tree_ensemble_model.joblib')

linear_model_size_kb = linear_model_size_bytes / 1024
tree_ensemble_model_size_kb = tree_ensemble_model_size_bytes / 1024

print("Linear model size: {:.4f} bytes, {:.4f} KB".format(linear_model_size_bytes, linear_model_size_kb))
print("Tree ensemble model size: {:.4f} bytes, {:.4f} KB".format(tree_ensemble_model_size_bytes, tree_ensemble_model_size_kb))

#measure peak memory for training and inference for each model using memory_profiler


#peak memory for training logistic regression
peak_memory_training_linear = max(memory_usage((linear_model.fit, (x_train, y_train)), max_usage=True))


#peak memory for training random forest
peak_memory_training_tree = max(memory_usage((tree_ensemble_model.fit, (x_train, y_train)), max_usage=True))

#peak memory for inference logistic regression
peak_memory_inference_linear = max(memory_usage((linear_model.predict, (x_test[0].reshape(1, -1),)), max_usage=True))

#peak memory for inference random forest
peak_memory_inference_tree = max(memory_usage((tree_ensemble_model.predict, (x_test[0].reshape(1, -1), )), max_usage=True))

#printing peak memory
print("Peak memory for training logistic regression: {:.4f} MB".format(peak_memory_training_linear))
print("Peak memory for training random forest: {:.4f} MB".format(peak_memory_training_tree))
print("Peak memory for inference logistic regression: {:.4f} MB".format(peak_memory_inference_linear))
print("Peak memory for inference random forest: {:.4f} MB".format(peak_memory_inference_tree))