import numpy as np # numerical python
import pandas as pd # data analysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

iris = pd.read_csv('C:/Users/Alex/Desktop/PYTHON/PycharmProjects/Iris/irisData.csv')
# print(iris.head(5))
# print(iris.describe())
# print(iris.shape)
# print(type(iris))

train, test = train_test_split(iris, test_size = 0.2)

# Specify headers
features = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth']
labels = ['flowerClass']

################     OPTIONAL SCALER      ################
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# trainbeforescale = train.as_matrix(features)
# testbeforescale = test.as_matrix(features)
# trainArray = scaler.fit_transform(trainbeforescale)
# testArray = scaler.fit_transform(testbeforescale)

#############      NON SCALED FEATURE VARIABLES      ##################
# Turn pandas dataframe into numpy array
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.as_matrix.html
trainArray = train.as_matrix(features) # training array into a numpy array
trainResults = train.as_matrix(labels) # training results into a numpy array
testArray = test.as_matrix(features)
testResults = test.as_matrix(labels)
# For Feature Extraction
irisArray = iris.as_matrix(features) # dataset into a numpy array
irisResults = iris.as_matrix(labels) # dataset into a numpy array

## Training!
randomForest = RandomForestClassifier(n_estimators=100) # initialize
randomForest.fit(trainArray, trainResults.ravel())

## Testing!
# using the numpy dependent variables, predict the flower classes
testPredictions = randomForest.predict(testArray)
# add predictions back to the data frame,
# so true results and predictions can be compared side-by-side
test = pd.DataFrame(test)
test['predictions'] = testPredictions

# Including probabilities in random forest answer
print("Random Forest classes: %s" %randomForest.classes_) # %s is for strings while %d is for numbers
testProbabilities = randomForest.predict_proba(testArray)
qwerty = np.column_stack((test, testProbabilities))
excel = pd.DataFrame(qwerty)

excel.to_excel('C:/Users/Alex/Desktop/PYTHON/PycharmProjects/Iris/results.xlsx', \
              sheet_name='sheet1', index=False)

# Create a confusion matrix with the actual and predicted values
df_confusion = pd.crosstab(testResults.ravel(), testPredictions, \
                           rownames=['Actual'], colnames=['Predicted'], margins=True)
df_conf_norm = df_confusion / df_confusion.sum(axis=1)

print(df_confusion)
print(df_conf_norm)

# compute accuracy score
# http://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
from sklearn.metrics import accuracy_score
print(accuracy_score(testResults.ravel(), testPredictions))
from sklearn.metrics import classification_report
print(classification_report(testResults.ravel(), testPredictions))
# Lets say there are 12 dogs.  The computer program selects 8 of the 12 dogs.  5 of the 8 are actually dogs.
# Precision is 5/8.  Recall is 5/12.

import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(iris[['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth']])
plt.show()

# Principal Component Analysis
# https://www.dataquest.io/blog/python-vs-r/
from sklearn.cluster import KMeans
kmeans_model = KMeans(n_clusters=5, random_state=1)
good_columns = iris._get_numeric_data().dropna(axis=1)
kmeans_model.fit(good_columns)
labels = kmeans_model.labels_
from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(good_columns)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
plt.show()

# Feature Extraction with Recursive Feature Elimination
from sklearn.feature_selection import RFE
model = RandomForestClassifier(n_estimators=100)
rfe = RFE(model, 1)
fit = rfe.fit(irisArray, irisResults.ravel())
print(("Num Features: %d") % fit.n_features_)
print(("Selected Features: %s") % fit.support_)
print(("Feature Ranking: %s") % fit.ranking_)

sortedFeatures = [x for y,x in sorted(zip(fit.ranking_,features))]
print("Random Forest Classifier (Most to least important): %s" %sortedFeatures)




