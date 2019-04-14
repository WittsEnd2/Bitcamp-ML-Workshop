from sklearn import datasets, svm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Demographic_Statistics_By_Zip_Code.csv")
x_train = df[['PERCENT MALE']]
y_train = df[['PERCENT RECEIVES PUBLIC ASSISTANCE']]
#train data (everything but last 20)
x_test = x_train[len(x_train)-20:]
y_test = y_train[len(y_train)-20:]
print(x_train, y_train)

#testing data (last 20)
x_train = x_train[:len(x_train)-20]
y_train = y_train[:len(y_train)-20]





regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)

#Get predictions
predictions = regr.predict(x_test)

# Plot outputs
plt.scatter(x_test, y_test,  color='black')
plt.plot(x_test, predictions, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()