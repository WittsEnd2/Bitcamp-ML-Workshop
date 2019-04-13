from sklearn import datasets, svm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Demographic_Statistics_By_Zip_Code.csv")
x_train = df[['PERCENT FEMALE']]
y_train = df[['PERCENT RECEIVES PUBLIC ASSISTANCE']]
x_test = x_train[len(x_train)-20:]
y_test = y_train[len(y_train)-20:]
x_train = x_train[:len(x_train)-20]
y_train = y_train[:len(y_train)-20]
print(x_train, y_train)





regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)
# predictions = np.array()
# predictions = predictions.reshape(-1,1)
# Make predictions using the testing set
jurisdiction_prediction = regr.predict(x_test)
# print(jurisdiction_prediction)

# # Plot outputs
plt.scatter(x_test, y_test,  color='black')
plt.plot(x_test, jurisdiction_prediction, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()