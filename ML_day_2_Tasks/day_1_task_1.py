from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
import numpy as np


diabetes = load_diabetes(as_frame=True)

X = diabetes.data
y = diabetes.target

X_train, X_test , y_train , y_test = train_test_split(
    X,y,test_size=0.2,random_state=42)

print("Total Records:", len(X))
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print(y_pred[:5])

coefficients = model.coef_

feature_names = X.columns

coefficient_data = list(zip(feature_names,coefficients))

sorted_coefficients = sorted(coefficient_data,key=lambda x: abs(x[1]),reverse=True)


print("Top 5 Largest Coefficients:")

for name, value in sorted_coefficients[:5]:
    print(name, ":", round(value,2))

ridge_model=Ridge(alpha=1.0)

ridge_model.fit(X_train,y_train)

ridge_pred=ridge_model.predict(X_test)

ridge_coefficients =ridge_model.coef_

print("Ridge Coefficients:")
print(ridge_coefficients)

ridge_rmse=np.sqrt(mean_squared_error(y_test,ridge_pred))

linear_rmse = np.sqrt(mean_squared_error(y_test, y_pred))


lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train,y_train)
lasso_pred = lasso_model.predict(X_test)

lasso_rmse= np.sqrt(mean_squared_error(y_test,lasso_pred))

linear_zero_count = sum(model.coef_ == 0)

ridge_zero_count = sum(ridge_model.coef_ == 0)

lasso_zero_count = sum(lasso_model.coef_ == 0)

print("\nModel Comparison Table")
print("Model             | Test RMSE | Number of Zero Coefficients")

print("Linear Regression |", round(linear_rmse, 2),"    |", linear_zero_count)
print("Ridge Regression  |", round(ridge_rmse, 2), "    |", ridge_zero_count)
print("Lasso Regression  |", round(lasso_rmse, 2), "     |", lasso_zero_count)
