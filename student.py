import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("Student.csv")

st.set_option("deprecation.showPyplotGlobalUse", False)

# <ins>Underlined Heading 1</ins>

st.markdown("<h1 style='text-align: center; color: red;'>Google Developer Student Clubs</h1>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: green;'>Student Performance Visualisation Web App</h1>", unsafe_allow_html=True)


st.sidebar.title("Student Data Visualisation")
if st.sidebar.button('Display Raw data'):
  st.subheader("Student Data Set", divider='rainbow')
  st.write(df)
  st.write("Number of Rows: ", df.shape[0])
  st.write("Number of Columns: ", df.shape[1])


plot_list = st.sidebar.multiselect('Select the plots/charts', ('Pie Chart', 'Box Plot', 'Scatter Plot', 'Count Plot'))
selected_column = st.sidebar.selectbox("Select a column for Visualisation:", df.columns)


if "Pie Chart" in plot_list:

    pie_data = df[selected_column].value_counts()
    plt.figure(figsize = (16, 6))
    plt.pie(pie_data, labels = pie_data.index, autopct = '%1.2f%%',startangle = 30,wedgeprops = {'edgecolor' : 'red'} )
    plt.title(selected_column, color = 'r')
    st.pyplot()

if "Count Plot" in plot_list:
  plt.figure(figsize = (20,12))
  sns.countplot(x = selected_column, data = df)
  plt.ylabel("Count", fontsize = 20)
  plt.xlabel(selected_column, fontsize = 20)
  plt.title(selected_column, color = 'r')
  st.pyplot()

if "Scatter Plot" in plot_list:
  fig, ax = plt.subplots()
  ax.scatter(df[selected_column], df["Performance Index"])
  ax.set_xlabel("Performance Index")
  ax.set_ylabel(selected_column)
  ax.set_title(f"Scatter Plot of {selected_column} vs Performance Index", color = "r")
  st.pyplot(fig)

if "Box Plot" in plot_list:
  plt.figure(figsize = (20,8))
  sns.boxplot(x = df[selected_column], y = "Performance Index", data = df)
  plt.ylabel("Performance Index", fontsize = 20)
  plt.xlabel(selected_column, fontsize = 20)
  st.pyplot()



y = df['Performance Index']
X = df.drop(['Performance Index'], axis = 1)

X['Extracurricular Activities'] = X['Extracurricular Activities'].replace({'Yes': 1, 'No': 0})
columns = X.columns

scaler = StandardScaler()
X = pd.DataFrame(X, columns = columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


st.sidebar.write("----------------")
if st.sidebar.button("Scaled DataFrame"):
  st.subheader("DataSet")
  st.write(X)
st.sidebar.write("----------------")

st.sidebar.title("Visualisation after prediction")
#Linear Regression

if st.sidebar.button("Linear Regression"):
  st.title("Linear Regression")

  lr = LinearRegression()
  lr.fit(X_train, y_train)
  train_prediction = lr.predict(X_train)
  test_prediction = lr.predict(X_test)
  train_mse = mean_squared_error(y_train, train_prediction)
  test_mse = mean_squared_error(y_test, test_prediction)
  st.write("Mean squared error for train set = ", train_mse)
  st.write("Mean squared error for test set = ", test_mse)

  plt.figure(figsize = (8,6))
  plt.scatter(y_train, train_prediction, c = 'blue', label = 'Model Prediction')
  plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], c = 'r', linestyle = '--', label = "Perfect Prediction")
  plt.xlabel('True Values')
  plt.ylabel('Predicted Values')
  plt.title('Training vs Predicted Values')
  plt.legend()
  plt.grid(True)
  st.pyplot()

  plt.figure(figsize=(8, 6))
  plt.scatter(y_test, test_prediction, c='blue', label='Model Predictions')
  plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Perfect Prediction')
  plt.xlabel('True Values')
  plt.ylabel('Predicted Values')
  plt.title('Testing True vs Predicted Values')
  plt.legend()
  plt.grid(True)
  st.pyplot()


#SVC Model
if st.sidebar.button("SVC"):
  st.title("Support vector Classifier")

  svc = SVC(kernel = 'linear')
  svc.fit(X_train, y_train)
  train_prediction = svc.predict(X_train)
  test_prediction = svc.predict(X_test)
  train_mse = mean_squared_error(y_train, train_prediction)
  test_mse = mean_squared_error(y_test, test_prediction)
  st.write("Mean squared error for train set = ", train_mse)
  st.write("Mean squared error for test set = ", test_mse)

  plt.figure(figsize = (8,6))
  plt.scatter(y_train, train_prediction, c = 'blue', label = 'Model Prediction')
  plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], c = 'r', linestyle = '--', label = "Perfect Prediction")
  plt.xlabel('True Values')
  plt.ylabel('Predicted Values')
  plt.title('Training vs Predicted Values')
  plt.legend()
  plt.grid(True)
  st.pyplot()

  plt.figure(figsize=(8, 6))
  plt.scatter(y_test, test_prediction, c='blue', label='Model Predictions')
  plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Perfect Prediction')
  plt.xlabel('True Values')
  plt.ylabel('Predicted Values')
  plt.title('Testing True vs Predicted Values')
  plt.legend()
  plt.grid(True)
  st.pyplot()

##RandomForestClassifier
if st.sidebar.button("RandomForestClassifier"):
  st.title("Random Forest Classifier")
  rf_clf = SVC(kernel = 'linear')
  rf_clf.fit(X_train, y_train)
  train_prediction = rf_clf.predict(X_train)
  test_prediction = rf_clf.predict(X_test)
  train_mse = mean_squared_error(y_train, train_prediction)
  test_mse = mean_squared_error(y_test, test_prediction)
  st.write("Mean squared error for train set = ", train_mse)
  st.write("Mean squared error for test set = ", test_mse)

  plt.figure(figsize = (8,6))
  plt.scatter(y_train, train_prediction, c = 'blue', label = 'Model Prediction')
  plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], c = 'r', linestyle = '--', label = "Perfect Prediction")
  plt.xlabel('True Values')
  plt.ylabel('Predicted Values')
  plt.title('Training vs Predicted Values')
  plt.legend()
  plt.grid(True)
  st.pyplot()

  plt.figure(figsize=(8, 6))
  plt.scatter(y_test, test_prediction, c='blue', label='Model Predictions')
  plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Perfect Prediction')
  plt.xlabel('True Values')
  plt.ylabel('Predicted Values')
  plt.title('Testing True vs Predicted Values')
  plt.legend()
  plt.grid(True)
  st.pyplot()

# Logistic Regression
if st.sidebar.button("LogisticRegression"):
  st.title("LogisticRegression")
  lg = SVC(kernel = 'linear')
  lg.fit(X_train, y_train)
  train_prediction = lg.predict(X_train)
  test_prediction = lg.predict(X_test)
  train_mse = mean_squared_error(y_train, train_prediction)
  test_mse = mean_squared_error(y_test, test_prediction)
  st.write("Mean squared error for train set = ", train_mse)
  st.write("Mean squared error for test set = ", test_mse)

  plt.figure(figsize = (8,6))
  plt.scatter(y_train, train_prediction, c = 'blue', label = 'Model Prediction')
  plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], c = 'r', linestyle = '--', label = "Perfect Prediction")
  plt.xlabel('True Values')
  plt.ylabel('Predicted Values')
  plt.title('Training vs Predicted Values')
  plt.legend()
  plt.grid(True)
  st.pyplot()

  plt.figure(figsize=(8, 6))
  plt.scatter(y_test, test_prediction, c='blue', label='Model Predictions')
  plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Perfect Prediction')
  plt.xlabel('True Values')
  plt.ylabel('Predicted Values')
  plt.title('Testing True vs Predicted Values')
  plt.legend()
  plt.grid(True)
  st.pyplot()







  



