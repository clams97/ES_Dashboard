# Data manipulation dependencies
import dash
import math
import datetime
import dash_table
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
from pandas import DataFrame
import plotly.graph_objects as go
#import dash_building_blocks as dbb
import dash_core_components as dcc
import dash_html_components as html
#from matplotlib import pyplot as plt
import dash_bootstrap_components as dbc
#from plotly.subplots import make_subplots
from dash.dependencies import Input, Output, State

# Clustering dependencies
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

# Feature Selection dependencies
from sklearn.feature_selection import SelectKBest # Selection method
from sklearn.feature_selection import f_regression # Score metric
from sklearn.feature_selection import mutual_info_regression # Score metric
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

# Forecasting dependencies
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor




# Loading the clean data data frame developed in project 1
#df_load = pd.read_csv("C:/Users/andre/Desktop/Project_2/IST_Total_Hourly_Clean.csv")
df_load = pd.read_csv("IST_Total_Hourly_Clean.csv")
df = pd.DataFrame()
df["Date"]=pd.to_datetime(df_load["Date"]) # Convert Date into datetime type
df["Power [KW]"] = df_load["Power_kW"]
df["Temperature [Cº]"] = df_load["Temperature [Cº]"]
df["Solar Radiation [w/m2]"] = df_load["Solar Radiation [w/m2]"]
df["Pressure [mbar]"] = df_load["Pressure [mbar]"]
df["Wind Speed [m/s]"] = df_load["Wind Speed [m/s]"]
df["Wind Gust [m/s]"] = df_load["Wind Gust [m/s]"]
df["Rain [mm/h]"] = df_load["rain_mm/h"]
df["HR"] = df_load["HR"]
df["Rain Day"] = df_load["Rain Day"]
df["Power-1"] = df["Power [KW]"].shift(1) # Previous hour consumption
df["Year"] = df["Date"].dt.strftime('%Y')
df["Holiday Day"] = df_load["Holiday"]
df = df.set_index("Date", drop=False) #!!!!!!!!! drop = True
df["Month"] = df.index.month
df["Month Name"] = df['Date'].dt.strftime("%B")
df["Week Day"] = df.index.dayofweek
df["Hour"] = df.index.hour
df["Date Day"] = df.index.date
df.dropna()
#df['Mon_Year'] = df['Date'].dt.strftime('%b-%Y')
#df['Mon_Year'] = df['Date'].dt.strftime('%b')   # the result is object/string unlike `.dt.to_period('M')` that retains datetime data type.
#available_years = df.index.year.unique()



##############################################_____IMAGES LOAD_____######################################################
PLOTLY_LOGO = Image.open("assets/logo.png")
menu = Image.open("assets/menu_2.png")







###########################################_____DATA PREPARATION_____###################################################

# Round numbers Table for Total table
df_tab=df[df.columns[0:9].values.tolist()].copy()
df_tab[df.columns[1:9].values.tolist()] = df_tab[df.columns[1:9].values.tolist()].apply(lambda x: x.round(2), axis=1)
# Definig a function to generate a table from a data frame
def generate_table(dataframe, max_rows=10):
    tab_total = dash_table.DataTable(
        data=dataframe.to_dict('records'),
        columns=[{'id': c, 'name': c} for c in dataframe.columns[0:8]],
        virtualization=True,
        fixed_rows={'headers': True},
        style_cell={'minWidth': 95, 'width': 95, 'maxWidth': 95},
        style_table={'height': 400}  # default is 500
    )
    return tab_total


# Definig a function to generate a table from a data frame
#def generate_table(dataframe, max_rows=10):
#    return html.Table([
#        html.Thead(
#            html.Tr([html.Th(col) for col in dataframe.columns])
#        ),
#        html.Tbody([
#            html.Tr([
#                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
#            ]) for i in range(min(len(dataframe), max_rows))
#        ])
#    ])


def generate_total_plot(start_date, end_date, value):
    start_date = pd.to_datetime(start_date)
    start_date = datetime.datetime(start_date.year, start_date.month, start_date.day, 0)
    end_date = pd.to_datetime(end_date)
    end_date = datetime.datetime(end_date.year, end_date.month, end_date.day, 0)
    df_1 = df.loc[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    s=[]
    for i in value:
        s.append(i.split("_")[0])
    fig1 = px.line(df_1, x="Date", y=s)
    return fig1



# Definig a function to generate a pie chart from a column
def generate_pie_chart(column):
    fig2 = px.pie(df, values=column, names='Year')
    return fig2


# Definig a function to generate a graph bar from a column
df_bar_total = df.copy()
df_bar_total.index=df_bar_total.index.map(lambda t: t.replace(year=2017))
df_bar_17_total = df_bar_total.loc[df_bar_total["Year"] == "2017"]
df_bar_18_total = df_bar_total.loc[df_bar_total["Year"] == "2018"]
def generate_graph_bar(column):
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=df_bar_17_total["Month Name"], y=df_bar_17_total[column], histfunc="avg", name="2017"))
    fig3.add_trace(go.Histogram(x=df_bar_18_total["Month Name"], y=df_bar_18_total[column], histfunc="avg", name="2018"))
    fig3.update_xaxes(ticklabelmode="period", dtick="M1")#, tickformat="%b")
    fig3.update_layout(barmode='group', title = 'Data displayed for each month:')
    return fig3



##############################################_____CLUSTERING_____######################################################
# Dataframe for the clustring section
df_cluster=df[df.columns[1:10].values.tolist() + df.columns[11:14].values.tolist() + df.columns[15:18].values.tolist()]
df_cluster = df_cluster.set_index("Date Day", drop=True)
df_cluster['Year'] = df_cluster["Year"].astype(float)


# Plot showing the optimal
#Nc = range(1, 20)
#kmeans = [KMeans(n_clusters=i) for i in Nc]
#score = [kmeans[i].fit(df_cluster).score(df_cluster) for i in range(len(kmeans))]

#print(f"\nFrom this graph we observe that for a number of clusters bigger than 4 there are no major improvemnts.   \n")

#df_kmeans=pd.DataFrame({"Number of Clusters":Nc, "Score":score})
#fig4 = px.line(df_kmeans, x="Number of Clusters", y="Score")
#fig4.update_layout(
#    title={
#        'text': "Performance Based on the Number of Clusters",
#        'y':0.95,
#        'x':0.5,
#        'xanchor': 'center',
#        'yanchor': 'top'})


def kmeans_pie(column_1, num_clusters=4):
  # Kmeans model
  model = KMeans(num_clusters).fit(df_cluster)
  pred = model.labels_
  # We add the respective cluster number to the dataframe
  df_cluster['Cluster'] = pred
  #Pie bar for the clusters
  fig5 = px.pie(df_cluster, values=column_1, names='Cluster')
  fig5.update_layout(
      title={
          'text': "Data Density by Clusters",
          'y': 0.95,
          'x': 0.5,
          'xanchor': 'center',
          'yanchor': 'top'})
  return fig5

def kmeans_scatter2d(column_1, column_2, num_clusters=4):
  model = KMeans(num_clusters).fit(df_cluster)
  pred = model.labels_
  # We add the respective cluster number to the dataframe
  df_cluster['Cluster']=pred
  fig6 = px.scatter(df_cluster, x=column_1, y=column_2, color="Cluster")#, size="Year", hover_data=['petal_width'])
  fig6.update_layout(coloraxis_colorbar=dict(title="Cluster Number", dtick=1, thicknessmode="pixels", thickness=20, yanchor="top",y=1))
  fig6.update_layout(
      title={
          'text': "Clusters 2D Visulaization",
          'y': 0.95,
          'x': 0.5,
          'xanchor': 'center',
          'yanchor': 'top'})
  return fig6

def kmeans_scatter3d(column_1, column_2, column_3, num_clusters=4):
  model = KMeans(num_clusters).fit(df_cluster)
  pred = model.labels_
  # We add the respective cluster number to the dataframe
  df_cluster['Cluster']=pred
  fig7 = px.scatter_3d(df_cluster, x=column_1, y=column_2, z=column_3, color="Cluster", size_max=10, opacity=0.8)#, size="Year", hover_data=['petal_width'])
  fig7.update_layout(coloraxis_colorbar=dict(title="Cluster Number", dtick=1, thicknessmode="pixels", thickness=20, yanchor="top",y=1))
  fig7.update_layout(title={'text': "Clusters 3D Visualization", 'y': 0.95,'x': 0.5,'xanchor': 'center','yanchor': 'top'})
  return fig7


# Identifying Daily Patterns
df_dailypatt=df[["Date Day", "Power [KW]", "Hour"]]

#Create a pivot table
df_pivot = df_dailypatt.pivot_table(index="Date Day" , columns='Hour', values="Power [KW]")
df_pivot = df_pivot.dropna()

sillhoute_scores = []
n_cluster_list = np.arange(2, 10).astype(int)

X = df_pivot.values.copy()

# Very important to scale!
sc = MinMaxScaler()
X = sc.fit_transform(X)

for n_cluster in n_cluster_list:
    kmeans = KMeans(n_clusters=n_cluster)
    cluster_found = kmeans.fit_predict(X)
    sillhoute_scores.append(silhouette_score(X, kmeans.labels_))

fig8 = px.line(x=n_cluster_list, y=sillhoute_scores)
fig8.update_layout(title={'text': "Sillhoute Scores", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})

kmeans = KMeans(n_clusters=3)
cluster_found = kmeans.fit_predict(X)
cluster_found_sr = pd.Series(cluster_found, name='cluster')
df_pivot = df_pivot.set_index(cluster_found_sr, append=True )

fig9=go.Figure()
cluster_values = sorted(df_pivot.index.get_level_values('cluster').unique())
for cluster in cluster_values:
  #df_pivot.xs(cluster, level=1).T.apply(lambda x: fig10.add_trace(go.scatter(x, mode="lines", axis=0)))
  fig9.add_trace(go.Scatter(x=df_pivot.xs(cluster, level=1).median().to_frame().reset_index()["Hour"], y=df_pivot.xs(cluster, level=1).median().to_frame().reset_index()[0], mode="lines"))
fig9.update_layout(title={'text': "Clusters 3D Visualization", 'y': 0.95,'x': 0.5,'xanchor': 'center','yanchor': 'top'})




#def kmeans_cluster(column, num_clusters=4):
#  model = KMeans(num_clusters).fit(df_cluster)
#  pred = model.labels_
#  # We add the respective cluster number to the dataframe
#  df_cluster['Cluster']=pred
#  #Pie bar for the clusters
#  df_bar_kmeans = df_cluster.copy()
#  df_bar_kmeans.index=df_bar_kmeans.index.map(lambda t: t.replace(year=2017))
#  df_bar_17_kmeans = df_bar_kmeans.loc[df_bar_kmeans["Year"] == "2017"]
#  df_bar_18_kmeans = df_bar_kmeans.loc[df_bar_kmeans["Year"] == "2018"]
#  fig5 = px.pie(df_bar_17_kmeans, values=column, names='Cluster')
#  fig6 = px.pie(df_bar_18_kmeans, values=column, names='Cluster')
#  return pred, fig5, fig6






##########################################_____FEATURES SELECTION_____##################################################

#df_fs = df # Data without outliers
#df_fs = df_fs.drop("Date Day", axis=1)
#df_fs = df_fs.drop("Date", axis=1)
#df_fs = df_fs.drop("Month Name", axis=1)
#df_fs['Year'] = df_fs["Year"].astype(float)
#df_fs = df_fs.drop("Date Day", axis=1)
#df_fs=df_fs.dropna()

# Copying the numerical values of the data frame to a matrix
#X=df_fs.values

# Define input and outputs
#Y=X[:,0] # Output of the model [Power_KW]
#X=X[:,[1,2,3,4,5,6,7,8,9,10,11,12]] # Input of the model - the rest of the features

######################____KBEST___##########################____f_Regression
#features_fr=SelectKBest(k=5,score_func=f_regression) # Test different k number of features, uses f-test ANOVA

# k=2 - Means that we want to select 2 features from the features available.
# score_func - Function taking two arrays X and y, and returning a pair of arrays (scores, pvalues) or a single array with scores.
# Types of score_func available:
#      -f_classif: ANOVA F-value between label/feature for classification tasks.
#      -mutual_info_classif: Mutual information for a discrete target.
#      -chi2: Chi-squared stats of non-negative features for classification tasks.
#      -f_regression: F-value between label/feature for regression tasks.
#      -mutual_info_regression: Mutual information for a continuous target.
#      -SelectPercentile: Select features based on percentile of the highest scores.
#      -SelectFpr: Select features based on a false positive rate test.
#      -SelectFdr: Select features based on an estimated false discovery rate.
#      -SelectFwe: Select features based on family-wise error rate.
#      -GenericUnivariateSelect: Univariate feature selector with configurable mode.

#fit_fr=features_fr.fit(X,Y) #calculates the f_regression of the features. In other words, it computes the correlation between the features and the output [Power_KW]
#features_fr_results=fit_fr.transform(X)
# Bar plot to better understand the dominant features.
#fig10 = px.bar(x=[i for i in range (len(fit_fr.scores_))], y=fit_fr.scores_, title="KBest F_Regression")
#fig10.update_layout(title={'text': "Data Distribution by Clusters - KBest f_Regression", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})



######################____KBEST___##########################____mutual_info_Regression
#features_mir=SelectKBest(k=5,score_func=mutual_info_regression)
#fit_mir=features_mir.fit(X,Y) #calculates the f_regression of the features. In other words, it computes the correlation between the features and the output [Power_KW]

#features_mir_results=fit_mir.transform(X)


# Bar plot to better understand the dominant features.
#fig11 = px.bar(x=[i for i in range (len(fit_mir.scores_))], y=fit_mir.scores_, title="KBest Mutual_Info_Regression")
#fig11.update_layout(title={'text': "Data Distribution by Clusters - KBest Mutual_Info_Regression", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})


######################____LINEAR REGRESSION___##########################
#model_LR=LinearRegression() # LinearRegression Model as Estimator
#rfe3_LR=RFE(model_LR,n_features_to_select=3)# using 3 features
#rfe4_LR=RFE(model_LR,n_features_to_select=4) # using 4 features
#rfe5_LR=RFE(model_LR,n_features_to_select=5) # using 5 features

#fit3_LR = rfe3_LR.fit(X,Y)
#fit4_LR = rfe4_LR.fit(X,Y)
#fit5_LR = rfe5_LR.fit(X,Y)

#fig12 = px.bar(x=[i for i in range(len(fit5_LR.ranking_))], y=fit5_LR.ranking_, title="RFE - Linear Regression_")
#fig12.update_layout(title={'text': "Data Distribution by Clusters - Linear Regression ", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
######################____RandomForestRegressor___##########################
#model_RFR = RandomForestRegressor()
#RFR=model_RFR.fit(X, Y)
#fig13 = px.bar(x=[i for i in range(len(RFR.feature_importances_))], y=RFR.feature_importances_, title="KBest Mutual_Info_Regression")
#fig13.update_layout(title={'text': "Data Distribution by Clusters - Random Forest Regressor", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})







##############################################_____FORECASTING_____#####################################################
df_model=df_fs.copy()
df_model=df_model.drop(columns=["Pressure [mbar]","Holiday Day","HR","Wind Speed [m/s]","Wind Gust [m/s]","Rain [mm/h]","Rain Day","Year","Month"])

#Input X and Output Y
X=df_model.values
Y=X[:,0]
X=X[:,[1,2,3,4,5]]

#Split training and test data
X_train, X_test, y_train, y_test = train_test_split(X,Y)

###############_________________Linear Regression___________________##################

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train,y_train)

# Make predictions using the testing set
y_pred_LR = regr.predict(X_test)

# Prediction Metrics
MAE_LR=metrics.mean_absolute_error(y_test,y_pred_LR)
MSE_LR=metrics.mean_squared_error(y_test,y_pred_LR)
RMSE_LR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y_test)

#Visualization
fig14 = px.line(y=y_test[1:200])
fig14.add_trace(go.Scatter(y=y_pred_LR[1:200], mode="lines"))
fig14.update_layout(title={'text': "Actual vs Predicted Comsumption", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
fig15=px.scatter(x=y_test,y=y_pred_LR)
fig15.update_layout(title={'text': "y_test vs y_predicted", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})

df_metrics_LR=pd.DataFrame({'MAE': [MAE_LR], 'MSE': [MSE_LR], "RMSE":[RMSE_LR], "cvRMSE":[cvRMSE_LR]})
#tab_total_LR = dash_table.DataTable(
#        data=df_metrics.to_dict('records'),
#        columns=[{'id': c, 'name': c} for c in df_metrics],
#        virtualization=True,
#        fixed_rows={'headers': True},
#        style_cell={'minWidth': 95, 'width': 95, 'maxWidth': 95},
#        style_table={'height': 100}  # default is 500
#    )


###############_________________Support Vector Regressor___________________##################
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_SVR = sc_X.fit_transform(X_train)
y_train_SVR = sc_y.fit_transform(y_train.reshape(-1,1))

regr = SVR(kernel='rbf')
regr.fit(X_train_SVR,y_train_SVR)

y_pred_SVR = regr.predict(sc_X.fit_transform(X_test))
y_test_SVR=sc_y.fit_transform(y_test.reshape(-1,1))
y_pred_SVR2=sc_y.inverse_transform(y_pred_SVR)

# Prediction Metrics
MAE_SVR=metrics.mean_absolute_error(y_test,y_pred_SVR2)
MSE_SVR=metrics.mean_squared_error(y_test,y_pred_SVR2)
RMSE_SVR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_SVR2))
cvRMSE_SVR=RMSE_SVR/np.mean(y_test)

#Visualization
fig16 = px.line(y=y_test[1:200])
fig16.add_trace(go.Scatter(y=y_pred_SVR2[1:200], mode="lines"))
fig16.update_layout(title={'text': "Actual vs Predicted Comsumption", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
fig17=px.scatter(x=y_test,y=y_pred_SVR2)
fig17.update_layout(title={'text': "y_test vs y_predicted", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})

df_metrics_SVR=pd.DataFrame({'MAE': [MAE_SVR], 'MSE': [MSE_SVR], "RMSE":[RMSE_SVR], "cvRMSE":[cvRMSE_SVR]})
#tab_total_SVR = dash_table.DataTable(
#        data=df_metrics.to_dict('records'),
#        columns=[{'id': c, 'name': c} for c in df_metrics],
#        virtualization=True,
#        fixed_rows={'headers': True},
#        style_cell={'minWidth': 95, 'width': 95, 'maxWidth': 95},
#        style_table={'height': 100}  # default is 500
#    )

###############_________________Decision Tree Regressor___________________##################
# Create Regression Decision Tree object
DT_regr_model = DecisionTreeRegressor()

# Train the model using the training sets
DT_regr_model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred_DT = DT_regr_model.predict(X_test)

# Prediction Metrics
MAE_DT=metrics.mean_absolute_error(y_test,y_pred_DT)
MSE_DT=metrics.mean_squared_error(y_test,y_pred_DT)
RMSE_DT= np.sqrt(metrics.mean_squared_error(y_test,y_pred_DT))
cvRMSE_DT=RMSE_DT/np.mean(y_test)

#Visualization
fig18 = px.line(y=y_test[1:200])
fig18.add_trace(go.Scatter(y=y_pred_DT[1:200], mode="lines"))
fig18.update_layout(title={'text': "Actual vs Predicted Comsumption", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
fig19=px.scatter(x=y_test,y=y_pred_DT)
fig19.update_layout(title={'text': "y_test vs y_predicted", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})

df_metrics_DT=pd.DataFrame({'MAE': [MAE_DT], 'MSE': [MSE_DT], "RMSE":[RMSE_DT], "cvRMSE":[cvRMSE_DT]})
#tab_total_DT = dash_table.DataTable(
#        data=df_metrics.to_dict('records'),
#        columns=[{'id': c, 'name': c} for c in df_metrics],
#        virtualization=True,
#        fixed_rows={'headers': True},
#        style_cell={'minWidth': 95, 'width': 95, 'maxWidth': 95},
#        style_table={'height': 100}  # default is 500
#    )

###############_________________Random forest___________________##################
parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 200,
              'min_samples_split': 15,
              'max_features': 'sqrt',
              'max_depth': 20,
              'max_leaf_nodes': None}
RF_model = RandomForestRegressor(**parameters)
RF_model.fit(X_train, y_train)
y_pred_RF = RF_model.predict(X_test)

# Prediction Metrics
MAE_RF=metrics.mean_absolute_error(y_test,y_pred_RF)
MSE_RF=metrics.mean_squared_error(y_test,y_pred_RF)
RMSE_RF= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y_test)

#Visualization
fig20 = px.line(y=y_test[1:200])
fig20.add_trace(go.Scatter(y=y_pred_RF[1:200], mode="lines"))
fig20.update_layout(title={'text': "Actual vs Predicted Comsumption", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
fig21=px.scatter(x=y_test,y=y_pred_RF)
fig21.update_layout(title={'text': "y_test vs y_predicted", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})

df_metrics_RF=pd.DataFrame({'MAE': [MAE_RF], 'MSE': [MSE_RF], "RMSE":[RMSE_RF], "cvRMSE":[cvRMSE_RF]})
#tab_total_RF = dash_table.DataTable(
#        data=df_metrics.to_dict('records'),
#        columns=[{'id': c, 'name': c} for c in df_metrics],
#        virtualization=True,
#        fixed_rows={'headers': True},
#        style_cell={'minWidth': 95, 'width': 95, 'maxWidth': 95},
#        style_table={'height': 100}  # default is 500
#    )


scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)

# Now apply the transformations to the data:
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)



###############_________________Gradient Boosting___________________##################
GB_model = GradientBoostingRegressor(validation_fraction=0.2, n_iter_no_change=5, tol=0.01)
GB_model.fit(X_train, y_train)
y_pred_GB =GB_model.predict(X_test)

# Prediction Metrics
MAE_GB=metrics.mean_absolute_error(y_test,y_pred_GB)
MSE_GB=metrics.mean_squared_error(y_test,y_pred_GB)
RMSE_GB= np.sqrt(metrics.mean_squared_error(y_test,y_pred_GB))
cvRMSE_GB=RMSE_GB/np.mean(y_test)


#Visualization
fig22 = px.line(y=y_test[1:200])
fig22.add_trace(go.Scatter(y=y_pred_GB[1:200], mode="lines"))
fig22.update_layout(title={'text': "Actual vs Predicted Comsumption", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
fig23=px.scatter(x=y_test,y=y_pred_GB)
fig23.update_layout(title={'text': "y_test vs y_predicted", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})

df_metrics_GB=pd.DataFrame({'MAE': [MAE_GB], 'MSE': [MSE_GB], "RMSE":[RMSE_GB], "cvRMSE":[cvRMSE_GB]})
#tab_total_GB = dash_table.DataTable(
#        data=df_metrics.to_dict('records'),
#        columns=[{'id': c, 'name': c} for c in df_metrics],
#        virtualization=True,
#        fixed_rows={'headers': True},
#        style_cell={'minWidth': 95, 'width': 95, 'maxWidth': 95},
#        style_table={'height': 100}  # default is 500
#    )

###############_________________Extreme Gradient Boosting___________________##################
XGB_model = XGBRegressor(validation_fraction=0.2, n_iter_no_change=5, tol=0.01)
XGB_model.fit(X_train, y_train)
y_pred_XGB =XGB_model.predict(X_test)
# Prediction Metrics
MAE_XGB=metrics.mean_absolute_error(y_test,y_pred_XGB)
MSE_XGB=metrics.mean_squared_error(y_test,y_pred_XGB)
RMSE_XGB= np.sqrt(metrics.mean_squared_error(y_test,y_pred_XGB))
cvRMSE_XGB=RMSE_XGB/np.mean(y_test)

#Visualization
fig24 = px.line(y=y_test[1:200])
fig24.add_trace(go.Scatter(y=y_pred_XGB[1:200], mode="lines"))
fig24.update_layout(title={'text': "Actual vs Predicted Comsumption", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
fig25=px.scatter(x=y_test,y=y_pred_XGB)
fig25.update_layout(title={'text': "y_test vs y_predicted", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})

df_metrics_XGB=pd.DataFrame({'MAE': [MAE_XGB], 'MSE': [MSE_XGB], "RMSE":[RMSE_XGB], "cvRMSE":[cvRMSE_XGB]})
#tab_total_XGB = dash_table.DataTable(
#        data=df_metrics.to_dict('records'),
#        columns=[{'id': c, 'name': c} for c in df_metrics],
#        virtualization=True,
#        fixed_rows={'headers': True},
#        style_cell={'minWidth': 95, 'width': 95, 'maxWidth': 95},
#        style_table={'height': 100}  # default is 500
#    )

###############_________________Bootstrapping___________________##################
BT_model = BaggingRegressor()
BT_model.fit(X_train, y_train)
y_pred_BT =BT_model.predict(X_test)

# Prediction Metrics
MAE_BT=metrics.mean_absolute_error(y_test,y_pred_BT)
MSE_BT=metrics.mean_squared_error(y_test,y_pred_BT)
RMSE_BT= np.sqrt(metrics.mean_squared_error(y_test,y_pred_BT))
cvRMSE_BT=RMSE_BT/np.mean(y_test)

#Visualization
fig26 = px.line(y=y_test[1:200])
fig26.add_trace(go.Scatter(y=y_pred_BT[1:200], mode="lines"))
fig26.update_layout(title={'text': "Actual vs Predicted Comsumption", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
fig27=px.scatter(x=y_test,y=y_pred_BT)
fig27.update_layout(title={'text': "y_test vs y_predicted", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})

df_metrics_BT=pd.DataFrame({'MAE': [MAE_BT], 'MSE': [MSE_BT], "RMSE":[RMSE_BT], "cvRMSE":[cvRMSE_BT]})
#tab_total_BT = dash_table.DataTable(
#        data=df_metrics.to_dict('records'),
#        columns=[{'id': c, 'name': c} for c in df_metrics],
#        virtualization=True,
#        fixed_rows={'headers': True},
#        style_cell={'minWidth': 95, 'width': 95, 'maxWidth': 95},
#        style_table={'height': 100}  # default is 500
#    )

###############_________________SKlearn NN___________________##################

#NN_model = MLPRegressor(hidden_layer_sizes=(10,10,10,10))
#NN_model.fit(X_train,y_train)
#y_pred_NN = NN_model.predict(X_test)

# Prediction Metrics
#MAE_NN =metrics.mean_absolute_error(y_test,y_pred_NN)
#MSE_NN =metrics.mean_squared_error(y_test,y_pred_NN)
#RMSE_NN = np.sqrt(metrics.mean_squared_error(y_test,y_pred_NN))
#cvRMSE_NN =RMSE_NN/np.mean(y_test)
##Visualization
#fig28 = px.line(y=y_test[1:200])
#fig28.add_trace(go.Scatter(y=y_pred_NN[1:200], mode="lines"))
#fig28.update_layout(title={'text': "Actual vs Predicted Comsumption", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
#fig29=px.scatter(x=y_test,y=y_pred_NN)
#fig29.update_layout(title={'text': "y_test vs y_predicted", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})

#df_metrics_SK_NN=pd.DataFrame({'MAE': [MAE_NN], 'MSE': [MSE_NN], "RMSE":[RMSE_NN], "cvRMSE":[cvRMSE_NN]})









app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

theme ={
    'dark': True,
    'detail': '#007439',
    'primary': '#00EA64',
    'secondary': '#6E6E6E',
}


#navbar = dbc.Navbar(
#    children=[
#        dbc.Button("Menu", outline=True, color="primary", className="mr-1", id="btn_sidebar"),
        #dbc.NavItem(dbc.NavLink("Home", href="#")),
        #dbc.DropdownMenu(
        #    children=[
        #        #dbc.DropdownMenuItem("More pages", header=True),
        #        dbc.DropdownMenuItem("Energy Data", href="#"),
        #        dbc.DropdownMenuItem("Clustering", href="#"),
        #        dbc.DropdownMenuItem("Features", href="#"),
        #        dbc.DropdownMenuItem("Forecasting", href="#"),
        #    ],
        #    nav=True,
        #    in_navbar=True,
        #    label="More",
        #),
#    ],
    # brand="Brand",
#    brand_href="/",
#    color="dark",
#    dark=True,
#    fluid=True,
#)


navbar = dbc.Navbar(
    [
        dbc.Button(html.Img(src=menu, height="30px"), className="mr-1", id="btn_sidebar"
                   ), # outline="True", color="secondary"
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(html.Img(src=PLOTLY_LOGO, height="40px")),
                        width={"size": 3, "order": "last", "offset": 3},
                    #dbc.Col(dbc.NavbarBrand("Navbar", className="ml-2")),
                    )
                ],
                align="center",
                no_gutters=True,
            ),
            href="/",

        ),
        dbc.NavbarToggler(id="navbar-toggler"),
        #dbc.Button("Menu", outline=True, color="primary", className="mr-1", id="btn_sidebar"),
        #dbc.Collapse(search_bar, id="navbar-collapse", navbar=True),
    ],
    color="dark",
    dark=True,
    fixed="top",
)

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 62.5,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "1rem 1rem",
    "background-color": "#f8f9fa",
}

SIDEBAR_HIDEN = {
    "position": "fixed",
    "top": 62.5,
    "left": "-16rem",
    "bottom": 0,
    "width": "16rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0rem 0rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "transition": "margin-left .5s",
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "margin-top":"4rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE1 = {
    "transition": "margin-left .5s",
    "margin-left": "2rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "margin-top":"4rem",
    "background-color": "#f8f9fa",
}

sidebar = html.Div(
    [
        html.H2("Energy Forecasting", className="display-5",), # Sidebar Title
        html.Hr(), # Empty line
        #html.P("Number of students per education Level", className="lead"), # Paragraf
        dbc.Nav(
            [
                dbc.NavLink("Exploratory Data Analysis", href="/", active="exact"),
                #dbc.NavLink("Exploratory Data Analysis", href="/page-1", active="exact"),
                #dbc.NavLink("Clustering", href="/page-2", active="exact"),
                #dbc.NavLink("Features", href="/page-3", active="exact"),
                #dbc.NavLink("Forecasting", href="/page-4", active="exact"),
                #html.Hr(), # Empty line
                #dbc.NavLink("About us", href="/page-5", active="exact"),
                #dbc.NavLink("Page 6", href="/page-6", active="exact"),
                dbc.DropdownMenu(
                    nav=True,
                    label="Clustering",
                    #bs_size="lsm",
                    #className="mb-3",
                    direction="down",
                    #color="primary",
                    right=False,
                    in_navbar=True,
                    children=[
                        #dbc.DropdownMenuItem("More pages", header=True),
                        dbc.DropdownMenuItem("Cluster Analysis", href="/page-2", id="Cluster_Analysis_dd_sb"),
                        dbc.DropdownMenuItem(divider=True),
                        dbc.DropdownMenuItem("Identifying daily Patterns", href="/page-3", id="Identifying_daily_Patterns_dd_sb"),
                    ],
                    id="dropdownmenu_sb_1",
                ),

                dbc.DropdownMenu(
                    nav=True,
                    label="Feature Selection",
                    #bs_size="lsm",
                    #className="mb-3",
                    direction="down",
                    #color="primary",
                    right=False,
                    in_navbar=True,
                    children=[
                        #dbc.DropdownMenuItem("More pages", header=True),
                        dbc.DropdownMenuItem("Feature Selection Methods", header=True),
                        dbc.DropdownMenuItem("Filter Methods", href="/page-4", id="Filter_Methods_dd_sb"),
                        dbc.DropdownMenuItem("Wrapper Methods", href="/page-5", id="Wrapper_Methods_dd_sb"),
                        dbc.DropdownMenuItem("Embedded Methods", href="/page-6", id="Embedded_Methods_dd_sb"),
                        #dbc.DropdownMenuItem(divider=True),
                        #dbc.DropdownMenuItem("Other Feature Extraction", href="/page-7", id="Other_Feature_Extraction_dd_sb"),
                    ],
                    id="dropdownmenu_sidebar_2",
                ),

                dbc.DropdownMenu(
                    nav=True,
                    label="Forecasting",
                    #bs_size="lsm",
                    #className="mb-3",
                    direction="down",
                    #color="primary",
                    right=False,
                    in_navbar=True,
                    children=[
                        #dbc.DropdownMenuItem("More pages", header=True),
                        dbc.DropdownMenuItem("Forecast Modeling", header=True),
                        #dbc.DropdownMenuItem("Training and Test Data", href="/page-11", id="Training_and_Test_Data_dd_sb"),
                        dbc.DropdownMenuItem("Training Models", href="/page-7", id="Training_Models_dd_sb"),
                        dbc.DropdownMenuItem(divider=True),
                        dbc.DropdownMenuItem("Neural Networks", href="/page-8", id="Neural_Networks_dd_sb"),
                    ],
                    id="dropdownmenu_sidebar_3",
                ),

            ],
            vertical=True, # Defines the menu as vertical instead of horizontal
            pills=True, # Is what giver the blue box around the option in the side manu
        ),


        html.Hr(), # Empty line
        dbc.Nav(
            [
                dbc.NavLink("About us", href="/page-10", active="exact"),
                dbc.NavLink("Cookie Policy", href="/page-11", active="exact"),
                dbc.NavLink("Contacts", href="/page-12", active="exact"),
                dbc.NavLink("Help", href="/page-13", active="exact"),
            ],

            vertical=True, # Defines the menu as vertical instead of horizontal
            pills=True, # Is what giver the blue box around the option in the side manu
            #fill=True,
        ),

    ],
    id="sidebar",
    style=SIDEBAR_STYLE,
)

content = html.Div(

    id="page-content",
    style=CONTENT_STYLE)

app.layout = html.Div(
    [
        dcc.Store(id='side_click'),
        dcc.Location(id="url"),
        navbar,
        sidebar,
        content,
    ],
)


@app.callback(
    [
        Output("sidebar", "style"),
        Output("page-content", "style"),
        Output("side_click", "data"),
    ],

    [Input("btn_sidebar", "n_clicks")],
    [
        State("side_click", "data"),
    ]
)
def toggle_sidebar(n, nclick):
    if n:
        if nclick == "SHOW":
            sidebar_style = SIDEBAR_HIDEN
            content_style = CONTENT_STYLE1
            cur_nclick = "HIDDEN"
        else:
            sidebar_style = SIDEBAR_STYLE
            content_style = CONTENT_STYLE
            cur_nclick = "SHOW"
    else:
        sidebar_style = SIDEBAR_STYLE
        content_style = CONTENT_STYLE
        cur_nclick = 'SHOW'

    return sidebar_style, content_style, cur_nclick

# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on


TAB_STYLE = {
    'width': 'inherit',
    'border': 'none',
    'boxShadow': 'inset 0px -1px 0px 0px lightgrey',
    'background': 'white',
    'paddingTop': 8,
    'paddingBottom': 0,
    'height': '42px',
}

SELECTED_STYLE = {
    'width': 'inherit',
    'boxShadow': 'none',
    'borderLeft': 'none',
    'borderRight': 'none',
    'borderTop': 'none',
    'borderBottom': '2px #004A96 solid',
    'background': 'white',
    'paddingTop': 8,
    'paddingBottom': 0,
    'height': '42px',
}


@app.callback(Output("page-content", "children"),
              [Input("url", "pathname")])
def render_page_content(pathname):
    #if pathname == "/":
    #    return html.Div(children=[html.H1('Hello Dash', style={"background-image": home_background})]),

        #html.Img(src=home_background, height="30px")
    if pathname == "/":
        return html.Div([
                dcc.Tabs(id='tabs', value='tab-1', children=[
                    dcc.Tab(label='Graphical Display', value='tab-1',), #className='os-tab',), #style=TAB_STYLE, selected_style = SELECTED_STYLE),
                    dcc.Tab(label='Table Display', value='tab-2',), #className='os-tab'), #style=TAB_STYLE, selected_style = SELECTED_STYLE),
                ],
                         colors={
                             "border": "Gray",
                             "primary": "blue",
                             "background": "#f8f9fa"
                         },
                ),
                html.Div(id='tabs-content')
        ])

    #elif pathname == "/page-1":
    #    return \
            #html.Div(
            #html.Div(
            #    dcc.Graph(id="graph_anim", figure=animation[0])
            #)
        #)
    elif pathname == "/page-2":
        return html.Div([
                dcc.Tabs(id='tabs_clustering', value='tab-4', children=[
                    dcc.Tab(label='KMeans', value='tab-4', ),# className='os-tab'), #style=TAB_STYLE, selected_style = SELECTED_STYLE),
                    dcc.Tab(label='Optimal Number of Clusters', value='tab-3',), #className='os-tab',), #style=TAB_STYLE, selected_style = SELECTED_STYLE),
                ],
                         colors={
                             "border": "Gray",
                             "primary": "blue",
                             "background": "#f8f9fa"
                         },
                ),
                html.Div(id='tabs-content-clustering')
        ])


    elif pathname == "/page-3":
        return html.Div(children=[
            html.Div([
                dcc.Graph(
                    id='g8',
                    figure=fig8,
                    style={"margin-bottom": 20, "margin-top": 20},
                ),
            ]),
            html.Div([
                dcc.Graph(
                    id='g9',
                    figure = fig9,
                    style={"margin-bottom": 30},
                ),
            ]),
        ])
    #elif pathname == "/page-4":
    #    return html.Div([
    #        dbc.Row([
    #            html.H5("Filter Methods"),
    #        ], justify="center", align="center", style={"top-margin": 100, "bottom-margin": 30}, ),
#
#            html.Div([
#                # html.P("Comparisson between 2017 and 2018"),
#                html.Div(
#                    dcc.Graph(id='g10',
#                              figure=fig10,
#                              style={"margin-left": "15px", "margin-right": "5px"}),
#                    style={"width": "50%", "left-margin": 10, 'display': 'inline-block'},
#                    className="six columns",
#                ),
#
#                #html.P("According to the KBest method, the 5 most influence features, in descending order, are:\n-Power\n-Solar Radiation\n-Week Day\n-HR\n-Temperature"),
#
#                html.Div(
#                    dcc.Graph(id='g11',
#                              figure=fig11,
#                              style={"margin-left": "5px", "margin-right": "15px"}),
#                    className="six columns",
#                    style={"width": "50%", "right-margin": 10, 'display': 'inline-block'}
#                ),

                #html.P("According to the KBest method, the 5 most influence features, in descending order, are:\n-Power\n-Hour\n-Solar Radiation\n-Week Day\n-Temperature"),

#            ], className="row", style={"right-margin": 20}),
#            html.Div(
#                "According to the KBest - f_Regression method, the 5 most influence features, in descending order, are:\n-Power\n-Solar Radiation\n-Week Day\n-HR\n-Temperature"),
#            html.Div(
#                "According to the KBest Mutual_Info_Regression method, the 5 most influence features, in descending order, are:\n-Power\n-Hour\n-Solar Radiation\n-Week Day\n-Temperature"),
#        ])

#    elif pathname == "/page-5":
#        return html.Div([
#            dbc.Row([
#                html.H5("Wrapper Methods"),
#            ], justify="center", align="center", style={"top-margin": 100, "bottom-margin": 30}, ),
#
#            html.Div([
#                dcc.Graph(
#                    id='g12',
#                    figure = fig12,
#                    style={"margin-bottom": 30},
#                ),
#            ]),
#            html.Div("According to this method, the 5 most influence features, in descending order, are:\n-Power\n-Hour\n-Solar Radiation\n-Week Day\n-Temperature")
#        ])
#    elif pathname == "/page-6":
#        return html.Div([
#            dbc.Row([
#                html.H5("Embedded Methods"),
#            ], justify="center", align="center", style={"top-margin": 100, "bottom-margin": 30}, ),
#
#            html.Div([
#                dcc.Graph(
#                    id='g13',
#                    figure = fig13,
#                    style={"margin-bottom": 30},
#                ),
 #           ]),
  #          html.Div("According to the Random Forest Regressor method, the 5 most influence features, in descending order, are:\n-Rain Day\n-Temperature\n-Power\n-Rain\n-Holiday Day")
  #      ])

    elif pathname == "/page-7":
        return html.Div([
                dcc.Tabs(id='tabs_training', value='tab-5', children=[
                    dcc.Tab(label='Linear Regression', value='tab-5',), #className='os-tab',), #style=TAB_STYLE, selected_style = SELECTED_STYLE),
                    dcc.Tab(label='Support Vector Regressor', value='tab-6',), #className='os-tab'), #style=TAB_STYLE, selected_style = SELECTED_STYLE),
                    dcc.Tab(label='Random Forest', value='tab-7', ),
                    dcc.Tab(label='Gradient Boosting', value='tab-8', ),
                    dcc.Tab(label='Extreme Gradient Boosting', value='tab-9', ),
                    dcc.Tab(label='Bootstrapping', value='tab-10', ),
                ],
                         colors={
                             "border": "Gray",
                             "primary": "blue",
                             "background": "#f8f9fa"
                         },
                ),
                html.Div(id='tabs-content-training')
        ])
    elif pathname == "/page-8":
        return html.Div([
                dcc.Tabs(id='tabs_nn', value='tab-12', children=[
                    dcc.Tab(label='Sklearn NN', value='tab-12',), #className='os-tab',), #style=TAB_STYLE, selected_style = SELECTED_STYLE),

                ],
                         colors={
                             "border": "Gray",
                             "primary": "blue",
                             "background": "#f8f9fa"
                         },
                ),
                html.Div(id='tabs-content-nn')
        ])
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.Div([
                html.Div([
                    html.Hr(),  # Empty line
                    dcc.Checklist(
                        id = "checklist_1",
                        options=[{"label": x, "value": x + "_cl1"} for x in df.columns[1:8]],
                        value=[df.columns[1] + "_cl1", df.columns[2] + "_cl1"],
                        labelStyle={'display': 'inline-block', "margin-left": 25},
                        #labelStyle=dict(display='block'),
                        inputStyle={"vertical-align": "middle", "margin": 10},
                        #labelStyle={"vertical-align": "middle"},
                        #style={"display": "inline-flex", "flex-wrap": "wrap", "justify-content": "space-between",
                        #       "line-height": "28px"},
                    ),
                ], style={'width': '100', 'margin-top': 40, "margin-bottom": 30}),

                html.Div([
                    html.Div([
                        html.H6('Time Interval'),
                    ], style={'margin-left': 35, "margin-top": 13}),
                    html.Div([
                        dcc.DatePickerRange(
                            id='datepickerrange',
                            start_date=df.index.date.min(),
                            end_date=df.index.date.max(),
                            min_date_allowed=df.index.date.min(),
                            max_date_allowed=df.index.date.max(),
                            display_format='D MMM YYYY'
                            #display_format='DD YYYY MM H:m:s'
                        ),
                        html.Div(id='output-container-range')
                    ], style={"with": 10, 'margin-left': 10}),
                ], className="row"),
                html.Hr(),  # Empty line

                    #dcc.RangeSlider(
                    #    id='rangeslider',
                    #    min=0,
                    #    max=df.index.nunique() - 1,
                    #    value=[0, df.index.nunique() - 1],
                    #    allowCross=False
                    #),

                    #dcc.RangeSlider(
                    #    id='my-range-slider',
                    #    min=2017,
                    #    max=2018,
                    #    step=0.01,
                    #    value=[2017, 2018],
                    #    allowCross=False,
                    #),
                    #html.Div(id='output-container-range-slider')
                    ], style={'width': '100', "margin-bottom": 30},
            ),

            html.Div([
                dcc.Graph(
                    id='total-data',
                    figure = {},
                    style={"margin-bottom": 30},
                ),
            ]),
            dbc.Row([
                html.H5("Advanced Search"),
            ],justify="center", align="center", style={"top-margin": 100, "bottom-margin": 30},),

            html.Div(
                className="row", children=[
                    html.Div(className='six columns', children=[
                        dcc.Dropdown(
                            id='dropdown_1',
                            #values_1 = ['ddm1_power', "ddm1_temperature", 'ddm1_HR', 'ddm1_wind_speed', 'ddm1_wind_gust', 'ddm1_pressure', 'ddm1_pressure', 'ddm1_solar_radiation', 'ddm1_rain'],
                            options=[{"label": x, "value": x + "_dd1"} for x in df.columns[1:8]],
                            value=df.columns[1] + "_dd1",
                            clearable=False,
                            searchable=False,
                            style={"margin-left": 7, "margin-right": 10, "margin-top": 15},
                        )], style=dict(width='50%'))
                    , html.Div(className='six columns', children=[
                        dcc.Dropdown(
                            id='dropdown_2',
                            #values_2 = ['ddm2_power', "ddm2_temperature", 'ddm2_HR', 'ddm2_wind_speed', 'ddm2_wind_gust', 'ddm2_pressure', 'ddm2_pressure', 'ddm2_solar_radiation', 'ddm2_rain'],
                            options=[{"label": x, "value": x + "_dd2"} for x in df.columns[1:8]],
                            value = df.columns[1] + "_dd2",
                            clearable=False,
                            searchable=False,
                            style={"margin-left": 2, "margin-right": 17, "margin-top": 15},
                        )], style=dict(width='50%'))
                ], style=dict(display='flex')),

            #html.Div([
            #    dcc.Dropdown(
            #        id="ticker",
            #        options=[{"label": x, "value": x}
            #                 for x in df.columns[1:]],
            #        value=df.columns[1],
            #        clearable=False,
            #    ),
            #    dcc.Graph(id="time-series-chart"),

            html.Div([
                #html.P("Comparisson between 2017 and 2018"),
                html.Div(
                    dcc.Graph(id='g1',
                              figure={},
                              style={"margin-left": "15px", "margin-right": "5px"}),
                              #{'data': [{'y': [1, 2, 3]}], 'layout': {'title': 'IST yearly electricity consumption (MWh)'}}),
                    style={"width": "50%", "left-margin": 10, 'display': 'inline-block'},
                    className="six columns",
                ),
                html.Div(
                    dcc.Graph(id='g2',
                              figure={},
                              style={"margin-left": "5px", "margin-right": "15px"}),
                              #{'data': [{'y': [1, 2, 3]}], 'layout': {'title': 'IST yearly electricity consumption (MWh)'}}),
                    className="six columns",
                    style={"width": "50%", "right-margin": 10, 'display': 'inline-block'}
                ),
            ], className="row", style={"right-margin": 20}),


        ])

    elif tab == 'tab-2':
        return html.Div([
            #html.H3('IST Energy yearly Concumption (kwh)'),
            html.Div([
                generate_table(df_tab, max_rows=10)
            ], style={'margin-top': 30, "width":"100%"}),
        ])
    else:
        return dash.no_update

@app.callback(Output('tabs-content-clustering', 'children'),
              Input('tabs_clustering', 'value'))

def render_tab_clustering(tab):
#    if tab == 'tab-3':
#        return html.Div([
#            html.Hr(),
#            html.Div([
#                html.Hr(),
#                dbc.Row([
#                    html.Div("To find out the optimal number of clusters, an iterative function was applied, capable of computing the score for each value of clusters. In this way, we can graphically observe the performance obtained."),
#                    html.Div("From this graph we observe that for a number of clusters bigger than 4/5 there are no major improvemnts"),
#                ], style={"top-margin": 100, "bottom-margin": 30}, ),
#            ],style={'marginBottom': 50, 'marginTop': 25}),
#            html.Hr(),
#            html.Div([
#                dcc.Graph(
#                    id='optimal_num_clusters',
#                    figure = fig4,
#                    style={"margin-bottom": 30},
#                ),
#            ]),
            #html.Hr(),
            #dbc.Row([
            #    html.Div("From this graph we observe that for a number of clusters bigger than 4/5 there are no major improvemnts"),
            #], style={"top-margin": 10, "bottom-margin": 30}, ),
            #html.Hr(),
#        ]),

    if tab == 'tab-4':
        return html.Div([
            html.Div([
            dbc.Row([
                html.H5("Search by Features and Number of Clusters",style={"top-margin": 500}),
            ],),#justify="center", align="right",style={"top-margin": 100, "bottom-margin": 30},),
            ],style={'marginBottom': 30, 'marginTop': 20, "marginLeft": 20}),

            html.Hr(),

            html.Div(
                className="row", children=[
                    html.Label(['X: '], style={'font-weight': 'bold', "text-align": "center"}),
                    html.Div(className='three columns', children=[
                        dcc.Dropdown(
                            id='dropdown_clust_1',
                            #values_clust = ['Pie Chart', "Histogram", 'Scatter 2D', 'Scatter 3D'],
                            options=[{"label": x, "value": x + "_cldd1"} for x in df_cluster.columns[0:14]],
                            value = df_cluster.columns[0] + "_cldd1",
                            clearable=False,
                            searchable=False,
                            style={"margin-left": 6, "margin-right": 10, "margin-top": 10, "margin-bottom": 10},
                        )], style=dict(width='30%')),
                    html.Label(['Y: '], style={'font-weight': 'bold', "text-align": "center"}),
                    html.Div(className='three columns', children=[
                        dcc.Dropdown(
                            id='dropdown_clust_2',
                            options=[{"label": x, "value": x + "_cldd2"} for x in df_cluster.columns[0:14]],
                            value=df_cluster.columns[1] + "_cldd2",
                            clearable=False,
                            searchable=False,
                            style={"margin-left": 6, "margin-right": 10, "margin-top": 10, "margin-bottom": 10},
                        )], style=dict(width='30%')),
                    html.Label(['Z: '], style={'font-weight': 'bold', "text-align": "center"}),
                    html.Div(className='three columns', children=[
                        dcc.Dropdown(
                            id='dropdown_clust_3',
                            # values_clust = ['Pie Chart', "Histogram", 'Scatter 2D', 'Scatter 3D'],
                            options=[{"label": x, "value": x + "_cldd3"} for x in df_cluster.columns[0:14]],
                            value=df_cluster.columns[2] + "_cldd3",
                            clearable=False,
                            searchable=False,
                            style={"margin-left": 6, "margin-right": 10, "margin-top": 10, "margin-bottom": 10},
                        )], style=dict(width='30%')),
                    ],style={'marginBottom': 30}
            ),

            html.Div(children=[
                html.Div(id='slider-output-container'),
                dcc.Slider(
                    id='my-slider',
                    min=0,
                    max=20,
                    step=1,
                    value=4,
                    marks={0:"0", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9", 10:"10", 11:"11", 12:"12", 13:"13", 14:"14", 15:"15", 16:"16", 17:"17", 18:"18", 19:"19", 20:"20"},
                    updatemode='drag'
                ),
                # html.Div(id='slider-output-container')
            ], style=dict(width='100%')),


                        #dcc.Dropdown(
                        #    id='dropdown_clust_2',
                        #    #values_2 = ['ddm2_power', "ddm2_temperature", 'ddm2_HR', 'ddm2_wind_speed', 'ddm2_wind_gust', 'ddm2_pressure', 'ddm2_pressure', 'ddm2_solar_radiation', 'ddm2_rain'],
                        #    options=[{"label": x, "value": x + "_cldd2"} for x in df.columns[1:8]],
                        #    #options=[
                        #    #    {'label': 'Pie Chart', 'value': 'pie_cldd1'}
                        #    #    {'label': "Histogram", 'value': 'hist_cldd1'},
                        #    #    {'label': 'Scatter 2D', 'value': 'scat2d_cldd1'},
                        #    #    {'label': 'Scatter 3D', 'value': 'scat3d_cld11'},
                        #    #],
                        #    value = df.columns[1] + "_cldd2",
                        #    clearable=False,
                        #    searchable=False,
                        #    style={"margin-left": 2, "margin-right": 17, "margin-top": 15},
                        #)
                    #], style=dict(width='50%'), style=dict(display='flex')),

            html.Div([
                # html.P("Comparisson between 2017 and 2018"),
                html.Div(
                    dcc.Graph(id='g5',
                              figure={},
                              style={"margin-left": "15px", "margin-right": "5px","margin-top": 20}),
                    # {'data': [{'y': [1, 2, 3]}], 'layout': {'title': 'IST yearly electricity consumption (MWh)'}}),
                    style={"width": "70%", "left-margin": 10, "top-margin":25, 'display': 'inline-block'},
                    className="six columns",
                ),
                html.Div(
                    dcc.Graph(id='g6',
                              figure=kmeans_pie("Power [KW]", num_clusters=4),
                              style={"margin-left": "5px", "margin-right": "15px", "margin-top": 20, "margin-bottom": 20}),
                    # {'data': [{'y': [1, 2, 3]}], 'layout': {'title': 'IST yearly electricity consumption (MWh)'}}),
                    className="six columns",
                    style={"width": "30%", "right-margin": 10, "top-margin":25, 'display': 'inline-block'}
                ),
            ], className="row", style={"right-margin": 20}),

            #dbc.Row([
            #    html.H5("Clusters 3D Scatter Plot"),
            #], justify="center", align="center", style={"top-margin": 100, "bottom-margin": 30}, ),

            html.Div([
                dcc.Graph(
                    id='g7',
                    figure={},
                    style={"margin-top": 5,"height":800},
                )],
            ),

            ])
    else:
        return dash.no_update


@app.callback(Output('tabs-content-training', 'children'),
              Input('tabs_training', 'value'))

def render_tab_training(tab):
    if tab == 'tab-5':
        return html.Div([
            #dbc.Row([
            #    html.p("To find out the optimal number of clusters, an iterative function was applied, capable of computing the score for each value of clusters. In this way, we can graphically observe the performance obtained."),
            #], justify="center", align="center", style={"top-margin": 100, "bottom-margin": 30}, ),
            html.Div([
                # html.P("Comparisson between 2017 and 2018"),
                html.Div(
                    dcc.Graph(id='g14',
                              figure=fig14,
                              style={"margin-left": "15px", "margin-right": "5px"}),
                    style={"width": "50%", "left-margin": 10, 'display': 'inline-block'},
                    className="six columns",
                ),

                # html.P("According to the KBest method, the 5 most influence features, in descending order, are:\n-Power\n-Solar Radiation\n-Week Day\n-HR\n-Temperature"),

                html.Div(
                    dcc.Graph(id='g15',
                              figure=fig15,
                              style={"margin-left": "5px", "margin-right": "15px"}),
                    className="six columns",
                    style={"width": "50%", "right-margin": 10, 'display': 'inline-block'}
                ),

                # html.P("According to the KBest method, the 5 most influence features, in descending order, are:\n-Power\n-Hour\n-Solar Radiation\n-Week Day\n-Temperature"),

            ], className="row", style={"right-margin": 20}),

            html.Hr(),

            dbc.Row([
                html.P("Prediction Metrics:"),
            ], style={"top-margin": 5, "bottom-margin": 5}, ),

            html.Div([
                generate_table(df_metrics_LR)
            ], style={'margin-top': 5, "width": "100%"}),

            #dbc.Row([
            #    html.P(f"\nPrediction Metrics:\n     MAE:  {MAE_LR}\n     MSE:  {MSE_LR}\n     RMSE:  {RMSE_LR}\n     cvRMSE:  {cvRMSE_LR}\n"),
            #], justify="center", align="center", style={"top-margin": 100, "bottom-margin": 30}, ),
        ]),

    elif tab == 'tab-6':
        return html.Div([
            #dbc.Row([
            #    html.p("To find out the optimal number of clusters, an iterative function was applied, capable of computing the score for each value of clusters. In this way, we can graphically observe the performance obtained."),
            #], justify="center", align="center", style={"top-margin": 100, "bottom-margin": 30}, ),
            html.Div([
                # html.P("Comparisson between 2017 and 2018"),
                html.Div(
                    dcc.Graph(id='g16',
                              figure=fig16,
                              style={"margin-left": "15px", "margin-right": "5px"}),
                    style={"width": "50%", "left-margin": 10, 'display': 'inline-block'},
                    className="six columns",
                ),

                # html.P("According to the KBest method, the 5 most influence features, in descending order, are:\n-Power\n-Solar Radiation\n-Week Day\n-HR\n-Temperature"),

                html.Div(
                    dcc.Graph(id='g17',
                              figure=fig17,
                              style={"margin-left": "5px", "margin-right": "15px"}),
                    className="six columns",
                    style={"width": "50%", "right-margin": 10, 'display': 'inline-block'}
                ),

                # html.P("According to the KBest method, the 5 most influence features, in descending order, are:\n-Power\n-Hour\n-Solar Radiation\n-Week Day\n-Temperature"),

            ], className="row", style={"right-margin": 20}),

            html.Hr(),

            dbc.Row([
                html.P("Prediction Metrics:"),
            ], style={"top-margin": 5, "bottom-margin": 5}, ),

            html.Div([
                generate_table(df_metrics_SVR)
            ], style={'margin-top': 5, "width": "100%"}),

            #dbc.Row([
            #    html.P(f"\nPrediction Metrics:\n"),
            #    html.P(f"\nMAE:  {MAE_LR}\n"),
            #    html.P(f"\nMSE:  {MSE_LR}\n"),
            #    html.P(f"\nRMSE:  {RMSE_LR}\n"),
            #    html.P(f"\ncvRMSE:  {cvRMSE_LR}\n"),
            #], justify="center", align="center", style={"top-margin": 100, "bottom-margin": 30}, ),

        ]),

    elif tab == 'tab-7':
        return html.Div([
            #dbc.Row([
            #    html.p("To find out the optimal number of clusters, an iterative function was applied, capable of computing the score for each value of clusters. In this way, we can graphically observe the performance obtained."),
            #], justify="center", align="center", style={"top-margin": 100, "bottom-margin": 30}, ),
            html.Div([
                # html.P("Comparisson between 2017 and 2018"),
                html.Div(
                    dcc.Graph(id='g18',
                              figure=fig18,
                              style={"margin-left": "15px", "margin-right": "5px"}),
                    style={"width": "50%", "left-margin": 10, 'display': 'inline-block'},
                    className="six columns",
                ),

                # html.P("According to the KBest method, the 5 most influence features, in descending order, are:\n-Power\n-Solar Radiation\n-Week Day\n-HR\n-Temperature"),

                html.Div(
                    dcc.Graph(id='g19',
                              figure=fig19,
                              style={"margin-left": "5px", "margin-right": "15px"}),
                    className="six columns",
                    style={"width": "50%", "right-margin": 10, 'display': 'inline-block'}
                ),

                # html.P("According to the KBest method, the 5 most influence features, in descending order, are:\n-Power\n-Hour\n-Solar Radiation\n-Week Day\n-Temperature"),

            ], className="row", style={"right-margin": 20}),

            html.Hr(),

            dbc.Row([
                html.P("Prediction Metrics:"),
            ], style={"top-margin": 5, "bottom-margin": 5}, ),

            html.Div([
                generate_table(df_metrics_DT)
            ], style={'margin-top': 5, "width": "100%"}),

            #dbc.Row([
            #    html.P(f"\nPrediction Metrics:\n     MAE:  {MAE_LR}\n     MSE:  {MSE_LR}\n     RMSE:  {RMSE_LR}\n     cvRMSE:  {cvRMSE_LR}\n"),
            #], justify="center", align="center", style={"top-margin": 100, "bottom-margin": 30}, ),
        ]),

    elif tab == 'tab-8':
        return html.Div([
            #dbc.Row([
            #    html.p("To find out the optimal number of clusters, an iterative function was applied, capable of computing the score for each value of clusters. In this way, we can graphically observe the performance obtained."),
            #], justify="center", align="center", style={"top-margin": 100, "bottom-margin": 30}, ),
            html.Div([
                # html.P("Comparisson between 2017 and 2018"),
                html.Div(
                    dcc.Graph(id='g20',
                              figure=fig20,
                              style={"margin-left": "15px", "margin-right": "5px"}),
                    style={"width": "50%", "left-margin": 10, 'display': 'inline-block'},
                    className="six columns",
                ),

                # html.P("According to the KBest method, the 5 most influence features, in descending order, are:\n-Power\n-Solar Radiation\n-Week Day\n-HR\n-Temperature"),

                html.Div(
                    dcc.Graph(id='g21',
                              figure=fig21,
                              style={"margin-left": "5px", "margin-right": "15px"}),
                    className="six columns",
                    style={"width": "50%", "right-margin": 10, 'display': 'inline-block'}
                ),

                # html.P("According to the KBest method, the 5 most influence features, in descending order, are:\n-Power\n-Hour\n-Solar Radiation\n-Week Day\n-Temperature"),

            ], className="row", style={"right-margin": 20}),

            html.Hr(),

            dbc.Row([
                html.P("Prediction Metrics:"),
            ], style={"top-margin": 5, "bottom-margin": 5}, ),

            html.Div([
                generate_table(df_metrics_RF)
            ], style={'margin-top': 5, "width": "100%"}),

            #dbc.Row([
            #    html.P(f"\nPrediction Metrics:\n     MAE:  {MAE_LR}\n     MSE:  {MSE_LR}\n     RMSE:  {RMSE_LR}\n     cvRMSE:  {cvRMSE_LR}\n"),
            #], justify="center", align="center", style={"top-margin": 100, "bottom-margin": 30}, ),
        ]),

    elif tab == 'tab-9':
        return html.Div([
            #dbc.Row([
            #    html.p("To find out the optimal number of clusters, an iterative function was applied, capable of computing the score for each value of clusters. In this way, we can graphically observe the performance obtained."),
            #], justify="center", align="center", style={"top-margin": 100, "bottom-margin": 30}, ),
            html.Div([
                # html.P("Comparisson between 2017 and 2018"),
                html.Div(
                    dcc.Graph(id='g22',
                              figure=fig22,
                              style={"margin-left": "15px", "margin-right": "5px"}),
                    style={"width": "50%", "left-margin": 10, 'display': 'inline-block'},
                    className="six columns",
                ),

                # html.P("According to the KBest method, the 5 most influence features, in descending order, are:\n-Power\n-Solar Radiation\n-Week Day\n-HR\n-Temperature"),

                html.Div(
                    dcc.Graph(id='g23',
                              figure=fig23,
                              style={"margin-left": "5px", "margin-right": "15px"}),
                    className="six columns",
                    style={"width": "50%", "right-margin": 10, 'display': 'inline-block'}
                ),

                # html.P("According to the KBest method, the 5 most influence features, in descending order, are:\n-Power\n-Hour\n-Solar Radiation\n-Week Day\n-Temperature"),

            ], className="row", style={"right-margin": 20}),

            html.Hr(),

            dbc.Row([
                html.P("Prediction Metrics:"),
            ], style={"top-margin": 5, "bottom-margin": 5}, ),

            html.Div([
                generate_table(df_metrics_GB)
            ], style={'margin-top': 5, "width": "100%"}),

            #dbc.Row([
            #    html.P(f"\nPrediction Metrics:\n     MAE:  {MAE_LR}\n     MSE:  {MSE_LR}\n     RMSE:  {RMSE_LR}\n     cvRMSE:  {cvRMSE_LR}\n"),
            #], justify="center", align="center", style={"top-margin": 100, "bottom-margin": 30}, ),
        ]),

    elif tab == 'tab-10':
        return html.Div([
            #dbc.Row([
            #    html.p("To find out the optimal number of clusters, an iterative function was applied, capable of computing the score for each value of clusters. In this way, we can graphically observe the performance obtained."),
            #], justify="center", align="center", style={"top-margin": 100, "bottom-margin": 30}, ),
            html.Div([
                # html.P("Comparisson between 2017 and 2018"),
                html.Div(
                    dcc.Graph(id='g24',
                              figure=fig24,
                              style={"margin-left": "15px", "margin-right": "5px"}),
                    style={"width": "50%", "left-margin": 10, 'display': 'inline-block'},
                    className="six columns",
                ),

                # html.P("According to the KBest method, the 5 most influence features, in descending order, are:\n-Power\n-Solar Radiation\n-Week Day\n-HR\n-Temperature"),

                html.Div(
                    dcc.Graph(id='g25',
                              figure=fig25,
                              style={"margin-left": "5px", "margin-right": "15px"}),
                    className="six columns",
                    style={"width": "50%", "right-margin": 10, 'display': 'inline-block'}
                ),

                # html.P("According to the KBest method, the 5 most influence features, in descending order, are:\n-Power\n-Hour\n-Solar Radiation\n-Week Day\n-Temperature"),

            ], className="row", style={"right-margin": 20}),

            html.Hr(),

            dbc.Row([
                html.P("Prediction Metrics:"),
            ], style={"top-margin": 5, "bottom-margin": 5}, ),

            html.Div([
                generate_table(df_metrics_XGB)
            ], style={'margin-top': 5, "width": "100%"}),

            #dbc.Row([
            #    html.P(f"\nPrediction Metrics:\n     MAE:  {MAE_LR}\n     MSE:  {MSE_LR}\n     RMSE:  {RMSE_LR}\n     cvRMSE:  {cvRMSE_LR}\n"),
            #], justify="center", align="center", style={"top-margin": 100, "bottom-margin": 30}, ),
        ]),

    elif tab == 'tab-11':
        return html.Div([
            #dbc.Row([
            #    html.p("To find out the optimal number of clusters, an iterative function was applied, capable of computing the score for each value of clusters. In this way, we can graphically observe the performance obtained."),
            #], justify="center", align="center", style={"top-margin": 100, "bottom-margin": 30}, ),
            html.Div([
                # html.P("Comparisson between 2017 and 2018"),
                html.Div(
                    dcc.Graph(id='g26',
                              figure=fig26,
                              style={"margin-left": "15px", "margin-right": "5px"}),
                    style={"width": "50%", "left-margin": 10, 'display': 'inline-block'},
                    className="six columns",
                ),

                # html.P("According to the KBest method, the 5 most influence features, in descending order, are:\n-Power\n-Solar Radiation\n-Week Day\n-HR\n-Temperature"),

                html.Div(
                    dcc.Graph(id='g27',
                              figure=fig27,
                              style={"margin-left": "5px", "margin-right": "15px"}),
                    className="six columns",
                    style={"width": "50%", "right-margin": 10, 'display': 'inline-block'}
                ),

                # html.P("According to the KBest method, the 5 most influence features, in descending order, are:\n-Power\n-Hour\n-Solar Radiation\n-Week Day\n-Temperature"),

            ], className="row", style={"right-margin": 20}),

            html.Hr(),

            dbc.Row([
                html.P("Prediction Metrics:"),
            ], style={"top-margin": 5, "bottom-margin": 5}, ),

            html.Div([
                generate_table(df_metrics_BT)
            ], style={'margin-top': 5, "width": "100%"}),

            #dbc.Row([
            #    html.P(f"\nPrediction Metrics:\n     MAE:  {MAE_LR}\n     MSE:  {MSE_LR}\n     RMSE:  {RMSE_LR}\n     cvRMSE:  {cvRMSE_LR}\n"),
            #], justify="center", align="center", style={"top-margin": 100, "bottom-margin": 30}, ),
        ]),

    else:
        return dash.no_update

@app.callback(Output('tabs-content-nn', 'children'),
              Input('tabs_nn', 'value'))

def render_content_nn(tab):
    if tab == 'tab-12':
        return html.Div([
            #dbc.Row([
            #    html.p("To find out the optimal number of clusters, an iterative function was applied, capable of computing the score for each value of clusters. In this way, we can graphically observe the performance obtained."),
            #], justify="center", align="center", style={"top-margin": 100, "bottom-margin": 30}, ),

            #html.Div([
                # html.P("Comparisson between 2017 and 2018"),
            #    html.Div(
            #        dcc.Graph(id='g28',
            #                  figure=fig28,
            #                  style={"margin-left": "15px", "margin-right": "5px"}),
            #        style={"width": "50%", "left-margin": 10, 'display': 'inline-block'},
            #        className="six columns",
            #    ),


             #   html.Div(
             #       dcc.Graph(id='g29',
             #                 figure=fig29,
             #                 style={"margin-left": "5px", "margin-right": "15px"}),
             #       className="six columns",
             #       style={"width": "50%", "right-margin": 10, 'display': 'inline-block'}
             #   ),



            #], className="row", style={"right-margin": 20}),

            #html.Hr(),

            #dbc.Row([
            #    html.P("Prediction Metrics:"),
            #], style={"top-margin": 5, "bottom-margin": 5}, ),

            #html.Div([
            #    generate_table(df_metrics_SK_NN)
            #], style={'margin-top': 5, "width": "100%"}),


        ]),

    else:
        return dash.no_update




#@app.callback(Output('datepickerrange', 'start_date'),
#              [Input('df', 'children'),
#               Input('rangeslider', 'value')])

#def update_daterangestart(df, rangeslider_value):
#    df_1 = pd.read_json(sorties, orient='split')
#    return np.sort(df_1.index.dt.date.unique())[rangeslider_value[0]]

#@app.callback(Output('top10-datepickerrange', 'end_date'),
#             [Input('df', 'children'),
#               Input('top10-rangeslider', 'value')])
#def update_daterangeend(df, rangeslider_value):
#    df_1 = pd.read_json(sorties, orient='split')
#    return np.sort(df_1.index.dt.date.unique())[rangeslider_value[1]]


@app.callback(Output('total-data', 'figure'),
                       [Input('datepickerrange', 'start_date'),
                        Input('datepickerrange', 'end_date'),
                        Input("checklist_1", 'value')])

def update_graph(start_date, end_date, value):
    if start_date is not None and end_date is not None:
        return generate_total_plot(start_date, end_date, value)

    else:
        return dash.no_update

@app.callback(
    [Output('g2', 'figure'), Output("g1", "figure")],
    [Input('dropdown_1', 'value')])

def update_output(value):
    if value == "Power [KW]_dd1":
        return generate_graph_bar("Power [KW]"), generate_pie_chart("Power [KW]")

    elif value == "Temperature [Cº]_dd1":
        return generate_graph_bar("Temperature [Cº]"), generate_pie_chart("Temperature [Cº]")

    elif value == "HR_dd1":
        return generate_graph_bar("HR"), generate_pie_chart("HR")

    elif value == "Wind Speed [m/s]_dd1":
        return generate_graph_bar("Wind Speed [m/s]"), generate_pie_chart("Wind Speed [m/s]")

    elif value == "Wind Gust [m/s]_dd1":
        return generate_graph_bar("Wind Gust [m/s]"), generate_pie_chart("Wind Gust [m/s]")

    elif value == "Pressure [mbar]_dd1":
        return generate_graph_bar("Pressure [mbar]"), generate_pie_chart("Pressure [mbar]")

    elif value == "Solar Radiation [w/m2]_dd1":
        return generate_graph_bar("Solar Radiation [w/m2]"), generate_pie_chart("Solar Radiation [w/m2]")

    elif value == "Rain [mm/h]_dd1":
        return generate_graph_bar("Rain [mm/h]"), generate_pie_chart("Rain [mm/h]")
        #fig3 = go.Figure()
        #fig3.add_trace(go.Histogram(x=df_bar_17["Month"], y=df_bar_17["Temperature [Cº]"], histfunc="avg", name="2017"))
        #fig3.add_trace(go.Histogram(x=df_bar_18["Month"], y=df_bar_18["Temperature [Cº]"], histfunc="avg", name="2018"))
        #fig3.update_xaxes(ticklabelmode="period", dtick="M1", tickformat="%b")
        #fig3.update_layout(barmode='group')
        #fig3.update_traces(opacity=0.75)
    else:
        return dash.no_update



@app.callback(
    [Output('g5', 'figure'), Output('g6', 'figure'), Output("g7", "figure")],
    [Input('dropdown_clust_1', 'value'), Input("dropdown_clust_2", "value"), Input("dropdown_clust_3", "value"), Input("my-slider", "value")])

def update_figure_kmean(column_1, column_2, column_3,num_clusters):
    if column_1 is not None and column_2 is not None and column_3 is None:
        column_1 = column_1.split("_")[0]
        column_2 = column_2.split("_")[0]
        column_3 = ["Solar Radiation [w/m2]"]
        return kmeans_scatter2d(column_1, column_2, num_clusters), kmeans_pie(column_1, num_clusters), kmeans_scatter3d(column_1, column_2, column_3, num_clusters)

    if column_1 is not None and column_2 is not None and column_3 is not None:
        column_1 = column_1.split("_")[0]
        column_2 = column_2.split("_")[0]
        column_3 = column_3.split("_")[0]
        return kmeans_scatter2d(column_1, column_2, num_clusters), kmeans_pie(column_1, num_clusters), kmeans_scatter3d(column_1, column_2, column_3, num_clusters)

    if (column_1 is None and column_2 is None and column_3 is None) or (column_1 is None and column_2 is None and column_3 is not None) or (column_1 is None and column_2 is not None and column_3 is not None) or (column_1 is not None and column_2 is None and column_3 is not None):
        return dash.no_update




    #elif type_graph == "hist_cldd1":
    #    return generate_graph_bar("Temperature [Cº]"), kmeans_pie(column, num_clusters)
#
#    elif type_graph == "scat2d_cldd1":
#        return generate_graph_bar("HR"), kmeans_pie(column, num_clusters)
#
#    elif type_graph == "scat3d_cld11":
#        return generate_graph_bar("Wind Speed [m/s]"), kmeans_pie(column, num_clusters)
#
#    else:
#        return dash.no_update

#'You have selected "{}"'.format(value)


#if __name__ == '__main__':
#    app.run_server(debug=True)
@app.callback(
    Output('slider-output-container', 'children'),
    [Input('my-slider', 'value')])
def update_output(value):
    value=int(value)
    return 'You have selected {} clusters'.format(value)



if __name__ == "__main__":
    app.run_server(debug=False)