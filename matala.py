import pandas as pd, numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from numpy import arange
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import neighbors
import statistics
# Explanation of the lines in the data:
#sex(M=0/F=1)
#race(W=0/B=1)
# sbp -> Systolic blood pressure    
# dbp -> Albumin promoter, a protein which in humans is encoded by the DBP gene. 
# race ->
     # American Indian or Alaska Native:
     # a person having origins in any of the original peoples of North and South America (including Central America),
     # and who maintains tribal affiliation or community attachment. -> 0
     # Black or African American:
     # a person having origins in any of the black racial groups of Africa. 
     # terms such as "Haitian" or "Negro" can be used in addition to "Black or African American." -> 1
# income -> Hourly wage of an employee
# education -> 1:primary school ,2:middle school ,3:secondary school ,4:postsecondary education , 5:Bachelor's or equivalent
# smokeintensity -< Amount of cigarettes per day 
# smokeyrs -> years of smoking
# bronch ->Connected to air pipes                       
# hbp -> High blood pressure
# pepticulcer -> Peptic ulcer disease (PUD) is a break in the inner lining of the stomach, the first part of the small intestine, or sometimes the lower esophagus.
# colitis -> Colitis is a chronic digestive disease characterized by inflammation of the inner lining of the colon.  
# hepatitis -> Liver inflammation
# chroniccough -> Chronic cough
# hay_fever -> Allergic reaction that causes sneezing, congestion, itchy nose and sore throat
# polio -> Disabling and life-threatening disease caused by the poliovirus. 
# pica -> Eating disorder that causes repeated eating of proper proper substances Food, such as dirt, ice, hair, coal, dirt, paper, sand, etc.
# lackpep -> Proteins with homology to papain that lack peptidase activity 
# hbpmed -> Medications for high blood pressure.

data = pd.read_csv('data11.csv', encoding='ISO-8859-1')
print("The data dimentions is: ", data.shape)
# Data exploration
data_des = data.describe()
countNull = data.isnull().sum()

# If there are more than third empty values, cut the column
k = 0
while k < data.shape[1]:
    if countNull.iloc[k] > data.shape[0] / 3:
        title = countNull.index[k]
        data.drop([title], axis=1, inplace=True)
    k += 1

print("The data dimentions after removing NA's is : ", data.shape)

# If there are only one value at the column drop it
dataUnique = data.nunique()
useless_column = dataUnique[dataUnique == 1].index
data = data.drop(useless_column, axis=1)

# We added a BMI column which is basically a common calculator built on a 
# person’s weight and height

data['BMI']= data['wt'] / (pow((data['ht']/100),2))
# Create a range column of BMI
# Underweight  
data['BMI_levels']= np.where(data['BMI'] < 18.5, 0,data['BMI'])
# A proper weight
data['BMI_levels']= np.where(data['BMI'].between(18.5,25) , 1, data['BMI_levels'])
# Over-weight
data['BMI_levels']= np.where(data['BMI'].between(25,30) , 2, data['BMI_levels'])
# Obesity
data['BMI_levels']= np.where(data['BMI'] > 30, 3, data['BMI_levels'])
data['BMI_levels'] = data['BMI_levels'].astype(int)

print('*'*35)
print(data.isnull().sum())
print('*' * 35)
# Features that removed
delete_features = ['alcoholhowmuch','death','hbp','diabetes','school','pica','age', 'BMI' ,'hbpmed', 'nervousbreak', 'chroniccough']

for i in data.columns:
    if i in delete_features:
        data.drop(i, axis=1, inplace=True)

# find the most frequent value
for column in data.columns:
    if column == 'birthplace':
        freq = data[column].value_counts().idxmax()
        data[column] = data[column].fillna(freq)
    else:
        mean_value = data[column].mean()
        data[column] = data[column].fillna(mean_value)


# The "place of birth" column is numeric and we do not want these numbers to be meaningful,
# so we will convert data in this feature into categories.
data["birthplace"] = data.birthplace.astype('category')

# Separate the data into 2 types of variables - categorical and numerical
# While calculting the corrlations we will use the coressponding corrleation method:
k=0
numerical_columns = list()
categorical_columns = list()

for i in data.dtypes:
    if i == float or statistics.mean(data.iloc[:,k])>10 and data.columns[k] != 'birthplace':
        numerical_columns.append(data.columns[k])
        k+=1  
    else:
        categorical_columns.append(data.columns[k])
        k+=1

dummies = data["birthplace"]
dum = pd.get_dummies(dummies)
data_numerical = pd.DataFrame(data[numerical_columns])
data_cetegorical = pd.DataFrame(data[categorical_columns])


from scipy import stats
# In order to remove outliers and normalize the data in the best way,
# we will check if the data is distributed normally
# Kolmogorov-Smirnov test:
# H0= The sample comes from a normal distribution.
# H1=The sample is not coming from normal distribution
alpha=0.05
for i in data_numerical.columns:
    stats.kstest(data_numerical[i], 'norm', alternative='less')
    stats.kstest(stats.norm.rvs(size=100), stats.norm.cdf)
    statistic,pvalue = stats.kstest(data_numerical[i], 'norm', alternative='greater')
    if pvalue <= alpha:
        print("Rejecting of H0 hypothesis. " ,i)
    else:
        print("Acceptance of H0 hypothesis. " ,i)

# After testing we saw that all the numerical data is normally distributed

# Normalize data , standardization :
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data_numerical)
data_numerical = scaler.transform(data_numerical)
data_numerical =pd.DataFrame(data = data_numerical, columns = numerical_columns)

# Remove outliers:
# We chose to remove outliers in numerical data only since we cannot know
# by our specific data, whether there are data interruptions or not.
import warnings
from scipy import stats
for Feature in data_numerical.columns:
    upper_limit = data_numerical[Feature].mean() + 3*data_numerical[Feature].std()
    lower_limit = data_numerical[Feature].mean() - 3*data_numerical[Feature].std()
    data_numerical[(data_numerical[Feature] > upper_limit) | (data_numerical[Feature] < lower_limit)]
    new_df = data_numerical[(data_numerical[Feature] < upper_limit) & (data_numerical[Feature] > lower_limit)]
    new_df
    data_numerical[Feature] = np.where(data_numerical[Feature]>upper_limit, upper_limit, np.where(
            data_numerical[Feature]<lower_limit, lower_limit, data_numerical[Feature]
        )
    )
    warnings.filterwarnings('ignore')
    plt.figure(figsize=(16,5))
    plt.subplot(1,2,1)
    sns.distplot(new_df[Feature])
    plt.show()
    
# We have presented plots of the categorical data in order
# to better understand them
for i in data_cetegorical.columns:
    sns.catplot(x=i, kind="count", palette="ch:.25", data=data)

# Pearson corelation
corr_pearson = data_numerical.corr()
sns.heatmap(corr_pearson)

# Spearman corelation
corr_spearman = data_cetegorical.corr(method="spearman")
sns.heatmap(corr_spearman)

data1 = pd.concat([data_numerical, data_cetegorical], axis=1, join='inner')
data1.drop('birthplace', axis=1)
data1 = pd.concat([data1, dum], axis=1, join='inner')


def Logistic_And_KNN(features_ , target):  
    
    ########################## Logistic Regreesion ##########################
    X_train, X_test, y_train, y_test = train_test_split(features_, target, test_size = 0.1, random_state = 0)
    reg = LogisticRegression()
    model1 = reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    
    # Calculating the confidence level of the prediction
    pred_proba = reg.predict_proba(X_test)
    # Tests if the training is good
    y_pred_tr = reg.predict(X_train)
    error = (y_pred-y_test)
    
    # Check that the model handles the data correctly and that there is a 
    # distribution to the level of security
    plt.figure(figsize=(16,5))
    plt.subplot(1,2,1)
    sns.distplot(pred_proba)
    plt.show()
    # print confusion matrix
    print(metrics.confusion_matrix(y_train, y_pred_tr))
    print(metrics.confusion_matrix(y_test, y_pred))
    # accuracy , recall, precision
    print(classification_report(y_train,y_pred_tr))
    print(classification_report(y_test,y_pred))
    
    intercept_logit =  model1.intercept_[0]
    classes_logit = model1.classes_
    coeff_logit = pd.DataFrame({'coeff': model1.coef_[0]}, index=X_train.columns)
    
    ########################## KNeighbors Regression ##########################
    # Error rate for different k values
    rmse_val = [] 
    for K in range(20):
        K = K+1
        model = neighbors.KNeighborsRegressor(n_neighbors = K)
        model.fit(X_train, y_train)  #fit the model
        pred=model.predict(X_test) #make prediction on test set
        error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
        rmse_val.append(error) #store rmse values
        print('RMSE value for k= ' , K , 'is:', error)
    curve = pd.DataFrame(rmse_val) #elbow curve 
    curve.plot()
    
    
    # It can be seen in the graph that a change occurs when K=6 , therefore we choose k=6
    # even though RMSE value is not the smallest, the difference in errors is not
    # too big and the result will be better
    
    from sklearn.neighbors import KNeighborsClassifier
    Classifier = KNeighborsClassifier(n_neighbors=6)
    model2 = Classifier.fit(X_train, y_train)
    y_pred_knn = Classifier.predict(X_test)
    # Calculating the confidence level of the prediction
    pred_proba_knn = Classifier.predict_proba(X_test)
    # Tests if the training is good
    y_pred_tr_knn = Classifier.predict(X_train)
    
    # Check that the model handles the data correctly and that there is a 
    # distribution to the level of security
    plt.figure(figsize=(16,5))
    plt.subplot(1,2,1)
    sns.distplot(pred_proba_knn)
    plt.show()
    #print confusion matrix
    print(metrics.confusion_matrix(y_train, y_pred_tr_knn))
    print(metrics.confusion_matrix(y_test, y_pred_knn))
    #accuracy , recall, precision
    print(classification_report(y_train,y_pred_tr_knn))
    print(classification_report(y_test,y_pred_knn))


features_=data1.iloc[:,data1.columns != 'headache']
target = data_cetegorical['headache']
Logistic_And_KNN(features_ , target)



