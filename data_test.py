import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error



def preprocess():
    '''
    In this part data was loaded and turned into numerical since the data had some categorical variables
    '''
    data = pd.read_csv ('sb_data.csv',encoding= 'unicode_escape')
    #data=pd.get_dummies(data,columns=['Seasons','Holiday','Functioning Day'],drop_first=True)

    data['Seasons']=np.where(data['Seasons'] == 'Spring', 1, data['Seasons'])
    data['Seasons']=np.where(data['Seasons'] == 'Summer', 2, data['Seasons'])
    data['Seasons']=np.where(data['Seasons'] == 'Autumn', 3, data['Seasons'])
    data['Seasons']=np.where(data['Seasons'] == 'Winter', 4, data['Seasons'])
    data['Holiday']=np.where(data['Holiday'] == 'No Holiday', 0, data['Holiday'])
    data['Holiday']=np.where(data['Holiday'] == 'Holiday', 1, data['Holiday'])
    data['Functioning Day']=np.where(data['Functioning Day'] == 'No', 0, data['Functioning Day'])
    data['Functioning Day']=np.where(data['Functioning Day'] == 'Yes', 1, data['Functioning Day'])
    data["Snowfall (cm)"]=np.where(data["Snowfall (cm)"]>0.1,1,data["Snowfall (cm)"])
    data["Snowfall (cm)"]=np.where(data["Snowfall (cm)"]==0,0,data["Snowfall (cm)"])
    data["Rainfall(mm)"]=np.where(data["Rainfall(mm)"]>0.1,1,data["Rainfall(mm)"])
    data["Rainfall(mm)"]=np.where(data["Rainfall(mm)"]==0,0,data["Rainfall(mm)"])
    data["Rented Bike Count"]=data["Rented Bike Count"]+1
    original_data =data["Rented Bike Count"]
    fitted_data, fitted_lambda = stats.boxcox(original_data) #normaazing target varaible

    X=data.drop(['Rented Bike Count','Date','Seasons','Holiday','Functioning Day','Dew point temperature(°C)'],axis =1) #drop features taht we do not need
    y=fitted_data #target



    X_train, X_test, y_train, y_test = train_test_split(X, y) #splitting data into traning  and test

    return X_train, X_test, y_train, y_test


'''
From here code is straigh forward, each fucntion is one model, to run the models go to main function type model name and run
'''

def lin_model(X_train, X_test, y_train, y_test):
    regression_model = LinearRegression()
    fit_regr=regression_model.fit(X_train, y_train)
    reg_train=fit_regr.predict(X_train)
    reg_test=fit_regr.predict(X_test)


    train_set_rmse = (np.sqrt(mean_squared_error(reg_train,y_train )))
    train_set_r2 = r2_score(y_train, reg_train)
    print('Linear model, R2 train score is : {} and the train root mean square is: {}'
         .format(train_set_r2,train_set_rmse))

    test_set_rmse = (np.sqrt(mean_squared_error(reg_test,y_test )))
    test_set_r2 = r2_score(y_test, reg_test)
    print('Linear model, R2 test score is : {} and the test root mean square is: {}'
         .format(test_set_r2,test_set_rmse))

def sv(X_train, X_test, y_train, y_test):  #SVR CLASSIFIER WITH CROSS VALL
    scalerTrain = preprocessing.StandardScaler().fit(X_train)
    scalerTest = preprocessing.StandardScaler().fit(X_test)
    X_train=scalerTrain.transform(X_train)
    X_test=scalerTest.transform(X_test)

    C=[ 10,50,80,100, 500,600,800,1000,1400,1800,2200, 2400 ]
    for i in C:
        svr_Model = SVR(C = i).fit(X_train, y_train)
        r2_train_svr = svr_Model.score(X_train, y_train)
        r2_test_svr=svr_Model.score(X_test, y_test)
        pred_train=svr_Model.predict(X_train)
        rmse_train = np.sqrt(mean_squared_error(y_train,pred_train))
        pred_test=svr_Model.predict(X_test)
        rmse_test = np.sqrt(mean_squared_error(y_test,pred_test))
        print('C = {}\n \
    SVR R2 training: {}, R2 test: {}\n RMSE Train:{}\n RMSE Test:{}'
             .format(i, r2_train_svr, r2_test_svr,rmse_train,rmse_test))

    svr_Model=SVR()

    param = {'C' : [800,1000,1400,1800,2200,2400]}

    gridSearchSVR=GridSearchCV(svr_Model,param,scoring='r2',cv=5)
    gridSearchSVR.fit(X_train,y_train)


    best_SVR=gridSearchSVR.best_estimator_
    bestSVR_testScore=best_SVR.score(X_test,y_test)


    pred_train=gridSearchSVR.predict(X_train)
    rmse_train = np.sqrt(mean_squared_error(y_train,pred_train))

    pred_test=gridSearchSVR.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test,pred_test))

    print('The best R2 train score is : {} with C = {}\n \
    '.format(gridSearchSVR.best_score_,gridSearchSVR.best_params_['C']))
    print('RMSE Train:{}\n'.format(rmse_train))
    print('The best R2 test score is : {}\n with Alpha = {}\n \
    '.format(bestSVR_testScore,gridSearchSVR.best_params_['C']))
    print('RMSE Test:{}\n'.format(rmse_test))





def ridge(X_train, X_test, y_train, y_test):
    alphas=[-2000,-1000, 0, 10, 20, 50, 6000, 1000000]
    for alpha in alphas:
        linridge = Ridge(alpha = alpha).fit(X_train, y_train)
        r2_train = linridge.score(X_train, y_train)
        r2_test = linridge.score(X_test, y_test)
        pred_t=linridge.predict(X_train)
        rmse_t = np.sqrt(mean_squared_error(y_train,pred_t))
        pred=linridge.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test,pred))
        print('Alpha = {}\n \
    R2 training: {}, R2 test: {}\n RMSE Train:{}\n RMSE Test:{}'
             .format(alpha, r2_train, r2_test,rmse_t,rmse))
    ridge=Ridge()
    #parameters={'alpha':[-10000,-6000,-100,-200, 1, 10, 20, 50,100,150,200, 600, 1000000]}
    parameters={'alpha':list(range(-300,300))}

    gridSearchRidge=GridSearchCV(ridge,parameters,scoring='r2',cv=3)
    gridSearchRidge.fit(X_train,y_train)
    best_ridge=gridSearchRidge.best_estimator_
    bestridge_testScore=best_ridge.score(X_test,y_test)

    pred_train=gridSearchRidge.predict(X_train)
    rmse_train = np.sqrt(mean_squared_error(y_train,pred_train))

    pred_test=gridSearchRidge.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test,pred_test))


    print('The best R2 train score is : {}\n with Alpha = {}\n \
    '.format(gridSearchRidge.best_score_,gridSearchRidge.best_params_['alpha']))
    print('The best R2 test score is : {}\n with Alpha = {}\n \
    '.format(bestridge_testScore,gridSearchRidge.best_params_['alpha']))
    print("RMSE for best Ridge Train:{}\nRMSE for ebst Ridge Test {} \n".format(rmse_train,rmse_test))





def lasso_cheker(X_train, X_test, y_train, y_test):
    alphas=[-2000,-1000, 0, 10, 20, 50, 6000, 1000000]
    for alpha in alphas:
        linridge = Lasso(alpha = alpha).fit(X_train, y_train)
        r2_train = linridge.score(X_train, y_train)
        r2_test = linridge.score(X_test, y_test)

        pred_t=linridge.predict(X_train)
        rmse_t = np.sqrt(mean_squared_error(y_train,pred_t))

        pred=linridge.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test,pred))
        print('Alpha = {}\n \
    R2 training: {}, R2 test: {}\n RMSE Train:{}\n RMSE Test:{}'
             .format(alpha, r2_train, r2_test,rmse_t,rmse))

    lasso=Lasso()
    parameters={'alpha':list(range(-300,300))}

    gridSearchLasso=GridSearchCV(lasso,parameters,scoring='r2',cv=4)
    gridSearchLasso.fit(X_train,y_train)



    best_Lasso=gridSearchLasso.best_estimator_
    bestLasso_testScore=best_Lasso.score(X_test,y_test)

    pred_train=gridSearchLasso.predict(X_train)
    rmse_train = np.sqrt(mean_squared_error(y_train,pred_train))

    pred_test=gridSearchLasso.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test,pred_test))


    print('The best R2 train score is : {:.2f} with Alpha = {:.2f}\n \
    '.format(gridSearchLasso.best_score_,gridSearchLasso.best_params_['alpha']))
    print('The best R2 test score is : {:.2f}\n with Alpha = {:.2f}\n \
    '.format(bestLasso_testScore,gridSearchLasso.best_params_['alpha']))

def dec_tree(X_train, X_test, y_train, y_test):
    decisionTree = DecisionTreeRegressor()

    parameters = {'max_depth' : [1,4,5,6,7,10,15,20]}



    gridSearch_decisionTree=GridSearchCV(decisionTree,parameters,scoring='r2',cv=6)
    gridSearch_decisionTree.fit(X_train,y_train)


    best_DecisionTree=gridSearch_decisionTree.best_estimator_
    bestDecisionTree_testScore=best_DecisionTree.score(X_test,y_test)


    pred_train=gridSearch_decisionTree.predict(X_train)
    rmse_train = np.sqrt(mean_squared_error(y_train,pred_train))

    pred_test=gridSearch_decisionTree.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test,pred_test))

    print('The best Decision Tree R2 score is : {} with max depth = {} \n \
    '.format(gridSearch_decisionTree.best_score_,gridSearch_decisionTree.best_params_['max_depth'] ))
    print('RMSE Train:{}\n'.format(rmse_train))
    print('The best R2 test score is : {}\n with max depth = {}\n \
    '.format(bestDecisionTree_testScore,gridSearch_decisionTree.best_params_['max_depth']))
    print('RMSE Test:{}\n'.format(rmse_test))


def RFR(X_train, X_test, y_train, y_test):
    randomForestAlgo = RandomForestRegressor (n_estimators = 1000, random_state = 42)





    parameters = {'max_depth' : [50,60,70,80,90,100]}


    gridSearch_RandomForest=GridSearchCV(randomForestAlgo,parameters,scoring='r2',cv=5)
    gridSearch_RandomForest.fit(X_train,y_train)

    best_randomForest=gridSearch_RandomForest.best_estimator_
    bestRandomForest_testScore=best_randomForest.score(X_test,y_test)



def run():

    X_train, X_test, y_train, y_test = preprocess()

    model = dec_tree(X_train, X_test, y_train, y_test)

    print(model)

    return None


if __name__ == '__main__':
    run()
