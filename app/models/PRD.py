import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle
from sklearn.metrics import make_scorer
from app.utils import custom_metric, encoder
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier
import numpy as np
import json




class TreeBasedModel:

    def __init__(self, model_type, filter_type, dataset= None, filepath = "./model/model_store/ML_Classifier.pkl"  , label_path = "./model/model_store/ML_Classifier.json", hyperparameter_tuning = True, only_parts = True, dimensionality_reduction = False, freq = 'None', overbook_grid = True, warranty = False):
        self.model_type = model_type
        self.dataset = dataset
        self.filter_type = filter_type
        self.hyperparameter_tuning = hyperparameter_tuning
        self.model = False
        self.X_test = False
        self.y_test = False
        self.filepath = filepath
        self.dimensionality_reduction = dimensionality_reduction
        self.only_parts = only_parts
        self.label_path = label_path
        self.freq = freq
        self.ovb = overbook_grid
        self.warranty = warranty

    def train_supervised(self):
        # TODO: check temp comment
        """
        Preprocessing and training supervised model depending on object variables. Returns test predictions.
        
        Keyword arguments:
        self -- initialized TreeBasedModel object
        """

        # Adding Encoding Method Logic
        
        self.tktno = self.dataset['ID'].unique()
        self.dataset.loc[self.dataset['VERSION'].isnull(), 'VERSION'] =  'NONE'

        if self.filter_type == 'label_version_all':
            input_df, output_df, label = encoder.label_version_encoder(self.dataset,only_parts = True, symptom3= True, freq = self.freq, warranty = self.warranty)


        if label:
            with open(self.label_path, 'w', encoding='utf-8') as f:
                json.dump(label, f, ensure_ascii=False, indent=4)
       

        X_train, X_test, y_train, y_test = train_test_split(input_df, output_df, test_size=0.2, random_state=42)
        X_train = X_train.astype(int)
        y_train = y_train.astype(int)
        X_test = X_test.astype(int)
        y_test = y_test.astype(int)


        self.tkt_test =  self.tktno[X_test.index]

        if self.model_type == 'RF':
            model = RandomForestClassifier(random_state=42)
            parameters = {
                        'max_depth': [None],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1,2,4],
                        'n_estimators': [100, 200, 400]
                        ,'criterion' : ['gini']}

        elif self.model_type == 'XGB':
            model = xgb.XGBClassifier(random_state=42)
            parameters = {
                        'colsample_bytree' : [0.1, 0.3,1],
                        'eta': [0.1, 0.3, 0.5],
                        'n_estimators': [100,200],
                        'subsample': [0.8,1]

            }
        

        if self.hyperparameter_tuning == True:
            if self.ovb:
                custom_scorer = make_scorer(custom_metric.overbooking_similarity, tktno=self.tktno)
                grid = GridSearchCV(model, param_grid= parameters, scoring= custom_scorer, verbose=10)
            else:
                grid = GridSearchCV(model, param_grid= parameters, verbose=10)
                
            grid.fit(X_train,y_train)
            model = grid.best_estimator_

        else:
            model.fit(X_train,y_train)
        

        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        return self.model, self.X_test, self.y_test, self.tkt_test,self.tktno
    



    def train_new_supervised(self):
        # TODO: check temp comment
        """
        Preprocessing and training supervised model depending on object variables, catching errors in hyperparameter tuning. Returns test data.
        
        Keyword arguments:
        self -- initialized TreeBasedModel object
        """
        self.tktno = self.dataset['ID'].unique()
        self.dataset.loc[self.dataset['VERSION'].isnull(), 'VERSION'] =  'NONE'

        if self.filter_type == 'label_version_all':
            input_df, output_df, label = encoder.label_version_encoder(self.dataset,only_parts = True, symptom3= True, freq = self.freq, warranty = self.warranty)
        

        if label:
            with open(self.label_path, 'w', encoding='utf-8') as f:
                json.dump(label, f, ensure_ascii=False, indent=4)


        input_df = input_df.astype(int)
        output_df = output_df.astype(int)

        self.inflow_tkt =  self.tktno[input_df.index]


        if self.model_type == 'RF':
            model = RandomForestClassifier(random_state=42)
            parameters = {
                        'max_depth': [None],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1,2,4],
                        'n_estimators': [100, 200, 400]
                        ,'criterion' : ['gini']}

        elif self.model_type == 'XGB':
            model = xgb.XGBClassifier(random_state=42)
            parameters = {
                        'colsample_bytree' : [0.1, 0.3,1],
                        'eta': [0.1, 0.3, 0.5],
                        'n_estimators': [100,200],
                        'subsample': [0.8,1]

            }
        


        if self.hyperparameter_tuning == True:
            try:
                if self.ovb:
                    custom_scorer = make_scorer(custom_metric.overbooking_similarity, tktno=self.tktno)
                    grid = GridSearchCV(model, param_grid= parameters, scoring= custom_scorer, verbose=10)
                else:
                    grid = GridSearchCV(model, param_grid= parameters, verbose=10)
                grid.fit(input_df,output_df)
                model = grid.best_estimator_
            except:
                model =  xgb.XGBClassifier(eta= 0.1,alpha= 0 ,random_state=42)
                model.fit(input_df,output_df)

        else:
            model.fit(input_df, output_df)
        self.input = input_df
        self.model = model
        return self.model, output_df, self.inflow_tkt

    def test_prediction(self):
        # TODO: check temp comment
        """
        Returns prediction of trained model on testing data.
        
        Keyword arguments:
        self -- initialized TreeBasedModel object
        """
        pred = self.model.predict(self.X_test)
        return pred




    def prediction(self, new_df, ML_path =  "./model/model_store/ML_Classifier.pkl",  label_encoder_path= "./model/model_store/ML_Classifier.json"):
        # TODO: check temp comment
        """
        Preprocesses input dataframe and outputs model predictions on that dataframe.
        
        Keyword arguments:
        self -- initialized TreeBasedModel object
        new_df -- dataframe to be modelled on
        ML_path -- path to saved model pkl file (default "./model/model_store/ML_Classifier.pkl")
        label_encoder_path -- path to saved label encoder JSON file (default "./model/model_store/ML_Classifier.json")
        """
        new_df.loc[new_df['VERSION'].isnull(), 'VERSION'] =  'NONE'
        new_df['VERSION'] = new_df['VERSION'].astype(str)


        if self.filter_type == 'label_version_all':
            input_df, removed_tkts = encoder.inflow_label_version_encoder(new_df= new_df, symptom3= True, label_encoder_path=label_encoder_path, warranty = self.warranty)


        with open(ML_path, 'rb') as f:
            pred_model = pickle.load(f)


        pred = pred_model.predict(input_df)

        return pred, removed_tkts
        
    def pickle_store(self):
        # TODO: check temp comment
        """
        Saves model to a pkl file.
        
        Keyword arguments:
        self -- initialized TreeBasedModel object
        """
        model_pkl_file = self.filepath
        
        with open(model_pkl_file, 'wb') as file:  
            pickle.dump(self.model, file)
