##takes in a dataframe, an x_feature as a string, and y_feature as a string
##undersamples the dataframe (lowers number of majority points to equal number of minority points in  dataframe)
##returns the undersampled dataframe
##note: cannot take in na values, na's must be erased prior to inputting a dataframe into undersampling method

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from datasets.models import RawFlower, RawUNM, RawDAR
from django.contrib.auth.models import User

from api import adapters

from api.analysis import merge2CohortFrames, merge3CohortFrames
import os

def undersampling(df, x_feature, y_feature):
    
    #set features and target for undersampling
    feats= df[x_feature].values.reshape(-1, 1)
    targs = df[y_feature]
    ##undersample
    undersample = RandomUnderSampler(sampling_strategy='majority')

    feats_under, targs_under = undersample.fit_resample(feats, targs)
    ##final undersampled dataframe
    return pd.DataFrame({x_feature: list(feats_under.reshape(1,feats_under.shape[0])[0]),y_feature: targs_under})

    
##takes in a dataframe, an x_feature as a string, and y_feature as a string
##oversamples the dataframe (raises number of minority points to equal number of majority points in  dataframe)
##returns the oversampled dataframe
##note: cannot take in na values, na's must be erased prior to inputting a dataframe into oversampling method

def oversampling(df, x_feature, y_feature):
    
    oversample = RandomOverSampler(sampling_strategy='minority')
    #set features and target for undersampling
    feats= df[x_feature].values.reshape(-1, 1)
    targs = df[y_feature]
    ##undersample
    feats_under, targs_under = oversample.fit_resample(feats, targs)
    ##final undersampled dataframe
    return pd.DataFrame({x_feature: list(feats_under.reshape(1,feats_under.shape[0])[0]),y_feature: targs_under})


##This method takes in df as a dataframe, x_feature as a string, y_feature as a string, 
##adjust dilution either "True" or "False" denoting if Result value should have dilution adjusment,
##enocde_cats which is boolean True or False denoting if categorical variables should be one hot encoded, 
##and type_model as one of "Logistic Regression", "Decision Trees", or "kNN" denoting the type of model to run.
##It outputs a dictionary stroing the type of model ran and metrics including Accuracy, AUC, and F1 score.
## crude_regressions(...) will be used run_crude_reg_analysis() below, where crude_reggresions(...)
##is run on each dataframe indivdually 

from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

def turntofloat(df):

    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except:
            pass
    return df

def crude_classification(df, x_feature, y_feature, adjust_dilution, encode_cats, type_model):
    import numpy as np

    ##gets rid of na values in the feature and target columns and keeps different ways
    ##1 and 0 could be written in the target column (Outcome)
    df = df.replace(-9,np.nan).replace('-9',np.nan).replace(999,np.nan).replace(888,np.nan)

    print(df[y_feature].unique())
    
    df = df[(~df[x_feature].isna()) & (~df[y_feature].isna()) & 
        (df[y_feature].isin([0.0,1.0,0,1, '0', '1']))]

    print('inside classification / after filter ******************************************')
    print(df.shape)


    
    #make sure all concentrations are above 0 - assuption is ok because lowest conc should have LOD
    #df = df[df[x_feature]> 0]

    """
    if adjust_dilution == 'True':
        df[x_feature] = df[x_feature] / df['UDR']

        ##old if had len(split_covars) > 1 & 
    if encode_cats == True:
        data = add_confound(df, x_feature, y_feature, split_covars)
        """
    
    if encode_cats == False:
        data = df[[x_feature]+ [y_feature]]
        data = data.dropna(axis = 'rows')
        data = turntofloat(data)

    else:
        data = df[[x_feature]+ [y_feature]]
        data = data.dropna(axis = 'rows')
        data = turntofloat(data)

    data = data[(data[x_feature]> 0) & (~data[x_feature].isna())  ]
    data = data.dropna(how = 'any')
    
    ## ??JAG SHOULD intercept be 1??
    
    # set intercept to 1
    data['intercept'] = 1
    
    ##?? Can I delete this ??
    """#TODO: clean up
    try:
        data['babySex'] = data['babySex'].astype(float)
    except:
        xsd = 1
    try:
        data['parity'] = data['parity'].astype(float)
    except:
        xsd = 1"""
    
    """data = data.select_dtypes(include = ['float','integer'])
    print('Data shape after intselect')
    print(data.shape)"""
    
    ##set features and shape them for the model
    features = data[x_feature].values.reshape(-1, 1)

    ##get the target variable 
    target = data[y_feature]

    # prepare the repeated kFold cross-validation procedure with 10 splits and 10 repeates
    cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=1)
    
    ##run logistic regression model on dataframe if type_model is "Logistic Regression" 
    if type_model=="Logistic Regression":
        # create logistic regression model
        log_reg = LogisticRegression()
        
        ##take the log of features
        log_features=[]
        for feature in features:
            new=np.log(feature)
            log_features.append([new[0]])
        
        ##calculate accuracy, auc, and f1 score metrics for logistic regression model
        accuracy = cross_val_score(log_reg, X=log_features, y=target, scoring='accuracy', cv=cv)
        auc = cross_val_score(log_reg, X=log_features, y=target, scoring="roc_auc", cv = cv)    
        f1 = cross_val_score(log_reg, X=log_features, y=target, cv=cv, scoring='f1')
        
        ##save metrics in a dictionary and return the dictionary
        results_dict={"Type of Model" : type_model, "Accuracy" : format(mean(accuracy), ".2f"),
                      "AUC" : format(mean(auc), ".2f"), "F1" : format(mean(f1), ".2f")}
        return accuracy, auc, f1
    ##?? Do I use log features or just features for dtc and knn ??
    
    elif type_model == "Decision Trees":
        from sklearn.tree import DecisionTreeClassifier
        dtc= DecisionTreeClassifier() ## depth, split crietrion?
        ##calculate accuracy, auc, and f1 score metrics for Decision Trees model
        accuracy = cross_val_score(dtc, X=features, y=target, scoring='accuracy', cv=cv)
        auc = cross_val_score(dtc, X=features, y=target, scoring="roc_auc", cv = cv)    
        f1 = cross_val_score(dtc, X=features, y=target, cv=cv, scoring='f1')
        
        ##save metrics in a dictionary and return the dictionary
        results_dict={"Type of Model" : type_model, "Accuracy" : format(mean(accuracy), ".2f"),
                      "AUC" : format(mean(auc), ".2f"), "F1" : format(mean(f1), ".2f")}
        return accuracy, auc, f1
        
    elif type_model == "kNN":
        knn=KNeighborsClassifier() ## how many neighbours?? 
        ##calculate accuracy, auc, and f1 score metrics for kNN model
        accuracy = cross_val_score(knn, X=features, y=target, scoring='accuracy', cv=cv)
        auc = cross_val_score(knn, X=features, y=target, scoring="roc_auc", cv = cv)    
        f1 = cross_val_score(knn, X=features, y=target, cv=cv, scoring='f1')
        
        ##save metrics in a dictionary and return the dictionary
        results_dict={"Type of Model" : type_model, "Accuracy" : format(mean(accuracy), ".2f"),
                      "AUC" : format(mean(auc), ".2f"), "F1" : format(mean(f1), ".2f")}
        return accuracy, auc, f1
    
    ##else the type of model was written incorretly so output this error explaining what happened
    else:
        print("Type of Model incorrectly inputted. Should be one of 1. Logistic Regression 2.Decision Trees 3. kNN")

    
##print(crude_classification(original, "Result", "Outcome", "True",  True, "Logistic Regression"))
##print(crude_classification(original, "Result", "Outcome", "True", True, "Decision Trees"))
##print(crude_classification(original, "Result", "Outcome", "True", True, "kNN"))

##This runs every type of model (Logistic Reression, Decision Trees, and k-Nearest Nieghbors) on every dataset.
## The original datasets are NEU (Northeastern), UNM (University of New Mexico), and DAR (Dartmouth data).
##Each datasets are stored individually and in combinations in the follwoing way: 

## NEU    UNM    DAR    NEU_UNM    NEU_DAR    UNM_DAR    NEU_UNM_DAR

##then the data is oversampled and added as a seperate copy

## NEU_oversampled    UNM_oversampled    DAR_oversampled  NEU_UNM_oversampled   
##NEU_DAR_oversampled    UNM_DAR_oversampled   NEU_UNM_DAR_oversampled

## and then data is undersampled and added as a seperate copy

## NEU_undersampled    UNM_undersampled    DAR_undersampled  NEU_UNM_undersampled   
##NEU_DAR_undersampled    UNM_DAR_undersampled   NEU_UNM_DAR_undersampled

##then ALL  above steps are repeated, but this time excluding participants who consume seafood in their diet.
##Then adjsuted and non adjusted version of each of the three models are run on all datasets.
##In all, there are 42 datasets and 6 models run on each (3  with diltions adjusmens and 3 without).
##Meaning there are 252 model metrics taken in total.
##the Model metrics from the above crude_regressions(....) is saved in seperate files stored locally

## main analysis

from imblearn.over_sampling import SMOTE
import pandas as pd
##from k_fold_imblearn import KFoldImblearn
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def run_crude_reg_analysis():
    
    #set analysis parameters

    x_feature = 'UTAS'
    Y_features_binary  = ['Outcome']
    y_feature= Y_features_binary[0]
  
    ## Get original data 
    df_NEU = adapters.neu.get_dataframe()
    df_UNM = adapters.unm.get_dataframe()
    df_DAR = adapters.dar.get_dataframe_pred()
    #df_DAR = adapters.unm.get_dataframe()
    
    ##get rid of na values in "Outcome" and "Result" columns in above dataframes
    df_NEU = df_NEU[df_NEU["Outcome"].notna() & df_NEU["UTAS"].notna()]
    df_UNM = df_UNM[df_UNM["Outcome"].notna() & df_UNM["UTAS"].notna()]
    df_DAR = df_DAR[df_DAR["Outcome"].notna() & df_DAR["UTAS"].notna()]

    ## merge data frames, don't need to get rid of na's because we're combining
    ##dataframes without na values alreayd, took care of it above
    ## merge data frames
    df_NEUUNM = merge2CohortFrames(df_NEU,df_UNM)
    df_NEUDAR = merge2CohortFrames(df_NEU,df_DAR)
    df_UNMDAR = merge2CohortFrames(df_UNM,df_DAR)
    df_merged_3 = merge3CohortFrames(df_NEU,df_UNM,df_DAR)
    
    ## Get original data only with particiapnts who DO NOT consume seafood
    
    df_NEU_nofish = adapters.neu.get_dataframe_nofish()
    df_UNM_nofish = adapters.unm.get_dataframe_nofish()
    df_DAR_nofish = adapters.dar.get_dataframe_nofish()
    
    ##get rid of na values in "Outcome" and "Result" columns in above dataframes
    df_NEU_nofish = df_NEU_nofish[df_NEU_nofish["Outcome"].notna() & df_NEU_nofish["UTAS"].notna()]
    df_UNM_nofish = df_UNM_nofish[df_UNM_nofish["Outcome"].notna() & df_UNM_nofish["UTAS"].notna()]
    df_DAR_nofish = df_DAR_nofish[df_DAR_nofish["Outcome"].notna() & df_DAR_nofish["UTAS"].notna()]


    ## merge data frames
    df_NEUUNM_nofish = merge2CohortFrames(df_NEU_nofish,df_UNM_nofish)
    df_NEUDAR_nofish = merge2CohortFrames(df_NEU_nofish,df_DAR_nofish)
    df_UNMDAR_nofish = merge2CohortFrames(df_DAR_nofish,df_UNM_nofish)
    df_merged_3_nofish = merge3CohortFrames(df_NEU_nofish,df_UNM_nofish,df_DAR_nofish)
   
    ##make undersampled Majority dataframes

    under_df_NEU = undersampling(df_NEU, x_feature, y_feature)
    under_df_UNM = undersampling(df_UNM, x_feature, y_feature)
    under_df_DAR = undersampling(df_DAR, x_feature, y_feature)
    
    under_df_NEUUNM = undersampling(df_NEUUNM, x_feature, y_feature)
    under_df_NEUDAR = undersampling(df_NEUDAR, x_feature, y_feature)
    under_df_UNMDAR = undersampling(df_UNMDAR, x_feature, y_feature)
    under_df_merged_3 = undersampling(df_merged_3, x_feature, y_feature)
    
    ##make undersampled Majority dataframes only with particiapnts who DO NOT consume seafood 
    
    under_df_NEU_nofish = undersampling(df_NEU_nofish, x_feature, y_feature)
    under_df_UNM_nofish = undersampling(df_UNM_nofish, x_feature, y_feature)
    under_df_DAR_nofish = undersampling(df_DAR_nofish, x_feature, y_feature)
    
    under_df_NEUUNM_nofish = undersampling(df_NEUUNM_nofish, x_feature, y_feature)
    under_df_NEUDAR_nofish = undersampling(df_NEUDAR_nofish, x_feature, y_feature)
    under_df_UNMDAR_nofish= undersampling(df_UNMDAR_nofish, x_feature, y_feature)
    under_df_merged_3_nofish = undersampling(df_merged_3_nofish, x_feature, y_feature)
     
    ##get oversampled Outcome Majority dataframes

    over_df_NEU = oversampling(df_NEU, x_feature, y_feature)
    over_df_UNM = oversampling(df_UNM, x_feature, y_feature)
    over_df_DAR = oversampling(df_DAR, x_feature, y_feature)
    
    over_df_NEUUNM = oversampling(df_NEUUNM, x_feature, y_feature)
    over_df_NEUDAR = oversampling(df_NEUDAR, x_feature, y_feature)
    over_df_UNMDAR = oversampling(df_UNMDAR, x_feature, y_feature)
    over_df_merged_3 = oversampling(df_merged_3, x_feature, y_feature)
    
    ##get oversampled Outcome Majority dataframes only with particiapnts who DO NOT consume seafood
    
    over_df_NEU_nofish = oversampling(df_NEU_nofish, x_feature, y_feature)
    over_df_UNM_nofish = oversampling(df_UNM_nofish, x_feature, y_feature)
    over_df_DAR_nofish = oversampling(df_DAR_nofish, x_feature, y_feature)
    
    over_df_NEUUNM_nofish = oversampling(df_NEUUNM_nofish, x_feature, y_feature)
    over_df_NEUDAR_nofish = oversampling(df_NEUDAR_nofish, x_feature, y_feature)
    over_df_UNMDAR_nofish= oversampling(df_UNMDAR_nofish, x_feature, y_feature)
    over_df_merged_3_nofish = oversampling(df_merged_3_nofish, x_feature, y_feature)
    
    ##add all of above dataframes to a list
    frames_for_analysis = [
        ('NEU', df_NEU),
        ('UNM', df_UNM),
        ('DAR', df_DAR),
        ('NEUUNM', df_NEUUNM),
        ('NEUDAR', df_NEUDAR),
        ('UNMDAR', df_UNMDAR),
        ('UNMDARNEU', df_merged_3),
        ('NEU NO Fish', df_NEU_nofish),
        ('UNM NO Fish', df_UNM_nofish),
        ('DAR NO Fish', df_DAR_nofish),
        ('NEUUNM NO Fish', df_NEUUNM_nofish),
        ('NEUDAR NO Fish', df_NEUDAR_nofish),
        ('UNMDAR NO Fish', df_UNMDAR_nofish),
        ('UNMDARNEU NO Fish', df_merged_3_nofish),
        
        ('NEU Undersampled', under_df_NEU),
        ('UNM Undersampled', under_df_UNM),
        ('DAR Undersampled', under_df_DAR),
        ('NEUUNM Undersampled', under_df_NEUUNM),
        ('NEUDAR Undersampled', under_df_NEUDAR),
        ('UNMDAR Undersampled', under_df_UNMDAR),
        ('UNMDARNEU Undersampled', under_df_merged_3),
        ('NEU NO Fish Undersampled', under_df_NEU_nofish),
        ('UNM NO Fish Undersampled', under_df_UNM_nofish),
        ('DAR NO Fish Undersampled', under_df_DAR_nofish),
        ('NEUUNM NO Fish Undersampled', under_df_NEUUNM_nofish),
        ('NEUDAR NO Fish Undersampled', under_df_NEUDAR_nofish),
        ('UNMDAR NO Fish Undersampled', under_df_UNMDAR_nofish),
        ('UNMDARNEU NO Fish Undersampled', under_df_merged_3_nofish),
        
        ('NEU Oversampled', over_df_NEU),
        ('UNM Oversampled', over_df_UNM),
        ('DAR Oversampled', over_df_DAR),
        ('NEUUNM Oversampled', over_df_NEUUNM),
        ('NEUDAR Oversampled', over_df_NEUDAR),
        ('UNMDAR Oversampled', over_df_UNMDAR),
        ('UNMDARNEU Oversampled', over_df_merged_3),
        ('NEU NO Fish Oversampled', over_df_NEU_nofish),
        ('UNM NO Fish Oversampled', over_df_UNM_nofish),
        ('DAR NO Fish Oversampled', over_df_DAR_nofish),
        ('NEUUNM NO Fish Oversampled', over_df_NEUUNM_nofish),
        ('NEUDAR NO Fish Oversampled', over_df_NEUDAR_nofish),
        ('UNMDAR NO Fish Oversampled', over_df_UNMDAR_nofish),
        ('UNMDARNEU NO Fish Oversampled', over_df_merged_3_nofish)
    ]
    
    
    ##printing the name and shape of each dataframe above
    for name, df in frames_for_analysis:
        print('Data Stats')
        print(name+ " : " + str(df.shape))



    ##??JAG :  What do we do for Daniel?? 
    # set output paths for results:
    output_path = '/usr/src/app/mediafiles/analysisresults/juliaresults/'
    ##output_path_model1_noadj = '/usr/src/app/mediafiles/analysisresults/model1noadj/'
    
   ## output_path_model1_adj = 'C:/Users/J-Dog/Desktop/model1adj/'
    ##output_path_model1_noadj = 'C:/Users/J-Dog/Desktop/model1noadj/'
    
    try:
        os.mkdir(output_path)
    except:
        print('Exists')
    
    
    #try:
   #     os.mkdir(output_path_model1_adj)
   #     os.mkdir(output_path_model1_noadj)
    #except:
    #    print('Exists')

    # start analysis with adjusted "Result" values
    ##main loop that iterates through every dataframe in 
    ##frames_for_analysis list

    all_results = []

    for name, frame in frames_for_analysis:
    
        print(name + ":")
        print('Min: {} Max: {}'.format(frame["UTAS"].min(), frame["UTAS"].max()))
        frame = frame[(frame["UTAS"] > 0) & (~frame["UTAS"].isna())]
        print()
        
        ##gets the Logsitic Regression Model results for the dataframe
        for y_feature in Y_features_binary:
            ##open file in specified path, .format... renames files
            ##text_file = open(output_path_model1_adj + "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature), "w")

            try:
                ##gets model results and write them to the ile in above path
                accuracy, auc, f1  = crude_classification(frame, x_feature, y_feature,'True', True, "Logistic Regression")
                dims = frame.shape
                for i in range(0, len(accuracy)):

                    all_results.append(["Logistic Regression",name, dims[0], i, accuracy[i], auc[i], f1[i]])
                ##dims = frame.shape
                ##text_file.write(name + "\n")
                ##text_file.write(str(frame[[x_feature] + [y_feature]].describe()))
                ##text_file.write('\n')
                ##text_file.write("Number of participants: {}\n".format(dims[0]))
                ##will be the list of results
                ##text_file.write(str(out))
                ##text_file.write("\n" +"\n" +"\n" +"**************")

            ##output error if file incorrectly written
            except Exception as e:
                print('Logistic Regression Error:**')
                print(e)
                ##text_file.write('Logistic Regression Error:*\n')
                ##text_file.write(str(e))
            ##text_file.close()
       
        ##gets the Decision Trees Model results for the dataframe
        for y_feature in Y_features_binary:
            ##open file in specified path, .format... renames files
            ##text_file = open(output_path_model1_adj + "decision_trees_classifier_{}_{}({}).txt".format(name, y_feature, x_feature), "w")
            try:
                ##gets model results
                accuracy, auc, f1  = crude_classification(frame, x_feature, y_feature,'True', True, "Decision Trees")
                dims = frame.shape
                for i in range(0, len(accuracy)):

                    all_results.append(["Decision Trees", name, dims[0], i, accuracy[i], auc[i], f1[i]])
                
                ##text_file.write(name + "\n")
                ##text_file.write(str(frame[[x_feature] + [y_feature]].describe()))
                ##text_file.write('\n')
                ##text_file.write("Number of participants: {}\n".format(dims[0]))
                ##will be the list of results
                ##text_file.write(str(out))
                ##text_file.write("\n" +"\n" +"\n" +"**************")
            except Exception as e:
                print('Decision Trees Error:**')
                print(e)
                ##text_file.write('Decision Trees Error:*\n')
                ##text_file.write(str(e))
            ##text_file.close()
        
        ##gets the k-Nearest Neighbor Model results for the dataframe
        for y_feature in Y_features_binary:
            ##open file in specified path, .format... renames files
            ##text_file = open(output_path_model1_adj + "kNN{}_{}({}).txt".format(name, y_feature, x_feature), "w")

            try:
                ##gets model results
                accuracy, auc, f1  = crude_classification(frame, x_feature, y_feature,'True', True, "kNN")
                dims = frame.shape
                for i in range(0, len(accuracy)):
                    all_results.append(["kNN", name, dims[0], i, accuracy[i], auc[i], f1[i]])
                

                ##text_file.write(name + "\n")
                ##text_file.write(str(frame[[x_feature] + [y_feature]].describe()))
                ##text_file.write('\n')
                ##text_file.write("Number of participants: {}\n".format(dims[0]))
                ##will be the list of results
                ##text_file.write(str(out))
                ##text_file.write("\n" +"\n" +"\n" +"**************")
            
            except Exception as e:
                print('kNN:**')
                print(e)
                ##text_file.write('kNN Error:*\n')
                ##text_file.write(str(e))
            ##text_file.close()
    final = pd.DataFrame(all_results, columns = ['Model','dataset', 'N', 'fold','accuracy','auc','f1'])
    final.to_csv(output_path + 'all_model_results.csv', index = False)
    print(all_results)
            
    
  
