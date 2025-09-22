import json
import pandas as pd
import numpy as np
import nltk
from utils import decoder #POD_preprocessing, decoder, POD_preprocessing
from nltk.corpus import words
import re
from collections import defaultdict, Counter
from model import PRD, PROD #POD, PRD, PROD
import json
from utils import decoder, special_configuration #decoder, sql_function, special_configuration
import os
#from dotenv import load_dotenv
#import oracledb

#load_dotenv() 

import warnings
warnings.filterwarnings("ignore")

## For Server Setup : requires nltk download

# nltk.download('punkt')
# nltk.download('words')

def js_r(filename: str):
# Function : opening up json file as a dictionary
    with open(filename) as f_in:
        return json.load(f_in)

def pkl_file_check(df,file_name_list):
# Function : goes over directory 'model_store' to check whether Triage ML pickle file exists
    excluded_df = df[df['MODELNO'].isin(file_name_list)]
    non_inputtable = df[~df['MODELNO'].isin(file_name_list)]
    non_inputtable_tkt = non_inputtable.ID.unique()
    return non_inputtable_tkt, excluded_df

def combination_check(df,directory,modelno):
# Function : goes over directory 'model_store' for encoder json file, for input combination
    ### Temporary Case
    ###
    df['COMBINATION'] = df['SYMPTOMDESCRIPTION1'].astype(str) + '+' + df['SYMPTOMDESCRIPTION2'].astype(str) + '+' + df['SYMPTOMDESCRIPTION3'].astype(str)
    ###
    modellist = df.MODELNO.unique()
    file_path = directory+modelno + '.json'
    with open(file_path, 'r') as file:
        enc = json.load(file)
    filtered_df = df[df['COMBINATION'].isin(enc['COMBINATION'])]
    non_inputtable = df[~df['COMBINATION'].isin(enc['COMBINATION'])]

    non_inputtable_tkt = non_inputtable.ID.unique()
    return non_inputtable_tkt, filtered_df

def prodtype_adder(df,final_df):
# Function : 
    tktlist =  df.ID.unique()

    for tkt in tktlist:
        prodtype = df['PRODUCTTYPE'][df['ID']==tkt].iloc[0]  
        final_df.loc[final_df['ID']==tkt,'PRODUCTTYPE'] = prodtype
    
    return final_df



def daily_run(df, rev1, rev2, prodtype, warranty = False):
    final_model_result_df = pd.DataFrame()
    df['VERSION'] = df['VERSION'].astype(str)

    pred = None
    model_train_all_df = df
    ML_path = "/home/ubuntu/KW_Triage_PR/model/model_store/" + rev1 + ".pkl"
    label_encoder_path = "/home/ubuntu/KW_Triage_PR/model/model_store/" + rev1 + ".json"
    # Adding a for loop to accommodate versions as well
    pod_versions = list(model_train_all_df['VERSION'].unique())
    model = PRD.TreeBasedModel(model_type = 'XGB', filter_type = 'label_version_all', filepath= ML_path, label_path= label_encoder_path, only_parts = True, warranty = warranty)
    for v in pod_versions:
    
        model_train_df = model_train_all_df[model_train_all_df['VERSION']==v]
        if v == 'None':
            print('No Version. Default Version')
            POD_path = f"/home/ubuntu/KW_Triage_PR/model/POD_jsons/{rev2}.json"
        else:
            POD_path = f"/home/ubuntu/KW_Triage_PR/model/POD_jsons/{rev2}_{v}.json"


        pred, removed_tkts = model.prediction(new_df=model_train_df, ML_path = ML_path , label_encoder_path= label_encoder_path)
        label_encoder = js_r(label_encoder_path)
        instruction_filepath = f'/home/ubuntu/KW_Triage_PR/data/instructions/{prodtype}.json'
        try:
            prod = js_r(instruction_filepath)
        except:
            print(f'No Instruction provided {prodtype}')
        try:
            # Adding a mechanism to add a duplicate of base version pod json file for a new encountered version
            try:
                POD_json = js_r(POD_path)
            except:
                ### New Version Method
                print("New Version Encountered. Reverting to Base Version")
                POD_path =  f"/home/ubuntu/KW_Triage_PR/model/POD_jsons/{rev2}.json"
                POD_json = js_r(POD_path)
                new_path = f"/home/ubuntu/KW_Triage_PR/model/POD_jsons/{rev2}_{v}.json"
                # with open(new_path, 'w', encoding='utf-8') as f:
                #     json.dump(POD_json, f, ensure_ascii=False, indent=4) 
                # print("Duplication of Base Version Completed")
                ### 

            prod_model = PROD.PROD(df= model_train_df, pred = pred, label_encoder= label_encoder, instruction = prod, POD_json = POD_json)
            model_result_df = prod_model.symptomcode_trigger()
            final_model_result_df = pd.concat([final_model_result_df, model_result_df], ignore_index=True)
        except:
            parts = json.loads(open(label_encoder_path,"r").read())
            try: 
                model_result_df= decoder.new_decode(df = model_train_df, pred=pred, parts = parts, wono=model_train_df)
                final_model_result_df = pd.concat([final_model_result_df, model_result_df], ignore_index=True)
            except:
                print("No ModelNo trained")
    
    final_model_result_df = decoder.part_descripton_add(final_model_result_df)


    return final_model_result_df


def single_daily_run(df, rev1, rev2, prodtype, warranty = False):
    final_model_result_df = pd.DataFrame()
    df['VERSION'] = df['VERSION'].astype(str)

    pred = None
    model_train_all_df = df
    ML_path = "/home/ubuntu/KW_Triage_PR/model/model_store/" + rev1 + ".pkl"
    label_encoder_path = "/home/ubuntu/KW_Triage_PR/model/model_store/" + rev1 + ".json"
    # Adding a for loop to accommodate versions as well
    pod_versions = list(model_train_all_df['VERSION'].unique())
    model = PRD.TreeBasedModel(model_type = 'XGB', filter_type = 'label_version_all', filepath= ML_path, label_path= label_encoder_path, only_parts = True, warranty = warranty)
    for v in pod_versions:
    
        model_train_df = model_train_all_df[model_train_all_df['VERSION']==v]
        if v == 'None':
            print('No Version. Default Version')
            POD_path = f"/home/ubuntu/KW_Triage_PR/model/POD_jsons/{rev2}.json"
        else:
            POD_path = f"/home/ubuntu/KW_Triage_PR/model/POD_jsons/{rev2}_{v}.json"


        pred, removed_tkts = model.prediction(new_df=model_train_df, ML_path = ML_path , label_encoder_path= label_encoder_path)
        label_encoder = js_r(label_encoder_path)
        instruction_filepath = f'/home/ubuntu/KW_Triage_PR/data/instructions/{prodtype}.json'
        try:
            prod = js_r(instruction_filepath)
        except:
            print(f'No Instruction provided {prodtype}')
        try:
            # Adding a mechanism to add a duplicate of base version pod json file for a new encountered version
            try:
                POD_json = js_r(POD_path)
            except:
                ### New Version Method
                print("New Version Encountered. Reverting to Base Version")
                POD_path =  f"/home/ubuntu/KW_Triage_PR/model/POD_jsons/{rev2}.json"
                POD_json = js_r(POD_path)
                new_path = f"/home/ubuntu/KW_Triage_PR/model/POD_jsons/{rev2}_{v}.json"
                # with open(new_path, 'w', encoding='utf-8') as f:
                #     json.dump(POD_json, f, ensure_ascii=False, indent=4) 
                # print("Duplication of Base Version Completed")
                ### 

            prod_model = PROD.PROD(df= model_train_df, pred = pred, label_encoder= label_encoder, instruction = prod, POD_json = POD_json)
            model_result_df = prod_model.symptomcode_trigger()
            final_model_result_df = pd.concat([final_model_result_df, model_result_df], ignore_index=True)
        except:
            parts = json.loads(open(label_encoder_path,"r").read())
            try: 
                model_result_df= decoder.new_decode(df = model_train_df, pred=pred, parts = parts, wono=model_train_df)
                final_model_result_df = pd.concat([final_model_result_df, model_result_df], ignore_index=True)
            except:
                print("No ModelNo trained")
    
    final_model_result_df = decoder.part_descripton_add(final_model_result_df)


    return final_model_result_df



# Multiple tickets
def prodrun():
    directory = '/home/ubuntu/KW_Triage_PR/model/model_store/'
    
    ### Temporary Seal Accommodation
    seal_dict = js_r('/home/ubuntu/KW_Triage_PR/data/special_case/seal.json')
    seal_modelno = list(seal_dict.keys())
    ###

    # File format
    file_format = '.pkl'
    file_names = [file[:-len(file_format)] for file in os.listdir(directory) if file.endswith(file_format)]
    file_names = [name.replace('+', '/') for name in file_names]


    #user=os.getenv('USER_DSN')
    #password=os.getenv('PASSWORD')
    #dsn = os.getenv('DSN')

    #connection = oracledb.connect(
    #user= user,
    #password= password,
    #dsn= dsn)
    
    #df= sql_function.extraction(connection)
    non_pkl_tkts, df = pkl_file_check(df=df, file_name_list= file_names)
    #sql_function.status_ml_exclusion(non_pkl_tkts, connection=connection)
    final_df = pd.DataFrame()
    modelList = df.MODELNO.unique()
    for modelNo in modelList:

        try:
            rev1 = modelNo.replace('/','+',1)
            rev2 = modelNo.replace('/','+',1)
            mdf = df[df['MODELNO'] == modelNo]
            prodtype = mdf['PRODUCTTYPE'].iloc[0] 
            ### Seal Temporary Logic
            if modelNo in seal_modelno:
                try:
                    seal_result_df = special_configuration.seal_system_temp(seal_dict=seal_dict, seal_df= mdf)
                    print('SEAL MODELNO CG')
                    final_df = pd.concat([final_df, seal_result_df], ignore_index=True)
                    continue
                except:
                    print('NON SEAL SYMPTOM')
            ### Seal Temporary Logic

            ### VDE LED Version Block
            if modelNo in ["QN77S90DAFXZA", "QN77S90DDFXZA"]:
                print(modelNo)
            else:
                if prodtype == 'VDE_LED':
                    print('VDE_LED case')
                    workorder_temp_list = list(mdf.WORKORDERID.unique())
                    for temp_wo in workorder_temp_list:
                        check_df = mdf[mdf['WORKORDERID'] == temp_wo]
                        versions_led =  list(check_df['VERSION'].unique())
                        version_led = str(versions_led[0])
                        if version_led == "None":
                            print('NO VERSION VDE_LED TICKET')
                            mdf = mdf[mdf['WORKORDERID']!= temp_wo] 
                            #sql_function.status_led(temp_wo,connection)
                            print('NO VERSION INSERTION SUCCESS')

            ### 

            non_inputtable_tkt, mdf = combination_check(df=mdf,directory=directory, modelno= rev1)
            if non_inputtable_tkt.any():
                #sql_function.status_fail(non_inputtable_tkt,connection)
                print('Ticket not inputtable, need to implement logic still')
            final_model_result_df = daily_run(mdf,rev1=rev1,rev2=rev2, prodtype=prodtype)

            ### Temporary request block
            if modelNo in ["QN77S90DAFXZA", "QN77S90DDFXZA"]:
                try:
                    final_model_result_df['PARTNO'][final_model_result_df['PARTNO'] == "BN44-01264A"] =  "BN44-01329A"
                    final_model_result_df['PARTDESC'][final_model_result_df['PARTNO'] == "BN44-01329A"] =  "DC VSS-POWER BOARD;L77QA9N_FVD,AC/DC,418"
                except ValueError as e:
                    # Print the error message
                    print(f"An error occurred: {e}")
            ###

            final_df = pd.concat([final_df, final_model_result_df], ignore_index=True)
        except Exception as e:
            print(rev1, 'does not have combination for ticket')
            # print(f"An error occurred: {e}")
            continue

    try:
        final_df['MODELTYPE'] = final_df['MODELTYPE'].fillna(value='PROGRAM')
        final_df['KEYWORD']= final_df['KEYWORD'].fillna(value='NONE')
        final_df['PARTNO']= final_df['PARTNO'].fillna(value='DIAGNOSIS')
        final_df['STATUS'] = 1

        reorder= ['ID', 'WORKORDERID', 'PRODUCTTYPE','PARTNO', 'MODELTYPE', 'KEYWORD','PARTDESC', 'STATUS']
        final_df = final_df[reorder]
        final_df =  prodtype_adder(df=df,final_df=final_df)
        
        # SPECIAL CONFIGURATION: DAMPER
        try:
            final_df = special_configuration.damper_add(final_df)
        except:
            print('not worked')
        
        # SPECIAL CONFIGURATION: ICEMAKER TWO PART NO ['DA97-22160A','DA97-22162A']
        try:
            final_df = special_configuration.icemaker_two_part_no(final_df)
            print('icemaker one of the two parts removed')
        except:
            print('icemaker parts not removed')
        

        # sql_function.status_unconverted(connection= connection)
        
        try:
            #sql_function.df_insert(final_df,connection=connection)
            print('INSERTION COMPLETED')
        except Exception as e:
            # Handle the exception and output the error message
            print('INSERTION FAILED')
            print(f"An error occurred: {e}")

        try:
            #sql_function.status_success(final_df, connection=connection)
            print('STATUS SUCCESS')
        except Exception as e:
            # Handle the exception and output the error message
            print(f"An error occurred: {e}")

    except:
        print('No Tickets to Predict!')
    
    connection.close()

def single_prodrun():
    directory = '/home/ubuntu/KW_Triage_PR/model/model_store/'
    
    ### Temporary Seal Accommodation
    seal_dict = js_r('/home/ubuntu/KW_Triage_PR/data/special_case/seal.json')
    seal_modelno = list(seal_dict.keys())
    ###

    # File format
    file_format = '.pkl'
    file_names = [file[:-len(file_format)] for file in os.listdir(directory) if file.endswith(file_format)]
    file_names = [name.replace('+', '/') for name in file_names]    

    # Example of single row
    test_row = pd.DataFrame({
        'MODELNO': ['RF263BEAESG/AA'], 
        'VERSION': ['4'],
        'MANUFACTUREMONTH': ['201708'],
        'SYMPTOMDESCRIPTION1': ['Ice/Water/Sparkling'],
        'SYMPTOMDESCRIPTION2': ['Ice making/ice bucket stuck issue'],
        'SYMPTOMDESCRIPTION3': ['Ice room bucket frost'],
        'STATUS': ['220'],
        'PRODUCTTYPE': ['REF_REF']
        })
    
    print(test_row)

    non_pkl_tkts, df = pkl_file_check(df=df, file_name_list= file_names)

    final_df = pd.DataFrame()
    modelList = df.MODELNO.unique()
    for modelNo in modelList:
        try:
            rev1 = modelNo.replace('/','+',1)
            rev2 = modelNo.replace('/','+',1)
            mdf = df[df['MODELNO'] == modelNo]
            prodtype = mdf['PRODUCTTYPE'].iloc[0] 
            ### Seal Temporary Logic
            if modelNo in seal_modelno:
                try:
                    seal_result_df = special_configuration.seal_system_temp(seal_dict=seal_dict, seal_df= mdf)
                    print('SEAL MODELNO CG')
                    final_df = pd.concat([final_df, seal_result_df], ignore_index=True)
                    continue
                except:
                    print('NON SEAL SYMPTOM')
            ### Seal Temporary Logic

            ### VDE LED Version Block
            if modelNo in ["QN77S90DAFXZA", "QN77S90DDFXZA"]:
                print(modelNo)
            else:
                if prodtype == 'VDE_LED':
                    print('VDE_LED case')
                    workorder_temp_list = list(mdf.WORKORDERID.unique())
                    for temp_wo in workorder_temp_list:
                        check_df = mdf[mdf['WORKORDERID'] == temp_wo]
                        versions_led =  list(check_df['VERSION'].unique())
                        version_led = str(versions_led[0])
                        if version_led == "None":
                            print('NO VERSION VDE_LED TICKET')
                            mdf = mdf[mdf['WORKORDERID']!= temp_wo]

            ### 

            non_inputtable_tkt, mdf = combination_check(df=mdf,directory=directory, modelno= rev1)
            if non_inputtable_tkt.any():
                #sql_function.status_fail(non_inputtable_tkt,connection)
                print('Ticket not inputtable, need to implement logic still')
            final_model_result_df = single_daily_run(mdf,rev1=rev1,rev2=rev2, prodtype=prodtype)

            # Stopping here to test output
            print(final_model_result_df)
            raise

            ### Temporary request block
            if modelNo in ["QN77S90DAFXZA", "QN77S90DDFXZA"]:
                try:
                    final_model_result_df['PARTNO'][final_model_result_df['PARTNO'] == "BN44-01264A"] =  "BN44-01329A"
                    final_model_result_df['PARTDESC'][final_model_result_df['PARTNO'] == "BN44-01329A"] =  "DC VSS-POWER BOARD;L77QA9N_FVD,AC/DC,418"
                except ValueError as e:
                    # Print the error message
                    print(f"An error occurred: {e}")
            ###

            final_df = pd.concat([final_df, final_model_result_df], ignore_index=True)
        except Exception as e:
            print(rev1, 'does not have combination for ticket')
            # print(f"An error occurred: {e}")
            continue
    
    try:
        final_df['MODELTYPE'] = final_df['MODELTYPE'].fillna(value='PROGRAM')
        final_df['KEYWORD']= final_df['KEYWORD'].fillna(value='NONE')
        final_df['PARTNO']= final_df['PARTNO'].fillna(value='DIAGNOSIS')
        final_df['STATUS'] = 1

        reorder= ['ID', 'WORKORDERID', 'PRODUCTTYPE','PARTNO', 'MODELTYPE', 'KEYWORD','PARTDESC', 'STATUS']
        final_df = final_df[reorder]
        final_df =  prodtype_adder(df=df,final_df=final_df)
        
        # SPECIAL CONFIGURATION: DAMPER
        try:
            final_df = special_configuration.damper_add(final_df)
        except:
            print('not worked')
        
        # SPECIAL CONFIGURATION: ICEMAKER TWO PART NO ['DA97-22160A','DA97-22162A']
        try:
            final_df = special_configuration.icemaker_two_part_no(final_df)
            print('icemaker one of the two parts removed')
        except:
            print('icemaker parts not removed')
        

        # sql_function.status_unconverted(connection= connection)
        
        try:
            #sql_function.df_insert(final_df,connection=connection)
            print('INSERTION COMPLETED')
        except Exception as e:
            # Handle the exception and output the error message
            print('INSERTION FAILED')
            print(f"An error occurred: {e}")

        try:
            #sql_function.status_success(final_df, connection=connection)
            print('STATUS SUCCESS')
        except Exception as e:
            # Handle the exception and output the error message
            print(f"An error occurred: {e}")

    except:
        print('No Tickets to Predict!')






if __name__ == '__main__':
    single_prodrun()