import json

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