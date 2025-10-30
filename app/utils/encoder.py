import pandas as pd
from prince import MCA
import json

def js_r(filename: str):
    # TODO: check temp comment
    """
    Opens JSON file
    
    Keyword arguments:
    filename -- name of file
    """
    with open(filename) as f_in:
        return json.load(f_in)

def combination_encoder(df,label_encoder):
    # TODO: check temp comment
    """
    Returns dataframe with column 'COMBINATION' containing unique combinations of symptoms
    
    Keyword arguments:
    df -- dataframe
    label_encoder -- output dataframe
    """
    df['SYMPCOMB'] = df['SYMPTOMDESCRIPTION1'].astype(str) + '+' + df['SYMPTOMDESCRIPTION2'].astype(str) + '+' + df['SYMPTOMDESCRIPTION3'].astype(str)

    # Get unique combinations
    unique_combinations = list(df['SYMPCOMB'].unique())
    label_encoder['COMBINATION'] = unique_combinations
    return label_encoder

    

def label_symptom_filter(df, only_parts = True, freq = 'None', alternative = True):
    # TODO: check temp comment
    """
    Returns list of all parts, unique ticket numbers, parts removed for all parts list, and list of alternative parts
    
    Keyword arguments:
    df -- dataframe
    only_parts -- indicates whether to remove specific items from all_parts (default True)
    freq -- indicates lower frequency bound of 'Other' parts (default 'None')
    alternative -- determines to group parts that have alternatives into one single partno (default True)
    """
    df = df.dropna(subset= ['PARTNO'])


    if freq == 'None':
        part_criterion = df['PARTNO'].value_counts()
        part_set =  part_criterion[part_criterion>0]
        all_parts = list(part_set.index)

        if only_parts == True:
            remove_parts = []
            for i in all_parts:
                if 'SQ'  in i:
                    remove_parts.append(i)
                elif 'JOB' in i:
                    remove_parts.append(i)
                elif 'CSP' in i:
                    remove_parts.append(i)
                elif 'MANA' in i:
                    remove_parts.append(i)
                elif 'PREV' in i:
                    remove_parts.append(i)    
                elif 'TV' in i:
                    remove_parts.append(i)  
                elif 'DA81-05595A' in i:
                    #epoxy
                    remove_parts.append(i)
                elif 'DIAGNOSIS' in i:
                    remove_parts.append(i)  
                elif 'DACOR' in i:
                    remove_parts.append(i)  
            all_parts = list(set(all_parts) - set(remove_parts))



    # alternative parts

    alt_parts = js_r('./data/alternative/final_altparts.json')
    conversion = {}

    if alternative:
        temp = all_parts.copy()
        conversion = {}
        for s in temp:
            if s in alt_parts.keys():
                for i in alt_parts[s]:
                    if i in all_parts:
                        all_parts.remove(i)
                        conversion[i] = s
    all_parts.sort()
    tkts = df.ID.unique()

    return all_parts, tkts, remove_parts, conversion
    







def label_version_processing(df, tkts, all_parts, symptom3 = False,  warranty= False, remove_parts = [], conversion = None):
    # TODO: check temp comment
    """
    Creates label encoder dictionary labelling parts in the part list and parts in the remove list. Outputs dataframes containing ticket numbers and indicating whether a part is in the list of all parts. Includes manufacture month in the dataframe. Could include lapse or warranty status.
    
    Keyword arguments:
    df -- dataframe
    tkts -- unique ticket numbers
    all_parts -- list of all parts
    symptom3 -- boolean indicating whether there are three symptoms (default False)
    warranty -- boolean indicating whether to include a warranty status (default False)
    remove_parts -- list of parts to be removed (default [])
    conversion -- part number pairs to be replaced in the dataframe (default None)
    """
    # Defaulted to use MFM
    if symptom3:
        cols = ['SYMPTOMDESCRIPTION1', 'SYMPTOMDESCRIPTION2', 'SYMPTOMDESCRIPTION3', 'VERSION', 'MFM']
    else:
        cols = ['SYMPTOMDESCRIPTION1', 'SYMPTOMDESCRIPTION2',  'VERSION', 'MFM']
    
    if warranty:
        cols = ['SYMPTOMDESCRIPTION1', 'SYMPTOMDESCRIPTION2', 'SYMPTOMDESCRIPTION3', 'VERSION', 'MFM', 'WARRANTYSTATUS']

    
        

    df['PARTNO'].replace(conversion,inplace= True )
    df['MFM'] = df['MANUFACTUREMONTH'].astype(str).str[:4]
    df = df.dropna(subset= ['PARTNO'])
    input_df =  pd.DataFrame(columns = cols)
    output_df = pd.DataFrame(columns = all_parts)
    parts_dict ={}
    for a in all_parts:
        parts_dict[a] =0

    for t in tkts:
        tkt_df = df[df['ID'] == t]
        p_dict = parts_dict.copy()
        s_dict = {}
        for col in cols:
            s_dict[col] = tkt_df[col].iloc[0]
        for i in range(len(tkt_df)):
            part = tkt_df['PARTNO'].iloc[i]
            if part in all_parts:
                p_dict[part] = 1
            elif part not in all_parts and part not in remove_parts:
                # print(part)
                continue
        input_df = pd.concat([input_df, pd.DataFrame([s_dict])], ignore_index=True)
        output_df = pd.concat([output_df, pd.DataFrame([p_dict])], ignore_index=True)

    label_encoder = {}

    if warranty:
        cols.remove('WARRANTYSTATUS')
        input_df['WARRANTY'] = 0
        input_df[input_df['WARRANTYSTATUS'] =='NO' ] =1
        input_df = input_df.drop('WARRANTYSTATUS', axis = 1)

    for col in cols:
        temp_dict = {}
        feature = input_df[col].unique().tolist()
        for index, element in enumerate(feature):
            temp_dict[element] = index
        input_df[col] = input_df[col].map(temp_dict)
        label_encoder[col] = temp_dict
    label_encoder['PARTS'] = all_parts
    label_encoder['REMOVE_PARTS'] = remove_parts

    return input_df, output_df, label_encoder



def inflow_label_version_processing(df, tkts, all_parts, symptom3 = False, warranty = False,  remove_parts = [], label_encoder = {}):
    # TODO: check temp comment
    """
    Creates dataframe which contains symptoms and manufacture month with corresponding ticket numbers and label encodings. Could contain lapse or warranty columns as well.

    Keyword arguments:
    df -- dataframe
    tkts -- unique ticket numbers
    all_parts -- list of all parts
    symptom3 -- boolean indicating whether there are three symptoms (default False)
    warranty -- boolean indicating whether to include a warranty status (default False)
    remove_parts -- list of parts to be removed (default [])
    label_encoder -- dictionary of labelled parts (default {})
    """

    if symptom3:
        cols = ['SYMPTOMDESCRIPTION1', 'SYMPTOMDESCRIPTION2', 'SYMPTOMDESCRIPTION3', 'VERSION', 'MFM']
    else:
        cols = ['SYMPTOMDESCRIPTION1', 'SYMPTOMDESCRIPTION2',  'VERSION', 'MFM']

    if warranty:
        cols = ['SYMPTOMDESCRIPTION1', 'SYMPTOMDESCRIPTION2', 'SYMPTOMDESCRIPTION3', 'VERSION', 'MFM', 'WARRANTYSTATUS']
    input_df =  pd.DataFrame(columns = cols)
    parts_dict ={}


    for a in all_parts:
        parts_dict[a] =0

    removed_tkts = tkts.copy()

    for t in tkts:
        tkt_df = df[df['ID'] == t]
        new_input_detect = len(tkt_df)
        if new_input_detect ==0:
            removed_tkts.remove(t)
            continue
        s_dict = {}
        for col in cols:
            s_dict[col] = tkt_df[col].iloc[0]


        input_df = pd.concat([input_df, pd.DataFrame([s_dict])], ignore_index=True)

    if warranty:
        cols.remove('WARRANTYSTATUS')
        input_df['WARRANTY'] = 0
        input_df[input_df['WARRANTYSTATUS'] =='NO' ] =1
        input_df = input_df.drop('WARRANTYSTATUS', axis = 1)

    for col in cols:
        input_df[col] = input_df[col].astype(str)

        if col == 'VERSION' or col == 'SYMPTOMDESCRIPTION3':
            lkeys = label_encoder[col].copy().keys()
            for key in lkeys:
                change_key = key.replace('.0','',1)
                label_encoder[col][change_key] = label_encoder[col].pop(key)
        input_df[col] = input_df[col].map(label_encoder[col])


    return input_df, removed_tkts







# ---- Final Encoding Process ----
    



def label_version_encoder(df,only_parts = True, symptom3 = False, warranty = False, mfm =True, freq = 'None'):
    all_parts, tkts, remove_parts, conversion = label_symptom_filter(df,only_parts= only_parts, freq=freq)
    input_df, output_df, label_encoder = label_version_processing(df =df, tkts=tkts, all_parts=all_parts, symptom3=symptom3, remove_parts=remove_parts, warranty=warranty, conversion=conversion)
    label_encoder= combination_encoder(df,label_encoder)

    return input_df, output_df, label_encoder



# ---- Inflow Encoder ----



def inflow_label_version_encoder(new_df,all_parts = [], remove_parts = [], symptom3 = True, warranty= False, label_encoder_path= './data/experiment.json'):
    # always defaulted to using symptom1 and symptom2
    if 'VERSION' in list(new_df.columns):
        new_df.loc[new_df['VERSION'].isnull(), 'VERSION'] =  'NONE'
   
    tkts = new_df.ID.unique()
    label_encoder = js_r(label_encoder_path)
    all_parts = label_encoder['PARTS']
    remove_parts = label_encoder['REMOVE_PARTS']
    new_df['MFM'] = new_df['MANUFACTUREMONTH'].astype(str).str[:4]
    new_input_df, removed_tkts  = inflow_label_version_processing(df= new_df, tkts= tkts,all_parts=all_parts,symptom3=symptom3,warranty=warranty, remove_parts=remove_parts, label_encoder=label_encoder)

    return new_input_df, removed_tkts
