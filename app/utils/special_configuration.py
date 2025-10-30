import json
import pandas as pd
from pandas.core.common import flatten
import oracledb
from dotenv import load_dotenv
import os
# import utils.sql_function
# from utils import sql_function
from datetime import datetime, timedelta

def js_r(filename: str):
# Function : opening up json file as a dictionary
    with open(filename) as f_in:
        return json.load(f_in)


def damper_add(df):
    damper_df = pd.read_excel('/home/ubuntu/KW-AIRO-PartRecommendation-API/app/data/damper.xlsx')
    factors = dict(zip(damper_df['PARTNO'], damper_df['QUANTITY']))
    rows = []
    for _, row in df.iterrows():
        count = factors.get(row['PARTNO'], 1)  # Default to 1 if no factor is found
        for _ in range(count):
            rows.append(row.copy())

    output_df = pd.DataFrame(rows)

    return output_df


def seal_system_temp(seal_dict, seal_df):

    wid = list(seal_df['WORKORDERID'].unique())

    final_df = pd.DataFrame()

    workorderid = seal_df.WORKORDERID.unique()

    reorder= ['ID', 'WORKORDERID', 'PRODUCTTYPE','PARTNO', 'MODELTYPE', 'KEYWORD','PARTDESC', 'STATUS']

    for wid in workorderid:
        wdf = seal_df[seal_df['WORKORDERID'] == wid]
        row_dict = {}
        row_dict['ID'] = wdf['ID'].iloc[0] 
        row_dict['WORKORDERID'] = wdf['WORKORDERID'].iloc[0]
        row_dict['PRODUCTTYPE'] = wdf['PRODUCTTYPE'].iloc[0]

        symptomdesc2 = wdf['SYMPTOMDESCRIPTION2'].iloc[0]
        symptomdesc3 = wdf['SYMPTOMDESCRIPTION3'].iloc[0]

        if symptomdesc2 in ['Weak cooling','No cooling']:
            key = 'cooling'
        if symptomdesc3 == '21 E or 21 C (Freezer Room Fan Error)':
            key = '21C'
        row_dict['PARTNO'] = seal_dict[wdf['MODELNO'].iloc[0]][key]
        temp_df = pd.DataFrame([row_dict]).explode('PARTNO')
        final_df = pd.concat([final_df, temp_df], ignore_index=True)

    final_df['MODELTYPE'] = 'SEAL'
    final_df['KEYWORD'] = 'NONE'

    final_df = decoder.part_descripton_add(final_df)


    
    return final_df

def icemaker_two_part_no(df):

    remove_parts = ['DA97-22160A','DA97-22162A']

    df_filtered = df[~df['PARTNO'].isin(remove_parts)]

    return df_filtered


