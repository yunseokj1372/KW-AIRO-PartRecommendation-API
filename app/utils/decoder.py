import pandas as pd
import numpy as np
from app.models import PRD

def part_descripton_add(df):

    #TODO: check temp comment
    """
    Returns a modified dataframe where part descriptions are added to corresponding part numbers

    Keyword arguments:
    df -- the dataframe to add part descriptions to
    """

    part_desc_df = pd.read_csv('/home/ubuntu/KW-AIRO-PartRecommendation-API/app/data/part_desc.csv')

    part_desc_df= part_desc_df.dropna()

    df['PARTDESC'] = 'None'

    partno = list(df['PARTNO'].unique())

    for p in partno:
        try:
            df['PARTDESC'][df['PARTNO'] == p] = part_desc_df['PARTDESC'][part_desc_df['PARTNO']==p].iloc[0]
        except:
            continue
    
    return df
    



# non inflow batch decoder
# returns: df --> ID | PARTNO
def decode(model, df):

    #TODO: check temp comment, check what df input it
    """
    Returns a new dataframe with ticket IDs in one column and part numbers in another, where the
    dataframe represents which part numbers to recommend for each ticket
    The PK is ['ID', 'PARTNO']

    Keyword arguments:
    model -- trained model for part replacement predictions
    df -- dataframe 
    """

    # get trained model and prediction of model
    ml, X_test, y_test = model.train_supervised()
    pred = model.test_prediction()

    predTList = pred_tickets(X_test, df)
    partList = y_test.columns.tolist()
    tktCol = []
    partCol = []

    # for each row of 0s, 1s, if 1, replace it with the corresponding part number based on it's column name (the part number)
    for row, ticket in zip(pred, predTList):
        for b, p in zip(row, partList):
            if b == 1:
                tktCol.append(ticket)
                partCol.append(p)
            
    data = list(zip(tktCol, partCol))
    new_df = pd.DataFrame(data, columns=['ID', 'PARTNO'])

    return new_df


def ob_decode( y_test,tktno):
    
    # TODO: check temp comment, check difference in decode methods
    """
    Returns a dictionary with ticket numbers as the key and a list of recommended parts as the value
    Any empty part lists are marked for diagnosis

    Keyword arguments:
    y_test -- the list of predictions from the test set
    tktno -- a list of ticket numbers to map the predicted parts to
    """

    partList = y_test.columns.tolist()
    final_dict ={}

    # predTList = pred_tickets(X_test, df)

    for tkt, row in zip(tktno, y_test.values):
        pList = []
        for b, p in zip(row, partList):
            if b == 1:
                pList.append(p)

        pcheck = len(pList)
        if pcheck ==0:
            pList.append('DIAGNOSIS')

        final_dict[tkt] = set(pList)

    return final_dict



def ob_pd_decode( y,tktno,partList):
    
    #TODO: check temp comment
    """
    Returns a dictionary with ticket numbers as the key and a list of recommended parts as the value

    Keyword arguments:
    y -- list of model predictions
    tktno -- list of ticket numbers
    partList -- list of predicted parts
    """

    final_dict ={}

    for tkt, row in zip(tktno, y.values):
        pList = []
        for b, p in zip(row, partList):
            if b == 1:
                pList.append(p)


        final_dict[tkt] = set(pList)

    return final_dict


def ob_np_decode(y,tktno,partList):

    #TODO: check temp comment
    """
    Returns a dictionary with ticket numbers as the key and a list of recommended parts as the value

    Keyword arguments:
    y -- list of model predictions
    tktno -- list of ticket numbers
    partList -- list of predicted parts
    """
    
    final_dict ={}

    for tkt, row in zip(tktno, y):
        pList = []
        for b, p in zip(row, partList):
            if b == 1:
                pList.append(p)

        final_dict[tkt] = set(pList)

    return final_dict



# for PRD inflow data
# returns: df --> ID | MODELNO | WORKORDERID | PARTNO
def new_decode(df, pred, parts, wono):

    #TODO: check temp comment
    """
    Returns a modified dataframe where columnns are added for model numbers, worder order IDs, and part numbers
    Any record with no recommended parts is marked for diagnosis

    Keyword arguments:
    df -- input dataframe to add columns to
    pred -- Model predictions for part replacement, organized as rows of 0s and 1s (1 indicating a part should be used)
    parts -- Loaded JSON file of parts data
    wono -- dataframe containing data for worker orders
    """

    predTList = df.ID.unique()
    partList = parts['PARTS']

    # empty lists to append to and define as returned df's columns
    tktCol = []
    partCol = []
    wonoCol = []
    # goes through each 0,1 row of the prediction
    for row, ticket in zip(pred, predTList):
        # if there are no 1s in the row (no suggested parts), label as 'DIAGNOSIS'
        if 1 not in row:
            tktCol.append(ticket)
            partCol.append('DIAGNOSIS')
            wonoCol.append(wono.loc[wono['ID'] == ticket, 'WORKORDERID'].iloc[0])


        # for every 1 in each ticket's prediction, append the ticketno, partno, and workorderno to the lists
        for b, p in zip(row, partList):
            if b == 1:
                tktCol.append(ticket)
                partCol.append(p)
                wonoCol.append(wono.loc[wono['ID'] == ticket, 'WORKORDERID'].iloc[0])

    # create the formatted df        
    data = list(zip(tktCol, wonoCol, partCol))
    new_df = pd.DataFrame(data, columns=['ID',  'WORKORDERID', 'PARTNO'])

    new_df = part_descripton_add(new_df)


    return new_df



# returns the list of random tickets used for the prediction
def pred_tickets(X_test, df):

    ticketIndices = X_test.index.tolist()
    ticketList = df.ID.unique()
    predTList = []

    for i in ticketIndices:
        add = ticketList[i]
        predTList.append(add)
    
    return predTList

def predy_tickets(y_test, df):

    ticketIndices = y_test.index.tolist()
    ticketList = df.ID.unique()
    predTyList = []

    for i in ticketIndices:
        add = ticketList[i]
        predTyList.append(add)
    
    return predTyList



# returns: result that is a json with ID, WORKORDERID, PARTNO
def PROD_reformat(df, pred, label_encoder):
    
    predTList = df.ID.unique()
    partList = label_encoder['PARTS']
    result = {}
    # for each ticket's prediction, 
    for row, ticket in zip(pred, predTList):
        temp_df = df[df['ID'] == ticket]
        wono = temp_df.WORKORDERID.unique()
        for w in wono:
            result[ticket] = {'WORKORDERID':w,  'SYMPTOMDESCRIPTION2': temp_df.loc[temp_df['ID'] == ticket, 'SYMPTOMDESCRIPTION2'].iloc[0],'PARTNO' : set() }
        for b, p in zip(row, partList):
            if b == 1:
                result[ticket]['PARTNO'].add(p)    

    return result

