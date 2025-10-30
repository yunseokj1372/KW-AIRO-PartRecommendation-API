import pandas as pd
from app.utils import POD_preprocessing, encoder
import re
import json
from app.utils import decoder

pd.set_option('display.max_columns', None)

class PROD:
    # Combination of PRD and POD
    # utilizes one of the decoder function
    def __init__(self, df= None, pred = None, label_encoder= None, instruction = None, POD_json = None, service_bulletin = None):
        self.PRD_pred = decoder.PROD_reformat(df, pred, label_encoder)
        self.part_converter = POD_preprocessing.part_converter(instruction=instruction, POD_json=POD_json)
        self.df = df
        self.instruction = instruction
        self.POD_dict = POD_json

    def symptomcode_trigger(self):
        # Triggers on specific symptom code with a corresponding instruction for certain product type
        # This function combines the PRD prediction with the instructed keywords which get associated with partno given modelno POD Json File
        # self.df['SYMPTOMCODE2'] = self.df['SYMPTOMCODE2'].astype(str)
        # self.df['SYMPCODE'] = self.df['SYMPTOMCODE1'] + self.df['SYMPTOMCODE2']
        self.df['SYMPCODE'] = self.df['SYMPTOMDESCRIPTION1'] + '+'+ self.df['SYMPTOMDESCRIPTION2']

        cols = ['ID', 'WORKORDERID', 'PRODUCTTYPE', 'SYMPCODE', 'PARTNO', 'MODELTYPE', 'KEYWORD']
        trigger_df =  pd.DataFrame(columns = cols)
        workorderno = self.df.WORKORDERID.unique()
        codes = self.part_converter.keys()
        for wono in workorderno:
            row_dict = {'PARTNO':()}
            temp_df = self.df[self.df['WORKORDERID'] == wono]
            tktno = temp_df['ID'].iloc[0]
            scode = temp_df['SYMPCODE'].iloc[0]
            ptype = temp_df['PRODUCTTYPE'].iloc[0]
            row_dict['ID'] = tktno
            row_dict['WORKORDERID'] = wono
            row_dict['SYMPCODE'] = scode
            row_dict['PRODUCTTYPE'] = ptype
            PRD_partno = self.PRD_pred[tktno]['PARTNO']


            if scode in codes:
                POD_partno = set(self.part_converter[scode].values())
            else:
                POD_partno = set()


            final_partno = POD_partno.union(PRD_partno)
            row_dict['PARTNO'] = final_partno

            # Accommodate Column with Where the parts come from. Prioritizes POD over PRD in terms of marking.
            temp_df = pd.DataFrame([row_dict]).explode('PARTNO')
            temp_df['MODELTYPE'] = 'PROGRAM'
            temp_df['KEYWORD'] = 'NONE'
            for i in range(len(temp_df)):
                det = temp_df['PARTNO'].iloc[i]
                if det in POD_partno and det in PRD_partno:
                    temp_df['MODELTYPE'].iloc[i] = 'BOTH'
                    for key, values in self.part_converter[scode].items():
                        if det in values:
                            temp_df['KEYWORD'].iloc[i] = key
                if det in POD_partno and det not in PRD_partno:
                    temp_df['MODELTYPE'].iloc[i] = 'MACRO'
                    for key, values in self.part_converter[scode].items():
                        if det in values:
                            temp_df['KEYWORD'].iloc[i] = key
            trigger_df = pd.concat([trigger_df, temp_df], ignore_index=True)
            
                

        return trigger_df


