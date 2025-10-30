from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from app.models.PRD import TreeBasedModel
from app.models.PROD import PROD
from app.core.config import settings
import logging
from typing import List, Optional
from app.utils.helper import js_r, pkl_file_check, combination_check, prodtype_adder
from app.utils.special_configuration import damper_add, icemaker_two_part_no, seal_system_temp
from app.utils.decoder import new_decode, part_descripton_add
import pandas as pd
import json
import os

# Set up file logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prediction.log'),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)

# Router definition
router = APIRouter(
    prefix="/recommend",
    tags=["recommendation"],
    responses={
        # TODO: add responses
    }
)

# Pydantic models for request/response
class RecommendationRequest(BaseModel):
    modelNo: str = Field(..., description="The model number of the product")
    productType: str = Field(..., description="The product type (e.g. REF_REF)")
    version: str = Field(..., description="The version of the model")
    mfm: str = Field(..., description="The manufacture month (YYYYMM format)")
    symptomDescription1: str = Field(..., description="Primary symptom description")
    symptomDescription2: str = Field(..., description="Secondary symptom description")
    symptomDescription3: str = Field(..., description="Tertiary symptom description")

class PartRecommendation(BaseModel):
    partNo: str = Field(..., description="The part number")
    partDescription: str = Field(..., description="The part description")
    quantity: Optional[int] = Field(None, description="Quantity of the part")

class RecommendationResponse(BaseModel):
    recommendation: List[PartRecommendation] = Field(..., description="The list of part recommendations")

class BatchRecommendationRequest(BaseModel):
    requests: List[RecommendationRequest] = Field(..., description="The list of recommendation requests")

class BatchRecommendationResponse(BaseModel):
    responses: List[RecommendationResponse] = Field(..., description="The list of recommendation responses")



def daily_run(df, rev1, rev2, prodtype, warranty=False):
    final_model_result_df = pd.DataFrame()
    df['VERSION'] = df['VERSION'].astype(str)

    pred = None
    model_train_all_df = df
    ML_path = settings.MODELS_PATH + rev1 + ".pkl"
    label_encoder_path = settings.MODELS_PATH + rev1 + ".json"
    # Adding a for loop to accommodate versions as well
    pod_versions = list(model_train_all_df['VERSION'].unique())
    model = TreeBasedModel(model_type = 'XGB', filter_type = 'label_version_all', filepath= ML_path, label_path= label_encoder_path, only_parts = True, warranty = warranty)
    for v in pod_versions:
    
        model_train_df = model_train_all_df[model_train_all_df['VERSION']==v]
        if v == 'None':
            print('No Version. Default Version')
            POD_path = f"{settings.POD_JSONS_PATH + rev2}.json"
        else:
            POD_path = f"{settings.POD_JSONS_PATH + rev2}_{v}.json"


        pred, removed_tkts = model.prediction(new_df=model_train_df, ML_path = ML_path , label_encoder_path= label_encoder_path)
        label_encoder = js_r(label_encoder_path)
        instruction_filepath = f'{settings.INSTRUCTIONS_PATH + prodtype}.json'
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
                POD_path =  f"{settings.POD_JSONS_PATH + rev2}.json"
                POD_json = js_r(POD_path)
                new_path = f"{settings.POD_JSONS_PATH + rev2}_{v}.json"
                # with open(new_path, 'w', encoding='utf-8') as f:
                #     json.dump(POD_json, f, ensure_ascii=False, indent=4) 
                # print("Duplication of Base Version Completed")
                ### 

            prod_model = PROD(df= model_train_df, pred = pred, label_encoder= label_encoder, instruction = prod, POD_json = POD_json)
            model_result_df = prod_model.symptomcode_trigger()
            final_model_result_df = pd.concat([final_model_result_df, model_result_df], ignore_index=True)
        except:
            parts = json.loads(open(label_encoder_path,"r").read())
            try: 
                model_result_df= new_decode(df = model_train_df, pred=pred, parts = parts, wono=model_train_df)
                final_model_result_df = pd.concat([final_model_result_df, model_result_df], ignore_index=True)
            except:
                print("No ModelNo trained")
    
    final_model_result_df = part_descripton_add(final_model_result_df)

    return final_model_result_df

def single_prodrun(request: RecommendationRequest, request_id: str = "1"):

    try:
        input_df = pd.DataFrame({
            'ID': [request_id],
            'WORKORDERID': [request_id],
            'MODELNO': [request.modelNo],
            'VERSION': [request.version],
            'MANUFACTUREMONTH': [request.mfm],
            'SYMPTOMDESCRIPTION1': [request.symptomDescription1],
            'SYMPTOMDESCRIPTION2': [request.symptomDescription2],
            'SYMPTOMDESCRIPTION3': [request.symptomDescription3],
            'PRODUCTTYPE': [request.productType]
        })

        directory = settings.MODELS_PATH

        # Load seal dictionary for special cases
        try:
            seal_dict = js_r(settings.SEAL_PATH)
            seal_modelno = list(seal_dict.keys())
        except FileNotFoundError:
            seal_dict = {}
            seal_modelno = []

        # Check if model files exist
        file_format = '.pkl'
        file_names = [file[:-len(file_format)] for file in os.listdir(directory) if file.endswith(file_format)]
        file_names = [name.replace('+', '/') for name in file_names]

        non_pkl_tkts, df = pkl_file_check(df=input_df, file_name_list=file_names)

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
                        seal_result_df = seal_system_temp(seal_dict=seal_dict, seal_df=mdf)
                        logger.info('SEAL MODELNO CG')
                        final_df = pd.concat([final_df, seal_result_df], ignore_index=True)
                        continue
                    except:
                        logger.info('NON SEAL SYMPTOM')
                ### Seal Temporary Logic

                non_inputtable_tkt, mdf = combination_check(df=mdf, directory=directory, modelno=rev1)
                if non_inputtable_tkt.any():
                    logger.info('Ticket not inputtable, need to implement logic still')
                final_model_result_df = daily_run(mdf, rev1=rev1, rev2=rev2, prodtype=prodtype)

                final_df = pd.concat([final_df, final_model_result_df], ignore_index=True)
            except Exception as e:
                logger.error(f'{rev1} does not have combination for ticket: {e}')
                continue

        try:
            final_df['MODELTYPE'] = final_df['MODELTYPE'].fillna(value='PROGRAM')
            final_df['KEYWORD'] = final_df['KEYWORD'].fillna(value='NONE')
            final_df['PARTNO'] = final_df['PARTNO'].fillna(value='DIAGNOSIS')
            final_df['STATUS'] = 1

            reorder = ['ID', 'WORKORDERID', 'PRODUCTTYPE', 'PARTNO', 'MODELTYPE', 'KEYWORD', 'PARTDESC', 'STATUS']
            final_df = final_df[reorder]
            final_df = prodtype_adder(df=df, final_df=final_df)

            # SPECIAL CONFIGURATION: DAMPER
            try:
                final_df = damper_add(final_df)
            except:
                logger.error('DAMPER CONFIGURATION DID NOT WORK')

            # SPECIAL CONFIGURATION: ICEMAKER TWO PART NO
            try:
                final_df = icemaker_two_part_no(final_df)
                logger.info('ICEMAKER ONCE OF THE TWO PARTS REMOVED')
            except:
                logger.error('ICEMAKER PARTS NOT REMOVED')

        except:
            logger.error('NO TICKETS TO PREDICT')

        logger.info(f'Result: {final_df}')

        recommendations = []
        if not final_df.empty:
            for _, row in final_df.iterrows():
                part_rec = PartRecommendation(
                    partNo=(row.get('PARTNO', 'DIAGNOSIS')),
                    partDescription=(row.get('PARTDESC', 'DIAGNOSIS')),
                    quantity=1
                )
                recommendations.append(part_rec)

        return RecommendationResponse(recommendation=recommendations)
    
    except Exception as e:
        logger.error(f'Error processing recommendation: {e}')
        return HTTPException(status_code=500, detail=str(e))

        
# API Endpoints
@router.post("/single", response_model=RecommendationResponse)
async def recommend_single(request: RecommendationRequest):
    """
    Get part recommendations for a single request
    """
    logger.info(f'Processing single recommendation for model: {request.modelNo}')

    try:
        result = single_prodrun(request)
        return result
    except Exception as e:
        logger.error(f'Error processing recommendation: {e}')
        return HTTPException(status_code=500, detail=str(e))

@router.post("/batch", response_model=BatchRecommendationResponse)
async def recommend_batch(request: BatchRecommendationRequest):
    """
    Get part recommendations for a batch of requests
    """
    logger.info(f'Processing batch recommendation for {len(request.requests)} requests')

    recommendations = []
    for i, req in enumerate(request.requests):
        try:
            recommendation = await recommend_single(req, request_id=f'{i+1}')
            recommendations.append(recommendation)
        except HTTPException as e:
            logger.error(f'Error processing request {i+1}: {e.detail}')
            recommendations.append(RecommendationResponse(recommendation=[]))
        except Exception as e:
            logger.error(f'Unexpected error processing request {i+1}: {e}')
            recommendations.append(RecommendationResponse(recommendation=[]))

    return BatchRecommendationResponse(recommendations=recommendations)