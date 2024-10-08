import re
import os
import time
import json
import pandas as pd
from pathlib import Path
import datetime
import numpy as np
import modelop.utils as utils
import modelop_sdk.restclient.moc_client as moc_client

##functions for certain tasks 
def change_date(date_string:str): #converts to Date format accepted by ModelOP
    datetime_object = datetime.datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
    return str(datetime_object)

def fix_numpy_nans_and_infs_in_dict(val: float) -> float: #removes nan and inf from numerical data
    # If value is numeric (not None), check for numpy.nan and numpy.inf
    # If True, change to None, else keep unchanged
    if val is not None:
        try:  # Some values are not numeric
            if np.isnan(val):
                val = None
            elif np.isinf(val):
                print("Infinity encountered while computing %s on column %s! Setting value to None.", val)
                val = None
        except TypeError:
            pass

    return float(val)



# modelop.init
def init(init_param):
    global DEPLOYABLE_MODEL_ID
    global VERSION
    global MODEL_NAME
    global CURRENT_VERSION_PROCESSED_DATE
    

    job = json.loads(init_param["rawJson"])
    
    # Get the deployable model we are targeting
    DEPLOYABLE_MODEL_ID = job.get('referenceModel', {}).get('id', None) 
    if not DEPLOYABLE_MODEL_ID:
        raise ValueError('You must provide a reference model for this job of the model to pull the test results from')
    
    VERSION= job.get('jobParameters', {}).get('version', "")
    if not VERSION:
        raise ValueError('No version parameter was found in the job parameters')
    
    CURRENT_VERSION_PROCESSED_DATE= job.get('jobParameters', {}).get('current_version_processed_date', "")
    if not CURRENT_VERSION_PROCESSED_DATE:
        raise ValueError('No version date was provided for this MTR')
    

    MODEL_NAME = job.get('referenceModel', {}).get('storedModel', {}).get('modelMetaData',{}).get('name',"SSCD")
   # modelMetaData name
    logger = utils.configure_logger()


#modelop.metrics
def metrics(data: pd.DataFrame):
    print("Running the metrics function") 

    finalResult = {}
    logger = utils.configure_logger()

    if data.empty:
        print("DataFrame Should not be Empty")
        finalResult["sample"] = {"no": "data provided"}
    else:
        #Open the file and read contents
        filename="model_performance.txt"
        if os.path.isfile(f'./{filename}'):
            logger.info(f'found {filename}')
            with open(f'./{filename}') as file:
                file_content = file.read()
                logger.info(f'Reading {filename}')
        else:
            logger.warning("Could not find the formatted text file and thus cannot set the version in the report")

        file=open(filename).read()
        contents=file.split("\n")

        #Extract the keys and values from the above file and append those to appropriate lists
        keys=[]
        vals=[]
        version_no=f'Version {VERSION}'
        print(version_no)
        for ind,line in enumerate(contents):
            if "|" in line:
                metrics_keys_init=contents[ind]
                metrics_keys=[key for key in re.split(r'\|{1,2}',metrics_keys_init)] # checks for both | and ||
                keys.append([key for key in metrics_keys if any(key)]) #remove empty strings
                vals.append(contents[ind+1].split()) # simply split using the whitespace

        #To produce a line graph, table as well as bar graph, the data should be saved in certain format as descrbed in the documentation here:
                #https://modelopdocs.atlassian.net/wiki/spaces/dv301/pages/1697154788/Model+Monitoring+Overview#Custom-Monitor-Output---Charts%2C-Graphs%2C-Tables

        #Remove the nessted structure from both keys and values lists and extract the unique keys to which values should be associated later
        keys_list_orig=[sub_key for key in keys for sub_key in key]
        # Some of the .txt contain a different name for F_Score for negative class and the next line checks for this and replaces it with the original name
        keys_list=['Fscore_neg_class' if key=='Fscore Negative_class' else key for key in keys_list_orig]
        
        keys_unique=sorted(set(keys_list),key=keys_list.index) #remove duplicates but preserve the order
        vals_list_o=[fix_numpy_nans_and_infs_in_dict(sub_val) for val in vals for sub_val in val]
        vals_list=["None" if np.isnan(val) else val for val in vals_list_o] #strings as nan would fail this criteria
        print(vals_list_o)
        print(vals_list)
        #Create a list of initial and evolving keys for the bar graph 
        keys_initial=[f+str("_initial") for f in keys_list[:10]] #since 10 metrics
        keys_evolving=[f+str("_evolving") for f in keys_list[10::]]
        keys_all=keys_initial+keys_evolving 
       
        
        # Since MOC reads the json object which fails for nan or inf values, therefore we need to remove these two kind of numerical values from our data
        #Iterate all the values for the given keys and put them in a ModelOp generic table format, i.e. list of dictionaries

        metrics_table=[{"metric": key, version_no: val} for key, val in zip(keys_all, vals_list)]
        print("current metrics table",metrics_table)
        
        #Set up the data in a structure needed by a time line graph with x axis as date created for each metric and y axis as metrics values at these tests. Separate the two set of data as initial and evolving metrics 
        #date_created=time.ctime(os.path.getctime(filename))

        data1=[{keys_unique[i]+"_initial": [[change_date(CURRENT_VERSION_PROCESSED_DATE),vals_list[i]]] for i, key in enumerate(keys_unique)}]
        data2=[{keys_unique[i]+"_evolving": [[change_date(CURRENT_VERSION_PROCESSED_DATE),vals_list[i+len(keys_unique)]]] for i, key in enumerate(keys_unique)}]
        time_graph_data=data1[0].copy()
        time_graph_data.update(data2[0])   
        #print("current time graph data:",time_graph_data)
        #Create a time line graph object with current data 
        time_line_graph=dict(title=f"{MODEL_NAME} Metrics Over Time",x_axis_label="X Axis",y_axis_label="Y Axis",data=time_graph_data)

        #Creating a horizontal bar graph with 20 categories corresponding to initial  and evolving  
        #bar graph data 
        bar_graph_data={version_no:vals_list}
        final_bar_graph=dict(title=f"{MODEL_NAME} Metrics Bar Chart",x_axis_label="X Axis",y_axis_label="Y Axis",rotated=True,data=bar_graph_data,categories=keys_all)
        #print("current bar graph:",final_bar_graph)


        ###############Add the aggregated table, bar and time graph##################
        # These will fetch all the test monitor results for the given deployable model and produce a table, a time line and a bar graph for all the MTRs
    
        #get data from previous MTR
        try:
            client = moc_client.MOCClient()
        except ValueError:
            print("can't find the client")
        path = f"model-manage/api/modelTestResultSummaries/search/findAllByDeployedModel_DeployableModel_Id?deployableModelId={DEPLOYABLE_MODEL_ID}&page=0&size=20"
        result = client.get(path)
        model_test_results = result.json.get("_embedded", {}).get("modelTestResultSummaries", [{}])
        #update the metrics table with previous model test results, if found
        previous_table_data=[test_result['testResults'][f"{MODEL_NAME}_Metrics_Per_Version_Table"] for test_result in model_test_results if f"{MODEL_NAME}_Metrics_Per_Version_Table" in test_result['testResults']]
        
        #print("previous table data from old MTRs:",previous_table_data)
        if len(previous_table_data)>0:
            for dict_item,val in zip(previous_table_data[0],vals_list):
                dict_item.update({version_no:val})        
                final_table=previous_table_data[0]
        else:
            final_table=metrics_table
        #print("final metrics table",final_table)

        #update  the bar graph with previous results
        #print(f"{MODEL_NAME}_Metrics_Per_Version_Bar")    
        previous_bar_graph_data=[test_result['testResults'][f"{MODEL_NAME}_Metrics_Per_Version_Bar"]['data'] for test_result in model_test_results if f"{MODEL_NAME}_Metrics_Per_Version_Bar" in test_result['testResults']]
        bar_graph_data_all={}
        bar_graph_data_all.update(bar_graph_data) #append the data from this date
        for i in range(len(previous_bar_graph_data)):
            bar_graph_data_all.update(previous_bar_graph_data[i])
         #update the bar graph with the old data appended to the current data
        final_bar_graph=dict(title=f"{MODEL_NAME} Metrics Per Version",x_axis_label="X Axis",y_axis_label="Y Axis",rotated=True,data=bar_graph_data_all,categories=keys_all)
        #print("final bar graph",final_bar_graph_data_all)

        #update the time graph with previous results
        #search for "performanceMetrics" key in the nested structure which is where this data object is written
        previous_time_graph_data=[test_result['testResults']['performanceMetrics'] for test_result in model_test_results if 'performanceMetrics' in test_result['testResults']]
        print(f"Total no. of MTRs with line graph including the current one ={len(previous_time_graph_data)+1}")
        #print("previous time graph data:",previous_time_graph_data)
        #concatenate all the MTRs and update the time graph object created above to produce a time line graph for each metric 
        for previous_data_dict in previous_time_graph_data:
            for key,val in previous_data_dict.items():
                time_graph_data[key]=time_graph_data.get(key,[])+val
        agg_line_graph=dict(title=f"{MODEL_NAME} Metrics Over Time",x_axis_label="X Axis",y_axis_label="Y Axis",data=time_graph_data)
        #print("updated time graph data:",time_graph_data)

        #append it all to the final object 
        finalResult[f"{MODEL_NAME}_Metrics_Per_Version_Table"] = final_table
        finalResult["performanceMetrics"] = time_graph_data
        finalResult[f"{MODEL_NAME}_Metrics_Per_Version_Bar"] = final_bar_graph
        finalResult[f"{MODEL_NAME}_Metrics_Over_Time"] = agg_line_graph


    yield finalResult
        
def main():
    #the metrics function requires a dataframe as an input and here we just provide a dummy dataframe for this program
    #to execute correctly. 
    data = {"data1":993,"data2":36,"data3":3959,"label_value":0,"score":1} #not used
    df = pd.DataFrame.from_dict([data])
    print(json.dumps(next(metrics(df)), indent=2))


if __name__ == '__main__':
	main()        
