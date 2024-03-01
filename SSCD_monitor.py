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
def change_date(date_string:str,time_seconds:int): #converts to Date format accepted by ModelOP
    datetime_object = datetime.datetime.strptime(date_string, '%a %b %d %H:%M:%S %Y')
    return str(datetime_object+datetime.timedelta(seconds=time_seconds))

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

    return val
# modelop.init
def init(init_param):
    global DEPLOYABLE_MODEL_ID

    job = json.loads(init_param["rawJson"])
    
    # Get the deployable model we are targeting
    DEPLOYABLE_MODEL_ID = job.get('referenceModel', {}).get('id', None) #7d5b3982-b2e5-444f-aab5-4de2727d1c3f
    if not DEPLOYABLE_MODEL_ID:
        raise ValueError('You must provide a reference model for this job of the model to pull the test results from')

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
        filename="model_performance_sample.txt"
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
        split_pattern=re.compile(r"\\||\\|| ")
        for ind,line in enumerate(contents):
            if "|" in line:
                #print(contents[ind],"\n",contents[ind+1])
                metrics_keys_init=contents[ind]
                metrics_keys=[key for key in re.split(r'\|{1,2}',metrics_keys_init)] # checks for both | and ||
                keys.append([key for key in metrics_keys if any(key)]) #remve empty strings
                vals.append(contents[ind+1].split()) # simply split using the whitespace

        #To produce a line graph, table as well as bar graph, the data should be saved in certain format as descrbed in the documentation here:
                #https://modelopdocs.atlassian.net/wiki/spaces/dv301/pages/1697154788/Model+Monitoring+Overview#Custom-Monitor-Output---Charts%2C-Graphs%2C-Tables

        #Remove the nessted structure from both keys and values lists and extract the unique keys to which values should be associated later
        keys_list=[sub_key for key in keys for sub_key in key]
        keys_unique=sorted(set(keys_list),key=keys_list.index) #remove duplicates but preserve the order
        vals_list=[fix_numpy_nans_and_infs_in_dict(sub_val) for val in vals for sub_val in val]
        # since MOC reads the json object which fails for nan or inf values, therefore we need to remove these two kind of numerical values from our data
        
        #Iterate all the values for the given keys and put them in a ModelOp generic table format, i.e. list of dictionaries
        generic_table=[{keys_unique[i]:float(vals_list[i+j]) for i in range(len(keys_unique))} for j in range(0,len(vals_list),len(keys_unique))]
        #print(generic_table)

        #Add the date to the table  in a way that it adds to the beginning of the dictionary
        date_created=time.ctime(os.path.getctime(filename))
        print(date_created)
        final_table=[]
        for gen in generic_table:
            items = list(gen.items())
            items.insert(0, ('date', date_created))
            gen = dict(items)
            final_table.append(gen)
        #print(final_table)    
            
        
        #Set up the data in a structure needed by a time line graph with x axis as date created for each metric and y axis as metrics values at these tests. Separate the two set of data as initial and evolving metrics 
        data1=[{keys_unique[i]+"_initial": [[change_date(date_created,0),float(vals_list[i])]] for i, key in enumerate(keys_unique)}]
        data2=[{keys_unique[i]+"_evolving": [[change_date(date_created,0),float(vals_list[i+len(keys_unique)])]] for i, key in enumerate(keys_unique)}]
        generic_line_graph_data=data1[0].copy()
        generic_line_graph_data.update(data2[0])   
       
        #creating a horizontal bar graph with categories as initial  and final 
        horizontal_bar_graph_data=dict(Initial_test=list(generic_table[0].values()),Final_test=list(generic_table[1].values()))
        #Render the required format 
        final_bar_graph=dict(title="SSCD Metrics Bar Chart",x_axis_label="X Axis",y_axis_label="Y Axis",rotated=True,data=horizontal_bar_graph_data,categories=keys_unique)
        #print(final_bar_graph)


        #the time line graph, which will fetch all the test monitor results for the given deployable model and produce a time line graph for all the MTRs
        try:
            client = moc_client.MOCClient()
        except ValueError:
            print("can't find the client")
        path = f"model-manage/api/modelTestResultSummaries/search/findAllByDeployedModel_DeployableModel_Id?deployableModelId={DEPLOYABLE_MODEL_ID}&page=0&size=20"
        result = client.get(path)
        model_test_results = result.json.get("_embedded", {}).get("modelTestResultSummaries", [{}])
        #search for "generic_line_graph" key in the nested structure
        line_graph_results=[test_result['testResults']['performanceMetrics'] for test_result in model_test_results if 'performanceMetrics' in test_result['testResults']]
        print(f"No. of MTRs with line graph ={len(line_graph_results)}")
        #concatenate all the MTRs to produce a time line graph for each metric 
        agg_data = {}
        for dict_o in line_graph_results:
            for key, val in dict_o.items():
                agg_data[key] = agg_data.get(key, []) + val 

        agg_line_graph=dict(title="SSCD Metrics Aggregate Line Graph",x_axis_label="X Axis",y_axis_label="Y Axis",data=agg_data)

        #Append it all to the final json object 
        finalResult["SSCD_metrics_table"] = final_table
        finalResult["performanceMetrics"] = generic_line_graph_data
        finalResult["SSCD_metrics_bar_graph"] = final_bar_graph
        finalResult["SSCD_metrics_time_line_graph"] = agg_line_graph


    yield finalResult
        
def main():
    data = {"data1":993,"data2":36,"data3":3959,"label_value":0,"score":1}
    df = pd.DataFrame.from_dict([data])
    print(json.dumps(next(metrics(df)), indent=2))


if __name__ == '__main__':
	main()        
