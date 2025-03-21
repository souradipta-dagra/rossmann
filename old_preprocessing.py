#1

import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

sts_chile_ns16_prepared = dataiku.Dataset("sts_vline_rrsmc_prepared")
sts_chile_ns16_prepared_df = sts_chile_ns16_prepared.get_dataframe()

sts_chile_ns16_prepared_df.columns


sts_chile_ns16_prepared_df['frequencyobs'] = sts_chile_ns16_prepared_df['observationcategory'].map(sts_chile_ns16_prepared_df['observationcategory'].value_counts())
sts_chile_ns16_prepared_df['frequencypercobs'] = sts_chile_ns16_prepared_df['observationcategory'].map(np.round(sts_chile_ns16_prepared_df['observationcategory'].value_counts(normalize=True)*100,2))


sts_chile_ns16_prepared_df['frequencysol'] = sts_chile_ns16_prepared_df['solutioncategory'].map(sts_chile_ns16_prepared_df['solutioncategory'].value_counts())
sts_chile_ns16_prepared_df['frequencypercsol'] = sts_chile_ns16_prepared_df['solutioncategory'].map(np.round(sts_chile_ns16_prepared_df['solutioncategory'].value_counts(normalize=True)*100,2))

#sts_chile_ns16_prepared_df['frequencypercsol'].value_counts()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df = sts_chile_ns16_prepared_df[[u'project', u'country', u'fleet', u'subsystem', u'database', u'observationcategory', u'observation', u'failureclass', u'problemcode',
u'problemcause', u'problemremedy', u'functionallocation', u'notificationsonumber', u'date', u'solutioncategory', u'solution', u'pbscode',
 u'symptomcode', u'rootcause', u'documentlink', u'language', u'resource', u'minresourcesneed', u'maxresourceneed', u'themostfrequentvalueforresource',
 u'time', u'mintimeperoneperson', u'maxtimeperoneperson', u'averagetime', u'frequencypercobs',
  u'frequencypercsol', u'minresourcesneedsol', u'maxresourceneedsol',
 u'themostfrequentvalueforresourcesol', u'mintimeperonepersonsol', u'maxtimeperonepersonsol', u'averagetimesol']]


df = df.rename({"observationcategory":"observation_category", "failureclass":"failure_class","problemcode":"problem_code","problemcause":"problem_cause",
 "problemremedy":"problem_remedy","solutioncategory":"solution_category","pbscode":"pbs_code","symptomcode":"symptom_code","rootcause":"root_cause",
 "functionallocation":"functional_location","notificationsonumber":"notifications_number","documentlink":"document_link","minresourcesneed":"min_resources_need",
 "maxresourceneed": "max_resource_need", "themostfrequentvalueforresource":"the_most_frequent_value_for_resource",
 "mintimeperoneperson": "min_time_per_one_person", "maxtimeperoneperson":"max_time_per_one_person", "averagetime":"average_time", "frequencypercobs":"frequency_obs",
  "frequencypercsol":"frequency_sol", "minresourcesneedsol":"min_resources_need_sol", "maxresourceneedsol":"max_resource_need_sol",
 "themostfrequentvalueforresourcesol":"the_most_frequent_value_for_resource_sol", "mintimeperonepersonsol":"min_time_per_one_person_sol",
 "maxtimeperonepersonsol":"max_time_per_one_person_sol", "averagetimesol":"average_time_sol"},axis=1)


if(df['notifications_number'].isnull().sum()>0):
    df['notifications_number'] = df['notifications_number'].apply(lambda x: 'NA' if pd.isna(x) else str(round(x)))

input_test_0103 = dataiku.Dataset("sts_vline_rrsmc_extract")
input_test_0103.write_with_schema(df)

#2
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from cosine_sim import clean_query
from translate import get_translation

# Read recipe inputs
xtrapolis_input_data = dataiku.Dataset("sts_vline_rrsmc_extract")
xtrapolis_input_data_df = xtrapolis_input_data.get_dataframe()
df = xtrapolis_input_data_df.copy()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
NaN = np.nan
df["obs_rating"]=NaN
df["no_times_obs_rated"]=NaN
df['obs_cat_rating'] = NaN
df["no_times_obs_cat_rated"] = NaN

df["observation_category"] = df["observation_category"].astype('str')
df["solution_category"] = df["solution_category"].astype('str')

df["observation_category"].fillna(value="NA",inplace=True)
df["problem_remedy"].fillna(value="NA",inplace=True)
df["pbs_code"].fillna(value="NA",inplace=True)
df["symptom_code"].fillna(value="NA",inplace=True)
df["root_cause"].fillna(value="NA",inplace=True)
df["failure_class"].fillna(value="NA",inplace=True)
df["solution"].fillna(value="NA",inplace=True)
df["functional_location"].fillna(value="NA",inplace=True)
df["notifications_number"].fillna(value="NA",inplace=True)
df["date"].fillna(value="NA",inplace=True)
df["document_link"].fillna(value="NA",inplace=True)



df["frequency_obs"].fillna(value=0,inplace=True)
df["frequency_sol"].fillna(value=0,inplace=True)
df["obs_rating"].fillna(value=0,inplace=True)
df["no_times_obs_rated"].fillna(value=0,inplace=True)
df['obs_cat_rating'].fillna(value=0,inplace=True)
df["no_times_obs_cat_rated"].fillna(value=0,inplace=True)
df['resource'].fillna(value=0,inplace=True)

df["frequency_obs"] = pd.to_numeric(df["frequency_obs"], downcast='float')
df["frequency_sol"] = pd.to_numeric(df["frequency_sol"], downcast='float')
df['resource'] = pd.to_numeric(df['resource'], downcast='integer')


df["obs_rating"] = pd.to_numeric(df["obs_rating"], downcast='integer')
df["no_times_obs_rated"] = pd.to_numeric(df["no_times_obs_rated"], downcast='integer')
df['obs_cat_rating'] = pd.to_numeric(df["obs_cat_rating"], downcast='integer')
df["no_times_obs_cat_rated"] = pd.to_numeric(df["no_times_obs_cat_rated"], downcast='integer')


df["observation_category"] = df["observation_category"].str.upper()
df["solution_category"].fillna(value="NA",inplace=True)
df["solution_category"] = df["solution_category"].str.upper()
df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_un = pd.unique(df.observation_category)
df_un_srt = np.sort(df_un)
cat_id = (np.arange(1,len(df_un,)+1)).tolist()
df_un_srt_df = pd.DataFrame(data=df_un_srt,columns=["observation_category"])
df_un_srt_df["cat_id"] = cat_id
df_final = pd.merge(df,df_un_srt_df,on="observation_category",how="left")
df_final.observation_category = df_final["observation_category"].apply(lambda x: x.upper())
df_final.solution_category = df_final["solution_category"].apply(lambda x: x.upper())
df_final=df_final.rename(columns={"cat_id": "category_id"})
df_final.obs_id = df_final.index.values+1

cols = list(df_final.columns)
cols = [cols[-1]] + cols[:-1]
cols
df_final = df_final[cols]
df_final.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_sun = pd.unique(df_final.solution_category)
df_sun_srt = np.sort(df_sun)
scat_id = (np.arange(1,len(df_sun,)+1)).tolist()
df_sun_srt_df = pd.DataFrame(data=df_sun_srt,columns=["solution_category"])
df_sun_srt_df["scat_id"] = scat_id
df_finals = pd.merge(df_final,df_sun_srt_df,on="solution_category",how="left")

# df_final.drop(columns=['category_id'],inplace=True)
df_finals=df_finals.rename(columns={"scat_id": "sol_category_id"})
df_finals.head()


df_final = df_finals.copy()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
idx = 7
new_col = np.arange(1,len(df_final)+1)  # can be a list, a Series, an array or a scalar
df_final.insert(loc=idx, column='obs_id', value=new_col)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE


if(len(df_final['language'].unique())==1):
    if(df_final['language'].unique()[0].lower()=='english'):
        lan = 'english'
    else:
        lan = 'nonenglish'
else:
    lan = 'nonenglish'


df_final['problem_cause'] = df_final['problem_cause'].replace(np.nan,'', regex=True)
df_final['problem_code'] = df_final['problem_code'].replace(np.nan,'', regex=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
for x in range(3):
    try:
        if(lan!='english'):
            dftn = pd.DataFrame()
            dftn['observation'] = df_final['observation'].copy()
            dftn['obs_translated'] = [get_translation(txt, 'en') for txt in df_final.observation]
            dftn['problem_cause_translated'] = [get_translation(txt, 'en') for txt in df_final.problem_cause]
            dftn['problem_code_translated'] = [get_translation(txt, 'en') for txt in df_final.problem_code]
            dftn['solution_translated'] = [get_translation(txt, 'en') for txt in df_final.solution]


            #   df_final["obs_translated"] = dftn['obs_translated'] + ' ' + dftn['problem_cause_translated']
            df_final["obs_translated"] = dftn["obs_translated"].apply(lambda x : clean_query(x))
            df_final["problem_cause_translated"] = dftn["problem_cause_translated"].apply(lambda x : clean_query(x))
            df_final["problem_code_translated"] = dftn["problem_code_translated"].apply(lambda x : clean_query(x))
            df_final["solution_translated"] = dftn["solution_translated"].apply(lambda x : clean_query(x))
            break

        else:
            df_final["obs_translated"] = df_final["observation"].apply(lambda x : clean_query(x))
            df_final["problem_cause_translated"] = df_final["problem_cause"].apply(lambda x : clean_query(x))
            df_final["problem_code_translated"] = df_final["problem_code"].apply(lambda x : clean_query(x))
            df_final["solution_translated"] = df_final["solution"].apply(lambda x : clean_query(x))
            break
    except:
        continue

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_final['problem_cause'] = df_final['problem_cause'].replace(r'^\s*$', np.nan, regex=True)
df_final["problem_cause"].fillna(value="NA",inplace=True)
df_final["problem_cause"]


col_str = ['category_id', 'project', 'country', 'fleet', 'subsystem', 'database', 'observation_category', 'obs_id', 'observation', 'failure_class', 'problem_code', 'problem_cause', 'problem_remedy', 'functional_location', 'notifications_number', 'date', 'solution_category', 'solution', 'pbs_code', 'symptom_code', 'root_cause', 'document_link', 'language', 'obs_translated']


for cols in col_str:
    df_final[cols] = df_final[cols].apply(str)


ind_input_procesed_df = df_final


# Write recipe outputs
ind_input_procesed = dataiku.Dataset("vline_rrsmc_processed")
ind_input_procesed.write_with_schema(ind_input_procesed_df)


#3
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

import gensim
import multiprocessing
from multiprocessing import Pool
import functools, multiprocessing
from scipy import sparse


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
nltk.download('averaged_perceptron_tagger')


import string
import random
from itertools import chain
import math
import csv
import time
import operator
from collections import Counter
from collections import defaultdict
import pandas as pd

stop = stopwords.words('english') + list(string.punctuation)
from nltk import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy import spatial

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from multiprocessing import Pool

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read recipe inputs
xtrapolis_chile_input_procesed = dataiku.Dataset("vline_rrsmc_processed")
xtrapolis_chile_input_procesed_df = xtrapolis_chile_input_procesed.get_dataframe()
df = xtrapolis_chile_input_procesed_df.copy()

df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def lst_to_str(str):
    str1 = " "
    return (str1.join(str))

def str_tagged(str):
    s_tagged = nltk.tag.pos_tag(str.split())
    s1 = [word for word,tag in s_tagged if tag == 'NN' or tag == 'NNS' or tag =='NNP' or tag == 'NNPS'or tag == 'CD']
    s1 = lst_to_str(s1)
    s2 = [word for word,tag in s_tagged if tag == 'JJ' or tag == 'JJR' or tag == 'JJS' or tag == 'RBR' or tag == 'RBS' or tag == 'RB' or tag == 'VB' or tag == 'VBN' or tag == 'VBG' or tag == 'VBD' or tag == 'VBP' or tag == 'VBZ']
    s2 = lst_to_str(s2)
    s = [s1, s2]
    return s

num_cores = multiprocessing.cpu_count()
def parallelize_dataframe(df, func, U=None, V=None):

    #blockSize = 5000
    num_partitions = 5 # int( np.ceil(df.shape[0]*(1.0/blockSize)) )
    blocks = np.array_split(df, num_partitions)

    pool = Pool(num_cores)
    if V is not None and U is not None:
        # apply func with multiple arguments to dataframe (i.e. involves multiple columns)
        df = pd.concat(pool.map(functools.partial(func, U=U, V=V), blocks))
    else:
        # apply func with one argument to dataframe (i.e. involves single column)
        df = pd.concat(pool.map(func, blocks))

    pool.close()
    pool.join()

    return df

def square(x):
    return x**2

def test_func(data):
    print("Process working on: ", data.shape)
    data["squareV"] = data["testV"].apply(square)
    return data

def vecProd(row, U, V):
    return np.sum( np.multiply(U[int(row["obsI"]),:], V[int(row["obsJ"]),:]) )

def mProd_func(data, U, V):
    data["predV"] = data.apply( lambda row: vecProd(row, U, V), axis=1 )
    return data

def generate_simulated_data():

    N, D, nnz, K = [302, 184, 5000, 5]
    I = np.random.choice(N, size=nnz, replace=True)
    J = np.random.choice(D, size=nnz, replace=True)
    vals = np.random.sample(nnz)

    sparseY = sparse.csc_matrix((vals, (I, J)), shape=[N, D])

    # Generate parameters U and V which could be used to reconstruct the matrix Y
    U = np.random.sample(N*K).reshape([N,K])
    V = np.random.sample(D*K).reshape([D,K])

    return sparseY, U, V

def main():
    Y, U, V = generate_simulated_data()

    # find row, column indices and obvseved values for sparse matrix Y
    (testI, testJ, testV) = sparse.find(Y)

    colNames = ["obsI", "obsJ", "testV", "predV", "squareV"]
    dtypes = {"obsI":int, "obsJ":int, "testV":float, "predV":float, "squareV": float}

    obsValDF = pd.DataFrame(np.zeros((len(testV), len(colNames))), columns=colNames)
    obsValDF["obsI"] = testI
    obsValDF["obsJ"] = testJ
    obsValDF["testV"] = testV
    obsValDF = obsValDF.astype(dtype=dtypes)

    print("Y.shape: {!s}, #obsVals: {}, obsValDF.shape: {!s}".format(Y.shape, len(testV), obsValDF.shape))

    # calculate the square of testVals
    obsValDF = parallelize_dataframe(obsValDF, test_func)

    # reconstruct prediction of testVals using parameters U and V
    obsValDF = parallelize_dataframe(obsValDF, mProd_func, U, V)

    print("obsValDF.shape after reconstruction: {!s}".format(obsValDF.shape))
    print("First 5 elements of obsValDF:\n", obsValDF.iloc[:5,:])

if __name__ == '__main__':
    main()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df['obs_translated'] = df['obs_translated'].fillna("")
df["obs_tagged"] = df.apply(lambda row:str_tagged(row['obs_translated']), axis = 1)
df['problem_cause_translated'] = df['problem_cause_translated'].fillna("")
df["problem_cause_tagged"] = df.apply(lambda row:str_tagged(row['problem_cause_translated']), axis = 1)
df['problem_code_translated'] = df['problem_code_translated'].fillna("")
df["problem_code_tagged"] = df.apply(lambda row:str_tagged(row['problem_code_translated']), axis = 1)
df["solution_tagged"] = df.apply(lambda row:str_tagged(row['solution_translated']), axis = 1)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df2 = pd.DataFrame(df.obs_tagged.tolist(), columns=['obs_entity', 'obs_desc'])
df = pd.concat([df, df2], axis=1, sort=False)

df3 = pd.DataFrame(df.problem_cause_tagged.tolist(), columns=['problem_cause_entity', 'problem_cause_desc'])
df = pd.concat([df, df3], axis=1, sort=False)

df4 = pd.DataFrame(df.problem_code_tagged.tolist(), columns=['problem_code_entity', 'problem_code_desc'])
df = pd.concat([df, df4], axis=1, sort=False)

df5 = pd.DataFrame(df.solution_tagged.tolist(), columns=['solution_entity', 'solution_desc'])
df = pd.concat([df, df5], axis=1, sort=False)

df.head()

df.fillna(value="NA",inplace=True)

# Write recipe outputs
xtrapolis_chile_input_tagged = dataiku.Dataset("vline_rrsmc_tagged")
xtrapolis_chile_input_tagged.write_with_schema(df)

#4

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import multiprocessing
import pandas as pd
import numpy as np
from multiprocessing import Pool
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import pickle
from api_keys import host,apiKey
import dataikuapi
import dataiku, os.path
from sentence_transformers import SentenceTransformer


# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
net2_project_tagged_prepared = dataiku.Dataset("vline_rrsmc_tagged_prepared")
net2_project_tagged_prepared_df = net2_project_tagged_prepared.get_dataframe()

vectorization_df = net2_project_tagged_prepared_df.copy()

vectorization_df.head()
vectorization_df.reset_index(drop=True,inplace=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def parallelize_dataframe(df, func):
    a = np.array_split(df, num_partitions)
    del df
    pool = Pool(num_cores)
    #df = pd.concat(pool.map(func, [a,b,c,d,e]))
    df = sp.vstack(pool.map(func, a), format='csr')
    pool.close()
    pool.join()
    return df


### Loading Sentence Transformer ###
encoder = SentenceTransformer("all-MiniLM-L12-v2")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ### Observation - Implemented Transformers ###
### Observation - Implemented Transformers ###
df_clean = vectorization_df.copy()
df_clean["obs_merged"] = df_clean["obs_entity"].fillna("") + df_clean["obs_desc"].fillna("")
df_clean = df_clean[(df_clean.obs_merged != "")]

text_ = df_clean['obs_merged'].tolist()
vectors = encoder.encode(text_)

#======================================
df_vectors = pd.DataFrame(vectors)
df_vectors['obs_id'] = df_clean['obs_id'].values

dataset = dataiku.Dataset('vectors_ds')
dataset.write_with_schema(df_vectors)
#======================================

vector_dimension = vectors.shape[1]
index = faiss.IndexFlatL2(vector_dimension)
#index.add(vectors)
index = faiss.IndexIDMap(index)
index.add_with_ids(vectors, df_clean.obs_id.values)

handle = dataiku.Folder('5E0Oitce')
path = handle.get_path()
path_in = os.path.join(path, 'vline_rrsmc_obs_transformers.index')

faiss.write_index(index,path_in)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ### Problem Cause - Implemented Transformers ###
### Problem Cause - Implemented Transformers ###
df_clean1 = vectorization_df.copy()

if not (df_clean1["problem_cause_entity"].isna().sum() == len(df_clean1) and \
        df_clean1["problem_cause_desc"].isna().sum() == len(df_clean1)):

    df_clean1["problem_cause_merged"] = df_clean1["problem_cause_entity"].fillna("") + df_clean1["problem_cause_desc"].fillna("")
    df_clean1 = df_clean1[(df_clean1.problem_cause_merged != "")]

    text1_ = df_clean1['problem_cause_merged'].tolist()
    vectors1 = encoder.encode(text1_)

    vector_dimension1 = vectors1.shape[1]
    index1 = faiss.IndexFlatL2(vector_dimension1)
    #index1.add(vectors1)
    index1 = faiss.IndexIDMap(index1)
    index1.add_with_ids(vectors1, df_clean1.obs_id.values)

    handle = dataiku.Folder('5E0Oitce')
    path = handle.get_path()
    path_in = os.path.join(path, 'vline_rrsmc_problem_cause_transformers.index')

    faiss.write_index(index1,path_in)

else:

    index1 = faiss.IndexFlatL2(0)

    handle = dataiku.Folder('5E0Oitce')
    path = handle.get_path()
    path_in = os.path.join(path, 'vline_rrsmc_problem_cause_transformers.index')

    faiss.write_index(index1,path_in)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ### Problem Code - Implemented Transformers ###
### Problem Code - Implemented Transformers ###
df_clean2= vectorization_df.copy()

if not (df_clean2["problem_code_entity"].isna().sum() == len(df_clean2) and \
        df_clean2["problem_code_desc"].isna().sum() == len(df_clean2)):

    df_clean2["problem_code_merged"] = df_clean2["problem_code_entity"].fillna("") + df_clean2["problem_code_desc"].fillna("")
    df_clean2= df_clean2[(df_clean2.problem_code_merged != "")]

    text2_ = df_clean2['problem_code_merged'].tolist()
    vectors2 = encoder.encode(text2_)

    vector_dimension2 = vectors2.shape[1]
    index2 = faiss.IndexFlatL2(vector_dimension2)
    #index2.add(vectors2)
    index2 = faiss.IndexIDMap(index2)
    index2.add_with_ids(vectors2, df_clean2.obs_id.values)

    handle = dataiku.Folder('5E0Oitce')
    path = handle.get_path()
    path_in = os.path.join(path, 'vline_rrsmc_problem_code_transformers.index')

    faiss.write_index(index2,path_in)

else:

    index2 = faiss.IndexFlatL2(0)

    handle = dataiku.Folder('5E0Oitce')
    path = handle.get_path()
    path_in = os.path.join(path, 'vline_rrsmc_problem_code_transformers.index')

    faiss.write_index(index2,path_in)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ### Solution - Implemented Transformers ###
### Problem Code - Implemented Transformers ###
df_clean3= vectorization_df.copy()
df_clean3["solution_merged"] = df_clean3["solution_entity"].fillna("") + df_clean3["solution_desc"].fillna("")
df_clean3= df_clean3[(df_clean3.solution_merged != "")]

text3_ = df_clean3['solution_merged'].tolist()
vectors3 = encoder.encode(text3_)

vector_dimension3 = vectors3.shape[1]
index3 = faiss.IndexFlatL2(vector_dimension3)
#index3.add(vectors3)
index3 = faiss.IndexIDMap(index3)
index3.add_with_ids(vectors3, df_clean3.obs_id.values)

handle = dataiku.Folder('5E0Oitce')
path = handle.get_path()
path_in = os.path.join(path, 'vline_rrsmc_solution_transformers.index')

faiss.write_index(index3,path_in)

