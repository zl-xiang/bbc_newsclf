import pickle
import os, os.path
import re
import random

def serialisation(file_name,data):
    # Open a file and use dump() 
    with open(file_name, 'wb') as file: 
        # A new file will be created 
        pickle.dump(data, file , pickle.HIGHEST_PROTOCOL)

def load(file_name):
    # Open the file in binary mode 
    with open(file_name, 'rb') as file:
     # Call load method to deserialze 
     data = pickle.load(file) 
     return data

# loading dataset from local directory
def load_local_dataset(dir,encoding='utf-8'):
    raw_dataset = []
    dataset = []
    with open(dir,'r', encoding='utf-8',
                 errors='ignore') as f:
        raw_dataset = f.readlines()
    for line in raw_dataset:
        if line != '\n' and line != '': dataset.append(line.replace('\n',''))
    return dataset
 
def dir_files_count(dir):
   return len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])


def does_file_exist(filename, dir="./"):
    return os.path.exists(os.path.join(dir,filename))
