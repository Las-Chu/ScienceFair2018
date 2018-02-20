import glob
import time
import random
import json
import csv
from pprint import pprint


def print_images():

    images = glob.glob('/home/ramesh/vidya/project_cancer/dataset/*.jpg')
    meta_json = glob.glob('/home/ramesh/vidya/project_cancer/dataset/*.json')
    #print(len(images))

    counter = 0
    for jsonPtr in meta_json:
        print(type(jsonPtr) is str, jsonPtr)
        data_from_json = json.load(open(jsonPtr)) # gets Dict
        print(type(data_from_json) is dict, data_from_json)
        print(data_from_json['name'], data_from_json['meta']['clinical']['benign_malignant'], data_from_json['meta']['clinical']['diagnosis'])
        '''
        #pythin Dict to JSON
        json_obj = json.dumps(data_from_json)
        print(type(json_obj) is str, json_obj)
        # change the JSON string into a python obj
        decoded_topythonobj = json.loads(json_obj)
        print(type(decoded_topythonobj) is dict, decoded_topythonobj)
        print(data_from_json)
        print(json_obj)
        #print (decoded_topythonobj)
        '''
        #open a file for writing
        fo = open("ds_metadata.csv", 'a')
        #create a csv writer object
        '''
        if counter == 0:
            for key in data_from_json:
                print key
                #condition = decoded_topythonobj.hasOwnProperty(key)
                #print condition
                #if (decoded_topythonobj.hasOwnProperty(key)):
                #   print key
                #    csv_writer = csv.DictWriter(fo, key)
                #    csv_writer.writeheader()
                #else:
                for innerkey in data_from_json[key]:
                    print innerkey
                    csv_writer = csv.DictWriter(fo, key)
                    csv_writer.writeheader()
            counter = +1
        #print ("test " , data_from_json.items())
        
        for attribute, value  in data_from_json.iteritems():
           print value
           csv_writer.writerows(value)
        '''
        fo.write(data_from_json['name']+','+ data_from_json['meta']['clinical']['benign_malignant']+','+ data_from_json['meta']['clinical']['diagnosis']+'\n')

        # close opened file
        fo.close()


img = print_images()