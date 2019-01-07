'''
This script reorganises the spectograms into 
a directory structure suitable for flow_from_directory
method of Keras's ImageDataGenerator
'''

import os
import argparse
import shutil
import csv


def organise(args):
    source = os.path.abspath(args.source)
    target = os.path.abspath(args.target)

    results = {"Already existed": 0, "Newly Added": 0}

    csvs = []
    for fname in os.listdir(source):
        if fname.endswith('.csv'):
            csvs.append(fname)


    for csv_item in csvs:
        set_name = csv_item.split('.')[0]   # training, testing or validation set        
        with open(os.path.join(source, csv_item), "r") as csvfile:
            for (file_path, label) in list(csv.reader(csvfile)):
                destination = os.path.join(target, set_name, label.strip())
                if not os.path.exists(destination):
                    os.makedirs(destination)
                try:
                    # shutil.move(file_path, destination)
                    shutil.copy(file_path, destination)
                    results["Newly Added"] += 1
                except shutil.Error as e:
                    print(e) 
                    results["Already existed"] += 1

    print("\n")
    print('The loaded csv files were: ')
    print(csvs)
    print("\n")
    print(results)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', dest='source', required=True)
    parser.add_argument('--target', dest='target', required=True)
    cli_args = parser.parse_args()

    organise(cli_args)
