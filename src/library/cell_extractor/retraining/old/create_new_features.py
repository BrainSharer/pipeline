from glob import glob
import pandas as pd
import pickle as pk
import numpy as np
import os
from cell_extractor.CellDetectorBase import CellDetectorBase

def create_combined_features(animal = 'DK55'):
    dir = '/net/birdstore/Active_Atlas_Data/cell_segmentation'
    files=glob(dir + f'/{animal}/CH3/*/punta*.csv')  
    df_list=[]
    for filei in files:
        if os.path.getsize(filei) == 1:
            continue
        df=pd.read_csv(filei)
        print(filei,df.shape)
        df_list.append(df)
    full_df=pd.concat(df_list)
    full_df.index=list(range(full_df.shape[0]))
    return full_df

def find_corresponding_row_index(all_segment,search_array):
    index_array = []
    i = 0
    for celli in search_array:
        section = celli[2]
        segments_in_section = all_segment[all_segment[:,2]==section]
        diff = segments_in_section[:,:2]-celli[:2]
        dist = np.sqrt(np.sum(np.square(diff),axis=1))
        cloest_segment = np.argmin(dist)
        if dist[cloest_segment]==0:
            index_array.append(cloest_segment)
        elif dist[cloest_segment]<20:
            index_array.append(cloest_segment)
            print(f'cannot find equal,subbing point with distance: {dist[cloest_segment]}')
        else:
            print('skipping')
            continue
        if i%100 == 0:
            print(i)
        i+=1
    return index_array

def change_positive_index_and_save_df(df):
    test_counts,train_sections = pk.load(open('/data/programming/categories.pkl','rb'))
    all_segment = np.array([df.col,df.row,df.section]).T

    cells = test_counts['detected by computer as sure, unmarked by human']
    cells = np.array([[ci[1]['x'],ci[1]['y'],ci[1]['section']] for ci in cells])
    cells_index = find_corresponding_row_index(all_segment,cells)

    original = train_sections['original training set after mind change']
    original = np.array([[ci[1]['x'],ci[1]['y'],ci[1]['section']] for ci in original])
    original_index = find_corresponding_row_index(all_segment,original)

    validated_unsure = test_counts['detected by computer as UNsure, marked by human as positive']
    validated_unsure = np.array([[ci[1]['x'],ci[1]['y'],ci[1]['section']] for ci in validated_unsure])
    validated_unsure_index = find_corresponding_row_index(all_segment,validated_unsure)

    pk.dump((cells_index,original_index,validated_unsure_index),open('positive_labels.pkl','wb'))

    # positive_index = cells_index+original_index+validated_unsure_index
    # positive_index = original_index
    # labels = np.zeros(len(df))
    # for i in positive_index:
    #     labels[i] = 1
    # df['label'] = labels
    # df.to_csv('/net/birdstore/Active_Atlas_Data/cell_segmentation/DK55/all_features.csv')


if __name__ == '__main__':
    base = CellDetectorBase('DK55',round=1)
    df = base.load_combined_features()
    change_positive_index_and_save_df(df)