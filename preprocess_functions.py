import numpy as np
import os
from skimage import img_as_ubyte
import warnings


'''
Get training/validation datasets' relative path
'''
def get_files_paths(directory):
    filenames = sorted([f for f in os.listdir(directory)])
    return ['{}/{}/{}_instance.mat'.format(directory, filename, filename) for filename in filenames]


'''
Get training/validation datasets' relative path
'''
def get_files_paths_nifiti(directory):
    filenames = sorted([f for f in os.listdir(directory)])
    return ['{}/{}'.format(directory, filename) for filename in filenames]

'''
Pad empty/black slices to patient's data so that each patient has a total of 160 slices, which
is evenly divisible by 16. If the total number of slices is not evenly divisible by 16, the FPN
would break during upsampling.
'''
def add_empty_slices(img, total_slices=160):
    if img.shape[2] != total_slices:
        empty_slices = np.zeros((img.shape[0], img.shape[1], total_slices - img.shape[2]))
        img = np.concatenate((img, empty_slices), axis=2)
    
    return img

'''
Preprocess Inf
'''
def preprocess_inf(data):
    # keys = instance ids
    inf_dict = {}
    
    # get keys
    keys = [key for key in data[0][1].dtype.fields.keys()]
    should_flatten = ['Slice', 'Radius', 'Area', 'SliceSort', 'AreaSort', 'RadiusSort', 'Volume']
    
    for i in range(len(data)):
        instance_id = data[i][0][0,0] # first element is instance id
        row_data = {}
        for key in keys:
            if key in should_flatten:
                row_data[key] = data[i][1][key][0][0].flatten()
            else:
                row_data[key] = data[i][1][key][0][0]
        inf_dict[instance_id] = row_data
    
    return inf_dict

'''
convert to cv2 compatiable format
'''
def to_cv2_compatiable(imgs):
    with warnings.catch_warnings():    # suppress warning
        for cur_slice in range(imgs.shape[2]):
            warnings.simplefilter("ignore")
            imgs[:,:,cur_slice] = img_as_ubyte(imgs[:,:,cur_slice])
    return imgs

'''
Find the slice number(s) that contain a target 
'''
def find_fg_slices(mask):
    slice_nums = []
    
    for i in range(mask.shape[0]):
        uni = np.unique(mask[i,:,:])
        if uni.shape[0] > 1:
            slice_nums.append(i)
            
    return slice_nums

'''
Get actual slices that contain the scan
'''
def get_actual_slices(data):
    for i in range(data['imageoriginal'].shape[2]):
        if len(np.unique(data['imageoriginal'][:,:,i])) == 1:
            return i
    return data['imageoriginal'].shape[2]