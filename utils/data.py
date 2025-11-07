import numpy as np
import pickle
import numpy as np
import pandas as pd
import torch
import os
import torch.nn.functional as F
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

work_dir = os.path.dirname(os.path.abspath(__file__))

def withdraw_9500():
    dataset_dir = "/home/xuke/lpf/all/utils/dataset/Sirinam/open/"

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_test_Unmon_NoDef.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle , encoding='bytes'))
    with open(dataset_dir + 'y_test_Unmon_NoDef.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle, encoding='bytes'))
    X_train=np.array(X_train)
    y_train=np.array(y_train)
    random_indices = np.random.choice(X_train.shape[0], 9500, replace=False)
    X_9500=X_train[random_indices]
    y_9500=y_train[random_indices]
    np.savez(dataset_dir+'Open_9500.npz', data=X_9500, label=y_9500)
    
def load_openSirinam_dataset( formatting=True,test_ratio=0.475,val_ratio = 0.25):
    
    # Point to the directory storing data
    dataset_dir = work_dir+"/dataset/Sirinam/open/"

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_test_Unmon_NoDef.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle , encoding='bytes'))
    with open(dataset_dir + 'y_test_Unmon_NoDef.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle, encoding='bytes'))
    
    print(len(X_train),len(y_train))
    a=input(" 112")
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_test_valid_split(X_train, y_train, valid_size=val_ratio, test_size=test_ratio)
    if formatting:
        return format_data_all(X_train, y_train, X_valid, y_valid, X_test, y_test, 200, 96)
    return X_train, y_train, X_valid, y_valid, X_test, y_test
  

def format_data(X, y, input_size, num_classes):
    """
    Format traces into input shape [N x Length x 1] and one-hot encode labels.
    """
    X = X[:, :input_size]
    X = X.astype('float32')
    X = X[:, :, np.newaxis]

    y = y.astype('int32')
    y = np.eye(num_classes)[y]

    return X, y


def format_data_all(X_train, y_train, X_valid, y_valid, X_test, y_test, input_size, num_classes):
    X_train, y_train = format_data(X_train, y_train, input_size, num_classes)
    X_valid, y_valid = format_data(X_valid, y_valid, input_size, num_classes)
    X_test, y_test = format_data(X_test, y_test, input_size, num_classes)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def LoadDataNoDefCW(input_size, num_classes,formatting=True,val_ratio=0.25,test_ratio=0.25):
    
    print("Loading non-defended dataset for closed-world scenario")
    # Point to the directory storing data
    dataset_dir = work_dir+"/dataset/Sirinam/"

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_train_NoDef.pkl', 'rb') as handle:
        
        X_train = np.array(pickle.load(handle , encoding='bytes'))
    with open(dataset_dir + 'y_train_NoDef.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle, encoding='bytes'))

    # Load validation data
    with open(dataset_dir + 'X_valid_NoDef.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle, encoding='bytes'))
    with open(dataset_dir + 'y_valid_NoDef.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle, encoding='bytes'))

    # Load testing data
    with open(dataset_dir + 'X_test_NoDef.pkl', 'rb') as handle:
        X_test = np.array(pickle.load(handle, encoding='bytes'))
    with open(dataset_dir + 'y_test_NoDef.pkl', 'rb') as handle:
        y_test = np.array(pickle.load(handle, encoding='bytes'))

    # print( "Data dimensions:")
    # print ("X: Training data's shape : ", X_train.shape)
    # print ("y: Training data's shape : ", y_train.shape)
    # print ("X: Validation data's shape : ", X_valid.shape)
    # print ("y: Validation data's shape : ", y_valid.shape)
    # print ("X: Testing data's shape : ", X_test.shape)
    # print ("y: Testing data's shape : ", y_test.shape)
    all_x = np.concatenate((X_train,X_valid,X_test),axis=0)
    all_y = np.concatenate((y_train,y_valid,y_test),axis=0)
    # category_count = Counter(all_y)
    # count_ = 0
    # for category, count in category_count.items():
    #     print(f"类别 {category} 的数量是: {count}")
    #     count_ += 1
    #     if count_ == 4:
    #         break
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_test_valid_split(all_x,all_y, valid_size=val_ratio, test_size=test_ratio)
    if formatting:
        return format_data_all(X_train, y_train, X_valid, y_valid, X_test, y_test, input_size, num_classes)
    else:
        return X_train, y_train, X_valid, y_valid, X_test, y_test
    
def train_test_valid_split(X, y, valid_size=0.1, test_size=0.1):
    """
    Split data into training, validation, and test sets.
    Set random_state=0 to keep the same split.
    """
    # Split into training set and others
    split_size = valid_size + test_size
    [X_train, X_, y_train, y_] = train_test_split(X, y,
                                    test_size=split_size,
                                    random_state=0,
                                    stratify=y)

    # Split into validation set and test set
    split_size = test_size / (valid_size + test_size)
    [X_valid, X_test, y_valid, y_test] = train_test_split(X_, y_,
                                            test_size=split_size,
                                            random_state=0,
                                            stratify=y_)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

#Rimmer数据集
def load_rimmer_dataset(input_size=5000, num_classes=100, formatting=True,test_ratio=0.25,val_ratio = 0.25):
    """
    Load Rimmer's (NDSS'18) dataset.
    """
    # Point to the directory storing data
    dataset_dir =  work_dir+"/dataset/Rimmer/"
    # datafile = '../Dataset/tor_100w_2500tr.npz'

    # Load data
    datafile = dataset_dir + 'tor_%dw_2500tr.npz' % num_classes
    with np.load(datafile, allow_pickle=True) as npzdata:
        data = npzdata['data']
        labels = npzdata['labels']

    
    
    # Convert website to integer
    y = labels.copy()
    websites = np.unique(labels)
    for w in websites:
        y[np.where(labels == w)] = np.where(websites == w)[0][0]
    # category_count = Counter(y)
    # count_ = 0
    # for category, count in category_count.items():
    #     count_ = count_ + 1
    #     print(f"类别 {category} 的数量是: {count}")
    #     if count_ == 5:
    #         break
    # Split data to fixed parts
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_test_valid_split(data, y, valid_size=val_ratio, test_size=test_ratio)
    with open(dataset_dir + 'tor_%dw_2500tr_test.npz' % num_classes, 'wb') as handle:
        pickle.dump({'X_test': X_test, 'y_test': y_test}, handle)

    if formatting:
        return format_data_all(X_train, y_train, X_valid, y_valid, X_test, y_test, input_size, num_classes)
    else:
        return X_train, y_train, X_valid, y_valid, X_test, y_test
    
#Drift数据集
def load_drift_dataset(input_size=5000, num_classes=92, formatting=True,test_ratio=0.25,val_ratio = 0.25):
    data = np.load('/home/xuke/lpf/all/utils/dataset/DF/Drift90.npz')
    
    # 从npz文件中获取数据
    X_inf = data['X_inferior']
    y_inf = data['y_inferior']
    X_sup = data['X_superior']
    y_sup = data['y_superior']
    label = np.zeros(np.max(data['y_superior'])+1)
    label[np.unique(y_sup)] = np.arange(len(np.unique(y_sup)))
    num_classes = len(np.unique(y_sup))
    X = np.vstack((X_inf,X_sup))
    Y = np.hstack((y_inf,y_sup))
    y = label[Y]
    # Split data to fixed parts
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_test_valid_split(X, y, valid_size=val_ratio, test_size=test_ratio)
    if formatting:
        return format_data_all(X_train, y_train, X_valid, y_valid, X_test, y_test, input_size, num_classes)
    else:
        return X_train, y_train, X_valid, y_valid, X_test, y_test




class MyDataset(Dataset):
    def __init__(self,data,labels):
        # print(data.dtype)
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
    
        return self.data[idx], self.labels[idx]
if __name__ == "__main__":
    # withdraw_9500()
    print("Load Data tool")