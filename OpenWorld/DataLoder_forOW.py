import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch
def load_sirinam_dataset(input_size=200,num_class=100,learn_param=None):
    """
    Load Sirinam's dataset for OW 完成开放场景Sirinam实验数据的加载
    """
    cw_train_data,cw_train_label,cw_test_data,cw_test_label=load_sirinam_dataset_cw(input_size, num_class, formatting=True)
    
    # 根据封闭世界的测试集数量，按照1:1加载开放世界数据
    cw_datanum=len(cw_test_label)
    ow_data,ow_label=load_sirinam_dataset_ow(input_size=200, flow_num=cw_datanum, type_num=100)
    
    # 合并cw和ow数据
    X_test=np.concatenate((cw_test_data,ow_data),axis=0)
    y_test=np.concatenate((cw_test_label,ow_label),axis=0)
    print('==>Finally, Load Sirinam dataset: X_train shape:',cw_train_data.shape,' y_train shape:',cw_train_label.shape)
    print('X_test shape:',X_test.shape,' y_test shape:',y_test.shape)
    return cw_train_data,cw_train_label,X_test,y_test
    
def load_rimmer_dataset(input_size=200,num_class=100,learn_param=None):
    """
    Load Rimmer's (NDSS'18) dataset. 完成开放场景下Rimmer实验数据的加载
    """
    # 先加载封闭世界的测试集
    X_train,y_train,cw_test_data,cw_test_label=load_rimmer_dataset_cw(input_size, num_class, formatting=True,test_ratio=learn_param.as_float('test_ratio'),val_ratio =learn_param.as_float('val_ratio'))
    
    # 根据封闭世界的测试集数量，按照1:1加载开放世界数据
    cw_datanum=len(cw_test_label)
    ow_data,ow_label=load_rimmer_dataset_ow(input_size=200, flow_num=cw_datanum, type_num=100)
    
    # 合并cw和ow数据
    X_test=np.concatenate((cw_test_data,ow_data),axis=0)
    y_test=np.concatenate((cw_test_label,ow_label),axis=0)
    print('==>Finally, Load Rimmer dataset: X_train shape:',X_train.shape,' y_train shape:',y_train.shape)
    print('==>Finally, Load Rimmer dataset: X_test shape:',X_test.shape,' y_test shape:',y_test.shape)
    return X_train,y_train,X_test,y_test

def load_sirinam_dataset_cw(input_size=200, num_classes=95, formatting=True):
    """
    Load Sirinam's (CCS'18) dataset. 共95个类别，每个类别有1000个样本，其中测试集默认100
    """
    # Point to the directory storing data
    dataset_dir = '/home/xuke/lpf/HAAD/utils/dataset/Sirinam/'
   
    # Load train data
    with open(dataset_dir + 'X_train_NoDef.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle, encoding='latin1'))
    with open(dataset_dir + 'y_train_NoDef.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle, encoding='latin1'))
    
    # Load testing data
    with open(dataset_dir + 'X_test_NoDef.pkl', 'rb') as handle:
        X_test = np.array(pickle.load(handle, encoding='latin1'))
    with open(dataset_dir + 'y_test_NoDef.pkl', 'rb') as handle:
        y_test = np.array(pickle.load(handle, encoding='latin1'))

    if formatting:
        X_train,y_train=format_data(X_train,y_train,input_size,num_classes)
        X_test, y_test=format_data(X_test, y_test, input_size, num_classes)
    print('Load Sirinam CW dataset: X_test shape:',X_test.shape,' y_test shape:',y_test.shape)
    print('X_train shape:',X_train.shape,' y_train shape:',y_train.shape)
    
    return  X_train,y_train,X_test, y_test

def load_rimmer_dataset_cw(input_size=200, num_classes=100, formatting=True,test_ratio=0.1,val_ratio =0.1):
    """
    Load Rimmer's (NDSS'18) dataset.  Rimmer数据集,100个类别,每个类别有2500个样本
    """
    # Point to the directory storing data
    dataset_dir = '/home/xuke/lpf/HAAD/utils/dataset/Rimmer/'
    # datafile = '../Dataset/tor_100w_2500tr.npz'

    # Load data
    datafile = dataset_dir + 'tor_%dw_2500tr.npz' % num_classes
    with np.load(datafile, allow_pickle=True) as npzdata:
        data = npzdata['data']
        labels = npzdata['labels']

    # Convert website to integer  将labels转为从0开始的整数
    y = labels.copy()
    websites = np.unique(labels)
    for w in websites:
        y[np.where(labels == w)] = np.where(websites == w)[0][0]

    # Split data to fixed parts
    X_train,y_train,X_test, y_test = train_test_valid_split(data, y, valid_size=val_ratio, test_size=test_ratio)
    
    if formatting:
        X_train,y_train=format_data(X_train,y_train,input_size, num_classes)
        X_test, y_test=format_data(X_test, y_test, input_size, num_classes)
    print('Load Rimmer CW dataset: X_train shape:',X_train.shape,' y_train shape:',y_train.shape)
    print('X_test shape:',X_test.shape,' y_test shape:',y_test.shape)
    return X_train,y_train,X_test, y_test
    
 
def train_test_valid_split(X, y, valid_size=0.1, test_size=0.1):
    """
    Split data into training, validation, and test sets.
    Set random_state=0 to keep the same split.
    """
    # Split into training set and others  划分训练集和others
    split_size = valid_size + test_size
    [X_train, X_, y_train, y_] = train_test_split(X, y,
                                    test_size=split_size,
                                    random_state=0,
                                    stratify=y)

    # Split into validation set and test set 将others划分为验证集和测试集
    split_size = test_size / (valid_size + test_size)
    [X_valid, X_test, y_valid, y_test] = train_test_split(X_, y_,
                                            test_size=split_size,
                                            random_state=0,
                                            stratify=y_)

    return  X_train,y_train,X_test, y_test

def format_data(X, y, input_size, num_classes):
    """
    Format traces into input shape [N x Length x 1] and one-hot encode labels.
    """
    X = X[:, :input_size]
    X = X.astype('float32')
    X = X[:, :, np.newaxis]

    y = y.astype('int32')
    # y = np.eye(num_classes)[y]

    return X, y


def load_rimmer_dataset_ow(input_size=200, flow_num=25000, type_num=100):
    """
    Load Rimmer's (NDSS'18) dataset for OW.
    """
    # rimmer数据集，400000个样本，均为开放场景下的流量
    # file_path="/home/zpc/lpf/utils/Dataset/Rimmer/tor_open_200w_2000tr.npz"
    file_path="/home/xuke/lpf/HAAD/utils/dataset/Rimmer/open/tor_open_400000w.npz"
    
    with np.load(file_path, allow_pickle=True) as npzdata:
        data = npzdata['data']
        labels = npzdata['labels']
    data=data[:flow_num]
    labels=labels[:flow_num]
    data = data[:, :input_size]
    data = data.astype('float32')
    data = data[:, :, np.newaxis]
    labels[:]=type_num   # 将开放场景数据的lable设置为 最大的类别
    labels = labels.astype('int32')
    print('Load Rimmer OW dataset: data shape:',data.shape,' labels shape:',labels.shape)
    return data,labels

def load_sirinam_dataset_ow(input_size=200, flow_num=25000, type_num=100):
    """
    Load Rimmer's (NDSS'18) dataset for OW.
    """
    data_path='/home/xuke/lpf/HAAD/utils/dataset/Sirinam/open/X_test_Unmon_NoDef.pkl'
    label_path="/home/xuke/lpf/HAAD/utils/dataset/Sirinam/open/y_test_Unmon_NoDef.pkl"
    # unmon_Nodef 目录下，有20000条开放世界的样本，label已经处理为 95

    with open(data_path, 'rb') as handle:
        data=pickle.load(handle , encoding='bytes')
    with open(label_path, 'rb') as handle:
        labels=pickle.load(handle , encoding='bytes')
    # 转np数组
    data=np.array(data)
    labels=np.array(labels) 
    
    # 按照给定num取开放场景样本数   
    data=data[:flow_num]
    labels=labels[:flow_num]
    
    # 数据包长度裁剪，类型转换
    data = data[:, :input_size]
    data = data.astype('float32')
    data = data[:, :, np.newaxis]
    labels[:]=type_num # 直接等于最大的类别   sirinam的开放场景数据本身已经预设为 95 了
    labels = labels.astype('int32')
    print('Load Sirinam OW dataset: data shape:',data.shape,' labels shape:',labels.shape)
    return data,labels


class MyDataset(Dataset):
    def __init__(self,data,labels):
        # print(data.dtype)
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
    
        return self.data[idx], self.labels[idx]

def load_datasets(data_name, num_classes, learn_param):
    val_ratio = learn_param.as_float('val_ratio')
    test_ratio = learn_param.as_float('test_ratio')

    if data_name == "sirinam95":
        loader = load_sirinam_dataset
    elif data_name == "rimmer100":
        loader = load_rimmer_dataset
    else:
        raise ValueError(f"Unknown dataset name: {data_name}")

    X_train,y_train,X_test, y_test = loader(
        input_size=200, num_class=num_classes, learn_param=learn_param
    )
    return X_train,y_train,X_test, y_test
    


