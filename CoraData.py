from collections import namedtuple
import itertools
import os
import pickle
import scipy.sparse as sp
import os.path as osp
import urllib.request
import numpy as np

Data = namedtuple('Data',['x','y','adjacency',
                          'train_mask','val_mask','test_mask'])

class CoraData(object):
    download_url = "https://github.com/kimiyoung/planetoid/raw/master/data"
    filenames = ["ind.cora.{}".format(name) for name in
                 ['x','tx','allx','y','ty','ally','graph','test.index']]
    
    def __init__(self, data_root="cora",rebuild=False):
        """
        including data downloading, processing and loading
        when data already existed, using the storing data 
        else repeat the action above

        Args:
        data_root: string, optional
                    used for data storage with path: {data_root}/raw
                    processed_data was stored in {data_root}/processed_cora.pkl
        rebuild: boolean, optional
                    whether to rebuild the data set,when set to True,
                    it will cover the proecessed data which already exists
        """
        self.data_root = data_root
        save_file = osp.join(self.data_root,"processed_cora.pkl")
        if osp.exists(save_file) and not rebuild:
            print("Using Cached file:{}".format(save_file))
            self._data = pickle.load(open(save_file,"rb"))
        else:
            self.maybe_download()
            self._data=self.process_data()
            with open(save_file,"wb") as f:
                pickle.dump(self.data,f)
            print("Cached file:{}".format(save_file))
        pass

    @property
    def data(self):
        """return data object, including x,y,adjacency,train/val/test_mask"""
        return self._data

    def maybe_download(self):
        save_path = osp.join(self.data_root,"raw")
        for name in self.filenames:
            if not osp.exists(osp.join(save_path,name)):
                self.download_data(
                    "{}/ind.cora.{}".format(self.download_url,name),save_path
                )
        return 

    @staticmethod
    def download_data(url, save_path):
        """download tools, recalled when raw data not exists"""
        if not osp.exists(save_path):
            os.makedirs(save_path)
        data = urllib.request.urlopen(url)
        filename = osp.splitext(url)

        with open(osp.join(save_path, filename),"wb") as f:
            f.write(data.read())

        return True
    
    def process_data(self):
        """
        process data to get node_features, label, adjacency_matrix, 
                            train/val/test_dataset
        """
        print("Process data ...")
        _, tx, allx, y, ty, ally, graph, test_index = [self.read_data(
            osp.join(self.data_root,"raw",name)) for name in self.filenames]
        train_index = np.arange(y.shape[0])
        val_index = np.arange(y.shape[0],y.shape[0]+500)
        sorted_test_index = sorted(test_index)

        x = np.concatenate((allx, tx), axis=0)
        y = np.concatenate((ally, ty), axis=0)

        x[test_index] = x[sorted_test_index]
        y[test_index] = y[sorted_test_index]
        num_nodes = x.shape[0]

        train_mask = np.zeros(num_nodes,dtype=np.bool_)
        val_mask = np.zeros(num_nodes,dtype=np.bool_)
        test_mask = np.zeros(num_nodes,dtype=np.bool_)
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True
        adjacency = self.build_adjacency(graph)
        print("node's feature shape: ", x.shape)
        print("node's label shape: ", y.shape)
        print("adjacency's shape: ", adjacency.shape)
        print("number of training nodes: ", train_mask.sum())
        print("number of validation nodes: ", val_mask.sum())
        print("number of test nodes: ", test_mask.sum())

        return Data(x=x,y=y,adjacency=adjacency,
                    train_mask=train_mask,val_mask=val_mask,test_mask=test_mask)
    
    @staticmethod
    def build_adjacency(adj_dict):
        """create adjacency_matrix by the adjacency form"""
        edge_index = []
        num_nodes = len(adj_dict)
        for src,dst in adj_dict.items():
            edge_index.extend([src,v] for v in dst)
            edge_index.extend([v,src] for v in dst)
        edge_index = list(k for k,_ in itertools.groupby(sorted(edge_index)))
        edge_index = np.asarray(edge_index)
        adjacency = sp.coo_matrix((np.ones(len(edge_index)),
                                   (edge_index[:,0],edge_index[:,1])),
                                   shape=(num_nodes,num_nodes),dtype="float 32")
        return adjacency
    
    @staticmethod
    def read_data(path):
        """read data in different methods for further process"""
        name = osp.basename(path)
        if name == "ind.cora.test.index":
            out = np.genfromtxt(path,dtype="int64")
            return out
        else:
            out = pickle.load(open(path,"rb"), encoding="latin1")
            out = out.toarray() if hasattr(out, "toarray") else out
            return out

    






