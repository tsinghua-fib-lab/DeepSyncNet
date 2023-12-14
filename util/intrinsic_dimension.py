import skdim
import numpy as np


class ID_Estimator:
    def __init__(self, method='MLE'):
        self.all_methods = ['MiND_ML', 'MLE', 'MADA', 'MOM', 'TLE']
        self.set_method(method)
    
    def set_method(self, method='MLE'):
        if method not in self.all_methods:
            assert False, 'Unknown method!'
        else:
            self.method = method
            
    def fit(self, X, k_list=20):

        if np.isscalar(k_list):
            k_list = np.array([k_list])
        else:
            k_list = np.array(k_list)

        dims = []
        for k in k_list:
            assert k>0, "k must be larger than 0"
            if self.method == 'MiND_ML':
                dims.append(np.mean(skdim.id.MiND_ML().fit_pw(X, n_neighbors=k).dimension_pw_))
            elif self.method == 'MLE':
                dims.append(np.mean(skdim.id.MLE().fit(X, n_neighbors=k).dimension_pw_))
            elif self.method == 'MADA':
                dims.append(np.mean(skdim.id.MADA().fit(X, n_neighbors=k).dimension_pw_))
            elif self.method == 'MOM':
                dims.append(np.mean(skdim.id.MOM().fit(X, n_neighbors=k).dimension_pw_))
            elif self.method == 'TLE':
                dims.append(np.mean(skdim.id.TLE().fit(X, n_neighbors=k).dimension_pw_))
            else:
                assert False, f"{self.method} not implemented!"
        if len(dims) == 1:
            return dims[0]
        else:
            return np.array(dims)


def eval_id_embedding(vars_filepath, method='MLE', is_print=False, max_point=1000, k_list=20, embedding=None):
    
    if embedding is None:
        embedding = np.load(vars_filepath+'/embedding.npy', allow_pickle=True)
    
    if len(embedding) > max_point:
        embedding = embedding[np.random.choice(len(embedding), max_point, replace=False)]

    embedding = np.unique(embedding, axis=0)
    
    estimator = ID_Estimator(method=method)
    dims = estimator.fit(embedding, k_list)

    if is_print:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.plot(k_list, dims, 'o-')
        plt.xlabel('k')
        plt.ylabel('ID')
        plt.title(f'ID Estimation ({method})')
        import os; os.makedirs(vars_filepath+f'/n{max_point}--k{k_list[0]}_{k_list[-1]}', exist_ok=True)
        plt.tight_layout(); plt.savefig(vars_filepath+f'/n{max_point}--k{k_list[0]}_{k_list[-1]}/{method}.png')

    return dims
    

def eval_id_data(data, method='MLE', max_point=5000, k_list=20):
    
    if len(data) > max_point: 
        data = data[np.random.choice(len(data), max_point, replace=False)]

    data = np.unique(data, axis=0)
    
    estimator = ID_Estimator(method=method)
    dims = estimator.fit(data, k_list)

    return np.mean(dims)