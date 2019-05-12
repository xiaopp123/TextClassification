import os

class ProjectPath(object):
    def __init__(self):
        self.data_dir = '../data'
        self.model_dir = '../model'
        self.result_dir = '../result'
        self.train_file = os.path.join(self.data_dir,"train_set.csv")
        self.test_file = os.path.join(self.data_dir,"test_set.csv")
        self.load_dir()

    def load_dir(self):
        if os.path.exists(self.data_dir)==False:
            os.makedirs(self.data_dir)
        if os.path.exists(self.model_dir)==False:
            os.makedirs(self.model_dir)
        if os.path.exists(self.result_dir)==False:
            os.makedirs(self.result_dir)


class FasttextParameter(ProjectPath):
    '''
    fasttext模型所需的要参数
    '''
    def __init__(self, column, fasttext_dim):
        ProjectPath.__init__(self)
        self.column = column
        self.fasttext_dim = fasttext_dim
        self.fasttext_train_file = os.path.join(self.data_dir, "fasttext_train.dat") 
        self.fasttext_val_file = os.path.join(self.data_dir, "fasttext_val.dat")
        self.fasttext_test_file = os.path.join(self.data_dir, "fasttext_test.dat")
        self.fasttext_dim = fasttext_dim
        self.prefix_label = "__label__"
        self.lr = 0.1
        self.epochs = 10
        self.num_outputs = 19
        self.bucket = 50000000
        self.thread = 56
        self.min_count = 3
        self.word_ngrams = 4
