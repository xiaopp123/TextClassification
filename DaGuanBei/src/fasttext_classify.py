import fasttext
from sklearn.metrics import f1_score
import os
import pandas as pd
from sklearn.metrics import f1_score
from Parameter import FasttextParameter
from utils import self_sample_by_class
from utils import transform_fasttext
from utils import logging
from utils import write_data

class FasttextClassify(FasttextParameter):
    def __init__(self, column, fasttext_dim):
        FasttextParameter.__init__(self, column, fasttext_dim)
        self.model_file = os.path.join(self.model_dir, \
            "{}_{}d.fasttext.bin".format(self.column, self.fasttext_dim))
        self.pretrained_vectors = os.path.join(self.model_dir, \
            "fasttext_{}_{}d.vec".format(self.column, self.fasttext_dim))
        self.model = None
        self.best_score = None
        self.num_outputs = 19

    def load_data(self):
        logging.info("加载数据与模型")
        self.train_set, self.val_set = self_sample_by_class(self.train_file, ratio = 0.85)
        print(self.train_set.shape)
        print(self.val_set.shape)
        transform_fasttext(self.train_set, self.fasttext_train_file, self.column)
        transform_fasttext(self.val_set, self.fasttext_val_file, self.column)

    def train_model(self):
        self.load_data()
        classify_model = None
        if os.path.exists(self.model_file):
            logging.info("fasttext分类模型")
            classify_model = fasttext.load_model(self.model_file, label_prefix=self.prefix_label)
        else:
            logging.info("fasttext分类模型不存在，开始重新训练...")
            if os.path.exists(self.pretrained_vectors):
                logging.info("存在预训练的词向量，从本地加载词向量进行训练...")
                classify_model = fasttext.supervised(self.fasttext_train_file, self.model_file[0 : -4], \
                                                     lr = 0.1, epoch = 100, dim = self.fasttext_dim, bucket = 50000000,\
                                                     loss = "softmax", thread = 56, min_count = 3, word_ngrams = 4,\
                                                     pretrained_vectors = self.pretrained_vectors, silent = False)
            else:
                logging.info("不存在预训练的词向量，重头开始训练...")
                classify_model = fasttext.supervised(self.fasttext_train_file, self.model_file[0:-4], lr=0.1,\
                                                     epoch = 100, dim = self.fasttext_dim, bucket = 50000000,\
                                                     loss = "softmax", thread = 56, min_count = 3, word_ngrams = 4,\
                                                     silent = False)

        self.model = classify_model
        self.best_score = self.evaluate()

    def predict(self):
        logging.info("对测试数据进行预测...")
        pred_x = pd.read_csv(self.test_file)[self.column].tolist()
        y_pred = self.model.predict(pred_x)
        probs_list = self.model.predict_proba(pred_x, self.num_outputs)
        y_probs, result_string, i = [], ['id,class\n'], 0
        logging.info("预测结果完毕，开始将预测结果和类别概率写入文件...")
        for label in y_pred:
            result_string.append("{},{}\n".format(i, int(label[0]) + 1))
            i += 1
        result_submit = os.path.join(self.result_dir, "fasttext_{}_{:.4f}.csv".format(self.column, self.best_score))
        write_data("".join(result_string), result_submit)
        logging.info("数据提交完毕，请查看：{}".format(self.result_dir))

    def evaluate(self):
        logging.info("开始在验证集合上测试准确率...")
        train_x, train_y = self.train_set[self.column].tolist(), (self.train_set['class'] - 1).tolist()
        val_x, val_y = self.val_set[self.column].tolist(), (self.val_set['class'] - 1).tolist()
        y_pred_train = [int(pred[0]) for pred in self.model.predict(train_x)]
        y_pred_eval = [int(pred[0]) for pred in self.model.predict(val_x)]
        f1 = f1_score(train_y, y_pred_train, average='macro')
        f2 = f1_score(val_y, y_pred_eval, average='macro')
        logging.info("模型在训练集f1: {:.4f}".format(f1))
        logging.info("模型在验证集f1: {:.4f}".format(f2))

        return f2

if __name__ == '__main__':
    fasttext_classify = FasttextClassify("word_seg", 20)
    fasttext_classify.train_model()
    fasttext_classify.predict()
