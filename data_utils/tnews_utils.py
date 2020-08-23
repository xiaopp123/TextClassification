#encoding=utf-8
import os
import sys

def load_data():
    print("start load data...")
    pass

def deal_raw_to_fasttext_data(file_in, file_out):
    if not os.path.exists(file_in):
        print("{} not exist!".format(file_in))
        sys.exit(-1)
    if os.path.exists(file_out):
        print("{} exist".format(file_out))
        return
    data_list = []
    with open(file_in, "r") as fr:
        for line in fr:
            temp = eval(line)
            label = temp["label"]
            sentence = temp["sentence"]
            data_list.append([label, sentence])
    with open(file_out, "w") as fw:
        for line in data_list:
            fw.write("__label__{}\t{}\n".format(line[0], line[1]))
    print("deal raw data to fasttext data finish!")
    
if __name__ == "__main__":
    file_in = sys.argv[1]
    file_out = sys.argv[2]
    deal_raw_to_fasttext_data(file_in, file_out)