import os
import random
import pickle
import tensorflow as tf
from read_utils import TextConverter
from model import Model,Config
from read_utils import get_excel_QAs, get_QAs_text

def main(_):

    model_path = os.path.join('models', Config.file_name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)


    input_file = 'data/去除2和null.xlsx'
    vocab_file = os.path.join(model_path, 'vocab_label.pkl')

    # 获取原始excel数据
    QAs = get_excel_QAs(input_file)  # 要求excel文件格式，第一个表，第一列id，第二列query,第三列response

    # 分配训练和验证数据集
    thres = int(0.8*len(QAs))
    train_QAs = QAs[:thres]
    val_QAs = QAs[thres:]


    # 数据处理
    text = get_QAs_text(train_QAs)
    converter = TextConverter(text, vocab_file, max_vocab=Config.vocab_max_size, seq_length=Config.seq_length)
    print('vocab size:',converter.vocab_size)


    # 产生训练样本
    train_QA_arrs = converter.QAs_to_arr(train_QAs)
    train_g = converter.batch_generator(train_QA_arrs, Config.batch_size)

    # 产生验证样本
    val_QA_arrs = converter.QAs_to_arr(val_QAs)
    val_g = converter.val_samples_generator(val_QA_arrs, Config.batch_size)

    # 加载上一次保存的模型
    model = Model(Config,converter.vocab_size)
    checkpoint_path = tf.train.latest_checkpoint(model_path)
    if checkpoint_path:
        model.load(checkpoint_path)

    print('start to training...')
    model.train(train_g, model_path, val_g)



if __name__ == '__main__':
    tf.app.run()
