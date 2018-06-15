
import tensorflow as tf
from read_utils import TextConverter, batch_generator
from model import DualBiLSTM
import os
import codecs

# FLAGS = tf.flags.FLAGS
#
# tf.flags.DEFINE_string('name', 'default', 'name of the model')
# tf.flags.DEFINE_integer('num_seqs', 100, 'number of seqs in one batch')
# tf.flags.DEFINE_integer('num_steps', 100, 'length of one seq')
# tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
# tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
# tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
# tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
# tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
# tf.flags.DEFINE_float('train_keep_prob', 0.5, 'dropout rate during training')
# tf.flags.DEFINE_string('input_file', '', 'utf8 encoded text file')
# tf.flags.DEFINE_integer('max_steps', 100000, 'max steps to train')
# tf.flags.DEFINE_integer('save_every_n', 1000, 'save the model every n steps')
# tf.flags.DEFINE_integer('log_every_n', 10, 'log to the screen every n steps')
# tf.flags.DEFINE_integer('max_vocab', 3500, 'max char number')


import argparse # 用于分析输入的超参数

## 英文诗（莎士比亚）
# args_in = '--input_file data/shakespeare.txt ' \
#           '--name shakespeare ' \
#           '--num_steps 50 ' \
#           '--num_seqs 32 ' \
#           '--learning_rate 0.01 ' \
#           '--max_steps 20000'.split()

# ## 中文诗词
# args_in = '--input_file data/poetry.txt ' \
#           '--name poetry ' \
#           '--num_steps 26 ' \
#           '--num_seqs 32 ' \
#           '--learning_rate 0.01 ' \
#           '--max_steps 20000'.split()

## thoth 问答
args_in = '--input_file data/去除2和null.xlsx ' \
          '--name thoth ' \
          '--num_steps 26 ' \
          '--num_seqs 32 ' \
          '--learning_rate 0.001 ' \
          '--sheetname Sheet1 ' \
          '--max_steps 20000'.split()
# ## 小黄鸡问答
# args_in = '--input_file data/xiaohuangji50w_fenciA.conv ' \
#           '--name xhj ' \
#           '--num_steps 26 ' \
#           '--num_seqs 32 ' \
#           '--learning_rate 0.001 ' \
#           '--max_steps 20000'.split()

def parseArgs(args):
    """
    Parse 超参数
    Args:
        args (list<stir>): List of arguments.
    """

    parser = argparse.ArgumentParser()
    test_args = parser.add_argument_group('test超参数')
    test_args.add_argument('--name', type=str, default='default',help='name of the model')
    test_args.add_argument('--num_seqs', type=int, default=100,help='number of seqs in one batch')
    test_args.add_argument('--num_steps', type=int, default=100,help='length of one seq')
    test_args.add_argument('--lstm_size', type=int, default=128,help='size of hidden state of lstm')
    test_args.add_argument('--num_layers', type=int, default=2,help='number of lstm layers')
    test_args.add_argument('--use_embedding', type=bool, default=False,help='whether to use embedding')
    test_args.add_argument('--embedding_size', type=int, default=128,help='size of embedding')
    test_args.add_argument('--learning_rate', type=float, default=0.001,help='learning_rate')
    test_args.add_argument('--train_keep_prob', type=float, default=0.8,help='dropout rate during training')
    test_args.add_argument('--input_file', type=str, default='',help='utf8 encoded text file')
    test_args.add_argument('--max_steps', type=int, default=100000,help='max steps to train')
    test_args.add_argument('--save_every_n', type=int, default=1000,help='save the model every n steps')
    test_args.add_argument('--log_every_n', type=int, default=10,help='log to the screen every n steps')
    test_args.add_argument('--max_vocab', type=int, default=8000,help='max char number')
    test_args.add_argument('--sheetname', type=str, default='default',help='name of the model')
    return parser.parse_args(args)

FLAGS = parseArgs(args_in)


from read_utils import get_excel_QAs, get_QAs_text

def main(_):
    model_path = os.path.join('models', FLAGS.name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)

    # excel data
    QAs = get_excel_QAs(FLAGS.input_file)  # 要求excel文件格式，第一个表，第一列id，第二列query,第三列response

    # # xhj data
    # from read_utils import loadConversations
    # QAs = loadConversations(FLAGS.input_file)

    text = get_QAs_text(QAs)

    if os.path.exists(os.path.join(model_path, 'converter.pkl')) is False:
        print('词库文件不存在,创建...')
        converter = TextConverter(text, FLAGS.max_vocab)
        converter.save_to_file(os.path.join(model_path, 'converter.pkl'))
    else:
        converter = TextConverter(filename=os.path.join(model_path, 'converter.pkl'))

    QA_arrs = converter.QAs_to_arrs(QAs, FLAGS.num_steps)
    samples = converter.samples_for_train(QA_arrs)
    g = batch_generator(samples, FLAGS.num_seqs)

    print(converter.vocab_size)
    model = DualBiLSTM(converter.vocab_size,
                     batch_size=FLAGS.num_seqs,
                     num_steps=FLAGS.num_steps,
                     lstm_size=FLAGS.lstm_size,
                     num_layers=FLAGS.num_layers,
                     learning_rate=FLAGS.learning_rate,
                     train_keep_prob=FLAGS.train_keep_prob,
                     use_embedding=FLAGS.use_embedding,
                     embedding_size=FLAGS.embedding_size
                     )
    model.train(g,
                FLAGS.max_steps,
                model_path,
                FLAGS.save_every_n,
                FLAGS.log_every_n,
                )


if __name__ == '__main__':
    tf.app.run()
