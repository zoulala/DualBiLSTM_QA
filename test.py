# ref:https://github.com/hzy46/Char-RNN-TensorFlow
import tensorflow as tf
from read_utils import TextConverter,batch_generator
from model import DualBiLSTM
import os

# FLAGS = tf.flags.FLAGS
#
# tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
# tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
# tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
# tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
# tf.flags.DEFINE_string('converter_path', '', 'model/name/converter.pkl')
# tf.flags.DEFINE_string('checkpoint_path', '', 'checkpoint path')
# tf.flags.DEFINE_string('start_string', '', 'use this string to start generating')
# tf.flags.DEFINE_integer('max_length', 30, 'max length to generate')
#




import argparse # 用于分析输入的超参数


## thoth
args_in = '--converter_path models/thoth/converter.pkl ' \
          '--input_file data/去除2和null.xlsx ' \
          '--sheetname Sheet1 ' \
          '--checkpoint_path models/thoth/ ' \
          '--num_steps 26 ' \
          '--num_seqs 32 ' \
          '--max_length 300'.split()

# ## 小黄鸡问答
# args_in = '--converter_path models/xhj/converter.pkl ' \
#           '--input_file data/xiaohuangji50w_fenciA.conv ' \
#           '--sheetname Sheet1 ' \
#           '--checkpoint_path models/xhj/ ' \
#           '--num_steps 26 ' \
#           '--num_seqs 32 ' \
#           '--max_length 300'.split()

def parseArgs(args):
    """
    Parse 超参数
    Args:
        args (list<stir>): List of arguments.
    """

    parser = argparse.ArgumentParser()
    test_args = parser.add_argument_group('test超参数')

    test_args.add_argument('--lstm_size',type=int, default=128, help='size of hidden state of lstm')
    test_args.add_argument('--num_layers', type=int, default=2, help='number of lstm layers')
    test_args.add_argument('--use_embedding', type=bool, default=False, help='whether to use embedding')
    test_args.add_argument('--embedding_size', type=int, default=128, help='size of embedding')
    test_args.add_argument('--converter_path',type=str, default='', help='model/name/converter.pkl')
    test_args.add_argument('--checkpoint_path', type=str, default='', help='checkpoint path')
    test_args.add_argument('--start_string', type=str, default='', help='use this string to start generating')
    test_args.add_argument('--max_length', type=int, default=30, help='max length to generate')
    test_args.add_argument('--sheetname', type=str, default='default',help='name of the model')
    test_args.add_argument('--input_file', type=str, default='',help='utf8 encoded text file')
    test_args.add_argument('--max_vocab', type=int, default=3500,help='max char number')
    test_args.add_argument('--num_seqs', type=int, default=100,help='number of seqs in one batch')
    test_args.add_argument('--num_steps', type=int, default=100,help='length of one seq')

    return parser.parse_args(args)

FLAGS = parseArgs(args_in)


from read_utils import get_excel_QAs, get_QAs_text,get_excel_libs
def main(_):
    # FLAGS.start_string = FLAGS.start_string#.decode('utf-8')
    converter = TextConverter(filename=FLAGS.converter_path)
    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path =\
            tf.train.latest_checkpoint(FLAGS.checkpoint_path)

    model = DualBiLSTM(converter.vocab_size, test=True,
                    batch_size=FLAGS.num_seqs,
                    num_steps=FLAGS.num_steps,
                    lstm_size=FLAGS.lstm_size, num_layers=FLAGS.num_layers,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size)

    model.load(FLAGS.checkpoint_path)


    # # ----------test3-------------
    # QAs = get_excel_QAs(FLAGS.input_file)  # 要求excel文件格式，第一个表，第一列id，第二列query,第三列response
    # QA_arrs = converter.QAs_to_arrs(QAs, FLAGS.num_steps)
    # all_samples = converter.samples_for_test3(QA_arrs)
    # indexs = model.test3(all_samples)
    # converter.index_to_QA_and_save(indexs,QAs,FLAGS.checkpoint_path)

    # ----------test4-------------
    libs = get_excel_libs('data/tianlong_libs.xlsx')
    libs_arrs = converter.libs_to_arrs(libs,FLAGS.num_steps)
    response_matul_state = model.test4_matul(libs_arrs)
    QAs = get_excel_QAs(FLAGS.input_file)  # 要求excel文件格式，第一个表，第一列id，第二列query,第三列response
    QAY = []
    k,n = 0,0
    for query,y_response in QAs:
        input_arr,input_len = converter.Q_to_arr(query,FLAGS.num_steps)
        indexs = model.test4(input_arr,input_len, response_matul_state)
        responses = converter.index_to_response(indexs, libs)
        # print('response:',responses)
        QAY.append((query, y_response, responses))
        if responses[0]==y_response:
            k += 1
            print(k)
        n += 1
    print('accuracy:',k/float(n))
    converter.save_to_excel(QAY, FLAGS.checkpoint_path)


    # ## ----------test5------------
    # from read_utils import loadConversations
    # QAs = loadConversations(FLAGS.input_file)
    # QA_arrs = converter.QAs_to_arrs(QAs, FLAGS.num_steps)
    # QA_arrs = converter.samples_for_test3(QA_arrs[:2000])  # 有454000个样本，花费时间很长
    # response_matul_state = model.test5_matul(QA_arrs)  # 有454000个样本，花费时间很长
    # while True:
    #     query = input('query:')
    #     input_arr,input_len = converter.Q_to_arr(query,FLAGS.num_steps)
    #     indexs = model.test4(input_arr,input_len, response_matul_state)
    #     responses = converter.index_to_response2(indexs, QAs)
    #     print('response:',responses)


if __name__ == '__main__':
    tf.app.run()
