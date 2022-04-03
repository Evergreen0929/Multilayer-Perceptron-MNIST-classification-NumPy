import os.path
import argparse
import numpy
import numpy as np
from dataloader import MNIST
import scipy
from scipy import special
import sys
from vis import vis, feature


def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("mode", choices=('train', 'inference', 'search'), default='inference',
                        help="Choose the mode to train, inference or search")

    parser.add_argument("--hidden_nodes", default=100, type=int,
                        help="Number of hidden nodes")

    parser.add_argument("--lr", default=0.01, type=float,
                        help="Learning rate")

    parser.add_argument("--lambda_w", default=0.01, type=float,
                        help="weight of L2-Normalization")

    parser.add_argument("--activate", choices=('sigmoid', 'tanh', 'relu', 'L_relu'), default='L_relu',
                        help="selection of activate functions")

    parser.add_argument("--vis_train", default=False, type=bool,
                        help="visualize train process")

    parser.add_argument("--vis_feature", default=False, type=bool,
                        help="visualize features")

    parser.add_argument("--show_result", default=False, type=bool,
                        help="show results directly")

    args = parser.parse_args()

    return args



class neuralNetwork:

    '''
    --------------------------------------初始化神经网络---------------------------------------
    参数列表：
    inputnodes:     输入层神经元数量
    hiddennodes:    隐含层神经元数量
    outputnodes:    输出层神经元数量
    learningrate:   学习率
    activate:       隐含层激活函数
    -----------------------------------------------------------------------------------------
    '''
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate, lambda_w, activate='sigmoid', Leaky_Relu_rate=0.01, final_required=True):

        self.inodes = inputnodes            # 输入层神经元节点
        self.hnodes = hiddennodes         # 隐含层神经元节点
        self.onodes = outputnodes           # 输出层神经元节点

        # 激活函数类型
        self.activate = activate

        # 全连接层权值矩阵
        # wxy代表x层和y层之间的全连接矩阵
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # 学习率
        self.lr = learningrate

        # 正则化系数
        self.lambda_w = lambda_w

        # leaky relu泄露率
        k = Leaky_Relu_rate

        self.final_required = final_required

        # 最后一层激活函数，指定sigmoid
        self.activation_function_full = lambda x: scipy.special.expit(x)

        # 激活函数选择，分别包括sigmoid，relu，tanh和leaky relu
        if activate == 'sigmoid':
            self.activation_function = lambda x: scipy.special.expit(x)
        elif activate == 'relu':
            self.activation_function = lambda x: np.maximum(0, x)
            self.d_activation_function = lambda x: np.where(x <= 0, 0, 1)
        elif activate == 'tanh':
            self.activation_function = lambda x: np.tanh(x)
        elif activate == 'L_relu':
            self.activation_function = lambda x: np.where(x <= 0, k * x, x)
            self.d_activation_function = lambda x: np.where(x <= 0, k, 1)


    '''训练神经网络'''
    def train(self, inputs_list, targets_list,epoch=0, max_epoch=20):
        lr = self.lr * (1-(epoch/max_epoch))**0.9
        # 初始化一个batch内权值矩阵的优化量d
        d_who = np.zeros_like(self.who)
        d_wih = np.zeros_like(self.wih)
        for inputs, targets in zip(inputs_list, targets_list):
            inputs = numpy.array(inputs, ndmin=2).T
            targets = numpy.array(targets, ndmin=2).T

            # 输入层-隐含层1
            hidden_inputs = numpy.dot(self.wih, inputs)
            # 激活函数
            hidden_outputs = self.activation_function(hidden_inputs)

            # 隐含层2-输出层
            final_inputs = numpy.dot(self.who, hidden_outputs)
            # 激活函数
            if self.final_required == True:
                final_outputs = self.activation_function_full(final_inputs)
            else:
                final_outputs = final_inputs

            # 反向传播
            if self.final_required == True:
                output_errors = (targets - final_outputs) * final_outputs * (1.0 - final_outputs) \
                                + self.lambda_w * (np.sum(np.square(self.wih)) / (self.wih.shape[0] * self.wih.shape[1])
                                                   + np.sum(np.square(self.who) / (self.who.shape[0] * self.who.shape[1])))
            else:
                output_errors = (targets - final_outputs) \
                                + self.lambda_w * (np.sum(np.square(self.wih)) / (self.wih.shape[0] * self.wih.shape[1])
                                                   + np.sum(np.square(self.who) / (self.who.shape[0] * self.who.shape[1])))

            if self.activate == 'sigmoid':
                hidden_errors = numpy.dot(self.who.T, output_errors) * hidden_outputs * (1.0 - hidden_outputs)
            elif self.activate == 'relu':
                hidden_errors = numpy.dot(self.who.T, output_errors) * self.d_activation_function(hidden_outputs)
            elif self.activate == 'tanh':
                hidden_errors = numpy.dot(self.who.T, output_errors) * (1.0 - hidden_outputs**2)
            elif self.activate == 'L_relu':
                hidden_errors = numpy.dot(self.who.T, output_errors) * self.d_activation_function(hidden_outputs)
            else:
                print("Activate function {} is not included".format(self.activate))
                raise NotImplementedError

            # batch内权重累计
            d_who += numpy.dot((output_errors), numpy.transpose(hidden_outputs))
            d_wih += numpy.dot((hidden_errors), numpy.transpose(inputs))

        # 权重更新
        self.who += lr * d_who
        self.wih += lr * d_wih

        pass

    '''模型推理: 一次前向传播'''
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        if self.final_required == True:
            final_outputs = self.activation_function_full(final_inputs)
        else:
            final_outputs = final_inputs

        return final_outputs

    '''模型测试'''
    def test(self, data_x, data_y, model, mode='test'):
        scorecard = []
        loss = 0.
        for i in range(data_x.shape[0]):
            inputs = data_x[i].reshape([self.inodes])
            outputs = model.query(inputs)
            label = numpy.argmax(outputs)
            loss += np.mean((numpy.argmax(data_y[i]) - label) ** 2)
            if (label == numpy.argmax(data_y[i])).all():
                scorecard.append(1)
            else:
                scorecard.append(0)
        scorecard_array = numpy.asarray(scorecard)
        print("{} performance = {:.5f}".format(mode, scorecard_array.sum() / scorecard_array.size))
        return scorecard_array.sum() / scorecard_array.size, loss / data_x.shape[0]

    '''模型保存为npy格式'''
    def save(self):
        if not os.path.exists('./save_model/model_{}_{}_{}'.format(self.inodes, self.hnodes,  self.onodes)):
            os.makedirs('./save_model/model_{}_{}_{}'.format(self.inodes, self.hnodes, self.onodes))
        np.save('./save_model/model_{}_{}_{}/weight_i_h.npy'.format(self.inodes, self.hnodes, self.onodes),
                np.array(self.wih, dtype=float))
        np.save('./save_model/model_{}_{}_{}/weight_h_o.npy'.format(self.inodes, self.hnodes, self.onodes),
                np.array(self.who, dtype=float))

    '''载入模型'''
    def load(self):
        self.wih = np.load('./save_model/model_{}_{}_{}/weight_i_h.npy'.format(self.inodes, self.hnodes, self.onodes))
        self.who = np.load('./save_model/model_{}_{}_{}/weight_h_o.npy'.format(self.inodes, self.hnodes, self.onodes))


'''实例化神经网络，用于数字分类'''
def mlp_fd(hidden_nodes, learning_rate, lambda_w, epochs=10, batchsize=64, pretrain=False, mode='train', activate='L_relu', vis_choice=False, feature_choice=False, show=False):
    data = MNIST(dimension=3)
    train_x = data.train_x_set
    train_y = data.train_y_set
    val_x = data.val_x_set
    val_y = data.val_y_set
    test_x = data.test_x_set
    test_y = data.test_y_set

    n = neuralNetwork(784, hidden_nodes, 10, learning_rate, lambda_w, activate=activate, Leaky_Relu_rate=0.05)

    '''load pretrained weights'''
    if pretrain == True:
        n.load()

    if mode == 'train':
        best_acc = 0.
        train_acc_set = []
        val_acc_set = []
        train_loss_set = []
        val_loss_set = []
        for e in range(epochs):
            train_x_batch = []
            train_y_batch = []
            print('\n')
            for i in range(train_x.shape[0]):

                r = int(i / train_x.shape[0] * 100)
                print("\r", end="")
                #print("Epoch[{}]: {}%: ".format(e+1, r), "▋" * (r // 2), end="")
                print("Epoch[{}]: {}%: ".format(e + 1, r), "|" * (r // 2), end="")
                sys.stdout.flush()

                if i % batchsize == 0:
                    train_x_batch = []
                    train_y_batch = []
                train_x_batch.append(train_x[i].reshape([784]))
                train_y_batch.append(train_y[i])
                if len(train_x_batch) == batchsize:
                    n.train(train_x_batch, train_y_batch, epoch=e, max_epoch=epochs + 10)
            print('\n')
            train_acc, train_loss = n.test(train_x, train_y, n, mode='train')
            acc, val_loss = n.test(val_x, val_y, n, mode='val')
            if acc > best_acc:
                best_acc = acc
                n.save()

            train_loss_set.append(train_loss)
            val_loss_set.append(val_loss)
            train_acc_set.append(train_acc)
            val_acc_set.append(acc)

        if vis_choice == True:
            vis(train_loss_set, val_loss_set, [hidden_nodes, learning_rate, lambda_w], type='loss', show=show)
            vis(train_acc_set, val_acc_set, [hidden_nodes, learning_rate, lambda_w], type='accuracy', show=show)

    print("\nTest the best model at last.")
    n.load()
    best = n.test(test_x, test_y, n, mode='test')

    if feature_choice == True:
        feature([n.wih, n.who], [hidden_nodes, learning_rate, lambda_w], smooth=True)

    return best

def mlp_fd_grid_search():
    best_acc_all = 0.
    best_layer = 100
    best_lr = 0.1
    best_w = 0.1
    lr_range = [0.01, 0.003, 0.001, 0.0003, 0.0001]
    w_range = [0.1, 0.03, 0.01, 0.003, 0]

    for layer in range(100, 1000, 100):
        for lr in lr_range:
            for lambda_w in w_range:
                best_acc = mlp_fd(layer, lr, lambda_w, epochs=15, batchsize=64, pretrain=False, mode='train', activate='L_relu')

                f = open('./logs.txt', 'a+', encoding='utf-8')
                f.write("hidden nodes: {}, learning rate: {:.6f},  L2-weight: {:.6f},  acc=[{:.5f}]\n".format(layer, lr, lambda_w, best_acc))
                f.close()

                if best_acc > best_acc_all:
                    best_acc_all = best_acc
                    best_layer = layer
                    best_lr = lr
                    best_w = lambda_w
                    print("\nNew best hyper-parameters: h-nodes: {}, lr: {}, w: {};  acc=[{:.5f}]\n".format(best_layer, best_lr, best_w, best_acc_all))
                else:
                    print("\nNo new best hyper-parameters: h-nodes: {}, lr: {}, w: {};   acc=[{:.5f}]\n".format(best_layer, best_lr, best_w, best_acc_all))


if __name__ == "__main__":
    np.random.seed(7)
    args = parse_args()
    if args.mode == 'train' or args.mode == 'inference':
        mlp_fd(args.hidden_nodes, args.lr, args.lambda_w, epochs=15, batchsize=64, pretrain=False, mode=args.mode,
               activate=args.activate, vis_choice=args.vis_train, feature_choice=args.vis_feature, show=args.show_result)

    elif args.mode == 'search':
        mlp_fd_grid_search()

    else:
        raise NotImplementedError

