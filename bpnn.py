import random
import numpy as np
import pickle
from MyError import *

# 生成(a,b)间的一个随机数
def _rand(a, b):
    return (b - a) * random.random() + a

# sigmoid function
def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

# derivative of sigmoid function
def _dsigmoid(x):
    return _sigmoid(x) * (1 - _sigmoid(x))

class Unit:
    """
    该类表示神经网络中的一个节点
    _weight - 各个输入的权重组成的列表
    _threshold - 该节点的阈值
    _step - 每一次迭代的变化量
    """
    def __init__(self, input_num):
        # 该节点的参数初始化
        self._weight = np.array([_rand(-1, 1) \
            for i in range(input_num)]) # 对应于input_num的权值
        self._threshold = _rand(-1, 1) # 该节点的阈值

        # 节点的变化量初始化
        # self._step = [0.0] * input_num

        self._input_num = input_num

    def _set_weight(self, weight):
        # 输入一组权值给节点
        weight = np.array(weight)
        self._weight = weight
    
    def _get_weight(self):
        # 获取节点的权值
        return self._weight

    def _set_threshold(self, threshold):
        # 输入一个阈值给节点
        self._threshold = threshold
    
    def _get_threshold(self):
        # 获取节点的阈值
        return self._threshold

    def _calculate_output(self, inputs):
        # 计算单个节点的output
        value_weight = sum([i * j for i, j in zip(inputs, self._weight)])
        output = _sigmoid(value_weight - self._threshold)
        return output

    def _update(self, delta_weight, delta_threshold):
        # delta_weight of input_num
        # delta_threshold of 1
        # 对单个节点进行update, 包括各权值及一个阈值
        self._weight = self._weight + delta_weight
        # self._threshold = self._threshold + delta_threshold

class Layer:
    """
    该类表示神经网络中的一层
    _input_num - 输入节点数
    _output_num - 输出节点数
    _unit_num - 实际使用节点数, 等于输出节点数 (因为update只是对输出节点做的)
    _inputs - 输入值列表
    _outputs - 输出值列表
    _units - 输出节点列表
    """
    def __init__(self, input_num, output_num):
        # 初始化_unit_num个Unit类
        # 每个Unit类有input_num个输入
        self._unit_num = output_num
        self._units = [Unit(input_num) for i in range(self._unit_num)]

        # 初始化该层的输入和输出
        self._input_num = input_num
        self._output_num = output_num
        self._inputs = np.array([0.0] * self._input_num)
        self._outputs = np.array([0.0] * self._output_num)

    def _set_inputs(self, inputs):
        # 输入一组输入给该层
        inputs = np.array(inputs)
        self._inputs = inputs
    
    def _get_inputs(self):
        # 取得该层的输入
        return self._inputs
    
    def _set_weights(self, weights):
        # 直接给Layer的weights赋值
        # shape = output * input
        weights = np.array(weights)
        if weights.shape == (self._output_num, self._input_num):
            pass
        else:
            raise weights_size_err()
        for i in range(self._unit_num):
            self._units[i]._set_weight(weights[i])

    def _get_weights(self):
        # 获取一层的weights
        # shape = output * input
        weights = np.empty([0, self._input_num])
        for i in range(self._unit_num):
            weight = self._units[i]._get_weight()
            weights = np.row_stack((weights, weight))
        return weights

    def _set_thresholds(self, thresholds):
        # 直接给Layer的threshold赋值
        # shape = output
        thresholds = np.array(thresholds)
        if len(thresholds) == self._output_num:
            pass
        else:
            raise thresholds_size_err()
        for i in range(self._unit_num):
            self._units[i]._set_threshold(thresholds[i])
    
    def _get_thresholds(self):
        # 获取一层的thresholds
        # shape = output
        thresholds = []
        for i in range(self._unit_num):
            threshold = self._units[i]._get_threshold()
            thresholds.append(threshold)
        thresholds = np.array(thresholds)
        return thresholds

    def _get_layer(self):
        thresholds = self._get_thresholds()
        weights = self._get_weights()
        return thresholds, weights

    def _calculate_outputs(self):
        # 根据参数及输入计算输出
        # 返回一个numpy.array
        for i in range(self._output_num):
            output = self._units[i]._calculate_output(self._inputs)
            self._outputs[i] = output
        return self._outputs

    def _update(self, delta_weight, delta_threshold):
        # delta_weight of output_num * input_num
        # delta_threshold of output_num
        # 对units中每个输出节点逐个进行update
        for i in range(self._unit_num):
            self._units[i]._update(delta_weight[i], delta_threshold[i])

class BPNN:
    """
    该类表示一个BP神经网络
    _input_num - 输入层节点数
    _hidden_num - 隐藏层节点数
    _output_num - 输出层节点数
    _hidden_layer - 输入层 --> 隐藏层
    _output_layer - 隐藏层 --> 输出层
    """
    def __init__(self, input_num, hidden_num, output_num, random_seed=3):
        # 初始化输入层, 隐藏层, 输出层
        self._input_num = input_num
        self._hidden_num = hidden_num
        self._output_num = output_num
        self._hidden_layer = Layer(self._input_num, self._hidden_num)
        self._output_layer = Layer(self._hidden_num, self._output_num)
        
        random.seed(random_seed)

    def _layer_load_data(self, layer, sample):
        """
        给layer传送sample数据
        计算outputs返回
        """
        layer._set_inputs(sample)
        return layer._calculate_outputs()

    # def _calculate_b(self):

    def _single_sample_epoch(self, sample, label, rate):
        """
        对于单个输入样本计算参数改变量
        相当于标准BP
        返回各项变化量
        """
        # 检查错误
        # for i in sample:
        #     # 输入非数值
        #     if isinstance(i, (int, float, np.int64)):
        #         pass
        #     else:
        #         raise data_type_err(i)
        # for i in label:
        #     # 输出非数值
        #     if isinstance(i, (int, float, np.int64)):
        #         pass
        #     else:
        #         raise data_type_err(i)

        # sample, label转化为np.ndarray
        sample = np.array(sample)
        label = np.array(label)

        # 各层输入, 并获取最终输出
        # shape = [hidden_num,]
        # shape = [output_num,]
        hidden_layer_outputs = \
            self._layer_load_data(self._hidden_layer, sample)
        output_layer_outputs = \
            self._layer_load_data(self._output_layer, hidden_layer_outputs)

        # 计算变化量
        
        # length = output_num
        # shape = [output_num,]
        g = output_layer_outputs * (1 - output_layer_outputs) \
            * (label - output_layer_outputs)
        
        # length = output_num * hidden_num
        # shape = [output_num, hidden_num]
        # each row - output node
        # each coloumn - hidden node
        delta_w = rate * np.dot(g.reshape(self._output_num, 1), \
            hidden_layer_outputs.reshape(1, self._hidden_num))

        # length = output_num
        # shape = [output_num,]
        delta_theta = - rate * g

        # length = hidden_num
        # shape = [hidden_num,]
        output_layer_weights = self._output_layer._get_weights()
        e = hidden_layer_outputs * (1 - hidden_layer_outputs) \
            * np.dot(g, output_layer_weights)

        # length = hidden_num * input_num
        # shape = [hidden_num, input_num]
        # each row - hidden node
        # each coloumn - input node
        delta_v = rate * np.dot(e.reshape(self._hidden_num, 1), \
            np.array(sample).reshape(1, self._input_num))
        
        # length = hidden_num
        # shape = [hidden_num,]
        delta_gamma = - rate * e

        # 均方误差
        E = 0.5 * sum((output_layer_outputs - label) ** 2)
        return delta_w, delta_theta, delta_v, delta_gamma, E

    def _single_sample_test(self, sample):
        """
        对于单个输入样本的测试
        返回Label
        """
        # sample, label转化为np.ndarray
        sample = np.array(sample)

        # 各层输入, 并获取最终输出
        # shape = [hidden_num,]
        # shape = [output_num,]
        hidden_layer_outputs = \
            self._layer_load_data(self._hidden_layer, sample)
        output_layer_outputs = \
            self._layer_load_data(self._output_layer, hidden_layer_outputs)

        return output_layer_outputs

    def _train_check(self, samples, labels):
        """
        各种训练前的输入检查
        """
        if len(labels.shape) == 1 and self._output_num == 1:
            pass
        elif len(samples.shape) == 1 and self._input_num == 1:
            pass
        elif samples.shape[1] != self._input_num or \
                labels.shape[1] != self._output_num:
            # 输入或输出的样本特征数不匹配BPNN初始化的值
            raise sample_size_err()
        elif samples.shape[0] != labels.shape[0]:
            # 输入的数据及label数目不等
            raise sample_label_err()
    
    def standard_train(self, samples, labels, rate=0.1, 
                            precision=0.1, epochs=100):
        """
        标准bpnn的训练
        目前退出条件是进行指定数目次epoch
        """
        # samples' shape = sample_num * input_num
        # labels' shape = sample_num * output_num
        # 0 < rate < 1.0
        
        # 检查错误
        samples = np.array(samples)
        labels = np.array(labels)
        self._train_check(samples, labels)
        sample_num = samples.shape[0]

        # start training - 标准BP
        count = 0
        while count < epochs:
            E = 0
            for i in range(sample_num):
                delta_w, delta_theta, delta_v, delta_gamma, e = \
                    self._single_sample_epoch(samples[i], labels[i], rate)
                E = E + e
                self._output_layer._update(delta_w, delta_theta)
                self._hidden_layer._update(delta_v, delta_gamma)

            if count % 100 == 0:
                print('Epoch: %d , Error %-.10f' % (count, E))
            count = count + 1  
        
        # end training  
        print("\nTotal epochs: " + str(count))
       
    def accumulate_train(self, samples, labels, rate=1, 
                            precision=0.1, epochs=100):
        """
        累积bpnn的训练
        """
        # samples' shape = sample_num * input_num
        # labels' shape = sample_num * output_num
        # 0 < rate < 1.0
        
        # 检查错误
        samples = np.array(samples)
        labels = np.array(labels)
        self._train_check(samples, labels)
        sample_num = samples.shape[0]

        # start training - 累积BP
        count = 0
        while count < epochs:
            E = 0 
            delta_ws, delta_thetas, delta_vs, delta_gammas = [], [], [], []
            for i in range(sample_num):
                delta_w, delta_theta, delta_v, delta_gamma, e = \
                    self._single_sample_epoch(samples[i], \
                                            labels[i], rate)
                E = E + e
                delta_ws.append(delta_w)
                delta_thetas.append(delta_theta)
                delta_vs.append(delta_v)
                delta_gammas.append(delta_gamma)
            delta_w = sum(delta_ws) / sample_num
            delta_theta = sum(delta_thetas) / sample_num
            delta_v = sum(delta_vs) / sample_num
            delta_gamma = sum(delta_gammas) / sample_num
            
            self._output_layer._update(delta_w, delta_theta)
            self._hidden_layer._update(delta_v, delta_gamma)
            if count % 100 == 0:
                print('Epoch: %d , Error %-.10f' % (count, E))
            count = count + 1
        
        # end training
        print("\nTotal epochs: " + str(count))
    
    def get_bpnn(self):
        """
        打印bpnn参数
        """
        hidden_layer = self._hidden_layer._get_layer()
        output_layer = self._output_layer._get_layer()
        print("Hidden Layer:")
        print("Thresholds:\n", hidden_layer[0], "\nWeights:\n", hidden_layer[1])
        print("Output Layer:")
        print("Thresholds:\n", output_layer[0], "\nWeights:\n", output_layer[1])

    def test(self, samples):
        """
        训练好的bpnn的测试
        """
        # samples' shape = sample_num * input_num
        samples = np.array(samples)
        if samples.shape[1] != self._input_num:
            # 输入或输出的样本特征数不匹配BPNN初始化的值
            raise sample_size_err()

        samples = np.array(samples)
        labels = np.empty([0,self._output_num])
        for i in range(samples.shape[0]):
            labels = np.row_stack((labels, \
                self._single_sample_test(samples[i])))
        return labels
    def save(self, filename):
        """
        保存bp神经网络
        """
        model = {
            "hidden_layer_weights": self._hidden_layer._get_weights(),
            "output_layer_weights": self._output_layer._get_weights(),
            "hidden_layer_thresholds": self._hidden_layer._get_thresholds(),
            "output_layer_thresholds": self._output_layer._get_thresholds()        
        }
        with open(filename, "wb") as f:
            pickle.dump(model, f)
        print("Model Saved!")

    def load(self, filename):
        """
        导入bp神经网络
        """
        with open(filename, "rb") as f:
            model = pickle.load(f)
            self._hidden_layer._set_weights(model["hidden_layer_weights"])
            self._hidden_layer._set_thresholds(model["hidden_layer_thresholds"])
            self._output_layer._set_weights(model["output_layer_weights"])
            self._output_layer._set_thresholds(model["output_layer_thresholds"])
        print("Model loaded!")

def demo():
    myBPNN = BPNN(2, 2, 1)
    data = [[1,0],[0,1],[0,1],[1,1]]
    label = [[0], [1], [1], [0]]
    # myBPNN.accumulate_train(samples=data, labels=label, rate=1, epochs=5000)
    # myBPNN.accumulate_train(samples=data, labels=label, rate=0.5, epochs=5000)
    myBPNN.standard_train(samples=data, labels=label, rate=1, epochs=1000)
    myBPNN.standard_train(samples=data, labels=label, rate=0.1, epochs=500)
    print("Training Finished!")

    myBPNN.get_bpnn()

    predict_label = myBPNN.test(data)
    for i in range(len(label)):
        print(label[i], '->', predict_label[i])

    myBPNN.save("myBPNN.m")

    test = BPNN(2, 2, 1)
    test.load("myBPNN.m")
    test.get_bpnn()

if __name__ == '__main__':
    demo()