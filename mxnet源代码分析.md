# mxnet源代码分析
整个分析的过程以example/image/image-classification/train_mnist.py 的分布式执行过程为线索。
启动命令：
'../../tools/launch.py -n 2 -H hosts --sync-dst-dir /tmp/mxnet python  train_mnist.py --kv-store dist_sync'
关于启动过程分析见**mxnet启动分析**.

## symbol的建立过程
train_mnist.py中使用符号式编程对网络进行了定义，函数get_mlp和get_lenet分别定义了两个网络一个是普通的多层网络，一个是带卷积的网络。以下是get_lenet的代码：

'''python
def get_lenet(add_stn=false):
	"""
	lecun, yann, leon bottou, yoshua bengio, and patrick
	haffner. "gradient-based learning applied to document recognition."
	proceedings of the ieee (1998)
	"""
	data = mx.symbol.variable('data')
	if(add_stn):
	    data = mx.sym.spatialtransformer(data=data, loc=get_loc(data), target_shape = (28,28),
	                                     transform_type="affine", sampler_type="bilinear")
	# first conv
	conv1 = mx.symbol.convolution(data=data, kernel=(5,5), num_filter=20)
	tanh1 = mx.symbol.activation(data=conv1, act_type="tanh")
	pool1 = mx.symbol.pooling(data=tanh1, pool_type="max",
	                          kernel=(2,2), stride=(2,2))
	# second conv
	conv2 = mx.symbol.convolution(data=pool1, kernel=(5,5), num_filter=50)
	tanh2 = mx.symbol.activation(data=conv2, act_type="tanh")
	pool2 = mx.symbol.pooling(data=tanh2, pool_type="max",
	                          kernel=(2,2), stride=(2,2))
	# first fullc
	flatten = mx.symbol.flatten(data=pool2)
	fc1 = mx.symbol.fullyconnected(data=flatten, num_hidden=500)
	tanh3 = mx.symbol.activation(data=fc1, act_type="tanh")
	# second fullc
	fc2 = mx.symbol.fullyconnected(data=tanh3, num_hidden=10)
	# loss
	lenet = mx.symbol.softmaxoutput(data=fc2, name='softmax')
	return lenet
'''
