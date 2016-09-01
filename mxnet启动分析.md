##编译测试环境##
用docker搭建了三个虚拟机.

##更改的地方：##
1.	文档的命令有错  --sync-dir 改为  --sync-dst-dir.
2.	我的docker用的是anacoda的python，所用在ssh_submit 中添加环境变量
export PATH="/root/anaconda2/bin:$PATH"。

##启动流程分析：##

		../../tools/launch.py -n 2 -H hosts --sync-dst-dir /tmp/mxnet python  train_mnist.py --kv-store dist_sync
以上是实例的启动命令。通过使用ssh 启动集群。
从lannch.py 跳转到 dmlc-core/tracker/dmlc_tracker/ssh.py

ssh.submit
1.	调用sync_dir将当前文件夹的文件全部拷贝到hosts文件指定的ip的/tmp/mxnet下。
2.	调用track.submit, track.submit 建立PSTracker实例，在PSTracker中给出环境变量

		env['DMLC_ROLE'] = 'scheduler'  // PS 中的角色
		env['DMLC_PS_ROOT_URI'] ＝str(self.hostIP)  //PS 中的ROOTIP地址
		env['DMLC_PS_ROOT_PORT'] = str(self.port)   //PS 中ROOT的端口

并在启动的主机上调用命令python  train_mnist.py --kv-store dist_sync。并得到以下环境变量：

		————————————————————————————————————－
		env {'DMLC_NUM_SERVER': '2', 'TERM': 'xterm', 'LESSCLOSE': '/usr/bin/lesspipe %s %s', 'SHLVL': '1',
		OLDPWD': '/project/mxnet/python', 'HOSTNAME': 'dc278d5190bd', 'LESSOPEN':       '| /usr/bin/lesspipe %s',
		'DMLC_ROLE': 'scheduler', 'PWD': '/project/mxnet/example/image-classification', 'DMLC_PS_ROOT_PORT': '9101',
		'PATH': '/root/bin:/root/anaconda2/bin:/      root/anaconda2/bin:/root/bin:/root/anaconda2/bin:/root/anaconda2/bin:/usr/local/sbin:
		/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin', 'DMLC_NUM_WORKER': '2', 'HOME': '/root', 'LS_COLORS':
		————————————————————————————————————－

3.	track.submit 完成本机的设置后，调用ssh.py 中 ssh_submit 函数启动server和worker，在不指定worker的情况下woker和server拥有相同多的数量。在这个例子中server的数量是2， 由－n 2指定。所以在ssh_submit 中将分别在另外的两台机器里分别起启动两个进程，使用以下命令：

		ssh -o StrictHostKeyChecking=no 172.17.0.5 -p 22 'export PATH="/root/anaconda2/bin:$PATH";  echo `env`;
		export DMLC_ROLE=server; export DMLC_PS_ROOT_PORT=9099; export DMLC_PS_ROOT_URI=172.17.0.3; export DMLC_NUM_SERVER=2;
		export DMLC_NUM_WORKER=2; cd /tmp/mxnet; python train_mnist.py --kv-store dist_sync'

		ssh -o StrictHostKeyChecking=no 172.17.0.4 -p 22 'export PATH="/root/anaconda2/bin:$PATH";  echo `env`;
		export DMLC_ROLE=server; export DMLC_PS_ROOT_PORT=9099; export DMLC_PS_ROOT_URI=172.17.0.3; export DMLC_NUM_SERVER=2;
		export DMLC_NUM_WORKER=2; cd /tmp/mxnet; python train_mnist.py --kv-store dist_sync'


		ssh -o StrictHostKeyChecking=no 172.17.0.5 -p 22 'export PATH="/root/anaconda2/bin:$PATH";  echo `env`;
		export DMLC_ROLE=worker; export DMLC_PS_ROOT_PORT=9099; export DMLC_PS_ROOT_URI=172.17.0.3; export DMLC_NUM_SERVER=2;
		export DMLC_NUM_WORKER=2; cd /tmp/mxnet; python train_mnist.py --kv-store dist_sync'


		ssh -o StrictHostKeyChecking=no 172.17.0.5 -p 22 'export PATH="/root/anaconda2/bin:$PATH";  echo `env`;
		export DMLC_ROLE=worker; export DMLC_PS_ROOT_PORT=9099; export DMLC_PS_ROOT_URI=172.17.0.3; export DMLC_NUM_SERVER=2;
		export DMLC_NUM_WORKER=2; cd /tmp/mxnet; python train_mnist.py --kv-store dist_sync'



##启动PS,计算任务##
自此已完成启动，控制权交给train_mnist.py.
train_mnist.py 获得以下参数：

		Namespace(batch_size=128, data_dir='mnist/', gpus=None, kv_store='dist_sync', load_epoch=None, lr=0.1, lr_factor=1,
		lr_factor_epoch=1, model_prefix=None, network='mlp', num_epochs=10, num_examples=60000, save_model_prefix=None)。


train_mnist.py 调用train_model.fit(args, net,get_iterator(data_shape))，args就是上面列出的参数，net就是我们的网络symbol。get_iterator是读取数据的函数，在这个函数里面有一个神奇的函数，我一直没有找到这个函数的定义mx.io.MNISTIter, 运行也不报错。

Train_model.fit 第一件事就是创建了kvstroe 调用 kvstroe.cc 中的
KVStore* KVStore::Create(const char *type_name)，然后调用kv = new kvstore::KVStoreDist();
KVStroeDist 在 mxnet/src/kvstore/kvstore_dist.h 定义， 里面会去调用IsWorkerNode()，会通过环境变量判断当前进程的角色就是前面得到的那些环境变量(worker, server)。

		171   static bool IsWorkerNode() {
		172 #if MXNET_USE_DIST_KVSTORE
		173     const char* role_str = ps::Environment::Get()->find("DMLC_ROLE");
		174     return (role_str == nullptr) || (!strcmp(role_str, "worker"));
		175 #else
		176     return true;
		177 #endif  // MXNET_USE_DIST_KVSTORE
		178   }



在 train_model.fit 的model中还会做一些变量的初始化。然后创建mxnet自己的model。

		79     model = mx.model.FeedForward(
		80         ctx                = devs,
		81         symbol             = network,
		82         num_epoch          = args.num_epochs,
		83         learning_rate      = args.lr,
		84         momentum           = 0.9,
		85         wd                 = 0.00001,
		86         initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34),
		87         **model_args)





		101     model.fit(
		102         X                  = train,
		103         eval_data          = val,
		104         eval_metric        = eval_metrics,
		105         kvstore            = kv,
		106         batch_end_callback = batch_end_callback,
		107         epoch_end_callback = checkpoint)

在model.fit 中用上面获取的kvstroe handle再创建了kvstore
		762         (kvstore, update_on_kvstore) = _create_kvstore(
		763             kvstore, len(self.ctx), self.arg_params)

这个地方没有完全看明白。在model.fit 中调用了 _train_multi_device这事真正做训练的地方。

		184     executor_manager = DataParallelExecutorManager(symbol=symbol,
		185                                                    sym_gen=sym_gen,
		186                                                    ctx=ctx,
		187                                                    train_data=train_data,
		188                                                    param_names=param_names,
		189                                                    arg_names=arg_names,
		190                                                    aux_names=aux_names,
		191                                                    work_load_list=work_load_list,
		192                                                    logger=logger

在得到executor_manager，就可以做，FB，BP

		229                 executor_manager.forward(is_train=True)
		230                 executor_manager.backward()


然后一些update_kvstore。

