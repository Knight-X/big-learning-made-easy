参照以下SFB的原理图，阅读poseidon的代码：

![](./images/SFBcaster.jpg)


图中Output SV Queue对应的是Context类中的local_sv_queues_

图中Input SV Queue对应的是Context类中的remote_sv_queues_

在Context::InitSVB中初始化local_sv_queues_和remote_sv_queues_，为每个layer创建一个caffe::SufficientVectorQueue


SVBWorker::Init中建立p2p连接，

	// client i connects to client [0, ..., i - 1]
	for (int i = 0; i < client_id_; ++i) {
	    int conn_msg = client_id_;
	    auto &other_host_info = host_map_[i];
	    int other_port = std::stoi(other_host_info.port, 0) + 1000;
	    comm_bus_->ConnectTo(
	        i, other_host_info.ip + ":" + std::to_string(other_port),
	        &conn_msg, sizeof(conn_msg));
	}

CommBus是本地线程和远程线程的共享总线，底层使用zmq

线程函数SVBWorker::Start，对应图中的Comunication Thread，其中Send()从local_sv_queues_ Get sv， 并发送到所有host；Receive()接收sv 并Add到remote_sv_queues_中

	while(!util::Context::svb_completed()) {
		Send();
		Receive();
	}  


线程函数Solver<Dtype>::ThreadSyncWithSVB，对应图中的Parameter Update Thread，用于在forward，backward调用之后，异步地更新参数

ThreadSyncWithSVB中，把某个layer的local_sv_queue和remote_sv_queue合并，然后调用ComputeGradientFromSV重构Weight并更新参数

图中SV Computing Thread 对应的自然是执行Solver<Dtype>::ForwardBackward的线程

在Context::InitSVB中初始化local_sv_queues_和remote_sv_queues_时，local_sv_queues_中SufficientVectorQueue允许读取num_app_thread + 1次，remote_sv_queues_允许读取num_app_thread次，读满这么多次才会pop


下面是关于卷积层和全连接层的性质：

One weird trick for parallelizing convolutional neural networks 文章指出

卷积层：包含90%~95%的计算，却只有大约5%的参数

全连接层：包含5-10%的计算，却有着95%的参数

所以poseidon只把SFB应用到更新全连接层的Weght参数

全连接层有Weight和bias两个参数，启动bias可能没有。看caffe中InnerProductLayer的LayerSetUp函数，它负责初始化参数

    if (bias_term_) {
      this->blobs_.resize(2);
      if (init_ps) {
        this->blob_global_id_.resize(2);
        (*layer_name_to_blob_global_idx)[name] = vector<int>(2);
      }  
    } else {
      this->blobs_.resize(1);
      if (init_ps) {
        this->blob_global_id_.resize(1);
        (*layer_name_to_blob_global_idx)[name] = vector<int>(1);
      }  
    }

所以在Solver<Dtype>::ForwardBackward中只有weights of a inner product layer走的是SFB

	if (util::Context::use_svb()
	  && type == LayerParameter_LayerType_INNER_PRODUCT
	  && j == 0) { // weights of a inner product layer
	sync_thread = new std::thread(&Solver::ThreadSyncWithSVB, this, 
	    net_->params()[param_id], param_id, layers[i], i, top_vecs[i],
	    bottom_vecs[i]);
	} else {
	sync_thread = new std::thread(&Solver::ThreadSyncWithPS, this, 
	    net_->params()[param_id], param_id, param_owner,
	    clock_counter_ - param_table_staleness_);
	}