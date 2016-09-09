# Bosen原理

## LazyTable基本架构
![](images/Architecture.png)

1. 存放parameters的table的rows分布在多个tablet servers上。
2. 执行一个App script后，PS会在每个Client都会运行一个App program (e.g., matrixfact.main())，每个App program可以生成多个app threads。App thread相当于MapReduce/Spark中的task。
3. App thread通过client library来访问相应的table servers获取所需的table中的rows。
4. Client library维护了一个多级cache和operation logs来减少与table server的交互。

## LazyTable数据模型与访问API

### Data model: Table[row(columns)]

由于ML中算法基本使用vector或者matrix，所以可以用Table来存储参数。

与二维表类似，一个Table（比如matrixfact中的`L_Table`）包含多个row，row一般是`denseRow`或者`sparseRow`，一个row包含多个column。具体的parameter存在table中的cell中。具体实现时，Table可以用`hashmap<rowid, row>`来实现。

由于Table中的paramters会被多个threads更新，所以row支持一些聚合操作，比如plus, multiply, union。

### LazyTable操作

因为要对Table进行读写更新操作，因此Table需要支持一些操作，LazyTable的操作接口借鉴了Piccolo的接口：

1. read(tableid, rowid, slack)

   读取row，如果local cache中存在该row且其slack满足staleness bound（也就是local cache中的参数足够新），那么从local cache读取该row，否则暂停读取线程（the calling thread waits）。这个API也是唯一可以block calling thread的API。

2. update(tableid, rowid, delta)

   更新table中row的参数，newParameter = oldParameter + delta，这三个都是vector。

3. refresh(tableid, rowid, slack)

   如果process cache（被多个app thread共享）中的table中的row已经old了，就更新之。

4. clock()

   调用后表示calling thread已经进入到下一个周期，因为SSP不存在固定的barrier，所以这个看似会synchronization的API并不会block calling thread。

### Data freshness and consistency guarantees

1. 数据新鲜度保证：

   每个row有一个data age field（也就是clock）用于表示该row的数据新鲜度。假设一个row的当前data age是t，那么表示该row里面的参数 contains all updates from all app threads for1, 2, ..., t.

   对于SSP来说，当calling thread在clock t的周期内发送`read(tableid, rowid, slack)`的请求时，如果相应row的`data age >= t-1-slack`，那么该row可以返回。

2. read-my-updates:

   ready-my-updates ensures that the data read by a thread contains all its own updates.

## LazyTable系统模块之Tablet Servers

### Tablet Servers基本功能

一个逻辑上的Table可以分布存放在不同的tablet server上，比如`L_Talbe`中的 i-th row 可以存在`tablet_server_id = i % total_num_of_servers`上。每个tablet server都将rows存放在内存中。

每个tablet server使用一个vector clock（也就是`vector<Clock>`）来keep track of rows的新鲜度。vector中第i个分量表示第i个row的clock，vector中最小的clock被定义为`global_clock_value`，比如`global_clock_value = t` 表示所有的app threads都已经完成了clock t周期的计算及参数更新。问题：每个tablet server只存储table中的一部分rows，一部分rows达到了clock t就能说所有的app threads都完成了clock t周期的计算？

### Table updates

由于tablet server会不断收到来自多个app thread的update请求，tablet server会先将update请求做一个本地cache（将update请求放到pending updates list中）。当且仅当收到client发送clock()请求时，tablet server才会集中处理将这些updates。这样可以保证row的新鲜度由vector clock唯一决定。

### Table read

当tablet server收到client端发来的read请求，会先查看`global_clock_value` （为什么不是该row的data age？），如果tablet server中的row新鲜度满足requested data age要求（`global_clock_value >= t-1-slack`），那么直接返回row给client。否则，将read request放到pending read list里面，并按照requested data age排序（从大到小？）。当`global_clock_value`递增到requested data age时，tablet server再将相应的row返回给client。除了返回row，tablet server还返回data age和requester clock。前者是`global_clock_value`，后者是client's clock（说明了which updates from this client have been applied to the row data，client可以利用这个信息来清除一些本地的oplogs）。

## LazyTable系统模块之Client library

Client library与app threads在同一个process，用于将LazyTable API的调用转成message发送到tablet server。Client library包含多层caches和operation logs。Client library会创建一个或多个background threads （简称为bg threds）来完成propagating updates和receiving rows的工作。

Client library由两层 caches/oplogs 组成：process cache/oplog和thread cache/oplog。Process cache/oplog被同在一个进程中的所有app thread和bg thread共享。Each thread cache/oplog is exclusively associated with one app thread.（实现好像不是这样的）。Thread cache的引入可以避免在process cache端有过多的锁同步，但是只能cache一些rows。

Client library也使用vector clock来track app thread的clock，第i个分量代表第i个app thread已经进入的clock周期。

### Client updates
App thread调用update(deltas)后，会先去访问对应的thread cache/oplog，如果cache中相应的row存在，那么`thread.cache.row += update.deltas`，同时会update写入到oplog中。不存在就直接存起来。当app thread调用clock()，那么在thread oplog中的updates都会被push到process oplog中，同时`process.cache.row += updates.deltas`。如果thread cache/oplog不存在，update会直接被push到process cache/oplog。

当一个client process中所有app threads都完成clock为 t 的计算周期，client library会使用一个bg thread（是head bg thread么？）向table server发送一个消息，这个消息包含clock t，process oplogs中clock为 t 的updates。这些process cache/oplogs中的updates会在发送该消息后一直保留，直到收到server返回的更新后的rows。

### Client read

在clock t周期内，如果一个app thread想要去读row r with a slack of s，那么client library会将这个请求翻译成`read row r with data age >= t-s-1`。接着，client library会先去thread cache中找对应的且满足条件的row，如果不存在就去process cache中找，如果还找不到就向tablet server发送要read row r的请求，同时block calling thread，直到server返回row r。在process cache中每个row有一个tag来表示是否有row request正在被处理，这样可以同步其它的request统一row的请求。

当tablet server返回row r时，client library端有一个bg thread会接受到row r，同时接受requester clock rc。rc表示该client提交的clock t的updates已经被处理。之后，process oplog就可以清除`clock <= rc` 的update日志。为了保证 read-my-updates，接收到row r 后，会将process oplog中`clock > rc`的操作作用到row r上，这样就可以得到本地最新的row r。最后，前面接受row r的bg thread会跟心row r的clock并将其返回到waiting app threads。


## Prefetching and fault-tolerance
### 数据预取

LazyTable提供了预取API refresh()，函数参数与read()一样，但与read()不一样的地方是refresh()不会block calling thread。

LazyTable支持两种预取机制：conservative prefetching和aggressive prefetching。前者只在必要的时候进行refresh，如果`cache_age < t-s-1`，prefetcher才会发送一个`request(row = r, age >= t-s-1)`。对于Aggressive prefetching，如果当前的row不是最新的会主动去更新。


# Bosen的线程启动过程分析

Start PS的第一个步骤就是初始化各个线程
```c++
petuum::PSTableGroup::Init(table_group_config, false)
```
其具体实现是
- 初始化每个node上的namenode，background及server threads
- 建立这些threads之间的通信关系
- 为createTables()做准备

## Namenode thread
一个Petuum cluster里面只有一个Namenode thread，负责协同各个节点上的bg threads和server threads。


## Server thread
角色是PS中的Server，负责管理建立和维护用于存放parameters的global tables。

## Background (Bg) thread
角色是PS中的Client，负责管理真正计算的worker threads，并与server thread通信。在每个node上，bg threads可以有多个，其中一个负责建立本地 table。

## 代码结构与流程
![init](figures/PSTableGroup-Init().png)


## Local 模式线程启动分析

启动流程

```c++
// main thread调用PSTableGroup::Init()后变成init thread并向CommBus注册自己
I1230 10:00:50.570231  9821 comm_bus.cpp:117] CommBus ThreadRegister()
// init thread创建Namenode thread，该向CommBus注册自己
I1230 10:01:16.210435 10014 comm_bus.cpp:117] CommBus ThreadRegister()
// Namenode thread启动
NameNode is ready to accept connections!
// cluster中bg thread的个数
I1230 10:05:09.398447 10014 name_node_thread.cpp:126] Number total_bg_threads() = 1
// cluster中的server thread的个数
I1230 10:05:09.398485 10014 name_node_thread.cpp:128] Number total_server_threads() = 1
// app中定义的table_group_config的consistency_model = SSPPush or SSP
I1230 10:06:24.141788  9821 server_threads.cpp:92] RowSubscribe = SSPPushRowSubscribe
// 启动（pthread_create）所有的local server threads，这里只有一个
I1230 10:09:50.340092  9821 server_threads.cpp:106] Create server thread 0
// Server thread获取cluster中的client个数
I1230 10:12:15.419473 10137 server_threads.cpp:239] ServerThreads num_clients = 1
// Server thread自己的thread id
I1230 10:12:15.419505 10137 server_threads.cpp:240] my id = 1
// Server thread向CommBus注册自己
I1230 10:12:15.419514 10137 comm_bus.cpp:117] CommBus ThreadRegister()
// 注册成功
I1230 10:12:15.419587 10137 server_threads.cpp:252] Server thread registered CommBus
// Bg thread启动，id = 100，Bg thread的id从100开始
I1230 10:12:51.534554 10171 bg_workers.cpp:889] Bg Worker starts here, my_id = 100
// Bg thread向CommBus注册自己
I1230 10:12:51.534627 10171 comm_bus.cpp:117] CommBus ThreadRegister()
// Bg thread先去connect Namenode thread
I1230 10:12:51.534677 10171 bg_workers.cpp:283] ConnectToNameNodeOrServer server_id = 0
// Bg thread去连接Namenode thread
I1230 10:12:51.534683 10171 bg_workers.cpp:290] Connect to local server 0
// Namenode thread 收到Bg thread id = 100的请求
I1230 10:12:51.534826 10014 name_node_thread.cpp:139] Name node gets client 100
// Server thread首先去连接Namenode thread
I1230 10:13:18.879250 10137 server_threads.cpp:141] Connect to local name node
// Namenode thread收到Server thread的请求
I1230 10:13:21.051105 10014 name_node_thread.cpp:142] Name node gets server 1
// Namenode已经收到所有的client和server的连接请求
I1230 10:13:33.913213 10014 name_node_thread.cpp:149] Has received connections from all clients and servers, sending out connect_server_msg
// Namenode向所有client (bg thread) 发送让其连接server thread的命令
I1230 10:13:33.913254 10014 name_node_thread.cpp:156] Send connect_server_msg done
// 发送connect_server_msg命令完毕
I1230 10:13:33.913261 10014 name_node_thread.cpp:162] InitNameNode done
// 每个bg thread去连接cluster中的所有的server threads，这里只有一个server thread
I1230 10:13:33.929790 10171 bg_workers.cpp:283] ConnectToNameNodeOrServer server_id = 1
// Bg thread连接上了server thread
I1230 10:13:33.929821 10171 bg_workers.cpp:290] Connect to local server 1
// 收到Namenode的连接反馈消息（client_start_msg表示连接成功)
I1230 10:13:33.929862 10171 bg_workers.cpp:368] get kClientStart from 0 num_started_servers = 0
// Server thread初始化完成
I1230 10:23:39.355000 10137 server_threads.cpp:187] InitNonNameNode done
// Bg thread收到server thread的反馈信息（client_start_msg表示连接成功)
I1230 10:23:39.355051 10171 bg_workers.cpp:368] get kClientStart from 1 num_started_servers = 1
// Bg thread id＝100收到CreateTable的请求
I1230 10:23:39.355198 10171 bg_workers.cpp:911] head bg handles CreateTable
Data mode: Loading matrix sampledata/9x9_3blocks into memory...
```
Thread Ids: （local模式下Namenode，Server及Bg thread都只有一个）
- 9821: main() thread
- 10014: Namenode thread
- 10137: Server thread
- 10171: Bg thread

图解如下：

![LocalThreads](images/LocalThreads.png)

## Distributed 模式线程启动分析

启动图解如下：

![DistributedThreads](images/DistributedThreads.png)

可以看到各个节点上的线程启动后，Server threads和Bg threads都与Namenode threads建立了连接。然后Namenode通知所有的bg threads与集群中的所有server threads建立连接。连接建立后，可以看到Server threads和Bg threads组成了一个二分图结构，也就是所谓的Parameter Server。

# ServerThreads

## 基本结构
1. 每个client上的app进程持有一个ServerThreads object，这个object管理该client上的所有server threads。这些server threads的启动过程：`app.main() => PSTableGroup::Init() => ServerThreads::Init() => ServerThreadMain(threadId) for each server thread`。
2. 每个server thread实际上是一个Server object。ServerThreads对象通过`vector<pthread_t> threads`和`vector<int> threads_ids`来引用server threads，通过其ServerContex指针用来访问每个server thread对应的Server object（`server_context_ptr->server_obj`）。
3. 对于每一个server thread，都持有一个ServerContext，其初始化时`server_context.bg_threads_ids`存储PS中所有bg threads的`bg_thread_id`，`server_context.server_obj`存储该server thread对应的Server object。
4. 每个Server object里面存放了三个数据结构：`client_bg_map<client_id, bg_id>`存放PS中有那些client，每个client上有那些bg threads；`client_ids`存放PS中有那些client；`client_clocks`是VectorClock，存放来自client的clock，初始化时clock为0。每个Server thread在初始化时会去connect PS中所有的bg thread，然后将`(client_id, 0)`添加到server thread对应的Server object中的`client_clocks`中。如果某个client上有多个bg thread，那么`(client_id, 0)`会被重复添加到`client_clocks: VectorClock`中，会做替换。注意`client_clocks: VectorClock`的长度为PS中client的总个数，也就是每一个client对应一个clock，而不是每个bg thread对应一个clock。Server object还有一个`client_vector_clock_map<int, VectorClock>`的数据结构，key为`client_id`，value为该client上所有bg thread的VectorClock。也就是说每个server thread不仅存放了每个client的clock，也存放了该client上每个bg thread的clock。
5. Server object还有一个`bg_version_map<bg_thread_id, int>`的数据结构，该结构用于存放server thread收到的bg thread的最新oplog版本。

## CreateTable

Server thread启动后，会不断循环等待消息，当收到Namenode发来的`create_table_msg`时，会调用`HandleCreateTable(create_table_msg)`来createTable，会经历以下步骤：

1. 从msg中提取出tableId。
2. 回复消息给Namenode说准备创建table。
3. 初始化TableInfo消息，包括table的`staleness, row_type, columnNum (row_capacity)`。
4. 然后调用server thread对应的Server object创建table，使用`Server.CreateTable(table_id, table_info)`。
5. Server object里面有个`map<table_id, ServerTable> tables`数据结构，`CreateTable(table_id)`就是new出一个ServerTable，然后将其加入这个map。
6. ServerTable object会存放`table_info`，并且有一个`map<row_id, ServerRow> storage`，这个map用来存放ServerTable中的rows。另外还有一个`tmp_row_buff[row_length]`的buffer。new ServerTable时，只是初始化一些这些数据结构。

## HandleClientSendOpLogMsg

当某个server thread收到client里bg thread发来的`client_send_oplog_msg`时，会调用ServerThreads的`HandleOpLogMsg(client_send_oplog_msg)`，该函数会执行如下步骤：

1. 从msg中抽取出`client_id`，判断该msg是否是clock信息，并提取出oplog的version。
2. 调用server thread对应的`ServerObj.ApplyOpLog(client_send_oplog_msg)`。该函数会将oplog中的updates requests都更新到本server thread维护的ServerTable。
3. 如果msg中没有携带clock信息，那么执行结束，否则继续下面的步骤：
4. 调用`ServerObj.Clock(client_id, bg_id)`，并返回`bool clock_changed`。该函数会更新client的VectorClock（也就是每个bg thread的clock），如果client的VectorClock中唯一最小的clock被更新，那么client本身的clock也需要更新，这种情况下`clock_changed`为true。
5. 如果`clock_changed == false`，那么结束，否则，进行下面的步骤：
6. `vector<ServerRowRequest> requests = serverObject.GetFulfilledRowRequests()`。
7. 对每一个request，提取其`table_id, row_id, bg_id`，然后算出bg thread的`version = serverObj.GetBgVersion(bg_id)`。
8. 根据提取的`row_id`去Server object的ServerTable中提取对应的row，使用方法`ServerRow server_row = ServerObj.FindCreateRow(table_id, row_id)`。
9. 调用`RowSubscribe(server_row, bg_id_to_client_id)`。如果consistency model是SSP，那么RowSubscribe就是SSPRowSubscribe；如果是SSP push，那么RowSubscribe就是SSPPushRowSubscribe。NMF使用是后者，因此这一步就是`SSPPushRowSubscribe(server_row, bg_id_to_client_id)`。该方法的意思是将`client_id`注册到该`server_row`，这样将该`server_row`在调用`AppendRowToBuffs`可以使用`callback_subs.AppendRowToBuffs()`。
10. 查看Server object中VectorClock中的最小clock，使用方法`server_clock = ServerObj.GetMinClock()`。
11. `ReplyRowRequest(bg_id, server_row, table_id, row_id, sersver_clock)`。
12. 最后调用`ServerPushRow()`。

### `Server.ApplyOpLog(oplog, bg_thread_id, version)`

1. check一下，确保自己`bg_version_map`中该bg thread对应的version比这个新来的version小1。
2. 更新`bg_version_map[bg_thread_id] = version`。
3. oplog里面可以存在多个update request，对于每一个update request，执行以下步骤：
4. 读取oplog中的`table_id, row_id, column_ids, num_updates, started_new_table`到updates。
5. 根据`table_id`从`ServerObj.tables`中找出对应的ServerTable。
6. 执行ServerTable的`ApplyRowOpLog(row_id, column_ids, updates, num_updates)`。该方法会找出ServerTable对应的row，并对row进行`BatchInc(column_ids, updates)`。如果ServerTable不存在该row，就先`CreateRow(row_id)`，然后`BatchInc()`。
7. 打出"Read and Apply Update Done"的日志。

### `ServerObj.Clock(client_id, bg_id)`

1. 执行`ServerObj.client_vector_clock_map[client_id].Tick(bg_id)`，该函数将client对应的VectorClock中`bg_id`对应的clock加1。
2. 如果`bg_id`对应的原始clock是VectorClock中最小值，且是唯一的最小值，那么clock+1后，需要更新client对应的clock，也就是对`client_clocks.Tick(client_id)`。
3. 然后看是否达到了snapshot的clock，达到就进行checkpoint。

## HandleRowRequestMsg

当某个server thread收到client里bg thread发来的`row_request_msg`时，会调用ServerThreads的`HandleRowRequest(bg_id, row_request_msg)`，该函数会执行如下步骤：

1. 从msg中提取出`table_id, row_id, clock`。
2. 查看ServerObj中的所有client的最小clock。使用`server_clock = ServerObj.GetMinClock()`。
3. 如果msg请求信息中的clock > `server_clock`，也就是说目前有些clients在clock时的更新信息还没有收到，那么先将这个msg的request存起来，等到ServerTable更新到clock时，再reply。具体会执行`ServerObj.AddRowRequest(sender_id, table_id, row_id, clock)`。
4. 如果msg请求信息中的clock  <= `server_clock`，也就是说ServerTable中存在满足clock要求的rows，那么会执行如下步骤：
5. 得到`bg_id`的version，使用`version = ServerObj.GetBgVersion(sender_id)`，`sender_id`就是发送`row_request_msg`请求的client上面的bg thread。
6. 将ServerTable中被request的row取出来到`server_row`。
7. 调用`RowSubscribe(server_row, sender_id_to_thread_id)`。
8. 将`server_row`reply给bg thread，具体使用`ReplyRowRequest(sender_id, server_row, table_id, row_id, server_clock, version)`。



### `ServerObj.AddRowRequest(sender_id, table_id, row_id, clock)`

当来自client的request当前无法被处理的时候（server的row太old），server会调用这个函数将请求先放到队列里。具体执行如下步骤：

1. 先new一个ServerRowRequest的结构体，将`bg_id, table_id, row_id, clock`放到这个结构体中。
2. 将ServerRowRequest放进`map<clock, vector<ServerRowRequest>> clock_bg_row_requests`中，该数据结构的key是clock，vector中的index是`bg_id`，value是ServerRowRequest。

### `ReplyRowRequest(sender_id, server_row, table_id, row_id, server_clock, version)`

1. 先构造一个`ServerRowRequestReplyMsg`，然后将`table_id, row_id, server_clock, version`填入这个msg中。
2. 然后将msg序列化后发回给`bg_id`对应的bg thread。

# CreateTable过程

## 基本流程

1. 每个App main Thread（比如每个节点上matrixfact.main()进程的main/init thread）调用`petuum::PSTableGroup::CreateTable(tableId, table_config)`来创建Table。
2. 该方法会调用同在一个Process里的head bg thread向NameNode thread发送创建Table的请求`create_table_msg`。
3. NameNode收到CreateTable请求，如果该Table还未创建，就在自己的线程里创建一个ServerTable。之后会忽略其他要创建同一Table的请求。
4. NameNode将CreateTable请求`create_table_msg`发送到cluster中的每个Server thread。
5. Server thread收到CreateTable请求后，先reply `create_table_reply_msg` to NameNode thread，表示自己已经知道要创建Table，然后直接在线程里创建一个ServerTable。
6. 当NameNode thread收到cluster中所有Server thread返回的reply消息后，就开始reply `create_table_reply_msg` to head bg thread说“Table已被ServerThreads创建”。
7. 当App main()里定义的所有的Table都被创建完毕（比如matrixfact里要创建三个Table），NameNode thread会向cluster中所有head bg thread发送“所有的Tables都被创建了”的消息，也就是`created_all_tables_msg`。

## 流程图
![CreateTable](images/CreateTableThreads.png)

## 代码结构图

![CreateTable](images/CreateTable.png)
