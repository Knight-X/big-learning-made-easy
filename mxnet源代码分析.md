# mxnet源代码分析
整个分析的过程以example/image/image-classification/train_mnist.py 的分布式执行过程为线索。
启动命令：
'../../tools/launch.py -n 2 -H hosts --sync-dst-dir /tmp/mxnet python  train_mnist.py --kv-store dist_sync'
关于启动过程分析见**mxnet启动分析**.

## symbol的建立过程
train_mnist.py中使用符号式编程对网络进行了定义，函数`get_mlp`和`get_lenet`分别定义了两个网络一个是普通的多层网络，一个是带卷积的网络。以下是get_lenet的代码：

```python
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
```

首先`data = mx.symbol.variable('date')`, 在symbol.py的variable函数调用了c_api.cc中的`MXSymbolCreateVariable`函数,继续调用进入symbol.cc中的CreateVariable，代码如下：

```c++
 Symbol Symbol::CreateVariable(const std::string &name) {
   Symbol s;
   s.heads_.push_back(DataEntry(std::make_shared<Node>(nullptr, name), 0));
   return s;
 }
```
Symbol在symbolic.h中被定义。

```c++
class Symbol {
  public:
   std::vector<std::string> ListArguments() const;
   /*! \return get the descriptions of outputs for this symbol */
   std::vector<std::string> ListOutputs() const;
   /*! \return get the descriptions of auxiliary data for this symbol */
   std::vector<std::string> ListAuxiliaryStates() const;

  ...

  protected:
   // Declare node, internal data structure.
   struct Node;
   /*! \brief an entry that represents output data from a node */
   struct DataEntry {
     /*! \brief the source node of this data */
     std::shared_ptr<Node> source;
     /*! \brief index of output from the source. */
     uint32_t index;
     /*! \brief enabled default copy constructor */
     DataEntry() {}
     /*! \brief constructor from index */
     DataEntry(std::shared_ptr<Node> source, uint32_t index)
         : source(source), index(index) {}
   };
   /*!
    * \brief the head nodes of Symbols
    * This head is only effective when
    */
   std::vector<DataEntry> heads_;

}

symbol.cc

 /*!
  * \brief Node is represents node of an operator in the symbolic graph.
  *
  * It stores connection to the inputs to function represented by OperatorProperty
  * NOTE on data structure: there are three types of node:
  * - Normal node: contains all the necessary elements of a graph.
  * - OperatorProperty: the inputs_ is empty, represents an OperatorProperty that has not been applied.
  * - Variable: the sym_ is nullptr, represents an named Variable of tensors that can be composed.
  */
 struct Symbol::Node {
   /*! \brief Operator of this node */
   std::unique_ptr<OperatorProperty> op;
   /*! \brief name of the node */
   std::string name;
   /*! \brief inputs to this node */
   std::vector<DataEntry> inputs;
   /*! \brief source node of the current node */
   std::shared_ptr<Symbol::Node> backward_source_node;
   /*!
    * \brief additional attributes about the node,
    *  Use pointer to save space, as attr can be accessed in a slow way,
    *  not every node will have attributes.
    */
   std::unique_ptr<std::map<std::string, std::string> > attr;
   /*!
     *\brief constructor
     *\param op the OperatorProperty to construct the Node
     *\param name the name of the symbol
    */
   explicit Node(OperatorProperty *op,
                 const std::string& name)
       : op(op), name(name) {}
   /*!
     *\brief copy constructor constructor
    */
   explicit Node(const Node& other)
       : name(other.name) {
     if (other.op != nullptr) {
       op.reset(other.op->Copy());
     }

```

createVariable 只是创建了一个空的node，并把名字付给了一个DataEntry并把它push进vector。这也就是符号编程的特点，定义的时候只是做了形的定义，并没有直接关联上具体的数据。

	> note c_api.cc 是python调用c++的一个入口。

现在回到`train_mnist.py`的`get_lenet`函数中。继续执行`conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)`，Convolution函数在Symbol.py中没有定义，所以接下来看一下Convolution函数从何而来。

为了讲清楚这个问题又需要引入一些定义。
在src/operator/convolution.cc中有两个宏，

```c++
 DMLC_REGISTER_PARAMETER(ConvolutionParam);

 MXNET_REGISTER_OP_PROPERTY(Convolution, ConvolutionProp)
 .add_argument("data", "Symbol", "Input data to the ConvolutionOp.")
 .add_argument("weight", "Symbol", "Weight matrix.")
 .add_argument("bias", "Symbol", "Bias parameter.")
 .add_arguments(ConvolutionParam::__FIELDS__())
 .describe("Apply convolution to input then add a bias.");
```

暂时先看第二个宏`MXNET_REGISTER_OP_PROPERTY(Convolution, ConvolutionProp)`被定义在了include/mxnet/operator.h中

```c++
 #define MXNET_REGISTER_OP_PROPERTY(name, OperatorPropertyType)          \
   DMLC_REGISTRY_REGISTER(::mxnet::OperatorPropertyReg, OperatorPropertyReg, name) \
   .set_body([]() { return new OperatorPropertyType(); })                \
   .set_return_type("Symbol") \
   .check_name()
```
这个宏又嵌套了`DMLC_REGISTRY_REGISTER(::mxnet::OperatorPropertyReg, OperatorPropertyReg, name)`,被定义在dmlc-core/include/dmlc/registry.h。

```c++
 #define DMLC_REGISTRY_REGISTER(EntryType, EntryTypeName, Name)          \
   static DMLC_ATTRIBUTE_UNUSED EntryType & __make_ ## EntryTypeName ## _ ## Name ## __ = \
       ::dmlc::Registry<EntryType>::Get()->__REGISTER__(#Name)           \
```

所以`MXNET_REGISTER_OP_PROPERTY(Convolution, ConvolutionProp)`展开后得到

```c++

   static DMLC_ATTRIBUTE_UNUSED EntryType & __make_OperatorPropertyReg_Convolution__ = \
       ::dmlc::Registry<::mxnet::OperatorPropertyReg>::Get()->__REGISTER__("Convolution") \
   .set_body([]() { return new ConvolutionProp(); })                \
   .set_return_type("Symbol") \
   .check_name()
   .add_argument("data", "Symbol", "Input data to the ConvolutionOp.")
   .add_argument("weight", "Symbol", "Weight matrix.")
   .add_argument("bias", "Symbol", "Bias parameter.")
   .add_arguments(ConvolutionParam::__FIELDS__())
   .describe("Apply convolution to input then add a bias.");

```

Registry是一个工具类用于把c++代码里面的函数分类别的统一注册。


```c++
 template<typename EntryType>
 class Registry {

  ...

     inline EntryType &__REGISTER__(const std::string& name) {
     CHECK_EQ(fmap_.count(name), 0)
         << name << " already registered";
     EntryType *e = new EntryType();
     e->name = name;
     fmap_[name] = e;
     const_list_.push_back(e);
     entry_list_.push_back(e);
     return *e;
   }

   private:
   /*! \brief list of entry types */
   std::vector<EntryType*> entry_list_;
   /*! \brief list of entry types */
   std::vector<const EntryType*> const_list_;
   /*! \brief map of name->function */
   std::map<std::string, EntryType*> fmap_;

 };

 #define DMLC_REGISTRY_ENABLE(EntryType)                                 \
   template<>                                                            \
   Registry<EntryType > *Registry<EntryType >::Get() {                   \
     static Registry<EntryType > inst;                                   \
     return &inst;                                                       \
   }
```
上面是Registry的数据结构，还列出了一个新的宏`DMLC_REGISTRY_ENABLE(EntryType)`, 这个宏很简单就是返回一个singleton的Registry<EntryType>。关于operator的这个`singleton在/operator/operator.cc`中被调用

```c++
DMLC_REGISTRY_ENABLE(::mxnet::OperatorPropertyReg);
```

static 说明这个`Registry<::mxnet::OperatorPropertyReg>`这个结构的实例在进main函数之前就已经ready,所以可以通过Get这个静态函数返回这个singleton在`__REGISTER__`函数中new了一个OperatorPropertyReg,将这个实例于"Convolution"对应存进`fmap_`中。

```c++
 /*!
  * \brief Registry entry for OperatorProperty factory functions.
  */
 struct OperatorPropertyReg
     : public dmlc::FunctionRegEntryBase<OperatorPropertyReg,
                                         OperatorPropertyFactory> {

	...

	std::string key_var_num_args;

    ...
}

template<typename EntryType, typename FunctionType>
 class FunctionRegEntryBase {
  public:
   /*! \brief name of the entry */
   std::string name;
   /*! \brief description of the entry */
   std::string description;
   /*! \brief additional arguments to the factory function */
   std::vector<ParamFieldInfo> arguments;
   /*! \brief Function body to create ProductType */
   FunctionType body;
   /*! \brief Return type of the function */
   std::string return_type;

   ...
}

```

OperatorPropertyReg继承自FunctionRegEntryBase, 在FunctionRegEntryBase中成员在`MXNET_REGISTER_OP_PROPERTY(Convolution, ConvolutionProp)`宏展开中被初始化。由于是staic，所以上述的过程在进libcrt0中，在进main之前被执行,或者如果是动态链接则在so被load时候被执行。

这些数据结构和python中Symbol.Convolution的关联发生在Symbol被import的时候。












