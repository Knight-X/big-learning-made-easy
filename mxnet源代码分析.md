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

这些数据结构和python中Symbol.Convolution的关联发生在Symbol被import的时候。当import Symbol时，下面的这两个函数会被执行，它将c++里面的代码入驻到了python。

```c++

 def _make_atomic_symbol_function(handle):
     """Create an atomic symbol function by handle and funciton name."""
     name = ctypes.c_char_p()
     desc = ctypes.c_char_p()
     key_var_num_args = ctypes.c_char_p()
     num_args = mx_uint()
     arg_names = ctypes.POINTER(ctypes.c_char_p)()
     arg_types = ctypes.POINTER(ctypes.c_char_p)()
     arg_descs = ctypes.POINTER(ctypes.c_char_p)()
     ret_type = ctypes.c_char_p()

     check_call(_LIB.MXSymbolGetAtomicSymbolInfo(
         handle, ctypes.byref(name), ctypes.byref(desc),
         ctypes.byref(num_args),
         ctypes.byref(arg_names),
         ctypes.byref(arg_types),
         ctypes.byref(arg_descs),
         ctypes.byref(key_var_num_args),
         ctypes.byref(ret_type)))
     param_str = ctypes2docstring(num_args, arg_names, arg_types, arg_descs)
     key_var_num_args = py_str(key_var_num_args.value)
     func_name = py_str(name.value)
     desc = py_str(desc.value)
     if key_var_num_args:
         desc += '\nThis function support variable length of positional input.'
     doc_str = ('%s\n\n' +
                '%s\n' +
                'name : string, optional.\n' +
                '    Name of the resulting symbol.\n\n' +
                'Returns\n' +
                '-------\n' +
                'symbol: Symbol\n' +
                '    The result symbol.')
     doc_str = doc_str % (desc, param_str)
     extra_doc = "\n" + '\n'.join([x.__doc__ for x in type.__subclasses__(SymbolDoc)
                                   if x.__name__ == '%sDoc' % func_name])
     doc_str += re.sub(re.compile("    "), "", extra_doc)
    def creator(*args, **kwargs):
         """Activation Operator of Neural Net.
         The parameters listed below can be passed in as keyword arguments.

         Parameters
         ----------
         name : string, required.
             Name of the resulting symbol.

         Returns
         -------
         symbol: Symbol
             the resulting symbol
         """
         param_keys = []
         param_vals = []
         symbol_kwargs = {}
         name = kwargs.pop('name', None)
         attr = kwargs.pop('attr', None)

         if key_var_num_args and key_var_num_args not in kwargs:
             param_keys.append(c_str(key_var_num_args))
             param_vals.append(c_str(str(len(args))))

         print "1051  ", kwargs
         for k, v in kwargs.items():
             if isinstance(v, Symbol):
                 symbol_kwargs[k] = v
             else:
                 param_keys.append(c_str(k))
                 param_vals.append(c_str(str(v)))
         # create atomic symbol
         param_keys = c_array(ctypes.c_char_p, param_keys)
         param_vals = c_array(ctypes.c_char_p, param_vals)
         sym_handle = SymbolHandle()
         check_call(_LIB.MXSymbolCreateAtomicSymbol(
             handle,
             mx_uint(len(param_keys)),
             param_keys, param_vals,
            ctypes.byref(sym_handle)))

         if len(args) != 0 and len(symbol_kwargs) != 0:
             raise TypeError(
                 '%s can only accept input'
                 'Symbols either as positional or keyword arguments, not both' % func_name)
         if key_var_num_args and len(symbol_kwargs) != 0:
             raise ValueError('This function supports variable length of Symbol arguments.\n' +
                              'Please pass all the input Symbols via positional arguments' +
                              ' instead of keyword arguments.')
         s = Symbol(sym_handle)
         attr = AttrScope.current.get(attr)
         if attr:
             s._set_attr(**attr)
         #get name here
         hint = func_name.lower()
         name = NameManager.current.get(name, hint)

         s._compose(*args, name=name, **symbol_kwargs)
         return s

     creator.__name__ = func_name
     creator.__doc__ = doc_str
     return creator

 def _init_symbol_module():
     """List and add all the atomic symbol functions to current module."""
     plist = ctypes.POINTER(ctypes.c_void_p)()
     size = ctypes.c_uint()

     check_call(_LIB.MXSymbolListAtomicSymbolCreators(ctypes.byref(size),
                                                      ctypes.byref(plist)))
     module_obj = sys.modules[__name__]
     module_internal = sys.modules["mxnet._symbol_internal"]
     for i in range(size.value):
         hdl = SymbolHandle(plist[i])
         function = _make_atomic_symbol_function(hdl)
         if function.__name__.startswith('_'):
             setattr(module_internal, function.__name__, function)
         else:
             setattr(module_obj, function.__name__, function)
```

`_init_symbol_module()`首先调用`_LIB.MXSymbolListAtomicSymbolCreators` 获取所用注册在`Registry<OperatorPropertyReg>`中的OperatorPropertyReg的指针,每一个operator都会注册一个自己的OperatorProertyReg。然后调用`_make_atomic_symbol_function`,用每个OperatorPropertyReg中的信息组装函数，在组装的时候用把Symbol和operator链接起来，最后返回一个Symbol。如果函数的名字以`_`开头，函数名被注册到`python/mxnet/_symbol_internal.py`空间下，这是一个空文件只用来注册一些函数。如果函数不以`_`开头则函数被注册到`python/mxnet/symbol.py`。

`_make_atomic_symbol_function` 首先调用`_LIB.MXSymbolGetAtomicSymbolInfo`获取注册的信息。最后返回了一个create函数，这个create函数就是operator的初始化函数，create函数是一个闭包结构,封印了相应的OperatorPropertyReg结构的的指针和从这个指针中获取出来的一些信息。首先对参数进行解析，首先把当先这个符号的命名从参数中pop出来，如果没有指定这个参数会在函数返回前根据当前的函数名生成一个符号命，接着把参数分成两组，一组是Symbol类型的参数`symbol_kwargs`用于compose符号，一组是当前的这个operator接收的参数，用于初始化当前的的符号。符号的初始化调用一下函数


```c++
 int MXSymbolCreateAtomicSymbol(AtomicSymbolCreator creator,
                                mx_uint num_param,
                                const char **keys,
                                const char **vals,
                                SymbolHandle *out) {
   Symbol *s = new Symbol();
   OperatorProperty *op = nullptr;

   API_BEGIN();
   OperatorPropertyReg *e = static_cast<OperatorPropertyReg *>(creator);
   op = e->body();
   std::vector<std::pair<std::string, std::string> > kwargs;
   for (mx_uint i = 0; i < num_param; ++i) {
     kwargs.push_back({std::string(keys[i]), std::string(vals[i])});
   }
   op->Init(kwargs);
   *s = Symbol::Create(op);
   *out = s;
   API_END_HANDLE_ERROR(delete s; delete op);
 }
```
首先创建Symbol实例，然后调用`OperatorPropertyReg->body()`,这个就是之前在`MXNET_REGISTER_OP_PROPERTY`展开中的`set_body()`函数设置的opartor的初始化函数。`op->Init(kwargs)`这个函数就是调用各个operator的Init函数，从python中传递过来的参数在这里被使用并初始化operator,关于operator的参数初始化后再细讲。最后调用`Symbol::Create(op)`


```c++
Symbol Symbol::Create(OperatorProperty *op)  {
   // use special representation for atomic symbol
   auto node = std::make_shared<Node>(op, "");
   size_t nret = op->NumVisibleOutputs();
   Symbol s;
   for (uint32_t i = 0; i < nret; ++i) {
     LOG(INFO) << i;
     s.heads_.push_back(DataEntry(node, i));
   }
   return s;
 }
```
使用op初始化一个Node，并将根据`op->NumvisibleOutputs()`的输出的数量生成DataEntry，并push到`heads_`。至此完成了Symbol和Operator的关系，python数据和到c++的调用，以及python函数在runtime的注册。

回到create函数中，继续调用`s._compose(*args, name=name, **symbol_kwargs)`,完成于之前符号的链接，就是在创建符号时参数中传进来的符号。
通过`_compose`, `_LIB.MXSymbolCompose`, 进入Symbol的compose。

```c++

void Symbol::Compose(const std::unordered_map<std::string, Symbol>& kwargs,
                      const std::string& name) {
   // CHECK_EQ(NumOutputs(), 1) << "Only composition of value function is supported currently";
   CHECK(!heads_[0].source->is_variable()) << "Variable cannot be composed";
   LOG(INFO) << name;
   heads_[0].source->name = name;
   for (const auto& kv : kwargs) {
     LOG(INFO) << kv.first;
     CHECK_EQ(kv.second.NumOutputs(), 1)
         << "Keyword Argument " << kv.first << " is a tuple, scalar is required";
   }
   size_t nmatched = 0;
   if (this->is_atomic()) {
     LOG(INFO) << "here";
     // atomic symbol do not have place holder for all the arguments
     std::vector<std::string> req_args = heads_[0].source->op->ListArguments();
     //use op's arguments list to resize input size with the node
     LOG(INOF) << "op->ListArguments req_args_size " <<
     heads_[0].source->inputs.resize(req_args.size());

     for (size_t i = 0; i < req_args.size(); ++i) {
       auto iter = kwargs.find(req_args[i]);
       if (iter != kwargs.end()) {
         heads_[0].source->inputs[i] = iter->second.heads_[0];
         ++nmatched;
       } else {
         heads_[0].source->inputs[i] = DataEntry(
             std::make_shared<Node>(nullptr, DefaultVarName(name, req_args[i])), 0);
         // also copy attribute of operator over to automatically created variable
         if (heads_[0].source->attr.get() != nullptr) {
           heads_[0].source->inputs[i].source->attr.reset(
               new std::map<std::string, std::string>(*(heads_[0].source->attr)));
         }
       }
     }
     // if things goes wrong recover the old state
     if (nmatched != kwargs.size()) {
       heads_[0].source->inputs.clear();
     }
   }

   ...

   if (nmatched != kwargs.size()) {
     std::vector<std::string> keys(kwargs.size());
     std::transform(kwargs.begin(), kwargs.end(), keys.begin(),
                    [](decltype(*kwargs.begin())& kv)->std::string { return kv.first; });
     KeywordArgumentMismatch("Symbol.Compose", keys, ListArguments());
   }
}

```

name 是之前符号的名字，首先获取当前的Symbol所需的arguments `ListArguments()`. 然后迭代所需的arguments,并在之前的符号中查找如果找到就讲当前的符号于传入的符号链接上。

```c++
heads_[0].source->inputs[i] = iter->second.heads_[0];
```

回头看看operator的参数的初始化。在src/operator/convolution.cc中还有一个宏的调用，`DMLC_REGISTER_PARAMETER(ConvolutionParam);`。

```c++
#define DMLC_REGISTER_PARAMETER(ConvolutionParam)                                  \
   ::dmlc::parameter::ParamManager *ComvolutionParam::__MANAGER__() {               \
     static ::dmlc::parameter::ParamManagerSingleton<ConvolutionParam> inst(#ConvolutionParam); \
     return &inst.manager;                                               \
   }                                                                     \
   static ::dmlc::parameter::ParamManager &__make__ ## ConvolutionParam ## ParamManager__ = \
       (*Convolution::__MANAGER__())
```

先看一下ConvolutionParam的定义。

```c++

struct ConvolutionParam : public dmlc::Parameter<ConvolutionParam> {
   TShape kernel;
   TShape stride;
   TShape dilate;
   TShape pad;
   uint32_t num_filter;
   uint32_t num_group;
   uint64_t workspace;
   bool no_bias;
   int cudnn_tune;
   bool cudnn_off;
   DMLC_DECLARE_PARAMETER(ConvolutionParam) {
     int shape[] = {1, 1};
     DMLC_DECLARE_FIELD(kernel).describe("convolution kernel size: (y, x) or (d, y, x)");
     DMLC_DECLARE_FIELD(stride).set_default(TShape(shape, shape + 2))
     .describe("convolution stride: (y, x) or (d, y, x)");
     DMLC_DECLARE_FIELD(dilate).set_default(TShape(shape, shape + 2))
     .describe("convolution dilate: (y, x)");
     shape[0] = shape[1] = 0;
     DMLC_DECLARE_FIELD(pad).set_default(TShape(shape, shape + 2))
     .describe("pad for convolution: (y, x) or (d, y, x)");
     DMLC_DECLARE_FIELD(num_filter).set_range(1, 100000)
     .describe("convolution filter(channel) number");
     DMLC_DECLARE_FIELD(num_group).set_default(1)
     .describe("Number of groups partition. "
               "This option is not supported by CuDNN, you can use SliceChannel to num_group,"
               "apply convolution and concat instead to achieve the same need.");
     DMLC_DECLARE_FIELD(workspace).set_default(1024).set_range(0, 8192)
     .describe("Tmp workspace for convolution (MB).");
     DMLC_DECLARE_FIELD(no_bias).set_default(false)
     .describe("Whether to disable bias parameter.");
     DMLC_DECLARE_FIELD(cudnn_tune)
     .add_enum("off", conv::kOff)
     .add_enum("limited_workspace", conv::kLimited)
     .add_enum("fastest", conv::kFastest)
     .set_default(dmlc::GetEnv("MXNET_CUDNN_AUTOTUNE_DEFAULT", 0))
     .describe("Whether to find convolution algo by running performance test."
               "Leads to higher startup time but may give better speed."
               "auto tune is turned off by default."
               "Set environment varialbe MXNET_CUDNN_AUTOTUNE_DEFAULT=1 to turn on by default.");
     DMLC_DECLARE_FIELD(cudnn_off).set_default(false)
     .describe("Turn off cudnn.");
   }
 };

```

继承自`dmlc::Parameter`,

```c++
 struct Parameter {
  public:
   /*!
    * \brief initialize the parameter by keyword arguments.
    *  This function will initialize the parameter struct, check consistency
    *  and throw error if something wrong happens.
    *
    * \param kwargs map of keyword arguments, or vector of pairs
    * \parma option The option on initialization.
    * \tparam Container container type
    * \throw ParamError when something go wrong.
    */
   template<typename Container>
   inline void Init(const Container &kwargs,
                    parameter::ParamInitOption option = parameter::kAllowUnknown) {
     PType::__MANAGER__()->RunInit(static_cast<PType*>(this),
                                   kwargs.begin(), kwargs.end(),
                                   NULL,
                                   option == parameter::kAllowUnknown);
   }
```

相似的手法，通过static拿到全局的handle。通过`DMLC_DECLARE_FIELD`定义了一些对每一个变量的描述和default，range等的属性，有些属性会用于runtime时候的check。这个里面有一个比较trick的地方是通过成员和类型`定义了一些对每一个变量的描述和default，range等的属性，有些属性会用于runtime时候的check。这个里面有一个比较trick的手法是地方是通过成员和类型`定义了一些对每一个变量的描述和default，range等的属性。

至此完成了Symbol创建的源代码分析。




