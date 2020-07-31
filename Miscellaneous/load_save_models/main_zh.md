# 由报错"Missing key(s) in state_dict"展开：如何合理的使用torch.save和torch.load

## 完整的报错信息

```shell
RuntimeError: Error(s) in loading state_dict for ResNet:
	Missing key(s) in state_dict: "conv1.weight", "bn1.weight", "bn1.bias", "bn1.running_mean", "bn1.running_var", ..., "fc.bias". 
	Unexpected key(s) in state_dict: "module.conv1.weight", "module.bn1.weight", "module.bn1.bias", "module.bn1.running_mean", "module.bn1.running_var", ..., "module.fc.bias". 
```

上面这条报错信息中间省略了网络的各个层，不过这不影响我们分析报错的原因，下面先理解报错的信息：
- `RuntimeError: Error(s) in loading state_dict for ResNet`：torch为`ResNet`这个对象读取模型权重`state_dict`时遇到错误。
- `Missing key(s) in state_dict: "conv1.weight", ...`：模型权重文件中并不存在诸如`conv1.weight`这样的key。
- `Unexpected key(s) in state_dict: "module.conv1.weight", ...`：模型权重文件中的key是类似`module.conv1.weight`的格式，但是我们定义的模型中并不存在这样格式的key。

当然，实际情况中，我们还可能会遇到与上面情况中的Missing key与Unexpected key调转的情况。不过问题发生和解决的原理是一致的。因此，下面先给出解决方法，然后我们再通过一个小实验来讨论讨论如何合理的使用torch.save和torch.load。

## 解决方法

上述的报错的原因是
1. 训练时，该model是经过`nn.DataParallel`方法包装的；
2. 保存训练好的model时，使用了`torch.save(model.state_dict())`而不是`torch.save(model.module.state_dict())`；
3. 读取训练好的model时，
    1. model没有经过`nn.DataParallel`方法包装，而用了`model.load_state_dict(torch.load('model.pth'))`来读取权重
    2. model经过`nn.DataParallel`方法包装，但用了`model.module.load_state_dict(torch.load('model.pth'))`来读取权重

清楚了报错原因之后，解决方法自然很明确了：
- 不改变现有的模型权重文件，将model经过`nn.DataParallel`方法包装，再用`model.load_state_dict(torch.load('model.pth'))`来读取模型权重。
- 如果不想用`nn.DataParallel`包装model，那么有两种方案：
    - 用上述先读取模型权重，再用`torch.save(model.module.state_dict, 'model_no_module.pth')`来保存新的key不含module的模型权重。
    - 用以下的代码去除key中`module.`：
        ```python
        state_dict = torch.load('model_with_module.pth')
        
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_k = k[7:]
            new_state_dict[new_k] = v
        print(new_state_dict.keys())

        model.load_state_dict(new_state_dict)
        ```

如果是Missing key和Unexpected key调转的情况下，只需要用`model.module.load_state_dict(torch.load('model.pth'))`就可以了。

下面的部分对第一部分提出的两个问题进行探讨（基于pytorch-1.5.1和torchvision-0.6.0）：
- 什么是`state_dict`?
- 为什么`state_dict`的key中没有`module`，而我们定义的model的key中有`module`？

## 什么是`state_dict`？

```python
import torch
import torchvision

model = torchvision.models.resnet.resnet18(pretrained=True)
# It is recommanded by pytorch official docs that we should save a model in the following way:
torch.save(model.state_dict())

print(type(model.state_dict()))
print(model.state_dict())
```
上面代码展示了`model.state_dict()`的一种使用情况。我们打印`model.state_dict()`的类型，得到的结果是`<class 'collections.OrderedDict'>`；而我们进一步打印`model.state_dict()`的内容，我们则可以得到一个key为各层参数名，value为各层参数值（即模型的权重weight和bias）的`OrderedDict`。因为每一个model所属的类都继承自`nn.Module`，我们可以追溯到pytorch（v1.5.1)的源码看看`nn.Module.state_dict()`是如何生成`OrderedDict`，位置在`$pytorch/torch/nn/modules/module.py`：

```python
def _save_to_state_dict(self, destination, prefix, keep_vars):
    r"""Saves module state to `destination` dictionary, containing a state
    of the module, but not its descendants. This is called on every
    submodule in :meth:`~torch.nn.Module.state_dict`.

    In rare cases, subclasses can achieve class-specific behavior by
    overriding this method with custom logic.

    Arguments:
        destination (dict): a dict where state will be stored
        prefix (str): the prefix for parameters and buffers used in this
            module
    """
    for name, param in self._parameters.items():
        if param is not None:
            destination[prefix + name] = param if keep_vars else param.detach()
    for name, buf in self._buffers.items():
        if buf is not None and name not in self._non_persistent_buffers_set:
            destination[prefix + name] = buf if keep_vars else buf.detach()

def state_dict(self, destination=None, prefix='', keep_vars=False):
        r"""Returns a dictionary containing a whole state of the module.

        Both parameters and persistent buffers (e.g. running averages) are
        included. Keys are corresponding parameter and buffer names.

        Returns:
            dict:
                a dictionary containing a whole state of the module

        Example::

            >>> module.state_dict().keys()
            ['bias', 'weight']

        """
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination._metadata[prefix[:-1]] = local_metadata = dict(version=self._version)
        self._save_to_state_dict(destination, prefix, keep_vars)
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + '.', keep_vars=keep_vars)
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, prefix, local_metadata)
            if hook_result is not None:
                destination = hook_result
        return destination
```

首先我们看`state_dict`函数。它主要分为四个部分：
- 第一个部分是`destination`的初始化。因为我们调用`model.state_dict()`的时候不带参数，所以第一次进入这个函数，`destination`就会被初始化。
- 第二个部分是往`destination._metadata`写入每一个部件版本号。
- 第三个部分是往`destination`通过`self._save_to_state_dict()`写入模型的权重。这里的for循环是通过递归的方式调用`self._save_to_state_dict()`将权重写入`destination`中。
- 第四个部分是往`destination`写入torch hook。

以`ResNet`为例，我们主要关注第三个部分。这里默认读者已经了解了`nn.Module`的定义。在第一次进入`state_dict`函数时，`self._modules.items()`包含了`ResNet`的所有层与自定义的`BasicBlock`。递归进入`state_dict`函数时，对于每一个例如`Conv2d`这些基础的层，`self._save_to_state_dict`都会被调用，从而把每一个基础层的`self._parameters`根据`prefix`正确写入到`destination`中；对而于每一个另外定义的块（类）例如`BasicBlock`，这个递归则会一直递归至块内定义的基础层再调用`self._save_to_state_dict`写入到`destination`。这样通过层层递归调用`self._save_to_state_dict`就把我们希望得到的模型权重以`OrderedDict`的形式写入完成了。

## 为什么有的`state_dict`的key中有`module`？

```python
import torch
import torchvision

model = torchvision.models.resnet.resnet18(pretrained=True)
# It is recommanded by pytorch official docs that we should save a model in the following way:
torch.save(model.state_dict())

print(type(model.state_dict()))
print(model.state_dict().keys())

parallel_model = torch.nn.DataParallel(model)
print(type(parallel_model.state_dict()))
print(parallel_model.state_dict().keys())
```

执行以上代码，可以发现，`print(model.state_dict())`打印出来的key是没有`module.`这一个部分的。那为什么有的`state_dict`会有`module.`这个部分呢？

原因是训练模型时，为了利用多GPU加速训练而使用了`nn.DataParallel`方法来包装model。我们观察`$torch/nn/parallel/data_parallel.py`的代码

```python
class DataParallel(Module):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallel, self).__init__()

        device_type = _get_available_device_type()
        if device_type is None:
            self.module = module
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = _get_all_device_indices()

        if output_device is None:
            output_device = device_ids[0]

        self.dim = dim
        self.module = module
        self.device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
        self.output_device = _get_device_index(output_device, True)
        self.src_device_obj = torch.device(device_type, self.device_ids[0])

        _check_balance(self.device_ids)

        if len(self.device_ids) == 1:
            self.module.to(self.src_device_obj)
```

可以发现，`DataParallel`的构造函数与model/module赋值相关的就只有`self.module=module`这一行。但是如果在这一行的下面添加一句打印`self.module`的内容的话，就可以发现其实`self.module`还是原来`module`的内容，即其key中没有`module.`这一字段。那`module.`字段是什么时候被添加到key中的呢？

可以注意到，`DataParallel`是继承了`Module`的，而`Module`类中有一个相关的重要方法`__setattr__`。关于`__setattr__`的详细介绍点击[这里](https://python-reference.readthedocs.io/en/latest/docs/dunderattr/setattr.html)。简单地说，`__setattr__(self, name, value)`这个方法会被每个类的构造函数中的赋值操作所触发，其中`name`的类型是`str`，即赋值目标的名字，例如`self.t=1`，那么`name='t'`；`value`则是所要赋的值，例如`self.t=1`，那么`value=1`。以下是`Module`的`__setattr__`方法的定义：

```python
def __setattr__(self, name: str, value: Union[Tensor, 'Module']) -> None:
        def remove_from(*dicts_or_sets):
            # ...

        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            # ...
        elif params is not None and name in params:
            # ...
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers, self._non_persistent_buffers_set)
                modules[name] = value
            # ...
```

考虑到篇幅，这里省略了不相关的代码。回到我们的疑问，`self.module=module`被执行的时候，`Module`的`__setattr__`被触发，其中`name='module', value=module`。这时，`type(module)`是`ResNet(nn.Module)`，它的`self._parameters`和`self._modules`都是一个空的`OrderedDict()`，那么我们就会进入第一层if-else的else中。由于`ResNet`是继承`Module`的，那么程序就会进入`if isinstance(value, Module):`这个条件中，继而执行到`modules[name]=value`这一行，此时相当于`self._modules['module']=ResNet()`被执行，从而给`ResNet`权重字典的key包装了一层module。

你可能还会有疑问，为什么`nn.DataParallel`要包装一层module？这里引用来自[PyTorch文档](https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html)的一段话：
> After wrapping a Module with DataParallel, the attributes of the module (e.g. custom methods) became inaccessible. This is because DataParallel defines a few new members, and allowing other attributes might lead to clashes in their names. For those who still want to access the attributes, a workaround is to use a subclass of DataParallel as below.

简而言之，原因是`nn.DataParallel`增加了新的类成员，如果不给model不包装一层的话，可能会导致类成员名字重复，因此`nn.DataParallel`就干脆把原来的model整一个包装在了`module`的下面。