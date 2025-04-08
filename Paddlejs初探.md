### 第3章 Paddle.js初探
第1章和第2章介绍的基础知识，通常在一本与深度学习及前端入门相关的书籍中都可以找到对应的内容。本书虽然并未涉及过多或太深的内容，但已足够支撑接下来的学习内容。

从本章开始，将正式揭开前端推理引擎的面纱。本章将会以在第1章和第2章提到的Paddle.js为例，介绍什么是AI全链路、Paddle.js在其上的位置和作用，以及Paddle.js的工作原理和基本操作方法。

#### 3.1 AI全链路
全链路实际上是一个极为抽象的概念，它并不只存在于软件领域，在设计、物流等行业中也同样存在。

本书提及的AI全链路，指的是在人工智能（AI）领域，从需求到最终产品交付的全部流程，完整地涵盖了从模型产出到最终业务集成的各个环节。

之所以要把AI全链路的概念明确提出来，而不是故弄玄虚地引出一个名词，是因为想强调Paddle.js作为前端推理引擎，在AI生态中所处的具体位置，从而更好地使读者了解其所能解决的具体问题。

##### 3.1.1 AI全链路基本介绍
完整的AI全链路主要分为上游和下游两部分。
- 上游主要负责模型产出，涵盖数据处理、算法设计、模型训练/评估模块。
- 下游主要负责模型转化与优化，涵盖模型转化、模型部署、推理预测、业务调用/监控模块，如图3-1所示。

数据处理->算法涉及->模型训练和评估->模型转换->模型部署->推理预测->业务调用和监控


### 图3-1 AI全链路
算法工程师：数据处理→算法设计→模型训练/评估（模型产出）
研发工程师：模型转化→模型部署→推理预测→业务调用监控（模型转化与优化）

上游的模型产出部分主要由算法工程师负责，下游的模型转化与优化部分主要由业务落地的研发工程师负责。当然，这样的分工模式只适用于一般情况，这样明显的区分是为了强调二者的工作内容有何不同。

无论是在模型产出的链路上游，还是在模型转化与优化的链路下游，除了最终的业务调用，传统的模型部署环境都维护在云端的服务器上。因为Paddle.js提供了可在Web平台上进行推理计算的运行环境，链路的下游部分（特别是在模型部署、推理预测环节）得以在浏览器等Web环境中运行。

所以，可以根据最终采用的技术方案，把运行环境分为服务侧（Server Side）和端侧（Client Side），Paddle.js就是一种端侧的推理引擎。

由于部署和执行环境的不同，推理过程依赖的模型格式也不尽相同，因此需要通过模型转化模块输出对应平台所能支持的格式。推理预测模块还要根据业务平台特性选择对应的推理引擎，Paddle.js作为一种端侧的推理引擎，目前已支持浏览器和Node.js两种运行时环境。

##### 3.1.2 前端推理引擎Paddle.js
Paddle.js于2021年年初发布了2.0版本，从单包单仓库升级为Monorepo（多包单仓库），其核心库如图3-2所示。

### 图3-2 Paddle.js核心库
Paddle.js：
- paddlejs-examples
- paddlejs-models
- paddlejs-mediapipe
- e2e
- benchmark
- paddlejs-extension
- paddlejs-core：
- paddlejs-backend-x：WebGL、WebGPU、WebAssembly、NodeGL、PlainJS
- paddlejs-converter

提示：Monorepo是一种代码管理方式，在这种方式下会摒弃传统的一个模块（Module）对应一个代码仓库（repository，repo）的方式，取而代之的是把所有模块放在一个代码仓库中管理。简而言之，就是用一个大的Git仓库管理所有代码。

目前，像React等开源项目都在使用类似的代码管理方式，而Monorepo的解决方案也层出不穷，如Yarn Workspace和Lerna。

Paddle.js采用Lerna进行多模块管理，包括代码维护和模块发布，这也是2.x版本和1.x版本的区别之一。感兴趣的读者可以通过Paddle.js源码深入了解。

图3-2列出了Paddle.js的核心模块，在下文会详细介绍，这里只需要知道Paddle.js是基于多模块的前端AI框架。

根据执行阶段和环境的不同，Paddle.js核心库分为离线模块和在线模块。离线模块主要负责离线模型处理，在线模块主要负责神经网络计算推理。

1. **离线模块**
拿到上游链路产出的模型之后，需要先对模型进行处理，因为在Web平台中需要使用浏览器支持的模型格式。

Paddle.js仓库中的paddlejs-converter模块是一个模型转换器，可以对模型进行格式转换和离线优化。优化方法主要包括算子融合、数据清理和模型量化及低精度应用。
- **算子融合**：可以将多个连续算子融合成单个等效算子，减少数据交换并简化图结构，加快推理速度。
- **数据清理**：清理无用的属性参数，减小模型体积。
- **模型量化及低精度应用**：支持将FP32精度模型量化为INT8模型和INT16模型，减小模型体积，加快推理速度。INT8模型精度降低，模型体积减小为原体积的1/4。将FP32精度模型权重转换为FP16半精度浮点数，可减小一半的模型体积。

提示：量化是指以低于浮点精度的比特宽度执行计算和存储张量。
浮点数精度（Float Precision，FP），FP32即单精度，是计算机常用的一种数据类型；FP16即半精度，是一种低精度类型。与计算中常用的单精度和双精度类型相比，FP16更适合在精度要求不高的场景中使用，且因为占用字节数更少，采用FP16可以提升计算吞吐量。

2. **在线模块**
paddlejs-core是前端推理引擎的核心部分，负责计算方案的注册和整个引擎推理流程的调度。

paddlejs-backend-x为Paddle.js的多个计算方案，目前支持WebGL、WebGPU、WebAssembly（WASM）、PlainJS（纯JavaScript版本）及NodeGL。

WebGL、WebGPU和NodeGL属于GPU计算方案，将数据存储为纹理，算子通过着色器（Vertex/Fragment/Compute Shader）实现，可以利用GPU并行计算的特性；而WASM和PlainJS属于CPU计算方案。不同的计算方案由于模型依赖、兼容性和性能等的不同，各有优缺点。本书第8章（计算方案） 

会详细介绍各种计算方案，使用者可根据具体场景选择合适的计算方案。

提示：WebAssembly（WASM）是一种使非JavaScript代码在浏览器中运行的方法。这些代码可以是C、C++或Rust等，将它们编译之后可以部署到浏览器，并且以二进制文件的方式，在CPU上以接近原生的速度运行，同时可以直接在JavaScript中将它们当成模块使用。 

Paddle.js同时提供封装好的模型工具库paddlejs - models，针对不同模型提供个性化的API封装，为前端工程师提供开箱即用的编程体验。 

对于资深开发人员，Paddle.js还提供了周边的工具库，如e2e（端到端测试）、benchmark（评估指标）等。如果想要了解模型推理的具体性能情况，则可使用paddlejs - benchmark产出性能测试报告，获取模型及算子的推理耗时等数据。

### 3.2 模型和神经网络拓扑结构
前端推理引擎所要实现的最重要的功能就是根据模型信息还原神经网络的拓扑结构，这里隐含有以下两点信息。
一是用规范的格式描述在第2章中所提及的网络结构和权重信息；二是需要针对前端推理引擎运行时环境，对原始模型格式进行处理。细心的读者可以发现，模型在这里拥有两种概念，一种是原始模型，另一种是推理模型。后文中如无特殊说明，提及的“模型”指的都是推理模型。

本节将会对两种模型转换和推理模型结构进行说明。

#### 3.2.1 模型结构文件与参数文件
Paddle.js不仅支持PaddlePaddle模型，还支持Caffe、TensorFlow和ONNX模型，只需要通过X2Paddle转换成Paddle支持的模型即可。 

TensorFlow模型转换命令如下：
```bash
# convert tensorflow
pip install x2paddle
x2paddle --framework=tensorflow --model=tf_model.pb --save_dir=pd_model
```
X2Paddle转换参数如表3 - 1所示。
### 表3 - 1 X2Paddle转换参数
| 参数 | 作用 |
| ---- | ---- |
| framework | 源模型类型（TensorFlow、Caffe、ONNX） |
| save_dir | 指定转换后的模型保存目录路径 |
| model | 当framework为TensorFlow/ONNX模型时，该参数指定TensorFlow模型的pb文件或ONNX模型的路径 |

目前，PaddlePaddle 2.x的模型结构和模型参数格式为.pdmodel和.pdiparams，Web平台并不支持这两个格式，需要经过paddlejs - converter转换为model.json和chunk.dat。其中，model.json为模型结构文件，用于存储模型的拓扑结构，包括模型中所有算子Op（Operator）的运算顺序和各个Op的详细信息；chunk.dat为模型参数文件，用于存储模型的权重数据。

#### 3.2.2 神经网络拓扑结构
执行前端引擎推理的第一步是要构建神经网络的拓扑结构。网络拓扑结构被表示为由神经网络层构成的有向无环图（Directed Acyclic Graph，DAG）。在有向无环图中，顶点表示模型算子，边表示算子的运算顺序。为了还原正确的执行顺序，Paddle.js在加载模型后，会根据模型结构信息进行拓扑排序。

1. **模型结构信息**
3.2.1节提到，model.json是用于存储模型的拓扑结构的，下面具体分析模型结构信息的内容。
```json
{
    "chunkNum": 2,
    "ops": [
        {
            "attrs": {},
            "inputs": {
                "X": ["feed"]
            },
            "outputs": {
                "Out": ["image"]
            },
            "type": "feed"
        },
        {
            "attrs": {
                "data_format": "NCHW",
                "dilations": [1, 1],
                "groups": 1,
                "paddings": [1, 1],
                "strides": [2, 2]
            },
            "inputs": {
                "Filter": ["conv2d_0.w_0"],
                "Input": ["image"]
            },
            "outputs": {
                "Output": ["conv2d_68.tmp_0"]
            },
            "type": "conv2d"
        },
        {
            "attrs": {
                "op_device": ""
            },
            "inputs": {
                "X": ["conv2d_68.tmp_0"]
            },
            "outputs": {
                "Out": ["relu_1.tmp_0"]
            },
            "type": "relu"
        },
        {
            "attrs": {
                "data_type": 1
            },
            "inputs": {
                "X": ["save_infer_model/scale_0.tmp_1"]
            },
            "outputs": {
                "Out": ["fetch"]
            },
            "type": "fetch"
        }
    ],
    "vars": [
        {
            "name": "conv2d_0.w_0",
            "persistable": true,
            "shape": [36, 3, 3, 3]
        },
        {
            "name": "conv2d_68.tmp_0",
            "persistable": false,
            "shape": [1, 36, 80, 144]
        }
    ]
}
```
可以看到，模型结构信息比较复杂，这里针对最外层的chunkNum、vars和ops分别进行说明。 

chunkNum代表参数文件个数，如果值为2，则表示参数文件为chunk_1.dat和chunk_2.dat。 

vars存储模型中的Tensor信息，每个Tensor都使用name提供具有唯一性的id。若persistable为true，则表示存储的Tensor为常量参数，参数数据存储在chunk.dat文件中；若persistable为false，则表示存储的Tensor为变量参数。shape定义数据 

个维度的大小。

ops存储模型中的算子信息，决定了模型的网络拓扑结构。ops中的Op个数代表了网络结构的层数，Op信息中的attrs描述了Op的输入参数。inputs表示输入张量，维度为[N, C, H, W] 的4维张量。对应的格式为NCHW，其中N是batch的大小，C是通道数，H是特征的高度，W是特征的宽度，数据类型为Float32或Float16。网络输入层Op为feed，输出层为fetch，根据Op的inputs和outputs信息可以找到当前Op的前后关联Op，进而生成网络拓扑图结构。

2. **模型结构可视化**
可以使用Netron或EasyAI Workbench实现模型结构可视化，方便分析模型，如图3-3所示。

### 图3-3 通过EasyAI Workbench可视化模型结构
（书中对应图片展示了相关界面）

（1）**Netron安装及使用**
对于Windows系统，可通过访问Netron项目的GitHub仓库，在已发布的版本中下载.exe文件，或者在终端执行winget install netron命令。
对于macOS系统，可通过访问Netron项目的GitHub仓库，在已发布的版本中下载.dmg文件，或者在终端执行brew install netron命令。
下载完文件后，打开Netron应用程序，将原始模型文件拖曳到Netron界面，即可查看模型结构。

（2）**工具**
EasyAI Workbench是一个AI工作台，如图3-4所示，里面内置了模型查看器、模型转换器、模型精度对齐等工具，其中通过定制化修改Netron实现Paddle.js和Paddle Lite模型可视化。可以在Paddle.js GitHub仓库下载此工具。

### 图3-4 EasyAI Workbench
（书中对应图片展示了相关界面）

### 3.3 推理过程与运行环境
下面介绍模型推理过程和模型所依赖的运行环境。
从用户数据到推理结果，Paddle.js的执行过程如图3-5所示。

### 图3-5 Paddle.js的执行过程
用户数据（图像、视频流、文字、语音）
↓
paddlejs - mediapipe模型前处理
↓
paddlejs - core（初始化、推理，通过API与paddlejs - backend - x交互）
  └─paddlejs - backend - x（WebGL、WebGPU、WebAssembly、NodeGL、PlainJS）
↓
paddlejs - mediapipe模型后处理
↓
推理结果
↑
paddlejs - models（gesture、humanseg、mobilenet、ocr等）

注意图3-5中的灰色背景部分，Paddle.js通过paddlejs - core模块封装推理过程，并通过paddlejs - backend - x模块维护计算方案供paddlejs - core注册使用，而具体使用哪一种计算方案，与Paddle.js的运行环境有关。

#### 3.3.1 推理过程
推理过程大致分为初始化和推理两个阶段。

在模型初始化过程中，首先加载推理模型并生成神经网络拓扑图，然后通过算子生成器根据用户注册的计算方案，将神经网络每一层算子生成对应的可执行单元。
在初始化的最后一步，引擎会默认执行预热过程，即传递与模型输入shape相同的Tensor——值全为1.0的二进制浮点数据，并在神经网络中完成推理计算。

提示：预热过程与使用真实数据进行推理预测的过程不同，它通过传递与模型输入shape相同的Tensor进行一次推理计算。

在预热过程中，完成权重上传和数据缓存，如果是GPU计算方案，则会进行着色器编译并缓存，以便加速真实用户数据的推理计算过程。后续推理耗时将会远小于预热耗时。
关于模型初始化的过程，这里只大致介绍，3.4节会通过代码详细说明。

在完成初始化过程后，就可以输入真实数据了。Paddle.js目前支持的用户输入包括图像及视频流。在直接使用这些数据之前，要进行数据的预处理工作，包含对图像进行拉伸、像素填充、数据归一化等操作，从而将用户输入转换成二进制浮点数据，变为符合模型要求的输入——Tensor。
这种预处理工作称为模型推理的前处理工作。

在完成初始化并通过前处理工作修改了原始数据后，就可以通过神经网络推理计算了。通过paddlejs - backend - x模块，在用户注册的backend环境中，通过神经网络层计算得出推理结果。

最后进行后处理，即把推理结果转化为业务侧所需要的数据。
在推理过程的前处理和后处理中包含着大量的计算工作，用户可使用Paddle.js提供的paddlejs - mediapipe模块来简化这些工作。

#### 3.3.2 运行环境
Paddle.js可以在浏览器和Node.js中运行。
运行环境与计算方案紧密相关，在浏览器中可以使用WebGL、WebGPU、WASM和PlainJS计算方案完成推理。截至本书完稿之时，主流浏览器（97.93%）都支持WebGL 1.0，WebGL 2.0的兼容性较低（77.29%），WebAssembly的兼容性为93.93%，在大部分浏览器中可以获得GPU/CPU加速；在Node.js环境中可以使用NodeGL、WASM和PlainJS计算方案完成推理。
根据终端性能、模型复杂度及想要兼容覆盖的范围，开发者可以作出最终的选择。3.4节将通过代码展示如何在模型初始化时选择计算方案。

![image](https://github.com/user-attachments/assets/c5c1e3e9-e44d-4758-9e5a-2949e7e82e7c)


### 3.4 使用Paddle.js
前面介绍了Paddle.js的核心模块和工作原理，本节将根据具体的代码介绍如何使用Paddle.js，在这个过程中可以对Paddle.js的设计有大致的了解。以一个简单的两层神经卷积网络为例，模型文件如下。 （此处未给出具体模型文件代码内容，按原文保留） 
