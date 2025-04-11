### 3.4 使用 Paddle.js
前面介绍了 Paddle.js 的核心模块和工作原理，本节将根据具体的代码介绍如何使用 Paddle.js，在这个过程中可以对 Paddle.js 的设计有大致的了解。以一个简单的两层神经卷积网络为例，模型文件如下。

```json
{
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
                "Scale_in": 1.0,
                "Scale_in_eltwise": 1.0,
                "Scale_out": 1.0,
                "Scale_weights": [1.0],
                "data_format": "AnyLayout",
                "dilations": [1, 1],
                "exhaustive_search": false,
                "force_fp32_output": false,
                "fuse_relu": false,
                "fuse_relu_before_depthwise_conv": false,
                "fuse_residual_connection": false,
                "groups": 1,
                "paddings": [0, 0],
                "strides": [1, 1]
            },
            "inputs": {
                "Filter": ["conv2d_0.w_0"],
                "Input": ["image"]
            },
            "outputs": {
                "Output": ["conv2d_0.tmp_0"]
            },
            "type": "conv2d"
        },
        {
            "attrs": {
                "Scale_in": 1.0,
                "Scale_in_eltwise": 1.0,
                "Scale_out": 1.0,
                "Scale_weights": [1.0],
                "data_format": "AnyLayout",
                "dilations": [1, 1],
                "exhaustive_search": false,
                "force_fp32_output": false,
                "fuse_relu": false,
                "fuse_relu_before_depthwise_conv": false,
                "fuse_residual_connection": false,
                "groups": 1,
                "paddings": [0, 0],
                "strides": [1, 1]
            },
            "inputs": {
                "Filter": ["conv2d_1.w_0"],
                "Input": ["iconv2d_0.tmp_0"]
            },
            "outputs": {
                "Output": ["conv2d_1.tmp_0"]
            },
            "type": "conv2d"
        },
        {
            "attrs": {},
            "inputs": {
                "X": ["conv2d_1.tmp_0"]
            },
            "outputs": {
                "Out": ["fetch"]
            },
            "type": "fetch"
        }
    ],
    "vars": [
        {
            "name": "image",
            "persistable": false,
            "shape": [1, 3, 3, 5]
        },
        {
            "name": "conv2d_0.tmp_0",
            "persistable": false,
            "shape": [1, 1, 2, 4]
        },
        {
            "name": "conv2d_0.w_0",
            "persistable": true,
            "shape": [1, 3, 2, 2]
        },
        {
            "name": "conv2d_1.w_0",
            "persistable": false,
            "shape": [1, 1, 2, 2]
        },
        {
            "name": "conv2d_1.tmp_0",
            "persistable": false,
            "shape": [1, 1, 1, 3]
        }
    ]
}
```

下面根据核心代码讲解 Paddle.js 执行推理的具体过程。

```javascript
import { Runner } from '@paddlejs/paddlejs-core';
import '@paddlejs/paddlejs-backend-webgl';

interface ModelConfig {
    path: string;
    feedShape: {
        fc: number;
        fw: number;
        fh: number;
    };
    mean?: number[];
    std?: number[];
    needPreheat?: boolean;
}

const modelConfig = {
    path: modelPath,
    feedShape: {
        fc: 3,
        fw: 3,
        fh: 5
    },
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
    needPreheat: true
} as ModelConfig;

// 模型加载和初始化
async function loadModel(modelConfig: modelConfig) {
    const runner = new Runner(modelConfig);
    await runner.init();
}

// 传入媒体资源进行推理计算
async function predict(input: HTMLImageElement | HTMLCanvasElement) {
    return await runner.predict(input);
}
```

① 引入核心框架@paddlejs/paddlejs-core，获取推理引擎的核心调度器——Runner 类。Paddle.js 在 Runner 类中封装了一些用于完成推理预测的重要模块，具体如下。
- Loader 模块，负责模型加载器。
- Graph 模块，负责生成模型的拓扑网络结构。 
- OpExecutor 模块，算子生成器，用于封装算子，方便在推理的过程中对其进行调用。 
- MediaProcessor 模块，负责模型的前处理工作。 

以上模块都作为 class 类在必要的时候被初始化并交由 Runner 调度，顾名思义，这个类像 “流水线” 一样，串起了整个 Paddle.js 的运行环境，其上的实例方法也对应了推理过程的重要步骤。

```javascript
import Loader from './loader';
import Graph from './graph';
...
import type OpExecutor from './opFactory/opExecutor';
import MediaProcessor from './mediaProcessor';
...

class Runner {
    // 根据 ModelConfig 初始化
    constructor(options: ModelConfig | null) {
    }

    // 模型初始化
    async init() {
        // 初始化计算方案
        if (!GLOBALS.backendInstance) {
            console.error('ERROR: Haven\'t register backend');
            return;
        }
        await GLOBALS.backendInstance.init();

        this.isExecuted = false;

        // 加载模型
        await this.load();

        // 生成模型输入数据
        this.genFeedData();

        // 生成拓扑结构
        this.genGraph();

        // 结构化算子信息
        this.genOpData();

        // 根据配置决定是否进行模型预热
        if (this.needPreheat) {
            return await this.preheat();
        }
    }

    // 加载模型，在模型初始化时被调用
    async load() {
    }

    // 生成模型输入数据，在模型初始化时被调用
    genFeedData() {
    }

    // 生成拓扑结构，在模型初始化时被调用
    genGraph() {
    }

    // 结构化算子信息，在模型初始化时被调用
    genOpData() {
    }

    // 模型预热，在模型初始化时可能被调用
    async preheat() {
    }

    // 判断模型是否已加载
    async checkModelLoaded() {
    }

    // 执行推理
    async predict(media, callback?: Function) {
    }

    // 针对特定输入数据执行推理
    async predictWithFeed(data: number[] | InputFeed[] | ImageData, callback?, shape?: number[]) {
    }

    // 执行 Op
    executeOp(op: OpExecutor) {
    }

    // 读取模型推理结果
    async read() {
    }
   ...
}
```

从以上代码可以清晰地看出，在模型推理的关键节点——从初始化到推理再到产出结果，Runner 类都起到了提纲挈领的作用。

② 引入 WebGL 计算方案@paddlejs/paddlejs-backend-webgl。

如 3.1.2 节所述，目前 Paddle.js 的计算方案还支持 WebGPU、WASM（WebAssembly）、PlainJS（纯 JavaScript 版本）及 NodeGL。本例使用 WebGL 计算方案。引入该 npm 包后，Paddle.js 自动完成计算方案注册和算子 Op 注册，利用 GPU 


加速进行模型推理。
③ 生成一个调度器 Runner 实例，需要传入以下模型配置参数。
- path 为模型文件地址，可以为网络路径或本地路径，模型结构文件与参数文件需要放在同一个目录里。
- feedShape 为模型支持的输入 Tensor shape，Runner 类根据该参数将输入媒体数据进行尺寸调整、数据填充等操作，将输入处理为模型所需的 Tensor shape。 
- mean 和 std 分别为平均值和标准差，用于处理输入媒体数据。 
- needPreheat 参数决定是否进行预热过程，true 为执行，false 为不执行。

④ 初始化过程。参照 Runner 类实现，该过程包括初始化计算方案、加载模型、生成神经网络拓扑、结构化算子信息和预热过程，下面分别进行说明。
- 初始化计算方案。在本例中，创建 WebGL 环境并完成相关 WebGL 参数的配置。
- 加载模型。Loader 模块根据模型地址下载模型结构文件 model.json 和模型参数文件 chunk.dat，并生成一个模型信息对象，代码如下。
```typescript
interface Model {
    ops: ModelOp[];
    vars: ModelVar[];
}
```
其中，ops 存储 model.json 中的模型结构信息，vars 存储 chunk.dat 中的权重数据。

- 生成神经网络拓扑。Graph 模块根据模型信息对象生成神经网络拓扑结构 weightMap。
    - 首先遍历模型结构信息 ops，使用算子生成器 OpExecutor 初始化每个 Op，生成 Op 对象。此时 Op 对象包含 id 和 next 属性，id 属性具有唯一性，作为当前 Op 索引；next 属性为下一个执行的 Op id，此时值为空字符串。
    - 然后根据模型结构信息中的 inputs 和 outputs 完成拓扑排序，所以本例中的执行顺序为 feed、conv2d、conv2d 和 fetch。
- 结构化算子信息。主要生成输入 Tensor 对象 inputTensors、输出 Tensor 对象 outputTensors 和算法程序 program。输入和输出 Tensor 对象主要包括 Tensor 数据维度和二进制数据。program 为 Op 的算法程序，不同的计算方案的算法程序实现不同，WebGL 计算方案通过着色器（Vertex/Fragment/Compute Shader）实现。
- 预热过程。needPreheat 根据模型配置参数决定是否执行此过程。
    - 如果执行，则生成一个与 feedShape 模型维度一致的 Tensor 数据——值全为 1.0 的二进制浮点数据，作为模型的输入 Tensor 数据。
    - feed 算子和 fetch 算子并不执行，只是作为模型的输入和输出标识，通过 feed 算子的 next 属性找到下一个算子（本例中为 conv2d 算子）开始计算，若遇到 fetch 对象算子，则直接返回。
    - 在预热过程中，完成模型权重数据缓存，WebGL 计算方案根据输入和输出 Tensor 对象的维度信息和权重数据生成纹理并缓存，这样在推理过程就不需要再上传数据，直接使用缓存即可，大大提高了推理速度。

⑤ 与预热过程不同的是，推理过程使用传入的媒体资源数据完成推理计算。推理前使用模型前处理模块（MediaProcessor）将媒体资源转换为 ImageData 对象，并根据 feedShape 信息进行尺寸调整和数据填充等操作，最终产生符合模型输入条件的 Tensor 数据。

以上的五个步骤是 Paddle.js 运行推理执行的几个重要环节，开发者只需要先引入核心框架@paddlejs/paddlejs-core 和合适的计算方案，再根据模型配置参数生成 Runner 实例，调用 init、predict 即可完成初始化和预测工作。

如果想要更加简单、快捷地使用 Paddle.js 实现 AI 效果，那么可以使用模型库 paddlejs-models。该库提供了多个模型软件开发工具包（Software Development Kit，SDK），如手势检测、人像分割、图像识别、文字识别等，开发者不需要再传入模型配置信息，只需要简单地调用 API 即可实现落地效果，具体如何使用将在第 4 章详细介绍。

### 3.5 总结
本章介绍了 Paddle.js 在 AI 全链路中的具体位置，并且根据具体代码讲解推理模型的结构和 Paddle.js 的执行机制，总结如下。
- Paddle.js 在 AI 全链路中是一种端侧（当然，可以利用 Node.js 使其运行在服务端）的推理引擎，一般帮具体的业务研发工程师完成从原始模型到最终业务结果的开发工作。 
- Paddle.js 的推理模型信息结构化地存储为 JSON 数据，使用 Netron 等可视化工具可以查看模型的具体结构信息。 
- 推理过程分为初始化和推理两个阶段，在数据处理和产出推理结果过程中，涉及前处理和后处理操作，主要用来修改模型推理的输入/输出数据。 
- Paddle.js 支持在浏览器和 Node.js 中运行，支持 WebGL、WebGPU、WASM、NodeGL 和 PlainJS 等计算方案，开发者可根据需要自行选择。 
- 在执行推理过程中，开发者只需要通过 Paddle.js 的核心模块 paddlejs-core 完成初始化和推理工作，整个流程只需要简单的五步。为了让整个过程更快捷，还可以使用模型库 paddlejs-models 来加速这一过程。 
