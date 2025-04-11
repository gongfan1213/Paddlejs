### 第 4 章 CV项目实战
第 1 章~第 3 章介绍了 Web AI 的基本要素，以及前端推理引擎 Paddle.js 的运行环境和推理过程，还介绍了神经网络模型是如何在前端完成推理预测的。本章延续第 3 章的内容，具体讲解如何利用模型库 paddlejs - models 快速实现计算机视觉（Computer Vision，CV）项目。

Paddle.js 模型库 paddlejs - models 在 CV 任务上封装了丰富的模型 SDK，覆盖图像分类、人像分割、关键点检测和文字识别等任务。本章首先从模型库讲起，然后介绍如何利用这些模型库实现经典的 CV 项目，最后介绍在小程序上如何实现 AI 效果。

#### 4.1 paddlejs - models 模型库
paddlejs - models 模型库是基于前端推理引擎 Paddle.js 面向前端工程师的 AI 功能解决方案，旨在提供开箱即用的 AI 功能，使产品快速接入 AI 功能；前端工程师还可以基于 SDK 进行二次开发，实现更多贴合业务场景的效果。目前，已经开源的模型 SDK 如表 4 - 1 所示。
| SDK | npm包 | 提供的AI功能 |
| ---- | ---- | ---- |
| mobilenet | @paddlejs - models/mobilenet | 1000种物品分类 |
| humanseg | @paddlejs - models/humanseg | 人像分割 |
| gesture | @paddlejs - models/gesture | 手势识别 |
| ocr | @paddlejs - models/ocr | 文字识别 |
| tinyYolo | @paddlejs - models/tinyYolo | 人脸检测（轻量模型） |
| facedetect | @paddlejs - models/facedetect | 人脸检测（适合多人像） |

##### 4.1.1 backend选择
paddlejs - models 模型库提供封装好的模型 SDK，不同的 SDK 引入不同的模型文件并提供不同的 AI 功能。每个 SDK 都集成 Paddle.js 核心框架（paddlejs - core）和合适的计算方案（backend），同时封装了模型前处理和通用的模型后处理，提供简单、易用的 API 接口供前端工程师调用。不同的模型 SDK 会根据模型结构和计算需求选择合适的计算方案。

其中，webgl - backend 支持 WebGL 1.0 和 WebGL 2.0，会根据当前设备对 WebGL 的支持情况自动切换版本；在 WebGPU 正式被浏览器支持之前，webgl - backend 仍是功能最强大的计算方案。经测试，大多数模型在 WebGL 上都能有较好的推理性能，但是由于 WebGL 编译着色器程序和将数据上传至纹理需要时间开销，因此对于一些超轻量模型来说，在 wasm - backend 上运行可能会获得更优的推理性能。表 4 - 2 显示了在 MacBook Pro(16inch,2019,A2141)Chrome 浏览器（版本为 95.0.4638.69）上，人脸模型（face，包含人脸检测 + 人脸关键点模型）、人像分割模型（humanseg）、物品分类模型（mobilenet）分别在 WebGL 2.0 和 WebAssembly 上的推理耗时。
| SDK | 模型文件大小 | WebGL 2.0耗时/ms | WebAssembly耗时/ms |
| ---- | ---- | ---- | ---- |
| mobilenet | 120.12KB + 13.31MB | 14.975 | 133.388 |
| humanseg | 127KB + 505KB | 21.49 | 180.25 |
| face | 227KB + 224KB/457KB + 2.2MB | 33.14 | 29.88 |

##### 4.1.2 引入模型library
在讲解如何引入模型 library 之前，先介绍 paddlejs - models 模型库中的每个模型是如何编译打包产出 SDK 的。
不同模型的编译打包的方式都是统一的，采用 Webpack 作为打包工具。这里以 mobilenet 为例，分析它的 Webpack 配置。
```javascript
/**
 * @file mobilenet webpack config
 */
const path = require('path');

module.exports = {
    mode: 'production',
    entry: {
        index: [path.resolve(__dirname, './src/index')]
    },
    resolve: {
        extensions: ['.ts']
    },
    module: {
        rules: [
            {
                test: /\.ts$/,
                loader: 'ts-loader',
                exclude: /node_modules/
            }
        ]
    },
    output: {
        filename: '[name].js',
        path: path.resolve(__dirname, 'lib'),
        globalObject: 'this',
        libraryTarget: 'umd',
        library: ['paddlejs','mobilenet'],
        publicPath: '/'
    }
};
```
其中，重点关注 output 打包配置项。配置项 libraryTarget:'umd'表示允许 mobilenet 以 CommonJS、AMD 方式加载或作为全局变量被引用。配置项 library:['paddlejs','mobilenet']表示可以通过以下方式获取全局变量。
```javascript
// 获取mobilenet
const mobilenet = paddlejs['mobilenet'];
```
可以看到，这里将 SDK 挂载到全局变量 paddlejs 下，是为了将所有模型的全局变量收敛到一个入口。其他模型的引入方式如下。
```javascript
// 获取humanseg
const humanseg = paddlejs['humanseg'];
// 获取ocr
const ocr = paddlejs['ocr'];
```

#### 4.2 经典CV模型实战
计算机视觉是一门让机器如何去 “看” 的学科，更进一步地说，就是让机器能够理解人类视觉系统完成的任务，从数字图像或视频中获取有效信息。计算机视觉已经在各领域得到广泛应用，如交通、安防、金融和医疗等。

接下来，从三个经典的 CV 项目（图像分类、图像分割和目标检测）出发，介绍如何封装一个模型 SDK，以及如何快速使用 SDK 实现 AI 效果。

##### 4.2.1 图像分类
图像分类（Image Classification）是根据图像的语义信息对不同类别的图像进行区分的，是计算机视觉的核心，更是物体检测、图像分割、物体跟踪、行为分析和人脸识别等其他高层次视觉任务的基础。图像分类在许多领域都有着广泛的应用，如安防领域的人脸识别和智能视频分析，交通领域的交通场景识别，互联网领域的基于内容的图像检索和相册自动归类，医疗领域的图像识别等。最经典的深度学习入门案例——手写数字识别，就是一个典型的图像分类问题。

一般来说，图像分类首先通过手工特征或特征学习方法对整个图像进行全部描述，然后使用分类器判别物体类别，因此如何提取图像的特征至关重要。在深度学习算法出现之前，使用较多的方法是基于词袋（Bag of Words）模型的物体分类方法。而基于深度学习的图像分类方法可以通过监督学习或无监督学习的方式学习层次化的特征描述，从而取代手工设计或选择图像特征的工作。近年来，深度学习模型中的卷积神经网络（Convolutional Neural Networks，CNN）在图像领域取得了惊人的成绩，CNN 直接利用图像像素信息作为输入，最大限度地保留了输入图像的所有信息。通过卷积操作进行特征的提取和高层抽象，模型输出直接是图像识别的结果。这种基于 “输入—输出” 直接端到端的学习方法取得了非常好的效果，得到了广泛应用。

图像分类是计算机视觉里基础且重要的一个领域，其研究成果一直影响着计算机视觉甚至深度学习的发展。图像分类有很多子领域，如单标签图像分类、多标签图像分类、细粒度图像分类等，此处只对单标签图像分类进行简要的叙述。

1. **1000种物品分类**
1000 种物品分类模型是基于 ImageNet - 1k 数据集训练出来的，有 1000 种类别，如 banana、pizza、cup 等。使用 CNN 实现图像分类的过程如图 4 - 4 所示。

@paddlejs - models/mobilenet 是对该模型分类功能的封装，暴露了两个 API——load 和 classify，调用过程如下。
```javascript
// 引入mobilenet
import * as mobilenet from '@paddlejs - models/mobilenet';

// 模型加载，初始化SDK
await mobilenet.load();

// 传入图像，获取图像分类结果
const res = await mobilenet.classify(img);
```
① 引入 mobilenet SDK。
② 调用 load 接口并完成初始化。具体实现如下。
```javascript
/**
 * 初始化SDK，加载模型，引擎初始化
 * @param {RunnerConfig} [config] SDK配置，可选，若不选，则表示使用imgNet1K分类模型配置，且在初始化期间完成预热
 */
export async function load(config?: RunnerConfig) {
    // 注册runner，imgNet1K为1000种物品分类模型配置
    runner = new Runner(config || imgNet1K);

    // 设置webgl_feed_process为true，将图像前处理过程在GPU上完成，提高图像前处理速度
    env.set('webgl_feed_process', true);
    // 设置webgl_pack_channel为true，conv2d开启计算向量化，提高推理性能
    env.set('webgl_pack_channel', true);

    // runner初始化
    await runner.init();
}
```
③ 调用 classify 接口完成推理。具体实现如下。
```javascript
/**
 * 获取输入图像的分类结果
 * @param {HTMLImageElement | HTMLVideoElement | HTMLCanvasElement} image 输入图像
 * @param {string[] | MobilenetMap} [map] 分类信息列表，可选
 * @return {string} result 分类结果，类别名
 */
export async function classify(image: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement, map?: string[] | MobilenetMap): string {
    // 传入图像，获取推理结果，推理结果表示图像成为1000种分类中的任何一种分类的可能性
    const res = await runner.predict(image);
    // 找到可能性最大的分类索引值
    const maxItemIndex = getMaxItemIndex(res);
    const curMap = map || imgNet1kMap;
    // 得到索引对应的分类名称
    const result = curMap[`${maxItemIndex}`];
    return result;
}
```

2. **任意图像分类模型**
对于图像分类应用来说，模型输入是图像资源，输出结果表示图像成为所有分类中的任何一种分类的可能性，其中最大输出值的索引就是模型推理的答案。正因如此，开发者可以提供自己训练好的模型和对应的分类来完成初始化，实现想要的分类效果，代码如下。
```javascript
// 开发者提供分类模型的地址，模型文件需要符合Paddle.js模型文件格式，包含引入的权重文件信息
// 模型参数mean和std
const customConfig = {
    path,
    mean,
    std
} as RunnerConfig;

// 使用开发者自己训练的模型配置
await mobilenet.load(customConfig);

// 传入图像，获取图像分类结果，传入对应的分类文件map
const res = await mobilenet.classify(img, map);
``` 
