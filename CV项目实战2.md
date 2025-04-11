### 4.2.2 图像分割
图像分割（Image Segmentation）是一种典型的计算机视觉任务，通过给出图像中每个像素点的标签，将图像分割成若干带类别标签的区块。图像分割是图像处理的重要组成部分，也是难点之一。随着人工智能（AI）的发展，图像分割技术已经在场景理解、医疗影像、人脸识别、机器人感知、视频监控和增强现实等领域获得了广泛的应用。

图像分割本质上是一种像素级别的图像分类任务，也就是对每个像素进行分类。图像分割通常分为语义分割（Semantic Segmentation）、实例分割（Instance Segmentation）、全景分割（Panoptic Segmentation）和人像分割（Portrait Segmentation）。

1. **语义分割**
把图像中每个像素赋予一个类别标签，如行人、汽车、建筑、地面、天空和树等。同一类别被标记为相同的颜色，语义分割只能区分类别，无法再区分同一类别下的不同实例。如图 4 - 5 所示，汽车类别都被标记为蓝色，无法区分停在一排的单辆汽车。

2. **实例分割**
实例分割包含了目标检测和语义分割的特点，只对图像中可区分类别的目标进行分割。相对目标检测的边界框，实例分割可以精确到物体的边缘信息；相对语义分割，实例分割并不需要对每个像素进行分类，只是对检测到的目标区域内的像素进行分类，并区分不同的实例。如图 4 - 6 所示，所有单辆汽车都被标记为不同的颜色。

3. **全景分割**
全景分割是语义分割和实例分割的结合，会对图像中每个像素进行分类，并且会区分同一类别的多个实例，用不同的颜色区分。如图 4 - 7 所示，对图像中每个像素都进行了分类，且相同类别的目标对象用不同的颜色区分了实例个体。

4. **人像分割**
人像分割是图像分割领域的经典应用，利用计算机视觉技术将人像从图像或视频流中分离出来，目前被广泛应用于虚拟背景、人像抠图美化、视频后期处理和视频会议背景替换等场景。

@paddlejs - models/humanseg 封装了 Paddle.js 核心库@paddlejs/paddlejs - core 和计算方案@paddlejs/paddlejs - backend - webgl，通过 WebGL 使 GPU 加速，采用 PaddleSeg 超轻量级人像分割模型 PP - HumanSeg - Lite，模型性能如表 4 - 3 所示。
| 模型名 | 输入尺寸 | 模型计算量 | 参数数量 | 推理耗时/ms | 原始模型大小/KB | 转换后模型大小/KB |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| PP - HumanSeg - Lite | 398像素×224像素 | 266M | 137K | 21.49 | 954 + 556 | 127 + 505 |
| PP - HumanSeg - Lite | 288像素×160像素 | 138M | 137K | 15.62 | 954 + 556 | 127 + 505 |

提示：测试环境使用 Paddle.js converter 优化图结构和参数裁剪，部署于 Web 端，显卡型号为 AMD Radeon Pro 5300M 4GB。模型大小为模型结构文件大小与参数文件大小总和。

@paddlejs - models/humanseg 暴露了四个 API：load、getSegValue、drawHumanSeg、blurBackground，调用过程如下。
```javascript
// 引入humanseg SDK
import * as humanseg from '@paddlejs - models/humanseg';

// 初始化SDK，下载398×224模型，默认执行预热
await humanseg.load();

// 获取分割结果
const segValue = await humanseg.getSegValue(img);

// 获取background canvas
const backCanvas = document.getElementById('background') as HTMLCanvasElement;
const destCanvas = document.getElementById('back') as HTMLCanvasElement;

// 背景替换，使用back_canvas作为新背景，实现背景替换
humanseg.drawHumanSeg(segValue, destCanvas, backCanvas);

const blurCanvas = document.getElementById('blur') as HTMLCanvasElement;
// 背景虚化
humanseg.blurBackground(segValue, blurCanvas);
```
① 引入 humanseg SDK。
② 初始化 SDK，下载 PP - HumanSeg - Lite 的 398×224 模型，并默认执行预热。具体实现如下。
```javascript
/**
 * 初始化SDK，默认下载398×224模型，默认执行预热
 * @param {boolean} [needPreheat=true] 是否在初始化阶段进行预热，默认为true，可选
 * @param {boolean} [enableLightModel=false] 是否使用288×160模型，默认为false，可选
 */
export async function load(needPreheat = true, enableLightModel = false) {
    const modelpath = 'https://paddlejs.bj.bcebos.com/models/fuse/humanseg/humanseg_398x224_fuse_activation';
    const lightModelPath = 'https://paddlejs.bj.bcebos.com/models/fuse/humanseg/humanseg_288x160_fuse_activation';
    const modelPath = enableLightModel? lightModelPath : modelpath;

    // 注册全局引擎Runner，并初始化
    runner = new Runner({
        modelPath: modelPath,
        needPreheat: needPreheat!== undefined? needPreheat : true,
        mean: [0.5, 0.5, 0.5],
        std: [0.5, 0.5, 0.5]
    });

    // 设置flag webgl_feed_process为true，在GPU上完成图像前处理过程，提高图像前处理速度
    env.set('webgl_feed_process', true);
    // 设置flag webgl_pack_channel为true，开启conv2d计算向量化，加快推理性能
    env.set('webgl_pack_channel', true);
    // 设置flag webgl_pack_output为true，按照四通道排布模型输出结果并读取，提高结果读取速度
    env.set('webgl_pack_output', true);

    // 引擎初始化
    await runner.init();
}
```
③ 对输入图像进行分割，并获取分割后的像素 alpha 值。返回的分割结果为 32 位的浮点型数组 Float32Array，数组中每个值代表对应像素是人像的置信度值（取值范围为0~1），如果该像素被分类为人像，则置信度越高，值越接近1。具体实现如下。
```javascript
/**
 * 获取分割结果
 * @param {HTMLImageElement | HTMLVideoElement | HTMLCanvasElement} input 分割对象，类型可以是image、canvas和video
 * @return {Float32Array} seg values of the input
 */
export async function getSegValue(input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement): Float32Array {
    // 传入Runner进行推理，获取推理结果，推理结果长度为2×398×224
    // 0 ~ 398×224为像素代表背景的置信度值
    // 398×224 ~ 2×398×224为像素代表人像前景的置信度值
    const backAndForeConfidence = await runner.predict(input);
    // 返回像素代表人像前景的置信度值数组
    return backAndForeConfidence.splice(backAndForeConfidence.length / 2);
}
```
④ 获取背景 canvas，将其作为新替换的背景。
⑤ 获取目标 canvas，分割效果将绘制在此。
⑥ 背景替换，使用 back_canvas 作为新背景，实现背景替换。该 API 主要实现绘制人像分割结果，可选是否替换背景。API 前两个参数分别是分割结果和目标 canvas，目标 canvas 为最终分割效果渲染的位置。第三个参数 backgroundCanvas 为可选参数，作为背景绘制在目标 canvas 中，实现背景替换效果。具体实现如下。
```javascript
/**
 * 人像分割，可选是否替换背景
 * @param {Float32Array} seg_values 输入图像的分割结果
 * @param {HTMLCanvasElement} canvas 为目标canvas，渲染结果将绘制于此
 * @param {HTMLCanvasElement} backgroundCanvas 为背景canvas，可选
 */
export function drawHumanSeg(
    seg_values: number[],
    canvas: HTMLCanvasElement,
    backgroundCanvas?: HTMLCanvasElement | HTMLImageElement
) {
    // 获取分割对象的原始大小
    const inputWidth = inputElement.naturalWidth || inputElement.width;
    const inputHeight = inputElement.naturalHeight || inputElement.height;

    const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
    canvas.width = WIDTH;
    canvas.height = HEIGHT;
    const tempCanvas = document.createElement('canvas') as HTMLCanvasElement;
    const tempContext = tempCanvas.getContext('2d') as CanvasRenderingContext2D;
    tempCanvas.width = WIDTH;
    tempCanvas.height = HEIGHT;

    const tempScaleData = ctx.getImageData(0, 0, WIDTH, HEIGHT);
    // 绘制原始输入图像
    tempContext.drawImage(inputElement, backgroundSize.x, backgroundSize.y, backgroundSize.sw, backgroundSize.sh);
    // 获取原始输入图像像素值
    const originImageData = tempContext.getImageData(0, 0, WIDTH, HEIGHT);

    for (let i = 0; i < WIDTH * HEIGHT; i++) {
        // 概率值×255，如果大于100，则认为是人像
        if (seg_values[i] * 255 > 100) {
            // 获取原始输入图像像素red通道值
            tempScaleData.data[i * 4] = originImageData.data[i * 4];
            // 获取原始输入图像像素green通道值
            tempScaleData.data[i * 4 + 1] = originImageData.data[i * 4 + 1];
            // 获取原始输入图像像素blue通道值
            tempScaleData.data[i * 4 + 2] = originImageData.data[i * 4 + 2];
            // 概率值×255作为该像素的alpha通道值
            tempScaleData.data[i * 4 + 3] = seg_values[i] * 255;
        }
    }

    tempContext.putImageData(tempScaleData, 0, 0);
    canvas.width = inputWidth;
    canvas.height = inputHeight;
    // 如果传入第三个参数，则将该参数作为背景绘制到目标canvas中，实现背景替换效果
    backgroundCanvas && ctx.drawImage(backgroundCanvas, -backgroundSize.bx, -backgroundSize.by, backgroundSize.bw, backgroundSize.bh);
    ctx.drawImage(tempCanvas, -backgroundSize.bx, -backgroundSize.by, backgroundSize.bw, backgroundSize.bh);
}
```
⑦ 获取背景虚化效果的目标 canvas。
⑧ 虚化背景。具体实现如下。
```javascript
/**
 * 虚化背景
 * @param {Float32Array} seg_values 输入图像的分割结果
 * @param {HTMLCanvasElement} dest_canvas 为目标canvas
 */
export function blurBackground(seg_values: number[], dest_canvas: HTMLCanvasElement) {
    const inputWidth = inputElement.naturalWidth || inputElement.width;
    const inputHeight = inputElement.naturalHeight || inputElement.height;
    const tempCanvas = document.createElement('canvas') as HTMLCanvasElement;
    const tempContext = tempCanvas.getContext('2d') as CanvasRenderingContext2D;
    tempCanvas.width = WIDTH;
    tempCanvas.height = HEIGHT;
    const dest_ctx = dest_canvas.getContext('2d') as CanvasRenderingContext2D;
    dest_canvas.width = inputWidth;
    dest_canvas.height = inputHeight;
    const tempScaleData = tempContext.getImageData(0, 0, WIDTH, HEIGHT);
    // 在tempCanvas上绘制原始图像
    tempContext.drawImage(inputElement, backgroundSize.x, backgroundSize.y, backgroundSize.sw, backgroundSize.sh);
    const originImageData = tempContext.getImageData(0, 0, WIDTH, HEIGHT);

    // 清空blurFilter
    blurFilter.dispose();
    // 使用blurFilter虚化原始图像
    const blurCanvas = blurFilter.apply(tempCanvas);

    for (let i = 0; i < WIDTH * HEIGHT; i++) {
        // 概率值×255，如果大于100，则认为是人像
        if (seg_values[i] * 255 > 100) {
            tempScaleData.data[i * 4] = originImageData.data[i * 4];
            tempScaleData.data[i * 4 + 1] = originImageData.data[i * 4 + 1];
            tempScaleData.data[i * 4 + 2] = originImageData.data[i * 4 + 2];
            tempScaleData.data[i * 4 + 3] = seg_values[i] * 255;
        }
    }
}
``` 


```javascript
tempContext.putImageData(tempScaleData, 0, 0);
// 在目标canvas上绘制虚化后的canvas
dest_ctx.drawImage(blurCanvas, 
    -backgroundSize.bx, -backgroundSize.by, 
    backgroundSize.bw, backgroundSize.bh);
// 在目标canvas上绘制分割后的人像
dest_ctx.drawImage(tempCanvas, 
    -backgroundSize.bx, -backgroundSize.by, 
    backgroundSize.bw, backgroundSize.bh);
```
人像分割效果如图4 - 8所示。

### 4.2.3 目标检测
目标检测（Object Detection）有两个任务：一个是识别出图像中的目标所属类别，这与图像分类目标是一样的；另一个是识别出目标的边界，也就是识别出目标的位置信息，目标可能有一个，也可能有多个。如图4 - 9所示，想要检测出图像中的猫，先要识别出图像中是否有猫这个类别，再推理出猫的位置信息。

#### 1. 目标检测概念和应用场景
目标检测是计算机视觉的主要研究方向之一，有着广泛的应用场景，如人脸检测、行人检测和文字检测等，目标检测是这些视觉任务的基础。既然为基础，就说明目标检测只是应用中的一个环节，先检测出图像中目标的类别及位置信息，再结合检测的结果对图像进行进一步处理，并作为关键点检测等其他模型的输入，以识别目标的详细信息。

#### 2. 实战：人脸检测
先识别出图像中是否有人脸及人脸的位置信息，再通过位置信息裁剪出图像中的人脸部位，通过关键点模型推理出面部的关键点，是人脸检测相关应用的常见思路。

简化版的表情识别应用，可根据唇部关键点的相对位置信息，判断出嘴角是上扬的还是下弯的，是紧闭的还是张开的，以此得出用户是开怀大笑、微笑的还是难过的。美妆应用可根据关键点信息描绘出面部各区域，结合增强现实渲染技术给面部区域上妆或美容。

这些应用都依赖人脸检测模型，检测图像中是否有人脸及人脸框的位置。@paddlejs - models/tinyYolo SDK封装了人脸检测的功能，是一种单目标检测，使用方式如下。
```javascript
// 引入tinyYolo
import * as tinyYolo from '@paddlejs - models/tinyYolo';

// 初始化SDK
await tinyYolo.load();

// 传入图像，获取人脸框坐标
const res = await tinyYolo.detect(img);
```
① 引入tinyYolo SDK。
② 调用load方法完成初始化。具体实现如下。
```javascript
export async function load(config: ModelConfig) {
    const {
        path = 'https://paddlejs.cdn.bcebos.com/models/tinyYolo',
        mean = [117.001 / 255, 114.697 / 255, 97.404 / 255],
        std = [1, 1, 1]
    } = config;

    const runner = new Runner({
        modelPath: path,
        feedShape: {
            fw: 320,
            fh: 320
        },
        mean: mean,
        std: std
    });

    await runner.init();
}
```
③ 调用detect方法完成推理。具体实现如下。
```javascript
/**
 * @param image传入图像
 * @return目标在图像中的位置信息
 */
export async function detect(image) {
    const res = await runner.predict(image);
    // 进一步处理推理结果，若在图像中检测到人脸，则返回长度为4的数组；若未检测到人脸，则返回空数组
    const result = process(res, image);
    return result;
}
```
人脸检测效果如图4 - 10所示。

#### 3. 实战：手势识别
手势识别任务首先识别出图像中的手势位置，进而识别出手势的分类。利用手掌的位置，可通过手部移动路径隔空控制页面中一些物体的移动，如图4 - 11所示，用手部移动路径隔空控制螺旋丸旋转。利用石头、剪子和布的手势分类可以做出一个如图4 - 12所示的猜拳小游戏。

@paddlejs - models/gesture SDK封装了手部检测及手势识别功能，SDK使用了手部检测及手势分类两个模型。通过手部检测，确定图像中是否有手部目标及手掌的位置，根据检测到的手部信息对原图像进行仿射变换并裁剪，将处理后的图像作为手势分类模型的输入，识别出最终的手势类别。SDK使用方式如下。
```javascript
// 引入gesture SDK
import * as gesture from '@paddlejs - models/gesture';

// 初始化SDK
await gesture.load();

// 传入图像，获取手势分类结果
const res = await gesture.classify(img);
```
① 引入gesture SDK。
② 调用load方法完成初始化。具体实现如下。
```javascript
export async function load() {
    const detectRunner = new Runner({
        modelPath: 'https://paddlejs.bj.bcebos.com/models/fuse/gesture/gesture_det_fuse_activation'
    });

    const recRunner = new Runner({
        modelPath: 'https://paddlejs.bj.bcebos.com/models/fuse/gesture/gesture_rec_fuse_activation'
    });

    // 初始化手势检测模型
    await detectRunner.init();
    // 初始化手势分类模型
    await recRunner.init();
}
```
③ 调用classify方法完成推理。具体实现如下。
```javascript
export async function classify(image) {
    // 手部检测推理
    const detectResult = await detectRunner.predict(image);
    const post = new DetectProcess(detectResult, canvas);
    const box = await post.outputBox(anchorResults);
    if (!box) {
        return '识别不到手';
    }
    // 手部框计算
    calculateBox(box);
    // 根据手部检测结果对原图像进行仿射变换和裁剪处理
    const feed = await post.outputFeed(recRunner);
    // 手势识别推理
    const recResult = await recRunner.predictWithFeed(feed);
    // 计算手势分类
    const lmProcess = new LMProcess(recResult);
    lmProcess.output();
    const type = lmProcess.type || '';
    return type;
}
```
手势识别效果如图4 - 13所示。 
