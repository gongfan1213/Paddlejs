### 4.3 小程序 CV 项目
本章前面内容主要介绍了Paddle.js在浏览器端的应用，本节主要介绍如何在微信小程序和百度智能小程序上实现AI效果。

#### 4.3.1 微信小程序插件paddlejsPlugin
要在微信小程序平台使用Paddle.js，需要引入微信小程序插件paddlejsPlugin。Paddle.js微信小程序模块架构如图4 - 14所示。

![Paddle.js微信小程序模块架构](图4 - 14 Paddle.js微信小程序模块架构)

![image](https://github.com/user-attachments/assets/f4b076fa-a116-4463-8426-5a61bc6dc205)



在图4 - 14中，paddlejsPlugin主要负责创建离屏canvas，为@paddlejs/paddlejs-backend-webgl提供WebGL入口，获取微信小程序端GPU算力。

1. **注册插件和npm包**
插件paddlejsPlugin可在小程序管理后台的“设置”→“第三方服务”→“插件管理”中搜索、添加，开发者需要在小程序的app.json中声明插件信息，代码如下。
```json
{
   ...
    "plugins": {
        "paddlejs-plugin": {
            "version": "2.0.1",
            "provider": "wx7138a7bb793608c3"
        }
    },
   ...
}
```
注：作者撰写本书时paddlejsPlugin为2.0.1版，实际练习时请使用paddlejsPlugin最新版本。

在小程序项目package.json文件中引入@paddlejs/paddlejs-core和@paddlejs/paddlejs-backend-webgl。

2. **核心代码实现**
首先引入各个模块，注册并实例化Paddle.js引擎，代码如下。
```javascript
// 引入paddlejs-core
import * as paddlejs from '@paddlejs/paddlejs-core';
// 引入paddlejs-backend-webgl
import '@paddlejs/paddlejs-backend-webgl';
// 引入plugin
const plugin = requirePlugin('paddlejs-plugin');
// 在插件内创建离屏canvas，提供WebGL入口
plugin.register(paddlejs, wx);

// 以mobilenet模型配置为例
export const PaddleJS = new paddlejs.Runner({
    modelPath: 'https://mms-voice-fe.cdn.bcebos.com/pdmodel/clas/fuse/v4_03082014',
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
    webglFeedProcess: true
});
```
接下来的使用方法与4.2节中的使用方法一致，配置模型参数、初始化引擎、完成推理等，示例代码如下。
```javascript
onLoad() {
    runner.init().then(() => {
        // 完成初始化
    });
}

/**
 * 推理
 * @param {Object} imgObj 图像信息
 * @param {Uint8Array} imgObj.data 像素数据
 * @param {number} imgObj.width 图像宽度
 * @param {number} imgObj.height 图像高度
 */
predict(imgObj) {
    runner.predict(imgObj, data => {
        // 完成推理，data是推理结果
    });
}
```

#### 4.3.2 百度智能小程序动态库paddlejs
百度智能小程序和微信小程序使用Paddle.js的方式并不相同，百度智能小程序开发者只需要引入小程序动态库paddlejs，以功能组件的形式添加到小程序内。动态库内部引入npm包@paddlejs/paddlejs-core和@paddlejs/paddlejs-backend-webgl，采用静默更新的方式，开发者不必关注。

1. **注册动态库**
在使用动态库前，开发者要在app.json中声明需要使用的动态库，代码如下。
```json
{
    "dynamicLib": {
        // 定义一个别名，小程序中用这个别名引用动态库
        "paddlejs": {
            "provider": "paddlejs"
        }
    }
}
```

2. **使用动态库**
使用动态库组件paddlejs与使用普通自定义组件的方式相仿，在JSON文件中配置如下信息。
```json
{
    "usingSwanComponents": {
        // 此处的paddlejs为自己定义的别名，本页面或本组件在模板中用此别名引用paddlejs动态库组件
        "paddlejs": "dynamicLib://paddlejs/paddlejs"
    }
}
```
在页面中可以使用动态库组件paddlejs。
```html
<view class="container">
    <view>下面这个自定义组件来自动态库</view>
    <!-- 这里的'paddlejs'就是本页面中对于动态库组件paddlejs的别名 -->
    <paddlejs options="{{options}}" 
              status="{{status}}" 
              imgBase64="{{imgBase64}}" 
              bindchildmessage="paddleStatusChange" />
</view>
```
组件props属性信息如表4 - 4所示。
| 名称 | 类型 | 是否必选 | 描述 |
| ---- | ---- | ---- | ---- |
| options | string | 是 | 模型配置项 |
| imgBase64 | string | 是 | 要预测的图像的Base64 |
| status | string | 是 | 当前状态，status变化触发组件调用相应的API。当status变为predict时，组件会读取imgBase64作为输入的图像，调用模型预测API |

bindchildmessage指定与paddlejs组件通信的方法（本例中以paddleStatusChange为例）。组件会分别在初始化过程和一次推理过程完成时触发方法paddleStatusChange，并传递对应的状态参数。
- 初始化完成时，event.detail.status为loaded，开发者可以选择图像触发推理。
- 当一次推理过程完成时，event.detail.status为complete，并传递推理结果event.detail.data，开发者可以根据需求来处理推理结果，以实现AI效果。

### 4.4 总结
本章介绍了Paddle.js模型库paddlejs-models，并且结合模型库实现了三个经典CV项目——图像分类、图像分割、目标检测，并介绍如何在小程序上实现AI效果，总结如下。
- 模型库paddlejs-models提供开箱即用的AI功能，开发者不需要自己提供模型，可简单地调用API使产品快速接入AI功能。
- 使用@paddlejs-models/mobilenet实现图像分类任务。
- 使用@paddlejs-models/humanseg实现人像分割任务，实现背景替换和背景虚化两种效果。 
- 使用@paddlejs-models/tinyYolo实现人脸检测，使用@paddlejs-models/gesture实现手势识别。
- 通过小程序插件paddlejsPlugin，以及npm包@paddlejs/paddlejs-core和@paddlejs/paddlejs-backend-webgl在微信小程序上开发CV任务。 
- 通过动态库功能组件paddlejs，在百度智能小程序上开发CV任务。

在第1章~第4章内容中，我们充分地了解了Web AI和其背后的技术原理，并且能够通过paddlejs前端推理引擎和其暴露的模型库来完成基本的业务开发。接下来，我们将以此为基础，了解如何深层定制相关技术栈以满足更复杂的需求。 
