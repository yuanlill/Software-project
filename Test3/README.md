
@[TOC](实现基本的图像分类APP
)

# 实验内容

按照教程构建基于TensorFlow Lite的Android花卉识别应
用。
• 查 看 该 应 用 的 代 码 框 架 ， 特 别 注 意 CameraX 库
(AndroidX.camera.*)和数据视图模型的使用。
• 上 传 完 成 既 定 功 能 的 代 码 至 Github ， 并 撰 写 详 细 的
Readme文档


## 实验步骤
### 1、下载初始代码
使用Git下载初始代码，选择文件夹->Git Bash Here->输入拷贝代码

```bash
git clone https://github.com/hoitab/TFLClassify.git
```

### 2、运行初始代码
1、打开Android Studio，打开已经存在项目TFLCLassify（刚下载的项目）


[![OmBj1J.jpg](https://img-blog.csdnimg.cn/img_convert/504d2cd24caf702315079f1d08f7fd22.png)](https://imgtu.com/i/OmBj1J)


1.5、可能存在的问题，打开后普遍出现***Gradle Sync build false***原因是目标项目需要的SDK为29，而主机中不包括。
解决方法：
Tools->SDK Manager->下载29版本即可


[![OmDI8e.jpg](https://img-blog.csdnimg.cn/img_convert/af6d487bdf654fc37864805fc2bdd529.png)](https://imgtu.com/i/OmDI8e)

2、手机通过USB接口连接开发平台，并设置手机开发者选项（荣耀手机为连续点击版本号5次）允许调试。


[![OmBXp4.jpg](https://img-blog.csdnimg.cn/img_convert/fa54e46db187925c801542f505bdaa37.png)](https://imgtu.com/i/OmBXp4)

3、允许应用获取手机摄像头的权限，查看最初效果

[![OmD976.jpg](https://img-blog.csdnimg.cn/img_convert/f593194b4dc97f5d61e77954b98b48f2.png)](https://imgtu.com/i/OmD976)


### 3、向应用中添加TensorFlow Lite
1、右键“start”模块，或者选择File，然后New>Other>TensorFlow Lite Model

[![OmBLhF.jpg](https://img-blog.csdnimg.cn/img_convert/55eeb71677fd25c78636075e8fad59e4.png)](https://imgtu.com/i/OmBLhF)

2、选择finish模块中ml文件下的FlowerModel.tflite

[![OmBqtU.jpg](https://img-blog.csdnimg.cn/img_convert/bcf94483502865399989c2bc7144226e.png)](https://imgtu.com/i/OmBqtU)

3、最终TensorFlow Lite模型被成功导入，并生成摘要信息

[![OmDp0x.jpg](https://img-blog.csdnimg.cn/img_convert/e5e397ab355a523f05bed167cba44d8e.png)](https://imgtu.com/i/OmDp0x)


### 4、检查代码中的TODO项
1、查看TODO列表视图，View>Tool Windows>TODO
2、默认情况下了列出项目所有的TODO项，进一步按照模块分组（Group By）

[![OmBxXR.jpg](https://img-blog.csdnimg.cn/img_convert/b1672378a1e98efc910b63913f92090f.png)](https://imgtu.com/i/OmBxXR)


### 5、添加代码重新运行APP
1、start->MainActivity.kt->TODO 1，添加初始化训练模型的代码

[![OmBvc9.jpg](https://img-blog.csdnimg.cn/img_convert/bfb806787680eada622e8345aae151b1.png)](https://imgtu.com/i/OmBvc9)

2、start->MainActivity.kt->TODO 2
在CameraX的analyze方法内部，需要将摄像头的输入ImageProxy转化为Bitmap对象，并进一步转化为TensorImage 对象

[![OmDSn1.jpg](https://img-blog.csdnimg.cn/img_convert/5ca1a10e7509197ec7f3cd50293ffb1a.png)](https://imgtu.com/i/OmDSn1)

3、start->MainActivity.kt->TODO 3
按照属性score对识别结果按照概率从高到低排序
列出最高k种可能的结果，k的结果由常量MAX_RESULT_DISPLAY定义


[![OmDPAK.jpg](https://img-blog.csdnimg.cn/img_convert/9aa6c3e55f6120e41c4f9ec841567355.png)](https://imgtu.com/i/OmDPAK)

4、start->MainActivity.kt->TODO 4
将识别的结果加入数据对象Recognition 中，包含label和score两个元素。后续将用于RecyclerView的数据显示

[![OmDitO.jpg](https://img-blog.csdnimg.cn/img_convert/1f9f9fe98c62def223cf25e3f18a29a0.png)](https://imgtu.com/i/OmDitO)

5、start->MainActivity.kt->TODO 5
将原先用于虚拟显示识别结果的代码注释掉或者删除

[![OmDFhD.jpg](https://img-blog.csdnimg.cn/img_convert/d7e92ab386a892e7f039d47c55275cb6.png)](https://imgtu.com/i/OmDFhD)


### 6、思考
#### 根据Android developers 官网概览：
CameraX 是 Jetpack 的新增库。利用该库，可以更轻松地向应用添加相机功能。该库提供了很多兼容性修复程序和解决方法，有助于在众多设备上打造一致的开发者体验。

ViewModel 类旨在以注重生命周期的方式存储和管理界面相关的数据。ViewModel 类让数据可在发生屏幕旋转等配置更改后继续留存。

LiveData 是一种可观察的数据存储器类。与常规的可观察类不同，LiveData 具有生命周期感知能力，意指它遵循其他应用组件（如 Activity、Fragment 或 Service）的生命周期。这种感知能力可确保 LiveData 仅更新处于活跃生命周期状态的应用组件观察者。

#### 思考：
真机模拟下调用摄像头就是使用了CameraX库，通过LiveData观察活动等的生命周期，通过ViewModel存储和管理界面相关数据，在摄像头发生翻转等情况下保存配置并留存。

### 7、修改代码

```java
            // TODO 4: Converting the top probability items into a list of recognitions
            var f = 100f;
            items.add(Recognition("这是上限",f))
            for (output in outputs) {
                if(output.label.equals("tulips")){
                    items.add(Recognition("郁金香", output.score))
                }else if(output.label.equals("roses")){
                    items.add(Recognition("玫瑰", output.score))
                }else{
                    items.add(Recognition("雏菊", output.score))
                }
            }
            f = 0f;
            items.add(Recognition("这是下限",f))
```

## 最终效果
#### 在物理设备运行start模块


[![ON4tyV.jpg](https://img-blog.csdnimg.cn/img_convert/fa69eb726fcfc70d74aa9c9552dcb913.png)](https://imgtu.com/i/ON4tyV)


#### 修改后在物理设备运行start模块（转换为中文，去除多余label）


[![ON4YQ0.jpg](https://img-blog.csdnimg.cn/img_convert/3c290fd3378ab5721b3c978edc5c7f55.png)](https://imgtu.com/i/ON4YQ0)



作者：员力

思考参考：
[CameraX概览
](https://developer.android.google.cn/jetpack/androidx/releases/camera)[ViewModel概览 ](https://developer.android.google.cn/topic/libraries/architecture/viewmodel)
[LiveData概览 ](https://developer.android.google.cn/topic/libraries/architecture/livedata.html)

