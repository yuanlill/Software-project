
@[TOC](构建首个Kotlin应用)

# 实验内容

• 安装Android Studio 4.1+以上版本，后续更好的使用
TensorFlow Lite
• 按照教程构建首个Kotlin交互应用
• 上传代码至Github，并撰写详细的Readme文档

## 实验目的
• 掌握Android Studio开发应用的基本流程
• 掌握Android Studio开发组件的基本用法
• 初始Kotlin语言的基本要素
• 掌握Android Navigation的基本用法
## 创建一个新的工程
### 1、打开Android Studio，创建Basic Activity
[![OPN3kT.png](https://img-blog.csdnimg.cn/img_convert/1e099730780391324c373e20501e5bf2.png)](https://imgtu.com/i/OPN3kT)
### 2、查看Android Studio整体布局
因之前学过移动开发故这步简单看过即可
[![OPNYp4.md.png](https://img-blog.csdnimg.cn/img_convert/35ab34ec051ee53e88b4afa7a1cda7e7.png)](https://imgtu.com/i/OPNYp4)
### 3、创建模拟器，运行程序
[![OPN8tU.png](https://img-blog.csdnimg.cn/img_convert/13de83c02557ee15772c357fe44babb8.png)](https://imgtu.com/i/OPN8tU)
### 4、查看与使用布局文件（.xml）
Android Studio中布局文件可使用拖动添加布局（较易）和编写代码布局，首先使用拖动添加布局
（1）整体布局页面（可使用右上角的Code/Split/Design选择界面，分别问纯代码界面/代码布局分离界面/纯布局界面）
[![OPNNc9.png](https://img-blog.csdnimg.cn/img_convert/788db6a032aea9b0bae8f59cb487613c.png)](https://imgtu.com/i/OPNNc9)
（2）先打开Design布局界面
[![OPNGhF.png](https://img-blog.csdnimg.cn/img_convert/7dc04b02236dc1591f58dc2958d59920.png)](https://imgtu.com/i/OPNGhF)
（3）首先更改文本内容，点击文本，在右侧找到Commen Attributes，可更改字体大小、颜色、粗细等属性
[![OPNUXR.png](https://img-blog.csdnimg.cn/img_convert/9dde532572dba416fc033756a13a1e27.png)](https://imgtu.com/i/OPNUXR)
（4）文本更改为Hello Kotlin！效果图
[![OPNdn1.png](https://img-blog.csdnimg.cn/img_convert/c07b855f73fac97bd843e89e8281a988.png)](https://imgtu.com/i/OPNdn1)
（5）添加按钮：从左侧拖出按钮放置在屏幕中即可
[![OPNw0x.png](https://img-blog.csdnimg.cn/img_convert/3fdc6e83bf7b69709a9550e3bd66e42f.png)](https://imgtu.com/i/OPNw0x)
（6）可在Split的代码页面中看见我们新添加的按钮（button_first）
[![OPN076.png](https://img-blog.csdnimg.cn/img_convert/e1e2c0b37528266531ca4d84f14bf3fb.png)](https://imgtu.com/i/OPN076)
(7)为button(重新)设置text属性（可直接设置，这里使用映射到资源文件）具体方法为：点击文本，左侧出现灯泡状的提示，选择 Extract string resource弹出对话框，令资源名为random_button_text，并点击OK
[![OPNDAK.png](https://img-blog.csdnimg.cn/img_convert/762803e4966d4ea7520f8b749f0c226f.png)](https://imgtu.com/i/OPNDAK)
（8）添加第三个按钮，并调好位置关系，其中点击按钮或文字会出现弹簧状的“链条”，这个就是布局的约束，初学时可直接在Design页面进行添加和删除等，较为方便
[![OPNrtO.png](https://img-blog.csdnimg.cn/img_convert/565f3c56ba49884feda25dccadb6d46b.png)](https://imgtu.com/i/OPNrtO)
（9）查看页面效果
[![OPNshD.png](https://img-blog.csdnimg.cn/img_convert/99e0d069358cffb99e8b6a3ab9c40eca.png)](https://imgtu.com/i/OPNshD)
（10）再次调整后的最终页面效果（其中颜色可在Design页面调整也可在Code页面添加text-color属性）
[![OPN69e.png](https://img-blog.csdnimg.cn/img_convert/e422b09eb91799d3d1bb0e7018523bb4.png)](https://imgtu.com/i/OPN69e)
## 完成应用程序交互
### 1、设置代码自动补全
（Android Studio中，依次点击File>New Projects Settings>Settings for New Projects…，查找Auto Import选项，在Java和Kotlin部分，勾选Add Unambiguous Imports on the fly。（如图）
[![OPNc1H.png](https://img-blog.csdnimg.cn/img_convert/3c070da0058696b7e7b9f7a4b85fc971.png)](https://imgtu.com/i/OPNc1H)
### 2、TOAST按钮添加一个toast消息（窗口弹出消息）

```java
        view.findViewById<Button>(R.id.toast_button).setOnClickListener {//为按钮toast_button绑定事件
            val Toast = Toast.makeText(context, "Hello Toast!", Toast.LENGTH_LONG)
            Toast.show()
        }
```
### 3、使Count按钮更新屏幕的数字
同样先给按钮绑定事件，其中调用Count函数（计数）

```java
        view.findViewById<Button>(R.id.count_button).setOnClickListener {
            count(view)
        }
        
    private fun count(view: View) {
        val showCountTextView = view.findViewById<TextView>(R.id.textview_first);
        val countString = showCountTextView.text.toString();
        var count = countString.toInt();
        count++;
        showCountTextView.text = count.toString();
    }
```
效果图
[![OPNWnI.png](https://img-blog.csdnimg.cn/img_convert/f4f09dac567ecdd2ab2af4e2c32a3758.png)](https://imgtu.com/i/OPNWnI)
## 完成导航与SafeArgs数据传输
### 1、完成第二界面样式
（1）完成第二界面xml文件编写（同第一界面）
### 2、启用SafeArgs
（较于传统的页面间数据传输，它的主要好处在于安全的参数类型）
①在build.gradle（Project）中添加

```java
buildscript {
    repositories {
        google()
    }
    dependencies {
        def nav_version = "2.4.2"
        classpath "androidx.navigation:navigation-safe-args-gradle-plugin:$nav_version"
    }
}
```
②在build.gradle（Module）中添加

```java
    id 'androidx.navigation.safeargs'
```
（注意不同版本的Android Studio（连同Gradle）所依赖的导航库不相同）
③我的配置

```java
android {
    compileSdk 32

    defaultConfig {
        applicationId "com.example.modol"
        minSdk 21
        targetSdk 32
        versionCode 1
        versionName "1.0"

        testInstrumentationRunner "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            minifyEnabled false
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        }
    }
    compileOptions {
        sourceCompatibility JavaVersion.VERSION_1_8
        targetCompatibility JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = '1.8'
    }
    buildFeatures {
        viewBinding true
    }
}
```
### 3、检查导航图，创建导航动作的参数
（1）打开nav_graph.xml文件（res>navigation>nav_graph.xml）
[![OPNgcd.png](https://img-blog.csdnimg.cn/img_convert/62768a9aaa4ffb517c062ded7e263d64.png)](https://imgtu.com/i/OPNgcd)
（2）在Fragment的属性栏，点击Arguments **+**符号，弹出的对话框中，添加参数myArgs，类型为整型Integer
[![OPN2jA.png](https://img-blog.csdnimg.cn/img_convert/7faadbd2cf97232319059fc0cc6802f8.png)](https://imgtu.com/i/OPN2jA)
### 4、FirstFragment添加代码，向SecondFragment发数据
（1）FirstFragment（currentCount作为参数传递给actionFirstFragmentToSecondFragment()）

```java
class FirstFragment : Fragment() {

    private var _binding: FragmentFirstBinding? = null

    // This property is only valid between onCreateView and
    // onDestroyView.
    private val binding get() = _binding!!

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {

        _binding = FragmentFirstBinding.inflate(inflater, container, false)
        return binding.root

    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        binding.buttonFirst.setOnClickListener {
//            findNavController().navigate(R.id.action_FirstFragment_to_SecondFragment)
            val showCountTextView = view.findViewById<TextView>(R.id.textview_first)
            val currentCount = showCountTextView.text.toString().toInt()
            val action = FirstFragmentDirections.actionFirstFragmentToSecondFragment(currentCount)
            findNavController().navigate(action)

        }

        view.findViewById<Button>(R.id.toast_button).setOnClickListener {
            val myToast = Toast.makeText(context, "Hello Toast!", Toast.LENGTH_LONG)
            myToast.show()
        }


        view.findViewById<Button>(R.id.count_button).setOnClickListener {
            countMe(view)
        }



    }

    private fun countMe(view: View) {
        val showCountTextView = view.findViewById<TextView>(R.id.textview_first);
        val countString = showCountTextView.text.toString();
        var count = countString.toInt();
        count++;
        showCountTextView.text = count.toString();
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
```
（2）SecondFragment（获取传递过来的参数列表，提取count数值，并在textview_header中显示，根据count值生成随机数并显示）

```java
class SecondFragment : Fragment() {

    private var _binding: FragmentSecondBinding? = null

    // This property is only valid between onCreateView and
    // onDestroyView.
    private val binding get() = _binding!!

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {

        _binding = FragmentSecondBinding.inflate(inflater, container, false)
        return binding.root

    }

    val args: SecondFragmentArgs by navArgs()


    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState);
        val count = args.myArgs
        val countText = getString(R.string.random_heading, count)
        view.findViewById<TextView>(R.id.textview_second).text = countText
        val random = java.util.Random()
        var randomNumber = 0
        if (count > 0) {
            randomNumber = random.nextInt(count + 1)
        }
        view.findViewById<TextView>(R.id.textview_random).text = randomNumber.toString()



        binding.buttonSecond.setOnClickListener {
            findNavController().navigate(R.id.action_SecondFragment_to_FirstFragment);
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
```
## 最终效果
[![OPNWnI.png](https://img-blog.csdnimg.cn/img_convert/fc90f778a365c2153efae4d760324e0b.png)](https://imgtu.com/i/OPNWnI)
[![OPNfBt.png](https://img-blog.csdnimg.cn/img_convert/7390fe2609be32de3d8d6a2f2636a093.png)](https://imgtu.com/i/OPNfBt)

作者：员力
