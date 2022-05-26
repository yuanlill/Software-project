@[TOC](TensorFlow Lite 模型生成
)
# 实验内容


• 了解机器学习基础
• 了解TensorFlow及TensorFlow Lite
• 按照教程完成基于TensorFlow Lite Model Maker的花卉
模型生成
• 使用实验三的应用验证生成的模型
• 将上述完成的Jupyter Notebook在Github上进行共享


## 实验步骤
### 预备工作

#### 1、安装程序运行必备的一些库


```python
!pip install tflite-model-maker
```

    Collecting tflite-model-maker
      Downloading tflite_model_maker-0.4.0-py3-none-any.whl (642 kB)
    Collecting neural-structured-learning>=1.3.1
      Downloading neural_structured_learning-1.3.1-py2.py3-none-any.whl (120 kB)
    Collecting tensorflow-model-optimization>=0.5
      Downloading tensorflow_model_optimization-0.7.2-py2.py3-none-any.whl (237 kB)
    Requirement already satisfied: PyYAML>=5.1 in e:\anaconda3\lib\site-packages (from tflite-model-maker) (6.0)
    Requirement already satisfied: six>=1.12.0 in e:\anaconda3\lib\site-packages (from tflite-model-maker) (1.16.0)
    Requirement already satisfied: matplotlib<3.5.0,>=3.0.3 in e:\anaconda3\lib\site-packages (from tflite-model-maker) (3.4.3)
    Collecting numba==0.53
      Downloading numba-0.53.0-cp39-cp39-win_amd64.whl (2.3 MB)
    Collecting librosa==0.8.1
      Downloading librosa-0.8.1-py3-none-any.whl (203 kB)
    Collecting tensorflow-datasets>=2.1.0
      Downloading tensorflow_datasets-4.5.2-py3-none-any.whl (4.2 MB)
    Collecting flatbuffers==1.12
      Downloading flatbuffers-1.12-py2.py3-none-any.whl (15 kB)
    Requirement already satisfied: lxml>=4.6.1 in e:\anaconda3\lib\site-packages (from tflite-model-maker) (4.6.3)
    Collecting tensorflow-addons>=0.11.2
      Downloading tensorflow_addons-0.17.0-cp39-cp39-win_amd64.whl (758 kB)
    Collecting sentencepiece>=0.1.91
      Downloading sentencepiece-0.1.96-cp39-cp39-win_amd64.whl (1.1 MB)
    Requirement already satisfied: pillow>=7.0.0 in e:\anaconda3\lib\site-packages (from tflite-model-maker) (8.4.0)
    Requirement already satisfied: Cython>=0.29.13 in e:\anaconda3\lib\site-packages (from tflite-model-maker) (0.29.24)
    Collecting tensorflow-hub<0.13,>=0.7.0
      Downloading tensorflow_hub-0.12.0-py2.py3-none-any.whl (108 kB)
    Collecting absl-py>=0.10.0
      Downloading absl_py-1.0.0-py3-none-any.whl (126 kB)
    Collecting tf-models-official==2.3.0
      Downloading tf_models_official-2.3.0-py2.py3-none-any.whl (840 kB)
    Collecting fire>=0.3.1
      Downloading fire-0.4.0.tar.gz (87 kB)
    Collecting urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1
      Downloading urllib3-1.25.11-py2.py3-none-any.whl (127 kB)
    Requirement already satisfied: numpy>=1.17.3 in e:\anaconda3\lib\site-packages (from tflite-model-maker) (1.20.3)
    Collecting tensorflow>=2.6.0
      Downloading tensorflow-2.9.1-cp39-cp39-win_amd64.whl (444.0 MB)
    Collecting tensorflowjs>=2.4.0
      Downloading tensorflowjs-3.18.0-py3-none-any.whl (77 kB)
    Collecting tflite-support>=0.4.0
      Downloading tflite_support-0.4.0-cp39-cp39-win_amd64.whl (439 kB)
    Collecting tflite-model-maker
      Downloading tflite_model_maker-0.3.4-py3-none-any.whl (616 kB)
    Collecting audioread>=2.0.0
      Downloading audioread-2.1.9.tar.gz (377 kB)
    Requirement already satisfied: packaging>=20.0 in e:\anaconda3\lib\site-packages (from librosa==0.8.1->tflite-model-maker) (21.0)
    Collecting pooch>=1.0
      Downloading pooch-1.6.0-py3-none-any.whl (56 kB)
    Requirement already satisfied: scikit-learn!=0.19.0,>=0.14.0 in e:\anaconda3\lib\site-packages (from librosa==0.8.1->tflite-model-maker) (0.24.2)
    Requirement already satisfied: scipy>=1.0.0 in e:\anaconda3\lib\site-packages (from librosa==0.8.1->tflite-model-maker) (1.7.1)
    Collecting soundfile>=0.10.2
      Downloading SoundFile-0.10.3.post1-py2.py3.cp26.cp27.cp32.cp33.cp34.cp35.cp36.pp27.pp32.pp33-none-win_amd64.whl (689 kB)
    Collecting resampy>=0.2.2
      Downloading resampy-0.2.2.tar.gz (323 kB)
    Requirement already satisfied: joblib>=0.14 in e:\anaconda3\lib\site-packages (from librosa==0.8.1->tflite-model-maker) (1.1.0)
    Requirement already satisfied: decorator>=3.0.0 in e:\anaconda3\lib\site-packages (from librosa==0.8.1->tflite-model-maker) (5.1.0)
    Collecting llvmlite<0.37,>=0.36.0rc1
      Downloading llvmlite-0.36.0-cp39-cp39-win_amd64.whl (16.0 MB)
    Requirement already satisfied: setuptools in e:\anaconda3\lib\site-packages (from numba==0.53->tflite-model-maker) (58.0.4)
    Requirement already satisfied: pandas>=0.22.0 in e:\anaconda3\lib\site-packages (from tf-models-official==2.3.0->tflite-model-maker) (1.3.4)
    Collecting py-cpuinfo>=3.3.0
      Downloading py-cpuinfo-8.0.0.tar.gz (99 kB)
    Collecting opencv-python-headless
      Downloading opencv_python_headless-4.5.5.64-cp36-abi3-win_amd64.whl (35.3 MB)
    Collecting google-cloud-bigquery>=0.31.0
      Downloading google_cloud_bigquery-3.1.0-py2.py3-none-any.whl (211 kB)
    Collecting google-api-python-client>=1.6.7
      Downloading google_api_python_client-2.49.0-py2.py3-none-any.whl (8.5 MB)
    Collecting dataclasses
      Downloading dataclasses-0.6-py3-none-any.whl (14 kB)
    Collecting gin-config
      Downloading gin_config-0.5.0-py3-none-any.whl (61 kB)
    Collecting kaggle>=1.3.9
      Downloading kaggle-1.5.12.tar.gz (58 kB)
    Requirement already satisfied: psutil>=5.4.3 in e:\anaconda3\lib\site-packages (from tf-models-official==2.3.0->tflite-model-maker) (5.8.0)
    Collecting tf-slim>=1.1.0
      Downloading tf_slim-1.1.0-py2.py3-none-any.whl (352 kB)
    Collecting termcolor
      Downloading termcolor-1.1.0.tar.gz (3.9 kB)
    Collecting httplib2<1dev,>=0.15.0
      Downloading httplib2-0.20.4-py3-none-any.whl (96 kB)
    Collecting google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.0,<3.0.0dev,>=1.31.5
      Downloading google_api_core-2.8.0-py3-none-any.whl (114 kB)
    Collecting google-auth-httplib2>=0.1.0
      Downloading google_auth_httplib2-0.1.0-py2.py3-none-any.whl (9.3 kB)
    Collecting google-auth<3.0.0dev,>=1.16.0
      Downloading google_auth-2.6.6-py2.py3-none-any.whl (156 kB)
    Collecting uritemplate<5,>=3.0.1
      Downloading uritemplate-4.1.1-py2.py3-none-any.whl (10 kB)
    Collecting protobuf>=3.12.0
      Downloading protobuf-4.21.0-cp39-cp39-win_amd64.whl (524 kB)
    Collecting googleapis-common-protos<2.0dev,>=1.52.0
      Downloading googleapis_common_protos-1.56.1-py2.py3-none-any.whl (211 kB)
    Requirement already satisfied: requests<3.0.0dev,>=2.18.0 in e:\anaconda3\lib\site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.0,<3.0.0dev,>=1.31.5->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (2.26.0)
    Collecting pyasn1-modules>=0.2.1
      Downloading pyasn1_modules-0.2.8-py2.py3-none-any.whl (155 kB)
    Collecting cachetools<6.0,>=2.0.0
      Downloading cachetools-5.1.0-py3-none-any.whl (9.2 kB)
    Collecting rsa<5,>=3.1.4
      Downloading rsa-4.8-py3-none-any.whl (39 kB)
    Collecting google-cloud-core<3.0.0dev,>=1.4.1
      Downloading google_cloud_core-2.3.0-py2.py3-none-any.whl (29 kB)
    Requirement already satisfied: python-dateutil<3.0dev,>=2.7.2 in e:\anaconda3\lib\site-packages (from google-cloud-bigquery>=0.31.0->tf-models-official==2.3.0->tflite-model-maker) (2.8.2)
    Collecting proto-plus>=1.15.0
      Downloading proto_plus-1.20.4-py3-none-any.whl (46 kB)
    Collecting google-cloud-bigquery-storage<3.0.0dev,>=2.0.0
      Downloading google_cloud_bigquery_storage-2.13.1-py2.py3-none-any.whl (180 kB)
    

    ERROR: Cannot uninstall 'llvmlite'. It is a distutils installed project and thus we cannot accurately determine which files belong to it which would lead to only a partial uninstall.
    

    Collecting grpcio<2.0dev,>=1.38.1
      Downloading grpcio-1.46.3-cp39-cp39-win_amd64.whl (3.5 MB)
    Collecting pyarrow<9.0dev,>=3.0.0
      Downloading pyarrow-8.0.0-cp39-cp39-win_amd64.whl (17.9 MB)
    Collecting google-resumable-media<3.0dev,>=0.6.0
      Downloading google_resumable_media-2.3.3-py2.py3-none-any.whl (76 kB)
    Collecting grpcio-status<2.0dev,>=1.33.2
      Downloading grpcio_status-1.46.3-py3-none-any.whl (10.0 kB)
    Collecting google-crc32c<2.0dev,>=1.0
      Downloading google_crc32c-1.3.0-cp39-cp39-win_amd64.whl (27 kB)
    Requirement already satisfied: pyparsing!=3.0.0,!=3.0.1,!=3.0.2,!=3.0.3,<4,>=2.4.2 in e:\anaconda3\lib\site-packages (from httplib2<1dev,>=0.15.0->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (3.0.4)
    Requirement already satisfied: certifi in e:\anaconda3\lib\site-packages (from kaggle>=1.3.9->tf-models-official==2.3.0->tflite-model-maker) (2021.10.8)
    Requirement already satisfied: tqdm in e:\anaconda3\lib\site-packages (from kaggle>=1.3.9->tf-models-official==2.3.0->tflite-model-maker) (4.62.3)
    Requirement already satisfied: python-slugify in e:\anaconda3\lib\site-packages (from kaggle>=1.3.9->tf-models-official==2.3.0->tflite-model-maker) (5.0.2)
    Requirement already satisfied: kiwisolver>=1.0.1 in e:\anaconda3\lib\site-packages (from matplotlib<3.5.0,>=3.0.3->tflite-model-maker) (1.3.1)
    Requirement already satisfied: cycler>=0.10 in e:\anaconda3\lib\site-packages (from matplotlib<3.5.0,>=3.0.3->tflite-model-maker) (0.10.0)
    Requirement already satisfied: attrs in e:\anaconda3\lib\site-packages (from neural-structured-learning>=1.3.1->tflite-model-maker) (21.2.0)
    Requirement already satisfied: pytz>=2017.3 in e:\anaconda3\lib\site-packages (from pandas>=0.22.0->tf-models-official==2.3.0->tflite-model-maker) (2021.3)
    Requirement already satisfied: appdirs>=1.3.0 in e:\anaconda3\lib\site-packages (from pooch>=1.0->librosa==0.8.1->tflite-model-maker) (1.4.4)
    Collecting pyasn1<0.5.0,>=0.4.6
      Downloading pyasn1-0.4.8-py2.py3-none-any.whl (77 kB)
    Requirement already satisfied: idna<4,>=2.5 in e:\anaconda3\lib\site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.0,<3.0.0dev,>=1.31.5->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (3.2)
    Requirement already satisfied: charset-normalizer~=2.0.0 in e:\anaconda3\lib\site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.0,<3.0.0dev,>=1.31.5->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (2.0.4)
    Requirement already satisfied: threadpoolctl>=2.0.0 in e:\anaconda3\lib\site-packages (from scikit-learn!=0.19.0,>=0.14.0->librosa==0.8.1->tflite-model-maker) (2.2.0)
    Requirement already satisfied: cffi>=1.0 in e:\anaconda3\lib\site-packages (from soundfile>=0.10.2->librosa==0.8.1->tflite-model-maker) (1.14.6)
    Requirement already satisfied: pycparser in e:\anaconda3\lib\site-packages (from cffi>=1.0->soundfile>=0.10.2->librosa==0.8.1->tflite-model-maker) (2.20)
    Collecting tensorflow-estimator<2.10.0,>=2.9.0rc0
      Downloading tensorflow_estimator-2.9.0-py2.py3-none-any.whl (438 kB)
    Collecting tensorflow-io-gcs-filesystem>=0.23.1
      Downloading tensorflow_io_gcs_filesystem-0.26.0-cp39-cp39-win_amd64.whl (1.5 MB)
    Collecting keras-preprocessing>=1.1.1
      Downloading Keras_Preprocessing-1.1.2-py2.py3-none-any.whl (42 kB)
    Collecting tensorboard<2.10,>=2.9
      Downloading tensorboard-2.9.0-py3-none-any.whl (5.8 MB)
    Collecting libclang>=13.0.0
      Downloading libclang-14.0.1-py2.py3-none-win_amd64.whl (14.2 MB)
    Requirement already satisfied: h5py>=2.9.0 in e:\anaconda3\lib\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (3.2.1)
    Requirement already satisfied: typing-extensions>=3.6.6 in e:\anaconda3\lib\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (3.10.0.2)
    Collecting keras<2.10.0,>=2.9.0rc0
      Downloading keras-2.9.0-py2.py3-none-any.whl (1.6 MB)
    Requirement already satisfied: wrapt>=1.11.0 in e:\anaconda3\lib\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (1.12.1)
    Collecting opt-einsum>=2.3.2
      Downloading opt_einsum-3.3.0-py3-none-any.whl (65 kB)
    Collecting gast<=0.4.0,>=0.2.1
      Downloading gast-0.4.0-py3-none-any.whl (9.8 kB)
    Collecting google-pasta>=0.1.1
      Downloading google_pasta-0.2.0-py3-none-any.whl (57 kB)
    Collecting astunparse>=1.6.0
      Downloading astunparse-1.6.3-py2.py3-none-any.whl (12 kB)
    Collecting protobuf>=3.12.0
      Downloading protobuf-3.19.4-cp39-cp39-win_amd64.whl (895 kB)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in e:\anaconda3\lib\site-packages (from astunparse>=1.6.0->tensorflow>=2.6.0->tflite-model-maker) (0.37.0)
    Collecting google-auth-oauthlib<0.5,>=0.4.1
      Downloading google_auth_oauthlib-0.4.6-py2.py3-none-any.whl (18 kB)
    Collecting tensorboard-data-server<0.7.0,>=0.6.0
      Downloading tensorboard_data_server-0.6.1-py3-none-any.whl (2.4 kB)
    Collecting tensorboard-plugin-wit>=1.6.0
      Downloading tensorboard_plugin_wit-1.8.1-py3-none-any.whl (781 kB)
    Requirement already satisfied: werkzeug>=1.0.1 in e:\anaconda3\lib\site-packages (from tensorboard<2.10,>=2.9->tensorflow>=2.6.0->tflite-model-maker) (2.0.2)
    Collecting markdown>=2.6.8
      Downloading Markdown-3.3.7-py3-none-any.whl (97 kB)
    Collecting requests-oauthlib>=0.7.0
      Downloading requests_oauthlib-1.3.1-py2.py3-none-any.whl (23 kB)
    Requirement already satisfied: importlib-metadata>=4.4 in e:\anaconda3\lib\site-packages (from markdown>=2.6.8->tensorboard<2.10,>=2.9->tensorflow>=2.6.0->tflite-model-maker) (4.8.1)
    Requirement already satisfied: zipp>=0.5 in e:\anaconda3\lib\site-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard<2.10,>=2.9->tensorflow>=2.6.0->tflite-model-maker) (3.6.0)
    Collecting oauthlib>=3.0.0
      Downloading oauthlib-3.2.0-py3-none-any.whl (151 kB)
    Collecting typeguard>=2.7
      Downloading typeguard-2.13.3-py3-none-any.whl (17 kB)
    Collecting promise
      Downloading promise-2.3.tar.gz (19 kB)
    Collecting dill
      Downloading dill-0.3.5.1-py2.py3-none-any.whl (95 kB)
    Collecting tensorflow-metadata
      Downloading tensorflow_metadata-1.8.0-py3-none-any.whl (50 kB)
    Collecting dm-tree~=0.1.1
      Downloading dm_tree-0.1.7-cp39-cp39-win_amd64.whl (90 kB)
    Collecting packaging>=20.0
      Downloading packaging-20.9-py2.py3-none-any.whl (40 kB)
    Collecting pybind11>=2.6.0
      Downloading pybind11-2.9.2-py2.py3-none-any.whl (213 kB)
    Collecting sounddevice>=0.4.4
      Downloading sounddevice-0.4.4-py3-none-win_amd64.whl (195 kB)
    Requirement already satisfied: text-unidecode>=1.3 in e:\anaconda3\lib\site-packages (from python-slugify->kaggle>=1.3.9->tf-models-official==2.3.0->tflite-model-maker) (1.3)
    Requirement already satisfied: colorama in e:\anaconda3\lib\site-packages (from tqdm->kaggle>=1.3.9->tf-models-official==2.3.0->tflite-model-maker) (0.4.4)
    Building wheels for collected packages: audioread, fire, kaggle, py-cpuinfo, resampy, termcolor, promise
      Building wheel for audioread (setup.py): started
      Building wheel for audioread (setup.py): finished with status 'done'
      Created wheel for audioread: filename=audioread-2.1.9-py3-none-any.whl size=23154 sha256=8ade2b09d7e6f6c985b14fcae1f32a0b59d4f78a4c7fe7e8aab85734f05e579c
      Stored in directory: c:\users\11095\appdata\local\pip\cache\wheels\a2\a3\bd\ec1568ce7515115a11ab686d509ad302124c782af065de47ee
      Building wheel for fire (setup.py): started
      Building wheel for fire (setup.py): finished with status 'done'
      Created wheel for fire: filename=fire-0.4.0-py2.py3-none-any.whl size=115943 sha256=ab486ec6fc40c47d165a1432a894f46c8e514986c4fddf0ef56d37764d97d8de
      Stored in directory: c:\users\11095\appdata\local\pip\cache\wheels\2a\93\86\8cd17bc6c40fb605c3ac549d0b860ef7e84ee5f67bf01a3287
      Building wheel for kaggle (setup.py): started
      Building wheel for kaggle (setup.py): finished with status 'done'
      Created wheel for kaggle: filename=kaggle-1.5.12-py3-none-any.whl size=73051 sha256=7c231380d03f4db469a4a81e335cb79c9c1da5cfbf05726e8cd4ac0ff4c130b7
      Stored in directory: c:\users\11095\appdata\local\pip\cache\wheels\ac\b2\c3\fa4706d469b5879105991d1c8be9a3c2ef329ba9fe2ce5085e
      Building wheel for py-cpuinfo (setup.py): started
      Building wheel for py-cpuinfo (setup.py): finished with status 'done'
      Created wheel for py-cpuinfo: filename=py_cpuinfo-8.0.0-py3-none-any.whl size=22258 sha256=29856ba3a11e39a2a520c81ba20116c57f0e793b26e2c7bdb75053e2e6c3cb27
      Stored in directory: c:\users\11095\appdata\local\pip\cache\wheels\a9\33\c2\bcf6550ff9c95f699d7b2f261c8520b42b7f7c33b6e6920e29
      Building wheel for resampy (setup.py): started
      Building wheel for resampy (setup.py): finished with status 'done'
      Created wheel for resampy: filename=resampy-0.2.2-py3-none-any.whl size=320731 sha256=a3493b67f6d3497673f9173e0162383a525aee8b8ca973ff6e5eb9e25bbd111d
      Stored in directory: c:\users\11095\appdata\local\pip\cache\wheels\86\2c\7d\46a32a246b0e5939cea2c5ec1492164073e0c5d16d666ae2cd
      Building wheel for termcolor (setup.py): started
      Building wheel for termcolor (setup.py): finished with status 'done'
      Created wheel for termcolor: filename=termcolor-1.1.0-py3-none-any.whl size=4847 sha256=a0f59a49f6d254f83430db0f0d57713dade390bd64794a12da3172e00ab67f41
      Stored in directory: c:\users\11095\appdata\local\pip\cache\wheels\b6\0d\90\0d1bbd99855f99cb2f6c2e5ff96f8023fad8ec367695f7d72d
      Building wheel for promise (setup.py): started
      Building wheel for promise (setup.py): finished with status 'done'
      Created wheel for promise: filename=promise-2.3-py3-none-any.whl size=21502 sha256=1a9a692f55f0403ad3dc9c9d557456162ee2d4e8a1f93760b7a50937c9e963aa
      Stored in directory: c:\users\11095\appdata\local\pip\cache\wheels\e1\e8\83\ddea66100678d139b14bc87692ece57c6a2a937956d2532608
    Successfully built audioread fire kaggle py-cpuinfo resampy termcolor promise
    Installing collected packages: pyasn1, urllib3, rsa, pyasn1-modules, protobuf, cachetools, oauthlib, grpcio, googleapis-common-protos, google-auth, requests-oauthlib, grpcio-status, google-api-core, tensorboard-plugin-wit, tensorboard-data-server, proto-plus, markdown, llvmlite, httplib2, google-crc32c, google-auth-oauthlib, absl-py, uritemplate, typeguard, termcolor, tensorflow-metadata, tensorflow-io-gcs-filesystem, tensorflow-estimator, tensorboard, pyarrow, promise, packaging, opt-einsum, numba, libclang, keras-preprocessing, keras, google-resumable-media, google-pasta, google-cloud-core, google-cloud-bigquery-storage, google-auth-httplib2, gast, flatbuffers, dm-tree, dill, astunparse, tf-slim, tensorflow-model-optimization, tensorflow-hub, tensorflow-datasets, tensorflow-addons, tensorflow, soundfile, sounddevice, sentencepiece, resampy, pybind11, py-cpuinfo, pooch, opencv-python-headless, kaggle, google-cloud-bigquery, google-api-python-client, gin-config, dataclasses, audioread, tflite-support, tf-models-official, tensorflowjs, neural-structured-learning, librosa, fire, tflite-model-maker
      Attempting uninstall: urllib3
        Found existing installation: urllib3 1.26.7
        Uninstalling urllib3-1.26.7:
          Successfully uninstalled urllib3-1.26.7
      Attempting uninstall: llvmlite
        Found existing installation: llvmlite 0.37.0
    

#### 2、出现Cannot uninstall 'llvmlite'报错，利用Anaconda Navigator中Environments组件管理卸载llvmlite包

[![XVEVat.jpg](https://img-blog.csdnimg.cn/img_convert/8e2a9207c6e159a072bf9cdc9d536d6d.png)](https://imgtu.com/i/XVEVat)


#### 3、再次安装


#### 4、出现conda-repo-cli未安装、pywinpty版本过高问题
conda-repo-cli 1.0.4 requires pathlib, which is not installed.
jupyter-server 1.13.5 requires pywinpty<2; os_name == "nt", but you have pywinpty 2.0.2 which is incompatible.'
报错，提示没安装conda-repo-cli 1.0.4，以及pywinpty过高，解决：安装conda-repo-cli 1.0.4，降低pywinpty版本

（1）安装conda-repo-cli 1.0.4
```python
!pip install conda-repo-cli==1.0.4
```

    Requirement already satisfied: conda-repo-cli==1.0.4 in e:\anaconda3\lib\site-packages (1.0.4)
    Requirement already satisfied: six in e:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (1.16.0)
    Requirement already satisfied: pytz in e:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (2021.3)
    Requirement already satisfied: PyYAML>=3.12 in e:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (6.0)
    Requirement already satisfied: clyent>=1.2.0 in e:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (1.2.2)
    Collecting pathlib
      Downloading pathlib-1.0.1-py3-none-any.whl (14 kB)
    Requirement already satisfied: requests>=2.9.1 in e:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (2.27.1)
    Requirement already satisfied: setuptools in e:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (61.2.0)
    Requirement already satisfied: nbformat>=4.4.0 in e:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (5.3.0)
    Requirement already satisfied: python-dateutil>=2.6.1 in e:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (2.8.2)
    Requirement already satisfied: jupyter-core in e:\anaconda3\lib\site-packages (from nbformat>=4.4.0->conda-repo-cli==1.0.4) (4.10.0)
    Requirement already satisfied: traitlets>=4.1 in e:\anaconda3\lib\site-packages (from nbformat>=4.4.0->conda-repo-cli==1.0.4) (5.1.1)
    Requirement already satisfied: jsonschema>=2.6 in e:\anaconda3\lib\site-packages (from nbformat>=4.4.0->conda-repo-cli==1.0.4) (4.4.0)
    Requirement already satisfied: fastjsonschema in e:\anaconda3\lib\site-packages (from nbformat>=4.4.0->conda-repo-cli==1.0.4) (2.15.1)
    Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in e:\anaconda3\lib\site-packages (from jsonschema>=2.6->nbformat>=4.4.0->conda-repo-cli==1.0.4) (0.18.0)
    Requirement already satisfied: attrs>=17.4.0 in e:\anaconda3\lib\site-packages (from jsonschema>=2.6->nbformat>=4.4.0->conda-repo-cli==1.0.4) (21.4.0)
    Requirement already satisfied: certifi>=2017.4.17 in e:\anaconda3\lib\site-packages (from requests>=2.9.1->conda-repo-cli==1.0.4) (2022.5.18.1)
    Requirement already satisfied: charset-normalizer~=2.0.0 in e:\anaconda3\lib\site-packages (from requests>=2.9.1->conda-repo-cli==1.0.4) (2.0.4)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in e:\anaconda3\lib\site-packages (from requests>=2.9.1->conda-repo-cli==1.0.4) (1.25.11)
    Requirement already satisfied: idna<4,>=2.5 in e:\anaconda3\lib\site-packages (from requests>=2.9.1->conda-repo-cli==1.0.4) (3.3)
    Requirement already satisfied: pywin32>=1.0 in e:\anaconda3\lib\site-packages (from jupyter-core->nbformat>=4.4.0->conda-repo-cli==1.0.4) (302)
    Installing collected packages: pathlib
    Successfully installed pathlib-1.0.1
    

（2）降低pywinpty版本

[![XVEEVI.jpg](https://img-blog.csdnimg.cn/img_convert/4be707de3c1fb63f17a6911d0cf6e1ae.png)](https://imgtu.com/i/XVEEVI)


#### 5、安装成功，导入相关的库



```python
import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt

```

### 模型训练

#### 1、获取数据


```python
image_path = tf.keras.utils.get_file(
      'flower_photos.tgz',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      extract=True)
image_path = os.path.join(os.path.dirname(image_path), 'flower_photos')

```

    Downloading data from https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    228813984/228813984 [==============================] - 39s 0us/step
    

#### 2、运行示例

##### （1）加载数据集，并将数据集分为训练数据和测试数据。


```python
data = DataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)
```

    INFO:tensorflow:Load image with size: 3670, num_label: 5, labels: daisy, dandelion, roses, sunflowers, tulips.
    

##### （2）训练Tensorflow模型
由于没有设置翻墙，程序执行如下语句时
model = image_classifier.create(train_data)
会因为模型下载超时而报错：urllib.error.URLError: <urlopen error [Errno 110] Connection timed out>
原因：在不指定模型路径情况相下，系统默认使用的是efficientnet_lite0模型，对应路径是
https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2 无法下载导致报错
解决方法：
修改访问网址：https://storage.googleapis.com/tfhub-modules/tensorflow/efficientnet/lite0/feature-vector/2.tar.gz


```python
inception_v3_spec = image_classifier.ModelSpec(uri='https://storage.googleapis.com/tfhub-modules/tensorflow/efficientnet/lite0/feature-vector/2.tar.gz')
inception_v3_spec.input_image_shape = [240, 240]
model = image_classifier.create(train_data, model_spec=inception_v3_spec)
```

    INFO:tensorflow:Retraining the models...
    

    INFO:tensorflow:Retraining the models...
    

    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     hub_keras_layer_v1v2_1 (Hub  (None, 1280)             3413024   
     KerasLayerV1V2)                                                 
                                                                     
     dropout_1 (Dropout)         (None, 1280)              0         
                                                                     
     dense_1 (Dense)             (None, 5)                 6405      
                                                                     
    =================================================================
    Total params: 3,419,429
    Trainable params: 6,405
    Non-trainable params: 3,413,024
    _________________________________________________________________
    None
    Epoch 1/5
    

    E:\anaconda3\lib\site-packages\keras\optimizers\optimizer_v2\gradient_descent.py:108: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
      super(SGD, self).__init__(name, **kwargs)
    

    103/103 [==============================] - 84s 801ms/step - loss: 0.9046 - accuracy: 0.7482
    Epoch 2/5
    103/103 [==============================] - 80s 776ms/step - loss: 0.6660 - accuracy: 0.8935
    Epoch 3/5
    103/103 [==============================] - 77s 743ms/step - loss: 0.6284 - accuracy: 0.9172
    Epoch 4/5
    103/103 [==============================] - 76s 740ms/step - loss: 0.6141 - accuracy: 0.9229
    Epoch 5/5
    103/103 [==============================] - 77s 743ms/step - loss: 0.5941 - accuracy: 0.9281
    

##### （3）评估模型


```python
loss, accuracy = model.evaluate(test_data)
```

    12/12 [==============================] - 10s 687ms/step - loss: 0.5974 - accuracy: 0.9183
    

##### （4）导出Tensorflow Lite模型


```python
model.export(export_dir='.')
```

    INFO:tensorflow:Assets written to: C:\Users\11095\AppData\Local\Temp\tmpabnr6auz\assets
    

    INFO:tensorflow:Assets written to: C:\Users\11095\AppData\Local\Temp\tmpabnr6auz\assets
    E:\anaconda3\lib\site-packages\tensorflow\lite\python\convert.py:766: UserWarning: Statistics for quantized inputs were expected, but not specified; continuing anyway.
      warnings.warn("Statistics for quantized inputs were expected, but not "
    

    INFO:tensorflow:Label file is inside the TFLite model with metadata.
    

    INFO:tensorflow:Label file is inside the TFLite model with metadata.
    

    INFO:tensorflow:Saving labels in C:\Users\11095\AppData\Local\Temp\tmpkqioa6nf\labels.txt
    

    INFO:tensorflow:Saving labels in C:\Users\11095\AppData\Local\Temp\tmpkqioa6nf\labels.txt
    

    INFO:tensorflow:TensorFlow Lite model exported successfully: .\model.tflite
    

    INFO:tensorflow:TensorFlow Lite model exported successfully: .\model.tflite
    

导出的模型存放在Jupyter Notebook当前的工作目录中。


### 实现效果


[![XVEkqA.jpg](https://img-blog.csdnimg.cn/img_convert/92a64a90d28a5dfd94d68c7d52e0cb2a.png)](https://imgtu.com/i/XVEkqA)




