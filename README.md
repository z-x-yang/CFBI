# CFBI: Collaborative Video Object Segmentation by Foreground-Background Integration
The official implementation of Collaborative Video Object Segmentation by Foreground-Background Integration (ECCV 2020, Spotlight). [[paper](https://arxiv.org/abs/2003.08333)] [[demo (YouTube)](https://www.youtube.com/watch?v=xdHi68UFt50)][[demo (Bilibili)](https://www.bilibili.com/video/av456456973/)]
Our journal extension, CFBI+, is availabel now. CFBI+ is more powerful and efficient than CFBI. [[paper](http://arxiv.org/abs/2010.06349)]

CFBI and CFBI+ work fine on both [PaddlePaddle](https://www.paddlepaddle.org.cn/) and PyTorch. Based on some necessary considerations, we only release the inference here.

**If you want to get the training code of CFBI, please contact us by email: <zongxin.yang@student.uts.edu.au>. 
And please inform us of your institution and the purpose of using CFBI in the email. 
Thank you for your understanding!**

Framework:
<div align=center><img src="https://github.com/z-x-yang/CFBI/raw/master/utils/overview.png" width="80%"/></div>

Some video segmentation results:
<div align=center><img src="https://github.com/z-x-yang/CFBI/raw/master/utils/quality.png" width="80%"/></div>

## Requirements
    1. Python3
    2. pytorch >= 1.3.0 and torchvision
    3. opencv-python and Pillow
## Getting Started
1. Prepare datasets:
    * Download the [validation split](https://drive.google.com/file/d/1-QrceIl5sUNTKz7Iq0UsWC6NLZq7girr/view?usp=sharing) of YouTube-VOS 2018, and decompress the file to `datasets/YTB/valid`. If you want to evaluate CFBI on YouTube-VOS 2019, please download this [split](https://drive.google.com/file/d/1o586Wjya-f2ohxYf9C1RlRH-gkrzGS8t/view?usp=sharing) instead.
    * Download 480p [TrainVal](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip) split of DAVIS 2017, and decompress the file to `datasets/DAVIS`.
2. Evaluating:
    * **YouTube-VOS**: Download pretrained CFBI, [ResNet101-CFBI](https://drive.google.com/file/d/1ZhoNOcDXGG-PpFXhCixs-L3yA255Wup8/view?usp=sharing), to `pretrain_models`, and then run `bash ytb_eval.sh`. After the evaluation, the result will be packed into a Zip file, which you need to send to [official evaluation server](https://competitions.codalab.org/competitions/19544) to calculate a score. For 2019 version, use this [server](https://competitions.codalab.org/competitions/20127) instead. The pretrained CFBI has been trained on YouTube-VOS using a larger batch size (16) and learning rate (0.02), which boosts the performance (J&F) to `81.8%` on the validation split of YouTube-VOS 2018.
    * **DAVIS**: Download pretrained CFBI, [ResNet101-CFBI-DAVIS](https://drive.google.com/file/d/1cRC-kEH5Is2dSnHrFoLIbTxXp4imZfj3/view?usp=sharing), to `pretrain_models`, and then run `bash davis_eval.sh`. After the evaluation, please use [official code](https://github.com/davisvideochallenge/davis2017-evaluation) to calculate a score, which should be `81.9%` (J&F).
    * **Fast CFBI**: To reduce memory usage, we also provide a fast setting in `ytb_eval_fast.sh`. The fast setting enables using `float16` in the matching process of CFBI. Besides, we apply an `atrous strategy` in the global matching of CFBI for further efficiency (The discussion of atrous matching will be submitted to our Arxiv paper soon). The fast setting will save a large amount of memory and significantly improve the inference speed of CFBI. However, this will only lose very little performance.
    * Another way for saving memory is to increase the number of `--global_chunks`. This will not affect performance but will make the network speed slightly slower.

## Model Zoo
**We recorded the inference speed of CFBI by using one NVIDIA Tesla V100 GPU. Besides, we used a multi-object speed instead of a single-object. A large portion of the sequences in VOS datasets contains multiple objects, and CFBI is good at processing multiple objects simultaneously.**

`F16` denotes using `float16` in the matching process. `Fast` means using both `float16` and `atrous strategy` in the inference stage. `MS` denotes using a multi-scale and flip strategy during inference.

**YouTube-VOS** (Eval on Val 2018):

In the inference stage, we restricted the long edge of each frame to be no more than 1040 (1.3 * 800) pixels, which is consistent with the biggest random-scale size (1.3 * 480p) in the training stage and is smaller than the original size of YouTube-VOS (720p).

**Name** | **Backbone**  | **J Seen** | **F Seen** | **J Unseen** | **F Unseen** | **Mean** | **Multi-Obj** <br> **FPS** | **Link** 
---------| :-----------: | :--------: | :--------: | :----------: | :----------: | :------: | :------------------------: | :------:
ResNet101-CFBI+ | ResNet101-DeepLabV3+ | 81.8 | **86.6** | **77.1** | **85.6** | **82.8** | 4.03 | Coming
ResNet101-MS-CFBI+ | ResNet101-DeepLabV3+ | **82.8** | **87.5** | **77.3** | **85.7** | **83.3** | 0.48 (88.1%↓) | The same as above
ResNet101-F16-CFBI+ | ResNet101-DeepLabV3+ | **81.9** | **86.6** | **77.1** | **85.6** | **82.8** | 4.93 (22.3%↑) | The same as above
ResNet101-Fast-CFBI+ | ResNet101-DeepLabV3+ | **81.9** | **86.6** | **77.1** | 85.5 | **82.8** | 5.06 (25.6%↑) | The same as above
ResNet101-CFBI | ResNet101-DeepLabV3+ | **81.9** | 86.3 | 75.6 | 83.4 | 81.8 | 3.48 | [Click](https://drive.google.com/file/d/1ZhoNOcDXGG-PpFXhCixs-L3yA255Wup8/view?usp=sharin) 
ResNet101-F16-CFBI | ResNet101-DeepLabV3+ |**81.9** | 86.4 | 75.6 | 83.3 | 81.8 | 4.62 (32.8%↑) | The same as above
ResNet101-Fast-CFBI | ResNet101-DeepLabV3+ | **81.9** | 86.4 | 75.6 | 83.1 |81.8 | **7.61 (118.7%↑)** | The same as above
MobileNetV2-CFBI | MobileNetV2-DeepLabV3+ | 80.4 | 84.7 | 74.9 | 82.6 | 80.6 | 3.88 | [Click](https://drive.google.com/file/d/1L_pA2UzBbOWyyJyNmgnXqnqDvFfkdXpO/view?usp=sharing)
MobileNetV2-Fast-CFBI | MobileNetV2-DeepLabV3+ | 80.2 | 84.6 | 74.7 | 82.7 | 80.6 | **9.69 (150.0↑%)** | The same as above

**DAVIS** (Eval on Val 2017):

In the inference stage, we ran using the default size of DAVIS (480p).

**Name** | **Backbone**  | **J score** | **F score** | **Mean** | **Multi-Obj** <br> **FPS** | **Link** 
---------| :-----------: | :---------: | :---------: | :------: | :------------------------: | :------:
ResNet101-CFBI+-DAVIS | ResNet101-DeepLabV3+ | **80.1** | **85.7** | **82.9** | **5.52** | Coming
ResNet101-CFBI-DAVIS | ResNet101-DeepLabV3+ | 79.3 | 84.5 | 81.9 | 5.88 | [Click](https://drive.google.com/file/d/1cRC-kEH5Is2dSnHrFoLIbTxXp4imZfj3/view?usp=sharing) 
ResNet101-F16-CFBI-DAVIS | ResNet101-DeepLabV3+ | 79.2 | 84.4 | 81.8 | 7.38 (25.5%↑) | The same as above
ResNet101-Fast-CFBI-DAVIS | ResNet101-DeepLabV3+ | 77.0 | 82.7 | 79.9 | **10.18 (73.1%↑)** | The same as above
MobileNetV2-CFBI-DAVIS | MobileNetV2-DeepLabV3+ | 76.5 | 80.3 | 78.4 | 6.94 | [Click](https://drive.google.com/file/d/1uuKlRqrPubJVRXKVr53cXuNFEQmmThGG/view?usp=sharing)
MobileNetV2-Fast-CFBI-DAVIS | MobileNetV2-DeepLabV3+ | 75.2 | 78.9 | 77.1 | **13.22 (90.5%↑)** | The same as above

## Citing
```
@inproceedings{yang2020collaborative,
  title={Collaborative video object segmentation by foreground-background integration},
  author={Yang, Zongxin and Wei, Yunchao and Yang, Yi},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2020}
}
@misc{yang2020collaborative,
      title={Collaborative Video Object Segmentation by Multi-Scale Foreground-Background Integration}, 
      author={Zongxin Yang and Yunchao Wei and Yi Yang},
      year={2020},
      eprint={2010.06349},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```




