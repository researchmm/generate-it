# Bi-directional Image and Text Generation

## UMT-BITG ([image & text generator](it-generator/README.md))
[Unifying Multimodal Transformer for Bi-directional Image and Text Generation](https://arxiv.org/abs/2110.09753), \
Yupan Huang, Bei Liu, Yutong Lu, in ACM MM 2021 (Industrial Track).

## UMT-DBITG ([diverse image & text generator](diverse-it-generator/README.md))
[A Picture is Worth a Thousand Words: A Unified System for Diverse Captions and Rich Images Generation](https://arxiv.org/abs/2110.09756), \
Yupan Huang, Bei Liu, Jianlong Fu, Yutong Lu, in ACM MM 2021 (Video and Demo Track).

Poster or slides are available in the `assets` folder by visiting [OneDrive](https://mail2sysueducn-my.sharepoint.com/:f:/g/personal/huangyp28_mail2_sysu_edu_cn/EkCFDwd2bQpKtYwyBi3A8ukBHWyNMQ_Tkw9ZeQhYOTMTBA?e=xsMWPO).

## Data & Pre-trained Models
Download preprocessed data and our pre-trained models by visiting [OneDrive](https://mail2sysueducn-my.sharepoint.com/:f:/g/personal/huangyp28_mail2_sysu_edu_cn/EkCFDwd2bQpKtYwyBi3A8ukBHWyNMQ_Tkw9ZeQhYOTMTBA?e=xsMWPO).
We suggest following our data structures, which is consistent with the paths in `config.py`. You may need to modify the `root_path` in `config.py`.
In addition, please following the instructions to prepare some other data:
* Download grid features in path `data/grid_features` provided by X-LXMERT or follow [feature extraction](https://github.com/allenai/x-lxmert/blob/master/feature_extraction/README.md) to extract these features.
  ```
  wget https://ai2-vision-x-lxmert.s3-us-west-2.amazonaws.com/butd_features/COCO/maskrcnn_train_grid8.h5 -P data/grid_features
  wget https://ai2-vision-x-lxmert.s3-us-west-2.amazonaws.com/butd_features/COCO/maskrcnn_valid_grid8.h5 -P data/grid_features
  wget https://ai2-vision-x-lxmert.s3-us-west-2.amazonaws.com/butd_features/COCO/maskrcnn_test_grid8.h5 -P data/grid_features
  ```
* For text-to-image evaluation on MSCOCO dataset, we need the real images to calculate the FID metric.
  For UMT-DBITG, we use MSCOCO karpathy split, which has been included in the OneDrive folder (`images/imgs_karpathy`).
  For UMT-BITG, please download [MSCOCO validation set](http://images.cocodataset.org/zips/val2014.zip) in path `images/coco_val2014`.


## Citation
If you like our paper or code, please generously cite us:
```
@inproceedings{huang2021unifying,
  author    = {Yupan Huang and Bei Liu and Yutong Lu},
  title     = {Unifying Multimodal Transformer for Bi-directional Image and Text Generation},
  booktitle = {Proceedings of the 29th ACM International Conference on Multimedia},
  year      = {2021}
}

@inproceedings{huang2021diverse,
  author    = {Yupan Huang and Bei Liu and Jianlong Fu and Yutong Lu},
  title     = {A Picture is Worth a Thousand Words: A Unified System for Diverse Captions and Rich Images Generation},
  booktitle = {Proceedings of the 29th ACM International Conference on Multimedia},
  year      = {2021}
}
```

## Acknowledgement
Our code is mainly based on [LaBERT](https://github.com/bearcatt/LaBERT) and [X-LXMERT](https://github.com/allenai/x-lxmert). 
Our text-to-image generation evaluation code is mainly based on [CLIP](https://github.com/openai/CLIP), [pytorch-fid](https://github.com/mseitzer/pytorch-fid/tree/802da3963113b5b5f8154e0e27580ee4c97460ab) and [inception_score](https://github.com/openai/improved-gan/blob/master/inception_score/README.md).
We sincerely thank them for their contributions!

Feel free to open issues or email to me for help to use this code. Any feedback is welcome!
