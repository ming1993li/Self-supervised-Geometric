# Self-supervised Geometric Features Discovery with Interpretable Attention for Vehicle Re-Identification and Beyond
## Introduction
This is the code of our [ICCV21 paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Self-Supervised_Geometric_Features_Discovery_via_Interpretable_Attention_for_Vehicle_Re-Identification_ICCV_2021_paper.pdf).

## Datasets
+ [veri776](https://github.com/VehicleReId/VeRidataset)
+ [vehicleID](https://pkuml.org/resources/pku-vehicleid.html)

## Tutorial
### train
Input arguments for the training scripts are unified in [args.py](./args.py).

```
python train.py
```
### test
Use --evaluate to switch to the evaluation mode.

## BibTeX
If you use this code in your project, please cite our paper:
```bibtex
@inproceedings{li2021self,
  title={Self-Supervised Geometric Features Discovery via Interpretable Attention for Vehicle Re-Identification and Beyond},
  author={Li, Ming and Huang, Xinming and Zhang, Ziming},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={194--204},
  year={2021}
}
```

## Thanks
Our code refers to [ReID strong baseline](https://github.com/michuanhaohao/reid-strong-baseline) and [D2Net](https://github.com/mihaidusmanu/d2-net).
