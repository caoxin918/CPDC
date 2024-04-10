# CPDC
# Note: If your work uses this algorithm or makes improvements based on it, please be sure to cite this paper. Thank you for your cooperation.

# 注意：如果您的工作用到了本算法，或者基于本算法进行了改进，请您务必引用本论文，谢谢配合



# CPDC-MFNet: Conditional Point diffusion completion network with Muti-scale Feedback Refine for 3D Terracotta Warriors

*XueLi Xu* <sup>1,2,3</sup>, *Da Song*<sup>1</sup>, *Guohua Geng*<sup>1,3</sup>,*Mingquan Zhou*<sup>1,3</sup>, *Jie Liu*<sup>4,5</sup>,*Kang LI*<sup>1,3</sup>, *Xin Cao<sup>1,3,*</sup>

1.School of Information Science and Technology, Northwest University, Xi’an, Shaanxi 710127, China

2.Yan’an University, Yan'an, Shaanxi 716000, China

3.National and Local Joint Engineering Research Center for cultural Heritage Digitization, Xi’an, Shaanxi 710127, China

4.College of Computer and Information Engineering, Henan Normal University, Xinxiang, Henan,453007, China

5.Big Data Engineering Laboratory for Teaching Resources & Assessment of Education Quality, Xinxiang Henan 453007, China

This repository is the official implementation of CPDC-MFNet: Conditional Point diffusion completion network with Muti-scale Feedback Refine for 3D Terracotta Warriors, Scientific Reports. 2024.

Please feel free to reach out for any questions or discussions!


## Requirements:

Make sure the following environments are installed.

```
python==3.6
pytorch==1.4.0
torchvision==0.5.0
cudatoolkit==10.1
matplotlib==2.2.5
tqdm==4.32.1
open3d==0.9.0
trimesh=3.7.12
scipy==1.5.1
```

Install PyTorchEMD by

```
cd metrics/PyTorchEMD
python setup.py install
cp build/**/emd_cuda.cpython-36m-x86_64-linux-gnu.so .
```

## Datasets 

We use ShapeNet rendering provided by [GenRe](https://github.com/xiumingzhang/GenRe-ShapeHD) for completion.

The dataset of Terracotta available from the corresponding author on reasonable request.

## Training

```bash
# Train a generator
python train_gen.py

# Train a feedbackNet
python FBNET.py 
```


## Testing

```bash
# Test a generator
python test_gen.py 
```

## Citation

```
@article{song2023CPDC,
  title={CPDC-MFNet: Conditional Point diffusion completion network with Muti-scale Feedback Refine for 3D Terracotta Warriors},
  author={Xueli Xu, Da Song, Guohua Geng, Mingquan Zhou, Jie Liu, Kang Li and Xin Cao},
}
```

## Acknowledgements

We would like to thank and acknowledge referenced codes from the following repositories:

https://github.com/WangYueFt/dgcnn

https://github.com/charlesq34/pointnet

https://github.com/charlesq34/pointnet2

https://github.com/AnTao97/dgcnn.pytorch
