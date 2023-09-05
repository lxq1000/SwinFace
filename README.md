# SwinFace
Official Pytorch Implementation of the paper, "SwinFace: A Multi-task Transformer for Face Recognition, Facial Expression Recognition, Age Estimation and Face Attribute Estimation"（https://arxiv.org/pdf/2308.11509.pdf）

##
In recent years, vision transformers have been introduced into face recognition and analysis and have achieved performance breakthroughs. 
However, most previous methods generally train a single model or an ensemble of models to perform the desired task, which ignores the synergy among different tasks and fails to achieve improved prediction accuracy, increased data efficiency, and reduced training time. 
This paper presents a multi-purpose algorithm for simultaneous face recognition, facial expression recognition, age estimation, and face attribute estimation (40 attributes including gender) based on a single Swin Transformer. 
Our design, the SwinFace, consists of a single shared backbone together with a subnet for each set of related tasks. 
To address the conflicts among multiple tasks and meet the different demands of tasks, a Multi-Level Channel Attention (MLCA) module is integrated into each task-specific analysis subnet, which can adaptively select the features from optimal levels and channels to perform the desired tasks. 
Extensive experiments show that the proposed model has a better understanding of the face and achieves excellent performance for all tasks.
Especially, it achieves 90.97\% accuracy on RAF-DB and 0.22 $\epsilon$-error on CLAP2015, which are state-of-the-art results on facial expression recognition and age estimation respectively.

<img src="https://github.com/lxq1000/SwinFace/blob/main/pictures/SwinFace.png" alt="Image" width="500">

## Evaluate
Here are some test results. For detailed experimental information, please refer to our paper.

- Face Recognition
<img src="https://github.com/lxq1000/SwinFace/blob/main/pictures/face%20recognition.png" alt="Image">

- Facial Expression Recognition
<img src="https://github.com/lxq1000/SwinFace/blob/main/pictures/facial%20expression%20recognition.png" alt="Image" width="500">

- Age Estimation
<img src="https://github.com/lxq1000/SwinFace/blob/main/pictures/age%20estimation.png" alt="Image" width="400">

- Facial Attribute Estimation  
<img src="https://github.com/lxq1000/SwinFace/blob/main/pictures/facial%20attribute%20estimation.png" alt="Image">



## Train and Inference

The `train.sh` file provides the necessary commands for training the model.

The `inference.py` file provides an example of using SwinFace for inference.

The trained model file can be downloaded from the following link：

  Google Drive：https://drive.google.com/drive/folders/1NjVN3Kp_Tmwt17hWCIWgHpuWzkHYaman?usp=sharing






