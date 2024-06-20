# Inject Backdoor in Measured Data to Jeopardize Full-Stack Medical Image Analysis System
The source code of "Inject Backdoor in Measured Data to Jeopardize Full-Stack Medical Image Analysis System" (Accepted by MICCAI 2024)

#### Abstract
Deep learning has achieved remarkable success in the medical domain, which makes it crucial to assess its vulnerabilities in medical systems. This study examines backdoor attack (BA) methods to evaluate the reliability and security of medical image analysis systems. However, most BA methods focus on isolated downstream tasks and are considered post-imaging attacks, missing a comprehensive security assessment of the full-stack medical image analysis systems from data acquisition to analysis. Reconstructing images from measured data for downstream tasks requires complex transformations, which challenge the design of triggers in the measurement domain. Typically, hackers only access measured data in scanners. To tackle this challenge, this paper introduces a novel Learnable Trigger Generation Method~(LTGM) for measured data. This pre-imaging attack method aims to attack the downstream task without compromising the reconstruction process or imaging quality. LTGM employs a trigger function in the measurement domain to inject a learned trigger into the measured data. To avoid the bias from handcrafted knowledge, this trigger is formulated by learning from the gradients of two key tasks: reconstruction and analysis. Crucially, LTGM's trigger strives to balance its impact on analysis with minimal additional noise and artifacts in the reconstructed images by carefully analyzing gradients from both tasks. Comprehensive experiments have been conducted to demonstrate the vulnerabilities in full-stack medical systems and to validate the effectiveness of the proposed method using the public dataset.

#### Citation
If our work is valuable to you, please cite our work:
```
@inproceedings{yang2023ccnet,
  author={Yang, Ziyuan and Chen, Yingyu and Sun, Mengyu and Zhang, Yi},
  title={Inject Backdoor in Measured Data to Jeopardize Full-Stack Medical Image Analysis System},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year={2024},
  organization={Springer}
```
