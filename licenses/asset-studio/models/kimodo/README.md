---
license: other
license_name: nvidia-open-model-license
license_link: >-
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license
library_name: kimodo
tags:
  - nvidia
  - kimodo
  - rigplay
  - SOMA
---

# Kimodo: Controllable Kinematic Motion Diffusion at Scale

**[Paper](https://research.nvidia.com/labs/sil/projects/kimodo/assets/kimodo_tech_report.pdf), [Project Page](https://research.nvidia.com/labs/sil/projects/kimodo/)**

### Description:
Kimodo (**Ki**nematic **Mo**tion **D**iffusi**o**n) generates three-dimensional (3D) skeletal body animations given a text prompt and/or constraints on the motion like full-body poses, end-effector joint positions, paths, and waypoints to follow.

The Kimodo model family includes models trained on different skeletons and datasets:
* Kimodo-SOMA-RP
    * Trained on the 30-joint SOMA skeleton with the proprietary Bones Rigplay dataset.
* Kimodo-SOMA-SEED
    * Trained on the 30-joint SOMA skeleton with the open Bones-SEED dataset.
* Kimodo-G1-RP
    * Trained on the proprietary Bones Rigplay dataset retargeted to the 34-joint Unitree G1 robot skeleton.
* Kimodo-G1-SEED
    * Trained on the open Bones-SEED dataset retargeted to the 34-joint Unitree G1 robot skeleton.
* Kimodo-SMPLX-RP
    * Trained on the proprietary Bones Rigplay dataset retargeted to the 22-joint SMPLX-body skeleton.

This release pertains to Kimodo-SOMA-RP-v1.2, an updated Kimodo-SOMA-RP model. This model is ready for commercial use.

### Changes in v1.2:
Kimodo-SOMA-RP-v1.2 is trained on **corrected training splits**: in v1.1, some BONES-SEED test
sequences had leaked into the training set, so its numbers on the
[Motion Generation Benchmark](https://github.com/nv-tlabs/kimodo) are not fully comparable to models
trained on the recommended splits. v1.2 removes the leaked sequences, making it the model to compare
against on the benchmark. Compared to v1.1 it reduces
foot skating and matches constraint accuracy; its text-following scores (R-precision/FID) are slightly
lower, partly reflecting the removed leakage. See the benchmark results page in the codebase for the
full tables.

### License:

This model is released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).

### Deployment Geography:
Global

### Use Case: <br>
The model is intended for users with any level of animation experience to create 3D human motion data for their application. This may include:
* Demonstrations for humanoid robots
* Digital human motion for digital twin and industrial simulations
* Digital human motion for synthetic data
* Animations for game and media development

### Release Date:  <br>
Github [10/15/2026] via [link](https://github.com/nv-tlabs/kimodo) <br>
HuggingFace [10/15/2026] via [link](https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.2) <br>

## References:
* Technical report: [Kimodo: Scaling Controllable Human Motion Generation](https://research.nvidia.com/labs/sil/projects/kimodo/assets/kimodo_tech_report.pdf)
* Webpage: [link](https://research.nvidia.com/labs/sil/projects/kimodo/)

## Model Architecture:
**Architecture Type:** Diffusion Model <br>
**Network Architecture:** Novel Two-Stage Transformer <br>
**Model Size:** 282 M parameters

## Inputs: <br>
**Input Types:** Text, Duration (Num Frames), Pose Constraints <br>

**Input Formats:**
- Text: String
- Duration: Integer
- Pose Constraints: Matrix

**Input Parameters:**
- Text: One-Dimensional (1D)
- Duration: One-Dimensional (1D)
- Pose Constraints:
    - One-Dimensional (1D) frame index of each constraint
    - Features to constrain may include Three-Dimensional (3D) joint positions, (3x3) joint rotation matrices, Two-Dimensional (2D) heading direction, and/or Two-Dimensional (2D) root position

**Other Properties Related to Input:** Maximum duration is 10 sec (300 frames at 30 frames per second).

## Outputs

**Output Type:** Skeleton Motion: Root Translation and Joint Rotations <br>

**Output Formats:**
- Root Translation: Matrix
- Joint Rotations: Matrix

**Output Parameters:**
- Root Translation: Two-Dimensional (`num_frames` x 3)
- Joint Rotations: Four-Dimensional (`num_frames` x 30 x 3 x 3)

**Other Properties Related to Outupt:**
* Motions are at 30 frames per second (30 fps)

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions. <br>

## Software Integration:
**Runtime Engines:**
* PyTorch

**Supported Hardware Microarchitecture Compatibility:** <br>
* NVIDIA Ampere
* NVIDIA Blackwell
* NVIDIA Lovelace

**Supported Operating Systems:**
* Linux
* Windows

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment. <br>

## Model Version
Kimodo-SOMA-RP-v1.2

## Training and Testing Datasets:

**Name**: Proprietary Bones Rigplay Dataset

**Data Modalities**
* Text
* Human Motion Capture

**Data Size**:
* Less than 1 Billion tokens of text
* 700 hours of human motion capture

**Data Collection Method** <br>
Automatic/Sensors

**Labeling Method** <br>
Hybrid: Automatic/Sensors, Human

**Properties:** 700 hours of captured human body motions on the SOMA skeleton with corresponding text descriptions. Split into 90%/10% train/test splits. Various augmentations were employed to expand text and motion variety.

**Quantitative Evaluation** <br>
For test set evaluation, please refer to the benchmark results in [the repo](https://github.com/nv-tlabs/kimodo).

# Inference:
**Acceleration Engine:** N/A<br>

**Test Hardware:** <br>
* GeForce RTX 3090
* GeForce RTX 4090
* GeForce RTX 5090
* NVIDIA A100
* NVIDIA L40S
* NVIDIA L4
* NVIDIA RTX 6000 Ada
* NVIDIA RTX A6000

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

For more detailed information on ethical considerations for this model, please see the Bias, Explainability, Safety & Security, and Privacy Subcards below. <br>

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail).

## Bias

Field                                                                                               |  Response
:---------------------------------------------------------------------------------------------------|:---------------
Participation considerations from adversely impacted groups [protected classes](https://www.senate.ca.gov/content/protected-classes) in model design and testing:  |  Gender
Measures taken to mitigate against unwanted bias:                                                   |  Our training data contains motion captured from a roughly equal number of male and female actors

## Explainability

Field                                                                                                  |  Response
:------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------
Intended Task/Domain:                                                                   |  Character Animation
Model Type:                                                                                            |  Diffusion Transformer
Intended Users:                                                                                        |  The model is intended for users with any level of animation experience to create 3D human motion data for their application. This may include demonstrations for humanoid robots, digital human motion for simulations and synthetic data, and animations for games and media.
Output:                                                                                                |  3D skeletal animation (root translation and joint rotations)
Describe how the model works:                                                                          |  Text input and pose constraints are processed and given to a transformer-based model that iteratively denoises a sequence of body poses.
Name the adversely impacted groups this has been tested to deliver comparable outcomes regardless of:  |  Gender
Technical Limitations & Mitigation:                                                                    |  Generated motions may include artifacts like foot skating where feet slide unnaturally when they should be in static contact with the ground. The motion does not always follow the given text prompt, and the model does not know how to perform certain types of actions (e.g., the model is best at locomotion, gestures, combat, dancing, and everyday activities). Each trained model currently outputs motion for a single character skeleton. The model is designed to output realistic human motions, so it cannot create cartoon motions or non-physically plausible motions. The model is not aware of objects in the scene around a character.
Verified to have met prescribed NVIDIA quality standards:  |  Yes
Performance Metrics:                                                                                   |  Pose Constraint Accuracy (joint distance error), Motion Quality (foot-skating error, FID, latent similarity), Text-Following Accuracy (R-precision, latent similarity)
Potential Known Risks:                                                                                 |  The model may output body motions that inadvertently reflect stereotypes related to age, gender, or physical characteristics. To mitigate this, prompts should describe actions in neutral, physical terms (e.g., “A person walks slowly with shuffled steps”) rather than relying on demographic adjectives.
Licensing:                                                                                             |  This model is released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)

## Privacy

Field                                                                                                                              |  Response
:----------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------
Generatable or reverse engineerable personal data?                                                     |  No
Personal data used to create this model?                                                                                       |  No
How often is dataset reviewed?                                                                                                     |  During dataset creation, model training, evaluation and before release
Was data from user interactions with the AI model (e.g. user input and prompts) used to train the model? |  No
Is there provenance for all datasets used in training?                                                                                |  Yes
Does data labeling (annotation, metadata) comply with privacy laws?                                                                |  Yes
Is data compliant with data subject requests for data correction or removal, if such a request was made?                           |  Not Applicable
Applicable Privacy Policy        | https://www.nvidia.com/en-us/about-nvidia/privacy-policy/

## Safety

Field                                               |  Response
:---------------------------------------------------|:----------------------------------
Model Application Field(s):                               |  Media & Entertainment, Industrial/Machinery and Robotics, Autonomous Vehicles
Describe the life critical impact (if present).   |  Not Applicable
Use Case Restrictions:                              |  Abide by the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
Model and dataset restrictions:            |  The Principle of least privilege (PoLP) is applied limiting access for dataset generation and model development.  Restrictions enforce dataset access during training, and dataset license constraints adhered to.
