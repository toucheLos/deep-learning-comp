1. Research Question
How do different GPU configurations and HPC settings affect the performance of various deep learning modules?

2. Hypothesis
Varying the number of GPUs and other HPC settings ()

3. Experiment Design

Categories and Models:
- CNN Models: ResNet-50, VGG16
- Object Detection and Segmentation Models: YOLOv3, Mask R-CNN
- RNN Models: LSTM, GRU
- Transformer Models: BERT, GPT-2
- Reinforcement Learning: DQN

Independent Variables
- Number of GPUs: 1, 2, 4
- Batch Size: Small, medium, large (e.g., 32, 64, 128)
- Precision FP32, FP16 (mixed precision)
- Learning Rate: Different Values (e.g., 0.001, 0.01, 0.1)
- Parallelism Strategy: Data parallelism, model parallelism
- CPU and Memory Utilization: Monitor and optimize during training
- Power Limit: Adjust the power limit of GPUs
- Clock Speeds: Adjust GPU and memory clock speeds

Dependent Variables (Testing for)
- Training time per epoch
- Inference time
- Model Accuracy
- Resource utilization (CPU/GPU usage, memory usage)
- Energy Consumtion

Modules and Tools
- Deep Learning Frameworks: TensorFlow, PyTorch
- GPU Management: CUDA, cuDNN
- Containerization: Singularity
- Monitoring Tools, NVIDIA Nsight, TensorBoard
- Statistical Analysis: pandas, matplotlib


Steps for the Experiment

    Load Required Modules:
        module load singularity
        module load cuda/12.2.2
        module load python/3.11.6

    Prepare the Environment:
        singularity pull docker://nvcr.io/nvidia/tensorflow:20.09-tf1-py3
        singularity pull docker://nvcr.io/nvidia/pytorch:20.09-py3
        source venv/bin/activate

    Run Experiments:
        Configure and run scripts for each model with different HPC settings.
        Use nvidia-smi to monitor and adjust GPU settings dynamically.

    Record Results:
        Log training time, inference time, accuracy, resource utilization, and energy consumption for each configuration.

    Analyze Results:
        Compare the performance metrics across different configurations to determine the optimal settings for each model.
