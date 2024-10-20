#!/bin/bash

# Kill GPU processes to avoid CUDA_OUT_OF_MEMORY errors
PROCESSES=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [ -n "$PROCESSES" ]; then 
  for PROCESS in ${PROCESSES[@]}; do 
    kill -9 $PROCESS
  done
  echo "Killed $open_processes processes."
fi

set_cuda_devices() {
  local devices=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | awk -F ' ' '$2 == 0 {print $1}' | head -n $GPU | paste -sd,)
  if [ -z "$devices" ]; then
    echo "Error: Not enough GPUs with 0 utilization. Exiting."
    exit 1
  else
    echo "$devices"
  fi
}

# Original iterations were:

# GPUS=(1 2 4)
# BATCH_SIZES=(32 64 128)
# PRECISIONS=(FP16 FP32)
# LEARNING_RATES=(0.001 0.01 0.1)

# But there were too many iterations

GPUS=(1 2)
BATCH_SIZES=(16 32)
PRECISIONS=(FP16 FP32)
LEARNING_RATES=(0.01 0.1)

# PROGRAMS=("vgg16" "yolov3" "mask_rcnn" "lstm" "gru" "bert" "gpt2" "dqn")
PROGRAMS=("resnet_50")

for PROGRAM in ${PROGRAMS[@]}; do
  for GPU in ${GPUS[@]}; do
    for BATCH_SIZE in ${BATCH_SIZES[@]}; do
      for PRECISION in ${PRECISIONS[@]}; do
        for LEARNING_RATE in ${LEARNING_RATES[@]}; do
          # Sleep to wait for GPU util to return to 0
          sleep 3
          devices=$(set_cuda_devices)
          echo "Selected GPUs: $devices"
          export CUDA_VISIBLE_DEVICES=$devices
          export BATCH_SIZE=$BATCH_SIZE
          export PRECISION=$PRECISION
          export LEARNING_RATE=$LEARNING_RATE
          export GPUS=$GPU
          CONFIG_ID="${GPUS}_${BATCH_SIZE}_${PRECISION}_${LEARNING_RATE}"


          LOG_FILE="logfile.txt"
          > $LOG_FILE

          nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits --loop=1 --id=$CUDA_VISIBLE_DEVICES >> $LOG_FILE &
          
          # Run deep learning models
          python ~/dl-comp/programs/monitor_resources.py ~/dl-comp/programs/${PROGRAM}.py $GPUS $BATCH_SIZE $PRECISION $LEARNING_RATE

          # Use nvidia-smi to monitor GPU usage, CPU usage, Power usage

          NVIDIA_SMI_PID=$(pgrep -n nvidia-smi)
            if [ ! -z "$NVIDIA_SMI_PID" ]; then
              kill -9 $NVIDIA_SMI_PID
            fi
          
          echo "Resource usage logged for $CONFIG_ID"
        done
      done
    done
  done
done

echo "Program killed, experiments complete."