#!/bin/bash

# Define configurations
GPUS=(1 2 4)
BATCH_SIZES=(32 64 128)
PRECISIONS=(FP32 FP16)
LEARNING_RATES=(0.001 0.01 0.1)

# Array of deep learning programs
programs=("resnet_50.py" "vgg16.py")

# Loop through each configuration and run each program
for GPU in ${GPUS[@]}; do
  for BATCH_SIZE in ${BATCH_SIZES[@]}; do
    for PRECISION in ${PRECISIONS[@]}; do
      for LEARNING_RATE in ${LEARNING_RATES[@]}; do
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU-1)))
        export BATCH_SIZE=$BATCH_SIZE
        export PRECISION=$PRECISION
        export LEARNING_RATE=$LEARNING_RATE
        export GPUS=$GPU        
        
        # Run each program
        for program in "${programs[@]}"; do
          echo "Running $program with GPU=$GPU, BATCH_SIZE=$BATCH_SIZE, PRECISION=$PRECISION, LEARNING_RATE=$LEARNING_RATE..."
          python ~/dl-comp/programs/$program
          if [ $? -ne 0 ]; then
            echo "Error occurred in $program. Continuing with the next program..."
          else
            echo "$program completed successfully."
          fi
        done
      done
    done
  done
done

echo "All programs have been executed."