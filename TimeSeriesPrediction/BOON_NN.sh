#!/bin/bash -l

# Batch script to run a serial job under SGE.

# Request 24 hours of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=01:30:0

# Request 1 terabyte of RAM (must be an integer followed by M, G, or T)
#$ -l mem=32G

# Request 1 GPU for the job
#$ -l gpu=1

# request a V100 node only, the line below says EF because only E-type and F-type nodes has nVidia Tesla V100s
# #$ -ac allow=EF

# Request 15 gigabyte of TMPDIR space (default is 10 GB - remove if cluster is diskless)
#$ -l tmpfs=15G

# Set the name of the job.
#$ -N LSTM_old_3

# Set the working directory to somewhere in your scratch space.  
#  This is a necessary step as compute nodes cannot write to $HOME.
# Replace "<your_UCL_id>" with your UCL user ID.
# the jobscript output and error files will be here
#$ -wd /home/zceesko/Scratch/script_outputs

# Your work should be done in $TMPDIR 
cd ../MEng-V2I/TimeSeriesPrediction

# Modules to load for the non-MKL GPU version of tensorflow
module unload compilers mpi
module load compilers/gnu/4.9.2
module load python3/recommended
module load cuda/10.0.130/gnu-4.9.2
module load cudnn/7.4.2.24/cuda-10.0
module load tensorflow/1.14.0/gpu
module load torch-deps

# install <python3pkg> on my directory
# pip3 install --user <python3pkg>

# Run the application and put the output into a file called SARIMA.txt, placed in the current directory, after any cd
/usr/bin/time --verbose python3 main.py --network 'lstm' --transfer_learning False --mode 'train' > LSTM_old_3.txt
pwd > ayoooo.txt

# Preferably, tar-up (archive) all output files onto the shared scratch area
# the $TMPDIR at the end means that the tar file is compressing the TMPDIR folder for that job
tar -zcvf $HOME/Scratch/script_outputs/files_from_job_$JOB_ID.tar.gz $TMPDIR

# Make sure you have given enough time for the copy to complete!