#!/bin/bash -l

# Batch script to run a serial job under SGE.

# Request 24 hours of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=24:00:0

# Request 1 terabyte of RAM (must be an integer followed by M, G, or T)
#$ -l mem=1T

# Request 15 gigabyte of TMPDIR space (default is 10 GB - remove if cluster is diskless)
#$ -l tmpfs=15G

# Set the name of the job.
#$ -N Serial_Job

# Set the working directory to somewhere in your scratch space.  
#  This is a necessary step as compute nodes cannot write to $HOME.
# Replace "<your_UCL_id>" with your UCL user ID.
# the jobscript output and error files will be here
#$ -wd /home/zceesko/Scratch/script_outputs

# Your work should be done in $TMPDIR 
cd ../MEng-V2I

# Load the python3 module bundle
module load python3/recommended

#When installing new packages to the central venv, uncomment the 3 lines below and change the package name
#module unload compilers
#module load compilers/gnu/4.9.2
#pip3 install --user <python3pkg>

# Run the application and put the output into a file called SARIMA.txt, placed in the current directory, after any cd
/usr/bin/time --verbose python3 src/train_TOU.py > SARIMA.txt

# Preferably, tar-up (archive) all output files onto the shared scratch area
# the $TMPDIR at the end means that the tar file is compressing the TMPDIR folder for that job
tar -zcvf $HOME/Scratch/script_outputs/files_from_job_$JOB_ID.tar.gz $TMPDIR

# Make sure you have given enough time for the copy to complete!