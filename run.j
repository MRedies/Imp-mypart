#!/usr/bin/env zsh
 
### Job name
#BSUB -J 2d
 
### File / path where STDOUT & STDERR will be written
###    %J is the job ID, %I is the array ID
#BSUB -o job_2d.%J.%I
 
### Request the time you need for execution in minutes
### The format for the parameter is: [hour:]minute,
### that means for 80 minutes you could also use this: 1:20
#BSUB -W 3:00
 
### Request memory you need for your job per PROCESS in MB
#BSUB -M 1024
 
### Request the number of compute slots you want to use
#BSUB -n 32 
#BSUB -N
 
### Use esub for Open MPI
#BSUB -a openmpi
 
### (OFF) load another Open MPI version than the default one
# module switch openmpi openmpi/1.7.4
 
### Change to the work directory
cd /home/mr071525/imp

 
### Execute your application
$MPIEXEC $FLAGS_MPI_BATCH ipython 2d_plot.py
mv *.npz ../cpy/
