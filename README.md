# MPI sorting algorithm
Implementation of Odd-Even Parallel and Parallel Sorting of Regular Sampling

## Requirement
### Install MPICH (require sudo permission)
```
sudo apt-get install libmpich-dev
```
Maybe install from source code is Ok, while the way you include the mpi.h file will be different. And the way you specify the config file(which machines to run, and how many processes on each machine) will also be different.

### Runing on multi-machine
If you want to run the code on multi machines, following the below steps.
1. Set up passwardless ssh login on all the machines (using ssh public key)
2. Build up a share file system/directory to all the machines. For example, nfs and cephfs. Then put all the executable files and the data files which the program needs when it runs. Also put the configure file here.
3. Write the configure file to specify which machines to run your code, and how many process on each machines. Here is an example: 
```
host1:4
host2:4
```
Here we run the code on 2 machines, host1 and host2 means the machine's host name, the number 4 means we will run the code using 4 processes on host1 and 4 processes on host2.

## Compile
```
mpicc.mpich -o odd_even -O3 odd_even.c
mpicc.mpich -o psrs -O3 psrs.c
```
## Running
**Single machine**
```
mpirun.mpich -np 8 ./odd_even
mpirun.mpich -np 8 ./psrs
```
**Multi machines**
```
mpirun.mpich -np 8 -f path/to/config_file ./odd_even
mpirun.mpich -np 8 -f path/to/config_file ./psrs
```
Arg **-np** means the total number of processes when running the code.<br/>
Arg **-f** specify your config file path.