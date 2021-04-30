#include <mpich/mpi.h>
#include <stdio.h>
#include <stdlib.h>


void main(int argc, char *argv[])

{

    int myid,num,name;

    char procname[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv); //实现对MPI环境的初始化

    MPI_Comm_rank(MPI_COMM_WORLD, &myid); //获取个人身份

    MPI_Comm_size(MPI_COMM_WORLD, &num); //获取给定组的大小

    MPI_Get_processor_name(procname, &name);//获取节点名称

    printf("Hello World! Process %d of %d on %s\n", myid, num, procname);

    MPI_Finalize();//结束MPI环境

}