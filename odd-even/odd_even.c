#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpich/mpi.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#define fourG (1ull << 32)
#define oneG (1ull << 30)
#define hhG (1ull << 28)

char *save_path_256M = "/mnt/cephfs/home/lizhenliang/mpi_random_files/ran256m";
char *save_path_1G = "/mnt/cephfs/home/lizhenliang/mpi_random_files/ran1g";
char *save_path_4G = "/mnt/cephfs/home/lizhenliang/mpi_random_files/ran4g";

int cmpfunc(const void *a, const void *b)
{
    float numA = *(float *)a;
    float numB = *(float *)b;
    if (numA > numB)
        return 1;
    else if (numA == numB)
        return 0;
    else
        return -1;
}

void rand_gen(float *a, int arr_len, int seed)
{
    srand(seed);
    for (int i = 0; i < arr_len; i++)
    {
        float s = rand() / (RAND_MAX + 1.0);
        a[i] = s;
    }
}


void print_time_diff(struct timespec *begin, struct timespec *end) {
    struct timespec result;
    result.tv_sec = end->tv_sec - begin->tv_sec;
    result.tv_nsec =  end->tv_nsec - begin->tv_nsec;
    if (end->tv_nsec < begin->tv_nsec) {
        result.tv_sec--;
        result.tv_nsec += (long)1e9;
    }
    
    printf("%ld.%09ld\n", result.tv_sec, result.tv_nsec);
}

void check_error(float *arr, int arr_len)
{
    int error_num = 0;
    for (int i = 1; i < arr_len; i++)
    {
        if (arr[i] < arr[i - 1])
            error_num++;
    }

    printf("Error num %d\n", error_num);
}

int get_partner(int phase, int rank);
void merge_low(float *local, float *recv, float *temp, int local_n);
void merge_high(float *local, float *recv, float *temp, int local_n);

int main(int argc, char *argv[])
{
    int world_rank;
    int world_size;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char procname[MPI_MAX_PROCESSOR_NAME];
    int pro_name_len;

    MPI_Get_processor_name(procname, &pro_name_len);

    //printf("World size is %d\n", world_size);
    printf("Process %d of %d on %s\n", world_rank, world_size, procname);


    uint64_t total_num;
    char *file_path;
    if(strcmp(argv[1], "0") == 0){
        total_num = hhG;
        file_path = save_path_256M;
    }else if (strcmp(argv[1], "1") == 0){
        total_num = oneG;
        file_path = save_path_1G;
    }else if (strcmp(argv[1], "2")==0){
        /* 4G will overflow, so we just using number little bit smaller than 4G in both algorithm*/
        total_num = fourG-4967296;
        file_path = save_path_4G;
    }else{
        printf("Unvalid para");
        exit(1);
    }

    int local_n = total_num / world_size;
    size_t total_length = total_num;

    FILE *fp = fopen(file_path, "rb");
    float *local_A = malloc(sizeof(float) * local_n);
    float *recv_B = malloc(sizeof(float) * local_n);
    float *temp_C = malloc(sizeof(float) * local_n);

    fseek(fp, sizeof(float) * local_n * world_rank, SEEK_SET);
    size_t read_num = fread(local_A, sizeof(float), local_n, fp);
    if (read_num != local_n){
        printf("Read error");
        return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    struct timespec begin;
    clock_gettime(CLOCK_MONOTONIC, &begin);

    qsort(local_A, local_n, sizeof(float), cmpfunc);

    for (int phase = 0; phase < world_size; phase++)
    {
        int partner = get_partner(phase, world_rank);
        if (0 <= partner && partner < world_size)
        {
            MPI_Sendrecv(local_A, local_n, MPI_FLOAT, partner, 0, recv_B, local_n, MPI_FLOAT, partner, 0, MPI_COMM_WORLD, &status);
            if (phase % 2 == 0)
            {
                if (world_rank % 2 == 0)
                    merge_low(local_A, recv_B, temp_C, local_n);
                else
                    merge_high(local_A, recv_B, temp_C, local_n);
            }
            else
            {
                if (world_rank % 2 == 0)
                    merge_high(local_A, recv_B, temp_C, local_n);
                else
                    merge_low(local_A, recv_B, temp_C, local_n);
            }
        }
    }

    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    print_time_diff(&begin, &end);

    MPI_Barrier(MPI_COMM_WORLD);

    float *total_arr = NULL;
    if (world_rank == 0)
    {
        total_arr = (float *)malloc(sizeof(float) * total_length);
        if (total_arr == NULL)
        {
            printf("total NULL\n");
            return 1;
        }
    }

    MPI_Gather(local_A, local_n, MPI_FLOAT, total_arr, local_n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        check_error(total_arr, total_length);
    }

    free(total_arr);
    free(local_A);
    free(recv_B);
    free(temp_C);

    MPI_Finalize();
    return 0;
}

int get_partner(int phase, int rank)
{
    int even_partner;
    int odd_partner;
    if (rank % 2 != 0)
    {
        even_partner = rank - 1;
        odd_partner = rank + 1;
    }
    else
    {
        even_partner = rank + 1;
        odd_partner = rank - 1;
    }

    return phase % 2 == 0 ? even_partner : odd_partner;
}

void merge_low(float *local, float *recv, float *temp, int local_n)
{
    int l_i, r_i, t_i;
    l_i = 0;
    r_i = 0;
    t_i = 0;
    while (t_i < local_n)
    {
        if (local[l_i] <= recv[r_i])
        {
            temp[t_i] = local[l_i];
            t_i++;
            l_i++;
        }
        else
        {
            temp[t_i] = recv[r_i];
            t_i++;
            r_i++;
        }
    }
    memcpy(local, temp, local_n * sizeof(float));
}

void merge_high(float *local, float *recv, float *temp, int local_n)
{
    int l_i, r_i, t_i;
    l_i = local_n - 1;
    r_i = local_n - 1;
    t_i = local_n - 1;
    while (t_i >= 0)
    {
        if (local[l_i] >= recv[r_i])
        {
            temp[t_i] = local[l_i];
            t_i--;
            l_i--;
        }
        else
        {
            temp[t_i] = recv[r_i];
            t_i--;
            r_i--;
        }
    }
    memcpy(local, temp, local_n * sizeof(float));
}