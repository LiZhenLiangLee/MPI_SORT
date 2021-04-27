#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpich/mpi.h>
#include <time.h>

#define fourG (1ull << 32)
#define oneG (1ull << 30)
#define hhG (1ull << 28)

const char *save_path_256M = "/mnt/cephfs/home/lizhenliang/mpi_random_files/ran256m";
const char *save_path_1G = "/mnt/cephfs/home/lizhenliang/mpi_random_files/ran1g";
const char *save_path_4G = "/mnt/cephfs/home/lizhenliang/mpi_random_files/ran4g";


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

void display(float *a, int length, int rank)
{
    printf("Rank %d ", rank);
    for (int i = 0; i < length; i++)
    {
        printf("%f  ", a[i]);
    }
    printf("\n");
}

void display_int(int *a, int length, int rank)
{
    printf("Rank %d ", rank);
    for (int i = 0; i < length; i++)
    {
        printf("%d  ", a[i]);
    }
    printf("\n");
}

void display_worank(float *a, int length)
{
    for (int i = 0; i < length; i++)
    {
        printf("%f  ", a[i]);
    }
    printf("\n");
}

int min(int x, int y)
{
    return x < y ? x : y;
}

void print_current_time(int begin)
{
    time_t rawtime;
    struct tm *timeinfo;

    time(&rawtime);
    timeinfo = localtime(&rawtime);
    if (begin == 1)
    {
        printf("Begin sort  %s", asctime(timeinfo));
    }
    else
    {
        printf("End sort  %s", asctime(timeinfo));
    }
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

void merge_samples(float *all_samples, int p);
int get_split_index(float *search_arr, float value, int base, int total_length);
int arr_sum(int *arr, int length);
int arr_part_sum(int *arr, int length, int begin, int end);
void merge_recv_arr(float *recv, int comm_size, int recv_length, int *recvCounts);

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
    if(strcmp(argv[1], "0") == 0){
        total_num = hhG;
    }else if (strcmp(argv[1], "1") == 0){
        total_num = oneG;
    }else if (strcmp(argv[1], "2")==0){
        total_num = fourG;
    }else{
        printf("Unvalid para");
        exit(1);
    }

    int local_n = total_num / world_size;
    // int local_n = 10000;
    int P = world_size;
    int N = local_n * P;

    FILE *fp = fopen(save_path_1G, "rb");

    float *local_arr = malloc(sizeof(float) * local_n);

    fseek(fp, sizeof(float) * local_n * world_rank, SEEK_SET);
    fread(local_arr, sizeof(float), local_n, fp);

    MPI_Barrier(MPI_COMM_WORLD);
    clock_t begin = clock();
    //print_current_time(1);

    qsort(local_arr, local_n, sizeof(float), cmpfunc);

    float *local_samples = malloc(sizeof(float) * P);

    int sample_index;
    for (int i = 0; i < P; i++)
    {
        sample_index = i * (N / (P * P));
        local_samples[i] = local_arr[sample_index];
    }

    float *all_samples = NULL;
    if (world_rank == 0)
    {
        all_samples = malloc(sizeof(float) * P * P);
    }

    MPI_Gather(local_samples, P, MPI_FLOAT, all_samples, P, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        merge_samples(all_samples, P);
    }

    float *fences = malloc(sizeof(float) * (P - 1));
    if (world_rank == 0)
    {
        for (int i = 1; i <= P - 1; i++)
        {
            int fence_index = i * P + P / 2 - 1;
            fences[i - 1] = all_samples[fence_index];
        }
    }

    MPI_Bcast(fences, P - 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // Split the local_arr into P segs using fences
    int *sendCounts = malloc(sizeof(int) * P);
    int *recvCounts = malloc(sizeof(int) * P);

    int base = 0;
    int split_index;
    for (int i = 0; i < P; i++)
    {
        if (i < P - 1)
        {
            split_index = get_split_index(local_arr, fences[i], base, local_n);
            sendCounts[i] = split_index - base;
            base = split_index;
        }
        else
        {
            sendCounts[i] = local_n - base;
        }
    }

    MPI_Alltoall(sendCounts, 1, MPI_INT, recvCounts, 1, MPI_INT, MPI_COMM_WORLD);

    int recv_length = arr_sum(recvCounts, P);
    float *recv_arr = malloc(sizeof(float) * recv_length);

    int *sdisp = malloc(sizeof(int) * P);
    int *rdisp = malloc(sizeof(int) * P);

    sdisp[0] = 0;
    for (int i = 1; i<world_size; i++)
    {
        sdisp[i] = sdisp[i-1] + sendCounts[i-1];
    }
    rdisp[0] = 0;
    for (int i = 1; i<world_size; i++)
    {
        rdisp[i] = rdisp[i-1] +  recvCounts[i-1];
    }

    MPI_Alltoallv(local_arr, sendCounts, sdisp, MPI_FLOAT, recv_arr, recvCounts, rdisp, MPI_FLOAT, MPI_COMM_WORLD);

    merge_recv_arr(recv_arr, P, recv_length, recvCounts);

    //print_current_time(0);
    clock_t end = clock();
    float used_time = 1.0 * (end - begin) / CLOCKS_PER_SEC;
    printf("used time %f on rank %d\n", used_time, world_rank);

    float *total_arr = NULL;
    int *total_recvCounts = NULL;
    if (world_rank == 0)
    {
        total_arr = malloc(sizeof(float) * N);
        total_recvCounts = malloc(sizeof(int) * P);
    }

    MPI_Gather(&recv_length, 1, MPI_INT, total_recvCounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    int *total_rdisp = NULL;
    if (world_rank == 0)
    {
        total_rdisp = malloc(sizeof(int) * P);
        total_rdisp[0] = 0;
        for (int i = 1; i<P; i++)
        {
            total_rdisp[i] = total_rdisp[i-1] +  total_recvCounts[i-1];
        }
    }

    MPI_Gatherv(recv_arr, recv_length, MPI_FLOAT, total_arr, total_recvCounts, total_rdisp, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if(world_rank == 0)
    {
        check_error(total_arr, N);
    }


    MPI_Finalize();
    return 0;
}

void merge_samples(float *all_samples, int p)
{
    int length = p * p;
    float *a = all_samples;
    float *b = malloc(sizeof(float) * length);

    int seg, start;
    for (seg = p; seg < length; seg += seg)
    {

        for (start = 0; start < length; start += seg + seg)
        {
            int low = start, mid = min(start + seg, length), high = min(start + seg + seg, length);
            int k = low;
            int start1 = low, end1 = mid;
            int start2 = mid, end2 = high;
            while (start1 < end1 && start2 < end2)
                b[k++] = a[start1] < a[start2] ? a[start1++] : a[start2++];
            while (start1 < end1)
                b[k++] = a[start1++];
            while (start2 < end2)
                b[k++] = a[start2++];
        }
        float *temp = a;
        a = b;
        b = temp;
    }
    if (a != all_samples)
    {
        int i;
        for (i = 0; i < length; i++)
            b[i] = a[i];
        b = a;
    }
    free(b);
}

int get_split_index(float *search_arr, float value, int base, int totoal_length)
{
    int i;
    for (i = base; i < totoal_length; i++)
    {
        if (search_arr[i] > value)
        {
            return i;
        }
    }

    return i;
}

int arr_sum(int *arr, int length)
{
    int sum = 0;
    for (int i = 0; i<length; i++)
    {
        sum += arr[i];
    }
    return sum;
}

int arr_part_sum(int *arr, int length, int begin, int end)
{
    int sum = 0;
    if (end > length) end = length;
    for (int i = begin; i<end; i++)
    {
        sum += arr[i];
    }
    return sum;
}

void merge_recv_arr(float *recv, int comm_size, int recv_length, int *recvCounts)
{
    float *a = recv;
    float *b = malloc(sizeof(float) * recv_length);
    int seg, start, i;
    for (seg = 1; seg < comm_size; seg += seg)
    {
        i = 1;
        start=0;
        while (start < recv_length)
        {
            int low = start;
            // int mid = min(recv_length, start + arr_part_sum(recvCounts, comm_size, (i-1)*2*seg, (i-1)*2*seg + seg));
            // int high = min(recv_length, start + arr_part_sum(recvCounts, comm_size, (i-1)*2*seg, i*2*seg));
            int mid = min(recv_length, arr_part_sum(recvCounts, comm_size, 0, (i-1)*2*seg + seg));
            int high = min(recv_length, arr_part_sum(recvCounts, comm_size, 0, i*2*seg));
            int k = low;
            int start1 = low, end1 = mid;
            int start2 = mid, end2 = high;
            while (start1 < end1 && start2 < end2)
                b[k++] = a[start1] < a[start2] ? a[start1++] : a[start2++];
            while (start1 < end1)
                b[k++] = a[start1++];
            while (start2 < end2)
                b[k++] = a[start2++];

            start += arr_part_sum(recvCounts, comm_size, (i-1)*2*seg, i*2*seg);
            i++;
        }
        float *temp = a;
        a = b;
        b = temp;   
    }
    if (a != recv)
    {
        int i;
        for (i = 0; i < recv_length; i++)
            b[i] = a[i];
        b = a;
    }
    free(b);
}