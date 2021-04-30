#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpich/mpi.h>
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

int min(int x, int y)
{
    return x < y ? x : y;
}

void print_time_diff(struct timespec *begin, struct timespec *end)
{
    struct timespec result;
    result.tv_sec = end->tv_sec - begin->tv_sec;
    result.tv_nsec = end->tv_nsec - begin->tv_nsec;
    if (end->tv_nsec < begin->tv_nsec)
    {
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

    if (argc != 2)
    {
        printf("Invalid parameter num\n");
        exit(1);
    }

    char procname[MPI_MAX_PROCESSOR_NAME];
    int pro_name_len;

    MPI_Get_processor_name(procname, &pro_name_len);

    //printf("World size is %d\n", world_size);
    printf("Process %d of %d on %s\n", world_rank, world_size, procname);

    size_t total_num;
    char *file_path;
    if (strcmp(argv[1], "0") == 0)
    {
        total_num = hhG;
        file_path = save_path_256M;
    }
    else if (strcmp(argv[1], "1") == 0)
    {
        total_num = oneG;
        file_path = save_path_1G;
    }
    else if (strcmp(argv[1], "2") == 0)
    {
        total_num = fourG - 4967296;
        file_path = save_path_4G;
    }
    else
    {
        printf("Unvalid para");
        exit(1);
    }

    int local_n = total_num / world_size;
    // int local_n = 10000;
    int P = world_size;
    size_t N = total_num;

    FILE *fp = fopen(file_path, "rb");

    float *local_arr = malloc(sizeof(float) * local_n);

    if (local_arr == NULL)
    {
        printf("local arr null rank %d\n", world_rank);
        return 1;
    }

    fseek(fp, sizeof(float) * local_n * world_rank, SEEK_SET);
    size_t read_num = fread(local_arr, sizeof(float), local_n, fp);
    if (read_num != local_n)
    {
        printf("Read error");
        return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    struct timespec begin;
    clock_gettime(CLOCK_MONOTONIC, &begin);

    qsort(local_arr, local_n, sizeof(float), cmpfunc);

    float *local_samples = malloc(sizeof(float) * P);

    size_t sample_index;
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

    if (recv_arr == NULL)
    {
        printf("recv arr NULL on rank %d\n", world_rank);
        return 1;
    }

    int *sdisp = malloc(sizeof(int) * P);
    int *rdisp = malloc(sizeof(int) * P);

    sdisp[0] = 0;
    for (int i = 1; i < world_size; i++)
    {
        sdisp[i] = sdisp[i - 1] + sendCounts[i - 1];
    }
    rdisp[0] = 0;
    for (int i = 1; i < world_size; i++)
    {
        rdisp[i] = rdisp[i - 1] + recvCounts[i - 1];
    }

    MPI_Alltoallv(local_arr, sendCounts, sdisp, MPI_FLOAT, recv_arr, recvCounts, rdisp, MPI_FLOAT, MPI_COMM_WORLD);

    merge_recv_arr(recv_arr, P, recv_length, recvCounts);


    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    print_time_diff(&begin, &end);

    MPI_Barrier(MPI_COMM_WORLD);

    // 4G的时候，total_rdisp最后加到一起会超出int, 4G时不能用Gatherv
    if (strcmp(argv[1], "2") != 0)
    {
        float *total_arr = NULL;
        int *total_recvCounts = NULL;
        if (world_rank == 0)
        {
            total_arr = malloc(sizeof(float) * N);
            if (total_arr == NULL)
            {
                printf("total NULL\n");
                return 1;
            }
            total_recvCounts = malloc(sizeof(int) * P);
            if (total_recvCounts == NULL)
            {
                printf("total_recvCount NULL\n");
                return 1;
            }
        }

        MPI_Gather(&recv_length, 1, MPI_INT, total_recvCounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        int *total_rdisp = NULL;
        if (world_rank == 0)
        {
            total_rdisp = malloc(sizeof(int) * P);
            total_rdisp[0] = 0;
            for (int i = 1; i < P; i++)
            {
                total_rdisp[i] = total_rdisp[i - 1] + total_recvCounts[i - 1];
            }
        }

        MPI_Gatherv(recv_arr, recv_length, MPI_FLOAT, total_arr, total_recvCounts, total_rdisp, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        if (world_rank == 0)
        {
            check_error(total_arr, N);
        }
    }
    else
    {
        check_error(recv_arr, recv_length);
        float next_head;
        if (world_rank == 0)
        {
            MPI_Recv(&next_head, 1, MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, &status);
            if (recv_arr[recv_length-1] > next_head){
                printf("sort error on %d\n", world_rank);
            }
        }
        else if (world_rank < world_size - 1)
        {
            MPI_Recv(&next_head, 1, MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, &status);
            if (recv_arr[recv_length-1] > next_head){
                printf("sort error on %d\n", world_rank);
            }
            MPI_Send(recv_arr, 1, MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD);
        }
        else
        {
            MPI_Send(recv_arr, 1, MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD);
        }
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
    for (int i = 0; i < length; i++)
    {
        sum += arr[i];
    }
    return sum;
}

int arr_part_sum(int *arr, int length, int begin, int end)
{
    int sum = 0;
    if (end > length)
        end = length;
    for (int i = begin; i < end; i++)
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
        start = 0;
        while (start < recv_length)
        {
            int low = start;
            // int mid = min(recv_length, start + arr_part_sum(recvCounts, comm_size, (i-1)*2*seg, (i-1)*2*seg + seg));
            // int high = min(recv_length, start + arr_part_sum(recvCounts, comm_size, (i-1)*2*seg, i*2*seg));
            int mid = min(recv_length, arr_part_sum(recvCounts, comm_size, 0, (i - 1) * 2 * seg + seg));
            int high = min(recv_length, arr_part_sum(recvCounts, comm_size, 0, i * 2 * seg));
            int k = low;
            int start1 = low, end1 = mid;
            int start2 = mid, end2 = high;
            while (start1 < end1 && start2 < end2)
                b[k++] = a[start1] < a[start2] ? a[start1++] : a[start2++];
            while (start1 < end1)
                b[k++] = a[start1++];
            while (start2 < end2)
                b[k++] = a[start2++];

            start += arr_part_sum(recvCounts, comm_size, (i - 1) * 2 * seg, i * 2 * seg);
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