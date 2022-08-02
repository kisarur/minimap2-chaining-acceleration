#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include <sys/time.h>
#include <sys/resource.h>
#include <x86intrin.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
using namespace std;

// parameters for hardware execution
#define NUM_HW_KERNELS 4 // Important: this should not exceed the no. of kernels in device/minimap2_chain.cl
#define DEVICE_MAX_N 332000000
#define BUFFER_MAX_N (DEVICE_MAX_N / 2)
#define BUFFER_N (BUFFER_MAX_N / 32)

// parameters for multi-threaded software execution
#define NUM_SW_THREADS 20
#define USE_SIMD

// parameters for HW/SW split
#define BETTER_ON_HW_THRESH 0.3

// for main.cpp
typedef struct {
    int start_call;
    int end_call;
    cl_long total_n;
} hw_fine_batch_t;

typedef struct {
    float total_time;
    float data_transfer_time;
} coarse_batch_timing_t;

#define STRING_BUFFER_LEN 1024

int chain_anchors(char *);
coarse_batch_timing_t process_coarse_grained_batch(vector<cl_ulong2> &a, cl_int* f, cl_int* p, vector<cl_long> &a_offsets,
                                                    vector<cl_ulong2> &a_hw, cl_int* f_hw, cl_int* p_hw, vector<cl_long> &a_hw_offsets,
                                                    vector<hw_fine_batch_t> &batches, vector<int> &calls_hw, vector<int> &calls_sw,
                                                    vector<cl_long> &n, vector<cl_int> &max_dist_x, vector<cl_int> &max_dist_y,
                                                    vector<cl_int> &bw, vector<cl_float> &avg_qspan);
void write_output(vector<cl_long> n, cl_int* f, cl_int* p, vector<cl_long> a_offsets, cl_int* f_hw, cl_int* p_hw, vector<cl_long> a_hw_offsets);
bool init(long);
void cleanup();
static void device_info_ulong( cl_device_id device, cl_device_info param, const char* name);
static void device_info_uint( cl_device_id device, cl_device_info param, const char* name);
static void device_info_bool( cl_device_id device, cl_device_info param, const char* name);
static void device_info_string( cl_device_id device, cl_device_info param, const char* name);
static void display_device_info( cl_device_id device );


// for multi-threaded-sw.cpp
#define WORK_STEAL 1    //simple work stealing enabled or not (no work stealing mean no load balancing)
#define STEAL_THRESH 1  //stealing threshold

#define NEG_CHK(ret) neg_chk(ret, __func__, __FILE__, __LINE__ - 1)

// core data structure that has information pthat are global to all the threads
typedef struct {
    int32_t num_thread;
} core_t;

// data structure for a batch of reads
typedef struct {
    int64_t n_batch;  //number of records in this batch

    int* calls_sw;

    cl_ulong2* a;
    int* f;
    int* p;
    cl_long* a_offsets;

    cl_long* n;
    cl_int* max_dist_x;
    cl_int* max_dist_y;
    cl_int* bw;
    cl_float* avg_qspan;
} db_t;

// argument wrapper for the multithreaded framework used for data processing 
typedef struct {
    core_t* core;
    db_t* db;
    int32_t starti;
    int32_t endi;
    void (*func)(core_t*, db_t*, int, int32_t);
    int32_t thread_index;
#ifdef WORK_STEAL
    void* all_pthread_args;
#endif
} pthread_arg_t;

void work_db(core_t* core, db_t* db);

void minimap2_opencl_sw(long n, int max_dist_x, int max_dist_y, int bw, cl_ulong2 * a,
                        int* f, int* p, float avg_qspan);

