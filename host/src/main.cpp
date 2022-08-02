#include "common.h"

// control whether the emulator should be used.
static bool use_emulator = false;

using namespace aocl_utils;

// OpenCL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_kernel kernels[NUM_HW_KERNELS] = {NULL};
static cl_command_queue kernel_queues[NUM_HW_KERNELS] = {NULL};
static cl_command_queue write_q = NULL;
static cl_command_queue read_q = NULL;
static cl_program program = NULL;

// For input and output buffers
cl_mem input_a_buf[2];
cl_mem output_f_buf[2];
cl_mem output_p_buf[2];

int main(int argc, char *argv[]) {
    if(argc != 2) {
        printf("Error! Not enough arguments specified.\nUsage: %s <chaining_anchor_data_file>\n", argv[0]);
        exit(1);
    }

    char * anchor_data_file = realpath(argv[1], NULL); // get absolute path

    int status;
  
    status = chain_anchors(anchor_data_file);

    return status;
}

// Read file input, process chaining anchors and write output to file
int chain_anchors(char* infile_name) {
    
    // initialize device stuff (input/output buffers, kernels, command queues, etc)
    if (!init(BUFFER_N)) {
        return -1;
    }

    // common (hw and sw) parameters
    vector<cl_long> n;
    vector<cl_int> max_dist_x;
    vector<cl_int> max_dist_y;
    vector<cl_int> bw;
    vector<cl_float> avg_qspan;
    vector<cl_long> a_offsets;
    vector<cl_ulong2> a;
    cl_int* f;
    cl_int* p;

    // hardware specific parameters
    vector<cl_long> a_hw_offsets;
    vector<cl_ulong2> a_hw;
    cl_int* f_hw;
    cl_int* p_hw;
    vector<int> calls_hw;

    // software specific parameters
    vector<int> calls_sw;

    // to store hardware fine-grained batch details
    vector<hw_fine_batch_t> batches;

    // to keep track of total time taken for the whole dataset
    coarse_batch_timing_t dataset_total_time = {0};

    // extract data from infile
    int coarse_grained_batch_count = 0;
    int call = 0;
    int hw_call = 0;
    cl_long total_n = 0;
    cl_long count = 0;
    cl_long count_hw = 0;
    cl_long hw_fine_batch_total_n = 0;
    string line;
    ifstream infile(infile_name);
    if (infile.is_open()) {
        while (getline(infile, line)) {
            // extract header data
            stringstream line_stream;
            line_stream << line;

            cl_long n_temp;
            line_stream >> n_temp;

            //fprintf(stderr, "n_temp = %ld\n", n_temp);

            if (n_temp > BUFFER_N) {
                fprintf(stderr, "Error: The size of the call (call = %d, n = %ld) exceeds buffer size (%d). Process this read on SW?\n", call, n_temp, BUFFER_N);
                exit(1);
            }

            // -----------------------------------------------------------------
            // Process a coarse-grained batch (if we have collected enough data)
            // -----------------------------------------------------------------
            if (total_n + n_temp > DEVICE_MAX_N) {
                // add details of remaining fine-grained batch (only if a new batch is created after the last batch [i.e. hw_fine_batch_total_n != 0])
                if (hw_fine_batch_total_n != 0) {
                    hw_fine_batch_t new_batch;
                    if (batches.size() > 0) {
                        new_batch.start_call = batches[batches.size() - 1].end_call + 1;
                    } else {
                        new_batch.start_call = 0;
                    }
                    new_batch.end_call = hw_call - 1;
                    new_batch.total_n = hw_fine_batch_total_n;
                    batches.push_back(new_batch);
                }
                

                // allocate memory for output arrays
                f = (cl_int*)malloc(total_n * sizeof(cl_int));
                p = (cl_int*)malloc(total_n * sizeof(cl_int));

                f_hw = (cl_int*)malloc(a_hw.size() * sizeof(cl_int));
                p_hw = (cl_int*)malloc(a_hw.size() * sizeof(cl_int));

                fprintf(stderr, "\n------------------------\n");
                fprintf(stderr, "Coarse-grained batch %d\n", coarse_grained_batch_count);
                fprintf(stderr, "------------------------\n");

                // process coarse-grained batch and write output to file
                coarse_batch_timing_t batch_timing = process_coarse_grained_batch(a, f, p, a_offsets, a_hw, f_hw, p_hw, a_hw_offsets, batches, calls_hw, calls_sw, n, max_dist_x, max_dist_y, bw, avg_qspan);
                dataset_total_time.total_time += batch_timing.total_time;
                dataset_total_time.data_transfer_time += batch_timing.data_transfer_time;

                // write output to stdout
                write_output(n, f, p, a_offsets, f_hw, p_hw, a_hw_offsets);
                

                // reset the variables/arrays to collect data for the next batch (coarse-grained batch)
                total_n = 0;
                call = 0;
                hw_call = 0; // 0 or -1?
                hw_fine_batch_total_n = 0;
                count = 0;

                free(f);
                free(p);
                free(f_hw);
                free(p_hw);
                n.clear();
                max_dist_x.clear();
                max_dist_y.clear();
                bw.clear();
                avg_qspan.clear();
                a_offsets.clear();
                a_hw_offsets.clear();
                a.clear();
                a_hw.clear();
                batches.clear();
                calls_hw.clear();
                calls_sw.clear();

                coarse_grained_batch_count++;
            }

            n.push_back(n_temp);

            string avg_qspan_unused;
            line_stream >> avg_qspan_unused;

            cl_int max_dist_x_temp;
            line_stream >> max_dist_x_temp;
            max_dist_x.push_back(max_dist_x_temp);

            cl_int max_dist_y_temp;
            line_stream >> max_dist_y_temp;
            max_dist_y.push_back(max_dist_y_temp);

            cl_int bw_temp;
            line_stream >> bw_temp;
            bw.push_back(bw_temp);

            a_offsets.push_back(count);

            // add current function call's anchors to "a", while calculating the parameters - sum_qspan, better_on_hw_count
            cl_ulong sum_qspan = 0;
            long better_on_hw_count = 0;
            long total_trip_count = 0;
            long st = 0;
            for (long i = 0; i < n_temp; i++) {
                cl_ulong2 a_temp;
                infile >> a_temp.x;
                infile >> a_temp.y;
                a.push_back(a_temp);

                // determine and store the inner loop's trip count (max is INNER_LOOP_TRIP_COUNT_MAX)
                while (st < i && a_temp.x > a[st + a_offsets[call]].x + max_dist_x[call]) ++st;
                int inner_loop_trip_count = i - st;
                if (inner_loop_trip_count < 0) { // trip count is 0 if (i - st) is negative
                    inner_loop_trip_count = 0;
                }
                if (inner_loop_trip_count > 64) { 
                    inner_loop_trip_count = 64;
                }

                total_trip_count += inner_loop_trip_count;

                sum_qspan += a_temp.y >> 32 & 0xff;

                count++;
            }

            // calculate the parameter that helps decide whether to schedule the call on hw or sw
            float better_on_hw_frac;
            if (n_temp > 0) { // can have some other threshold than 0 (eg. MIN_ANCHORS) to force processing small calls on software
                better_on_hw_frac = (float) total_trip_count / (n_temp * 64);
            } else {
                better_on_hw_frac = 0;
            }

            // schedule current call to be executed on hw, if it performs better on hardware
            if (better_on_hw_frac >= BETTER_ON_HW_THRESH) { // better_on_hw_frac >= 0.2 for human refmap

                // create a new fine-grained data batch if it's required (i.e. adding the current read to batch exceeds buffer capacity - BUFFER_N)
                if (hw_fine_batch_total_n + n_temp > BUFFER_N) {
                    hw_fine_batch_t new_batch;

                    if (batches.size() > 0) {
                        new_batch.start_call = batches[batches.size() - 1].end_call + 1;
                    } else {
                        new_batch.start_call = 0;
                    }
                    new_batch.end_call = hw_call - 1;
                    new_batch.total_n = hw_fine_batch_total_n;
                    batches.push_back(new_batch);

                    hw_fine_batch_total_n = 0;
                }

                a_hw_offsets.push_back(a_hw.size());

                for (long i = 0; i < n_temp; i++) {
                    a_hw.push_back(a[a_offsets[call] + i]);
                }

                calls_hw.push_back(call);
                hw_fine_batch_total_n += n_temp;
                hw_call++;

            } else {
                a_hw_offsets.push_back(-1); // -1 means the call is scheduled on software
                calls_sw.push_back(call);
            }

            // calculate avg_qspan and store it
            avg_qspan.push_back(.01 * ((cl_float)sum_qspan / n_temp));

            string line;
            if (n_temp > 0) {
                getline(infile, line);  // remaining new line
            }
            getline(infile, line);  // EOR line

            total_n += n_temp;
            call++;
        }
    } else {
        fprintf(stderr, "Error opening input file!\n");
        exit(1);
    }

    // ---------------------------------
    // Process last coarse-grained batch
    // ---------------------------------

    // add details of remaining fine-grained batch (only if a new batch is created after the last batch [i.e. hw_fine_batch_total_n != 0])
    if (hw_fine_batch_total_n != 0) {
        hw_fine_batch_t new_batch;
        if (batches.size() > 0) {
            new_batch.start_call = batches[batches.size() - 1].end_call + 1;
        } else {
            new_batch.start_call = 0;
        }
        new_batch.end_call = hw_call - 1;
        new_batch.total_n = hw_fine_batch_total_n;
        batches.push_back(new_batch);
    }

    // allocate memory for output arrays
    f = (cl_int*)malloc(total_n * sizeof(cl_int));
    p = (cl_int*)malloc(total_n * sizeof(cl_int));

    f_hw = (cl_int*)malloc(a_hw.size() * sizeof(cl_int));
    p_hw = (cl_int*)malloc(a_hw.size() * sizeof(cl_int));

    fprintf(stderr, "\n------------------------\n");
    fprintf(stderr, "Coarse-grained batch %d\n", coarse_grained_batch_count);
    fprintf(stderr, "------------------------\n");

    // process coarse-grained batch and write output to file
    coarse_batch_timing_t batch_timing = process_coarse_grained_batch(a, f, p, a_offsets, a_hw, f_hw, p_hw, a_hw_offsets, batches, calls_hw, calls_sw, n, max_dist_x, max_dist_y, bw, avg_qspan);
    dataset_total_time.total_time += batch_timing.total_time;
    dataset_total_time.data_transfer_time += batch_timing.data_transfer_time;

    // write output to stdout
    write_output(n, f, p, a_offsets, f_hw, p_hw, a_hw_offsets);

    free(f);
    free(p);
    free(f_hw);
    free(p_hw);

    infile.close();

    // free the resources allocated
    cleanup();

    // print total timing for the whole dataset
    fprintf(stderr, "\nTotal processing time for the whole dataset: %0.3f s\n", dataset_total_time.total_time);
    fprintf(stderr, "Total data transfer time for the whole dataset: %0.3f s\n", dataset_total_time.data_transfer_time);

    return 0;
}

/////// HELPER FUNCTIONS ///////

/* 
--------------------------------------
Process a Coarse-grained Batch of Data
--------------------------------------
*/

coarse_batch_timing_t process_coarse_grained_batch(vector<cl_ulong2> &a, cl_int* f, cl_int* p, vector<cl_long> &a_offsets,
                                                    vector<cl_ulong2> &a_hw, cl_int* f_hw, cl_int* p_hw, vector<cl_long> &a_hw_offsets,
                                                    vector<hw_fine_batch_t> &batches, vector<int> &calls_hw, vector<int> &calls_sw,
                                                    vector<cl_long> &n, vector<cl_int> &max_dist_x, vector<cl_int> &max_dist_y,
                                                    vector<cl_int> &bw, vector<cl_float> &avg_qspan) {
    // print details of coarse-grained batch
    fprintf(stderr, "Total no. of calls: %ld\n", n.size());
    fprintf(stderr, "Hardware calls: %ld\n", calls_hw.size());
    //for (int i = 0; i < calls_hw.size(); i++) { fprintf(stderr, "%d, ", calls_hw[i]); }; fprintf(stderr, "\n");
    fprintf(stderr, "Software calls: %ld\n", calls_sw.size());

    //fprintf(stderr, "a_hw_offsets size: %ld\n", a_hw_offsets.size());

    long mem_bytes = a.size() * (sizeof(cl_ulong2) + 2 * sizeof(cl_int));
    float mem_megabytes = (float)mem_bytes / (1024 * 1024);

    fprintf(stderr, "\nTotal n = %ld\n", a.size());
    fprintf(stderr, "Total host memory needed for data = %f MB (%ld bytes)\n", mem_megabytes, mem_bytes);

    fprintf(stderr, "\nHardware n = %ld\n", a_hw.size());
    fprintf(stderr, "Hardware n/Total n = %0.3f %% \n", (a_hw.size() * 100.0 / a.size()));

    /* // print fine details of coarse-grained batch
    for (int i = 0; i < n.size(); i++) {
        cout << n[i] << " " << max_dist_x[i] << " " << max_dist_y[i] << " " << bw[i] << " " << offsets[i] << " " << avg_qspan[i] << endl;

        for (long j = 0; j < 5; j++) {
            cout << a[offsets[i] + j].x << " " << a[offsets[i] + j].y << " " << endl;
        }
    }
    cout << endl; */

    // print details of fine-grained batches
    fprintf(stderr, "\nDetails of fine-grained hardware batches\n");
    fprintf(stderr, "----------------------------------------\n");
    for (int i = 0; i < batches.size(); i++) {
        hw_fine_batch_t batch = batches[i];
        fprintf(stderr, "batch = %d, start_call = %d, num_calls = %d, total_n = %ld\n", i, batch.start_call, (batch.end_call - batch.start_call + 1), batch.total_n);
    }

    cl_int status;

    struct timeval begin, end;
    gettimeofday(&begin, NULL);

    int batches_count = batches.size();

    cl_event dependencies[3];
    cl_event write_event[batches_count];
    cl_event kernel_event[batches_count][NUM_HW_KERNELS];
    int kernel_event_count_list[batches_count];
    cl_event f_read_event[batches_count];
    cl_event p_read_event[batches_count];

    for (int i = 0; i < batches_count; i++) {
        hw_fine_batch_t batch = batches[i];
        int calls_in_batch = batch.end_call - batch.start_call + 1;
        int calls_per_kernel = (calls_in_batch + NUM_HW_KERNELS - 1) / NUM_HW_KERNELS;

        int kernel_event_count = 0;

        if (i < 2) {
            clEnqueueWriteBuffer(write_q, input_a_buf[i % 2], CL_FALSE, 0, batch.total_n * sizeof(cl_ulong2), &a_hw[a_hw_offsets[calls_hw[batch.start_call]]], 0, NULL, &write_event[i]);
            clFlush(write_q);

            int kernel_start_call = batch.start_call;
            int issued_calls = 0;
            for (int j = 0; j < NUM_HW_KERNELS && issued_calls < calls_in_batch; j++) {  // (issued_calls < calls_in_batch) becomes false when there's no work to distribute to all the kernels
                for (int hw_call = kernel_start_call; hw_call < (kernel_start_call + calls_per_kernel) && hw_call <= batch.end_call; hw_call++) {
                    int call = calls_hw[hw_call]; // get original call index from hw_call index
                    cl_long call_n = n[call];
                    cl_int call_max_dist_x = max_dist_x[call];
                    cl_int call_max_dist_y = max_dist_y[call];
                    cl_int call_bw = bw[call];
                    cl_long call_offset = a_hw_offsets[call] - a_hw_offsets[calls_hw[batch.start_call]];
                    cl_float call_avg_qspan = avg_qspan[call];

                    cl_int status;

                    // set the kernel arguments.
                    status = clSetKernelArg(kernels[j], 0, sizeof(cl_long), &call_n);
                    checkError(status, "Failed to set argument 0");

                    status = clSetKernelArg(kernels[j], 1, sizeof(cl_int), &call_max_dist_x);
                    checkError(status, "Failed to set argument 1");

                    status = clSetKernelArg(kernels[j], 2, sizeof(cl_int), &call_max_dist_y);
                    checkError(status, "Failed to set argument 2");

                    status = clSetKernelArg(kernels[j], 3, sizeof(cl_int), &call_bw);
                    checkError(status, "Failed to set argument 3");

                    status = clSetKernelArg(kernels[j], 4, sizeof(cl_float), &call_avg_qspan);
                    checkError(status, "Failed to set argument 4");

                    status = clSetKernelArg(kernels[j], 5, sizeof(cl_long), &call_offset);
                    checkError(status, "Failed to set argument 5");

                    status = clSetKernelArg(kernels[j], 6, sizeof(cl_mem), &input_a_buf[i % 2]);
                    checkError(status, "Failed to set argument 6");

                    status = clSetKernelArg(kernels[j], 7, sizeof(cl_mem), &output_f_buf[i % 2]);
                    checkError(status, "Failed to set argument 7");

                    status = clSetKernelArg(kernels[j], 8, sizeof(cl_mem), &output_p_buf[i % 2]);
                    checkError(status, "Failed to set argument 8");

                    // launch the kernel.
                    if (hw_call == kernel_start_call + calls_per_kernel - 1 || hw_call == batch.end_call) {
                        status = clEnqueueTask(kernel_queues[j], kernels[j], 1, &write_event[i], &kernel_event[i][j]);
                        checkError(status, "Failed to launch kernel");
                    } else {
                        status = clEnqueueTask(kernel_queues[j], kernels[j], 1, &write_event[i], NULL);
                        checkError(status, "Failed to launch kernel");
                    }

                    issued_calls++;
                }
                kernel_start_call += calls_per_kernel;
                clFlush(kernel_queues[j]);
                kernel_event_count++;
            }

        } else {  // end (i < 2)

            clEnqueueWriteBuffer(write_q, input_a_buf[i % 2], CL_FALSE, 0, batch.total_n * sizeof(cl_ulong2), &a_hw[a_hw_offsets[calls_hw[batch.start_call]]], kernel_event_count_list[i - 2], kernel_event[i - 2], &write_event[i]);
            clFlush(write_q);

            dependencies[0] = write_event[i];
            dependencies[1] = f_read_event[i - 2];
            dependencies[2] = p_read_event[i - 2];

            int kernel_start_call = batch.start_call;
            int issued_calls = 0;
            for (int j = 0; j < NUM_HW_KERNELS && issued_calls < calls_in_batch; j++) {
                for (int hw_call = kernel_start_call; hw_call < (kernel_start_call + calls_per_kernel) && hw_call <= batch.end_call; hw_call++) {
                    int call = calls_hw[hw_call]; // get original call index from hw_call index
                    cl_long call_n = n[call];
                    cl_int call_max_dist_x = max_dist_x[call];
                    cl_int call_max_dist_y = max_dist_y[call];
                    cl_int call_bw = bw[call];
                    cl_long call_offset = a_hw_offsets[call] - a_hw_offsets[calls_hw[batch.start_call]];
                    cl_float call_avg_qspan = avg_qspan[call];

                    cl_int status;

                    // set the kernel arguments.
                    status = clSetKernelArg(kernels[j], 0, sizeof(cl_long), &call_n);
                    checkError(status, "Failed to set argument 0");

                    status = clSetKernelArg(kernels[j], 1, sizeof(cl_int), &call_max_dist_x);
                    checkError(status, "Failed to set argument 1");

                    status = clSetKernelArg(kernels[j], 2, sizeof(cl_int), &call_max_dist_y);
                    checkError(status, "Failed to set argument 2");

                    status = clSetKernelArg(kernels[j], 3, sizeof(cl_int), &call_bw);
                    checkError(status, "Failed to set argument 3");

                    status = clSetKernelArg(kernels[j], 4, sizeof(cl_float), &call_avg_qspan);
                    checkError(status, "Failed to set argument 4");

                    status = clSetKernelArg(kernels[j], 5, sizeof(cl_long), &call_offset);
                    checkError(status, "Failed to set argument 5");

                    status = clSetKernelArg(kernels[j], 6, sizeof(cl_mem), &input_a_buf[i % 2]);
                    checkError(status, "Failed to set argument 6");

                    status = clSetKernelArg(kernels[j], 7, sizeof(cl_mem), &output_f_buf[i % 2]);
                    checkError(status, "Failed to set argument 7");

                    status = clSetKernelArg(kernels[j], 8, sizeof(cl_mem), &output_p_buf[i % 2]);
                    checkError(status, "Failed to set argument 8");

                    // launch the kernel.
                    if (hw_call == kernel_start_call + calls_per_kernel - 1 || hw_call == batch.end_call) {
                        status = clEnqueueTask(kernel_queues[j], kernels[j], 3, dependencies, &kernel_event[i][j]);
                        checkError(status, "Failed to launch kernel");
                    } else {
                        status = clEnqueueTask(kernel_queues[j], kernels[j], 3, dependencies, NULL);
                        checkError(status, "Failed to launch kernel");
                    }

                    issued_calls++;
                }
                kernel_start_call += calls_per_kernel;
                clFlush(kernel_queues[j]);
                kernel_event_count++;
            }

        }  // end (i >= 2)

        clEnqueueReadBuffer(read_q, output_f_buf[i % 2], CL_FALSE, 0, batch.total_n * sizeof(cl_int), &f_hw[a_hw_offsets[calls_hw[batch.start_call]]], kernel_event_count, kernel_event[i], &f_read_event[i]);
        clEnqueueReadBuffer(read_q, output_p_buf[i % 2], CL_FALSE, 0, batch.total_n * sizeof(cl_int), &p_hw[a_hw_offsets[calls_hw[batch.start_call]]], kernel_event_count, kernel_event[i], &p_read_event[i]);
        clFlush(read_q);

        // save number of kernels that got active for this batch (since it's used later)
        kernel_event_count_list[i] = kernel_event_count;
    }
    //fprintf(stderr, "[INFO] Hardware work enqueuing done!\n");

    struct timeval sw_begin, sw_end;
    gettimeofday(&sw_begin, NULL);

    // multi-threaded software execution
	core_t core;
    db_t db;

    core.num_thread = NUM_SW_THREADS;

    db.n_batch = calls_sw.size();
    db.calls_sw = &calls_sw[0];

	db.a = &a[0];
	db.f = f;
	db.p = p;
    db.a_offsets = &a_offsets[0];

	db.n = &n[0];
	db.max_dist_x = &max_dist_x[0];
	db.max_dist_y = &max_dist_y[0];
	db.bw = &bw[0];
    db.avg_qspan = &avg_qspan[0];

    work_db(&core, &db); 

    gettimeofday(&sw_end, NULL);

    //fprintf(stderr, "[INFO] Multi-threaded software processing done!\n");

    // wait for all hardware work to finish
    status = clFinish(read_q);
    checkError(status, "Failed to finish read_q");

    //fprintf(stderr, "[INFO] Total work done!\n");

    coarse_batch_timing_t batch_timing;

    // print total time taken for processing the batch
    gettimeofday(&end, NULL);
    batch_timing.total_time = 1.0 * (end.tv_sec - begin.tv_sec) + 1.0 * (end.tv_usec - begin.tv_usec) / 1000000;
    fprintf(stderr, "\nTotal time for coarse-grained batch: %0.3f s\n", batch_timing.total_time);

    float sw_time = 1.0 * (sw_end.tv_sec - sw_begin.tv_sec) + 1.0 * (sw_end.tv_usec - sw_begin.tv_usec) / 1000000;
    fprintf(stderr, "\nTime for multi-threaded software execution: %0.3f s\n", sw_time);


    /* // print time taken only for execution on hardware
    cl_ulong exec_time_ns = 0;
    for (int i = 0; i < batches_count; i++) {
        for (int j = 0; j < kernel_event_count_list[i]; j++) {
            exec_time_ns += getStartEndTime(kernel_event[i][j]);
        }
    }
    fprintf(stderr, "Total execution time for hardware calls in coarse-grained batch\t: %0.3f s\n", double(exec_time_ns) * 1e-9); */

    // print times spent for data transfer
    cl_ulong time_ns = 0;
    cl_ulong total_time_ns = 0;

    for (int i = 0; i < batches_count; i++) {
        time_ns += getStartEndTime(write_event[i]);
    }
    total_time_ns += time_ns;
    fprintf(stderr, "Total transfer time (a)\t: %0.3f s\n", double(time_ns) * 1e-9);

    time_ns = 0;
    for (int i = 0; i < batches_count; i++) {
        time_ns += getStartEndTime(f_read_event[i]);
    }
    total_time_ns += time_ns;
    fprintf(stderr, "Total transfer time (f)\t: %0.3f s\n", double(time_ns) * 1e-9);

    time_ns = 0;
    for (int i = 0; i < batches_count; i++) {
        time_ns += getStartEndTime(p_read_event[i]);
    }
    total_time_ns += time_ns;
    fprintf(stderr, "Total transfer time (p)\t: %0.3f s\n", double(time_ns) * 1e-9);

    batch_timing.data_transfer_time = double(total_time_ns) * 1e-9;
    fprintf(stderr, "Total transfer time (a + f + p)\t: %0.3f s\n", batch_timing.data_transfer_time);

    // release events.
    for (int i = 0; i < batches_count; i++) {
        for (int j = 0; j < kernel_event_count_list[i]; j++) {
            clReleaseEvent(kernel_event[i][j]);
        }
        clReleaseEvent(write_event[i]);
        clReleaseEvent(f_read_event[i]);
        clReleaseEvent(p_read_event[i]);
    }

    return batch_timing;
}

void write_output(vector<cl_long> n, cl_int* f, cl_int* p, vector<cl_long> a_offsets, cl_int* f_hw, cl_int* p_hw, vector<cl_long> a_hw_offsets) {
    for (int call = 0; call < n.size(); call++) {
        printf("%ld\n", n[call]);
        if (a_hw_offsets[call] == -1) { // call scheduled on sw
            for (long i = a_offsets[call]; i < a_offsets[call] + n[call]; i++) {
                printf("%d\t%d\n", f[i], p[i]);
            }
        } else { // call scheduled on hw
            for (long i = a_hw_offsets[call]; i < a_hw_offsets[call] + n[call]; i++) {
                printf("%d\t%d\n", f_hw[i], p_hw[i]);
            }
        }
        
        printf("EOR\n");
    }
}

bool init(long buf_size) {
    cl_int status;

    if (!setCwdToExeDir()) {
        return false;
    }

    // Get the OpenCL platform.
    if (use_emulator) {
        platform = findPlatform("Intel(R) FPGA Emulation Platform for OpenCL(TM)");
    } else {
        platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
    }
    if (platform == NULL) {
        fprintf(stderr, "ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
        return false;
    }

    // User-visible output - Platform information
    {
        char char_buffer[STRING_BUFFER_LEN];
        fprintf(stderr, "Querying platform for info:\n");
        fprintf(stderr, "==========================\n");
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
        fprintf(stderr, "%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
        clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
        fprintf(stderr, "%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
        clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
        fprintf(stderr, "%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);
    }

    // Query the available OpenCL devices.
    scoped_array<cl_device_id> devices;
    cl_uint num_devices;

    devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));

    // We'll just use the first device.
    device = devices[0];

    // Display some device information.
    display_device_info(device);

    // Create the context.
    context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &status);
    checkError(status, "Failed to create context");

    // Create the write queue.
    write_q = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create main command write_q");

    // Create the read queue.
    read_q = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create main command read_q");

    // Create the program.
    std::string binary_file = getBoardBinaryFile("minimap2_opencl", device);
    fprintf(stderr, "Using AOCX: %s\n", binary_file.c_str());
    program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

    // Build the program that was just created.
    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    checkError(status, "Failed to build program");

    // Create the kernels - name passed in here must match kernel names in the
    // original CL file, that was compiled into an AOCX file using the AOC tool
    // This also creates a seperate command queue for each kernel
    for (int i = 0; i < NUM_HW_KERNELS; i++) {
        // Generate the kernel name (minimap2_opencl0, minimap2_opencl1, minimap2_opencl2, etc.), as defined in the CL file
        std::ostringstream kernel_name;
        kernel_name << "minimap2_opencl" << i;

        kernels[i] = clCreateKernel(program, kernel_name.str().c_str(), &status);
        checkError(status, "Failed to create kernel");

        kernel_queues[i] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
        checkError(status, "Failed to create kernel command queue");
    }

    // Input buffers.
    for (int i = 0; i < 2; i++) {
        input_a_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                        buf_size * sizeof(cl_ulong2), NULL, &status);
        checkError(status, "Failed to create buffer for input a");
    }

    // Output buffers.
    for (int i = 0; i < 2; i++) {
        output_f_buf[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                         buf_size * sizeof(cl_int), NULL, &status);
        checkError(status, "Failed to create buffer for f");
    }

    for (int i = 0; i < 2; i++) {
        output_p_buf[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                         buf_size * sizeof(cl_int), NULL, &status);
        checkError(status, "Failed to create buffer for p");
    }

    return true;
}

// Free the resources allocated during initialization
void cleanup() {
    if (kernels) {
        for (int i = 0; i < NUM_HW_KERNELS; i++) {
            clReleaseKernel(kernels[i]);
            clReleaseCommandQueue(kernel_queues[i]);
        }
    }

    if (input_a_buf) {
        clReleaseMemObject(input_a_buf[0]);
        clReleaseMemObject(input_a_buf[1]);
    }
    if (output_f_buf) {
        clReleaseMemObject(output_f_buf[0]);
        clReleaseMemObject(output_f_buf[1]);
    }
    if (output_p_buf) {
        clReleaseMemObject(output_p_buf[0]);
        clReleaseMemObject(output_p_buf[1]);
    }
    if (program) {
        clReleaseProgram(program);
    }
    if (write_q) {
        clReleaseCommandQueue(write_q);
    }
    if (read_q) {
        clReleaseCommandQueue(read_q);
    }
    if (context) {
        clReleaseContext(context);
    }
}

// Helper functions to display parameters returned by OpenCL queries
static void device_info_ulong(cl_device_id device, cl_device_info param, const char* name) {
    cl_ulong a;
    clGetDeviceInfo(device, param, sizeof(cl_ulong), &a, NULL);
    fprintf(stderr, "%-40s = %lu\n", name, a);
}
static void device_info_uint(cl_device_id device, cl_device_info param, const char* name) {
    cl_uint a;
    clGetDeviceInfo(device, param, sizeof(cl_uint), &a, NULL);
    fprintf(stderr, "%-40s = %u\n", name, a);
}
static void device_info_bool(cl_device_id device, cl_device_info param, const char* name) {
    cl_bool a;
    clGetDeviceInfo(device, param, sizeof(cl_bool), &a, NULL);
    fprintf(stderr, "%-40s = %s\n", name, (a ? "true" : "false"));
}
static void device_info_string(cl_device_id device, cl_device_info param, const char* name) {
    char a[STRING_BUFFER_LEN];
    clGetDeviceInfo(device, param, STRING_BUFFER_LEN, &a, NULL);
    fprintf(stderr, "%-40s = %s\n", name, a);
}

// Query and display OpenCL information on device and runtime environment
static void display_device_info(cl_device_id device) {
    fprintf(stderr, "Querying device for info:\n");
    fprintf(stderr, "========================\n");
    device_info_string(device, CL_DEVICE_NAME, "CL_DEVICE_NAME");
    device_info_string(device, CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR");
    device_info_uint(device, CL_DEVICE_VENDOR_ID, "CL_DEVICE_VENDOR_ID");
    device_info_string(device, CL_DEVICE_VERSION, "CL_DEVICE_VERSION");
    device_info_string(device, CL_DRIVER_VERSION, "CL_DRIVER_VERSION");
    device_info_uint(device, CL_DEVICE_ADDRESS_BITS, "CL_DEVICE_ADDRESS_BITS");
    device_info_bool(device, CL_DEVICE_AVAILABLE, "CL_DEVICE_AVAILABLE");
    device_info_bool(device, CL_DEVICE_ENDIAN_LITTLE, "CL_DEVICE_ENDIAN_LITTLE");
    device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE");
    device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE");
    device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE");
    device_info_bool(device, CL_DEVICE_IMAGE_SUPPORT, "CL_DEVICE_IMAGE_SUPPORT");
    device_info_ulong(device, CL_DEVICE_LOCAL_MEM_SIZE, "CL_DEVICE_LOCAL_MEM_SIZE");
    device_info_ulong(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, "CL_DEVICE_MAX_CLOCK_FREQUENCY");
    device_info_ulong(device, CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS");
    device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_ARGS, "CL_DEVICE_MAX_CONSTANT_ARGS");
    device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE");
    device_info_uint(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
    device_info_uint(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, "CL_DEVICE_MEM_BASE_ADDR_ALIGN");
    device_info_uint(device, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE");
    device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR");
    device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT");
    device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT");
    device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG");
    device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT");
    device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE");

    {
        cl_command_queue_properties ccp;
        clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &ccp, NULL);
        fprintf(stderr, "%-40s = %s\n", "Command queue out of order? ", ((ccp & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) ? "true" : "false"));
        fprintf(stderr, "%-40s = %s\n", "Command queue profiling enabled? ", ((ccp & CL_QUEUE_PROFILING_ENABLE) ? "true" : "false"));
    }
}
