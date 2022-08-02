#include "common.h"

static inline float mg_log2(float x) // NB: this doesn't work when x<2
{
	union { float f; uint32_t i; } z = { x };
	float log_2 = ((z.i >> 23) & 255) - 128;
	z.i &= ~(255 << 23);
	z.i += 127 << 23;
	log_2 += (-0.34484843f * z.f + 2.02466578f) * z.f - 0.67487759f;
	return log_2;
}

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <x86intrin.h>
#include "parallel_chaining_v2_22.h"

#define MM_SEED_SEG_SHIFT 48
#define MM_SEED_SEG_MASK (0xffULL << (MM_SEED_SEG_SHIFT))

// Die on error. Print the error and exit if the return value of the previous function is -1
static inline void neg_chk(int ret, const char* func, const char* file,
                           int line) {
    if (ret >= 0)
        return;
    fprintf(stderr,
            "[%s::ERROR]\033[1;31m %s.\033[0m\n[%s::DEBUG]\033[1;35m Error "
            "occured at %s:%d.\033[0m\n\n",
            func, strerror(errno), func, file, line);
    exit(EXIT_FAILURE);
}

static inline int32_t steal_work(pthread_arg_t* all_args, int32_t n_threads) {
    int32_t i, c_i = -1;
    int32_t k;
    for (i = 0; i < n_threads; ++i) {
        pthread_arg_t args = all_args[i];
        //fprintf(stderr,"endi : %d, starti : %d\n",args.endi,args.starti);
        if (args.endi - args.starti > STEAL_THRESH) {
            //fprintf(stderr,"gap : %d\n",args.endi-args.starti);
            c_i = i;
            break;
        }
    }
    if (c_i < 0) {
        return -1;
    }
    k = __sync_fetch_and_add(&(all_args[c_i].starti), 1);
    //fprintf(stderr,"k : %d, end %d, start %d\n",k,all_args[c_i].endi,all_args[c_i].starti);
    return k >= all_args[c_i].endi ? -1 : k;
}

void* pthread_single(void* voidargs) {
    //const double start_time = aocl_utils::getCurrentTimestamp();

    int32_t i;
    pthread_arg_t* args = (pthread_arg_t*)voidargs;
    db_t* db = args->db;
    core_t* core = args->core;
    int32_t thread_index = args->thread_index;  // kisaru

    //printf("[INFO] Thread %d started!\n", thread_index);

#ifndef WORK_STEAL
    for (i = args->starti; i < args->endi; i++) {
        args->func(core, db, i, thread_index);
    }
#else
    pthread_arg_t* all_args = (pthread_arg_t*)(args->all_pthread_args);
    //adapted from kthread.c in minimap2
    for (;;) {
        i = __sync_fetch_and_add(&args->starti, 1);
        if (i >= args->endi) {
            break;
        }
        args->func(core, db, i, thread_index);
    }
    while ((i = steal_work(all_args, core->num_thread)) >= 0) {
        args->func(core, db, i, thread_index);
    }
#endif

    //const double end_time = aocl_utils::getCurrentTimestamp();

    // Wall-clock time taken for the thread.
    //printf("[INFO] Thread %d Time (HW)\t: %0.3f ms\n", thread_index, (end_time - start_time) * 1e3);

    //fprintf(stderr,"Thread %d done\n",(myargs->position)/THREADS);
    pthread_exit(0);
}

void pthread_db(core_t* core, db_t* db, void (*func)(core_t*, db_t*, int, int32_t)) {
    //create threads
    pthread_t tids[core->num_thread];
    pthread_arg_t pt_args[core->num_thread];
    int32_t t, ret;
    int32_t i = 0;
    int32_t num_thread = core->num_thread;
    int32_t step = (db->n_batch + num_thread - 1) / num_thread;
    //todo : check for higher num of threads than the data
    //current works but many threads are created despite

    //set the data structures
    for (t = 0; t < num_thread; t++) {
        pt_args[t].core = core;
        pt_args[t].db = db;
        pt_args[t].starti = i;
        i += step;
        if (i > db->n_batch) {
            pt_args[t].endi = db->n_batch;
        } else {
            pt_args[t].endi = i;
        }
        pt_args[t].func = func;
        pt_args[t].thread_index = t;  // kisaru
#ifdef WORK_STEAL
        pt_args[t].all_pthread_args = (void*)pt_args;
#endif
        //fprintf(stderr,"t%d : %d-%d\n",t,pt_args[t].starti,pt_args[t].endi);
    }

    //double cpu1 = cputime();
    //double real1 = realtime();

    //create threads
    for (t = 0; t < core->num_thread; t++) {
        ret = pthread_create(&tids[t], NULL, pthread_single,
                             (void*)(&pt_args[t]));
        NEG_CHK(ret);
    }
    //printf("[INFO] pthread creation done!\n");

    //pthread joining
    for (t = 0; t < core->num_thread; t++) {
        int ret = pthread_join(tids[t], NULL);
        NEG_CHK(ret);
    }

    //double cpu2 = cputime();
    //double real2 = realtime();

    //printf("cpu time = %f\n", (cpu2 - cpu1));
    //printf("real time = %f\n", (real2 - real1));
}

/* process the ith read in the batch db */
void work_per_single_read(core_t* core, db_t* db, int32_t i, int32_t thread_index) {
    int call = db->calls_sw[i];

    long n = db->n[call];
    int max_dist_x = db->max_dist_x[call];
    int max_dist_y = db->max_dist_y[call];
    int bw = db->bw[call];
    long offset = db->a_offsets[call];
    float avg_qspan = db->avg_qspan[call];

    cl_ulong2* a = &(db->a[offset]);
    int* f = &(db->f[offset]);
    int* p = &(db->p[offset]);


#ifndef USE_SIMD

    minimap2_opencl_sw(n, max_dist_x, max_dist_y, bw, a, f, p, avg_qspan);

#else

    // Parallel chaining data-structures
	anchor_t* anchors = (anchor_t*)malloc(n* sizeof(anchor_t));
	for (i = 0; i < n; ++i) {
		uint64_t ri = a[i].x;
		int32_t qi = (int32_t)a[i].y, q_span = a[i].y>>32&0xff; // NB: only 8 bits of span is used!!!
		anchors[i].r = ri;
		anchors[i].q = qi;
		anchors[i].l = q_span;
	}
	num_bits_t *anchor_r, *anchor_q, *anchor_l;
	create_SoA_Anchors_32_bit(anchors, n, anchor_r, anchor_q, anchor_l);

    // dp_chain obj(max_dist_x, max_dist_y, bw, 25, 64, 0, 0, 0, 0, 0, 0); // dummy params
	dp_chain obj(max_dist_x, max_dist_y, bw, 25, 64, 3, 40, 0.12, 0.0, 0, 1); // correct params (note: max_iter is set to 64 to match with ours)

    //kisaru
    uint32_t* f_1 = (uint32_t*)malloc(n * sizeof(uint32_t));
    int* p_1 = (int*)malloc(n * sizeof(int));
    int* v_1;//= (int*)malloc(n * sizeof(int));

	obj.mm_dp_vectorized(n, &anchors[0], anchor_r, anchor_q, anchor_l, f_1, p_1, v_1, max_dist_x, max_dist_y, NULL, NULL);

	// -16 is due to extra padding at the start of arrays
	anchor_r -= 16; anchor_q -= 16; anchor_l -= 16;
	free(anchor_r); 
	free(anchor_q); 
	free(anchor_l);
	free(anchors);
	for(int i = 0; i < n; i++){
        f[i] = f_1[i];
        p[i] = p_1[i];
        //v[i] = v_1[i]; //kisaru
	}

    //kisaru
	free(f_1);
	free(p_1);
	//free(v_1);

#endif

    //printf("call %d finished\n", call);

    //fprintf(stderr,"n: %d, time(ms): %f\n", n, (realtime() - start) * 1000);

}

/* process all reads in the given batch db */
void * work_db(void * args) {
    struct timeval sw_begin, sw_end;
    gettimeofday(&sw_begin, NULL);

    sw_batch_t * sw_batch = (sw_batch_t *)args;
    core_t* core = &(sw_batch->core);
    db_t* db = &(sw_batch->db);

    /*
    printf("n_batch = %d\n", db->n_batch);
    printf("size(calls_sw) = %d\n", db->calls_sw.size());
    for (int i = 0; i < db->calls_sw.size(); i++) {
        printf("%d, ", db->calls_sw[i]);
    }
    printf("\n");
    */

    if (core->num_thread == 1) {
        int32_t i = 0;
        for (i = 0; i < db->n_batch; i++) {
            work_per_single_read(core, db, i, 0);
        }
    } else {
        pthread_db(core, db, work_per_single_read);
    }

    gettimeofday(&sw_end, NULL);

    float sw_time = 1.0 * (sw_end.tv_sec - sw_begin.tv_sec) + 1.0 * (sw_end.tv_usec - sw_begin.tv_usec) / 1000000;
    fprintf(stderr, "\nTime for multi-threaded software execution: %0.3f s\n", sw_time);
    
    pthread_exit(0);
}

const char LogTable256[256] = {
#define LT(n) n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n
    -1, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    LT(4), LT(5), LT(5), LT(6), LT(6), LT(6), LT(6),
    LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7)};

inline int ilog2_32(unsigned int v) {
    unsigned int t, tt;
    if ((tt = v >> 16)) return (t = tt >> 8) ? 24 + LogTable256[t] : 16 + LogTable256[tt];
    return (t = v >> 8) ? 8 + LogTable256[t] : LogTable256[v];
}

void minimap2_opencl_sw(long n, int max_dist_x, int max_dist_y, int bw, cl_ulong2 * a,
                        int* f, int* p, float avg_qspan) {
    long i, j, st = 0;
    unsigned long sum_qspan = 0;

    // fill the score and backtrack arrays
    for (i = 0; i < n; ++i) {
        unsigned long ri = a[i].x;
        long max_j = -1;
        int qi = (int)a[i].y, q_span = a[i].y >> 32 & 0xff;  // NB: only 8 bits of span is used!!!
        int max_f = q_span, min_d;
        int sidi = (a[i].y & MM_SEED_SEG_MASK) >> MM_SEED_SEG_SHIFT;
        while (st < i && ri > a[st].x + max_dist_x) ++st;
        int h = 65;
        for (j = i - 1; j >= st && j > i - h; --j) {
            long dr = ri - a[j].x;
            int dq = qi - (int)a[j].y, dd, sc, log_dd;
            int sidj = (a[j].y & MM_SEED_SEG_MASK) >> MM_SEED_SEG_SHIFT;
            if ((/*sidi == sidj*/ 1 && dr == 0) || dq <= 0) continue;  // don't skip if an anchor is used by multiple segments; see below
            if ((/*sidi == sidj*/ 1 && dq > max_dist_y) || dq > max_dist_x) continue;
            dd = dr > dq ? dr - dq : dq - dr;
            if (/*sidi == sidj*/ 1 && dd > bw) continue;
            if (/*n_segs > 1 kisaru*/ 1 && /*!is_cdna*/ 1 && /*sidi == sidj*/ 1 && dr > max_dist_y) continue;
            min_d = dq < dr ? dq : dr;
            sc = min_d > q_span ? q_span : dq < dr ? dq
                                                   : dr;
            log_dd = dd ? ilog2_32(dd) : 0;
            if (/*is_cdna*/ 0 || /*sidi != sidj*/ 0) {
                int c_log, c_lin;
                c_lin = (int)(dd * avg_qspan);
                c_log = log_dd;
                if (sidi != sidj && dr == 0)
                    ++sc;  // possibly due to overlapping paired ends; give a minor bonus
                else if (dr > dq || sidi != sidj)
                    sc -= c_lin < c_log ? c_lin : c_log;
                else
                    sc -= c_lin + (c_log >> 1);
            } else
                sc -= (int)(dd * avg_qspan) + (log_dd >> 1);
            sc += f[j];
            if (sc > max_f) {
                max_f = sc, max_j = j;
            }
        }
        f[i] = max_f, p[i] = max_j;
    }
}