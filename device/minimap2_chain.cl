#define MM_SEED_SEG_SHIFT  48
#define MM_SEED_SEG_MASK   (0xffULL<<(MM_SEED_SEG_SHIFT))

#define INNER_LOOP_TRIP_COUNT 64

__constant char LogTable256[256] = {
#define LT(n) n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n
	-1, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
	LT(4), LT(5), LT(5), LT(6), LT(6), LT(6), LT(6),
	LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7)
};

inline int ilog2_32(unsigned int v)
{
	unsigned int t, tt;
	if ((tt = v>>16)) return (t = tt>>8) ? 24 + LogTable256[t] : 16 + LogTable256[tt];
	return (t = v>>8) ? 8 + LogTable256[t] : LogTable256[v];
}

inline int compute_sc(unsigned long a_x, unsigned long a_y, int f, unsigned long ri, int qi, int q_span, float avg_qspan, int max_dist_x, int max_dist_y, int bw) 
{
	long dr = ri - a_x;
	int dq = qi - (int)a_y, dd, sc, log_dd;
	int sidj = (a_y & MM_SEED_SEG_MASK) >> MM_SEED_SEG_SHIFT;
	if (dr == 0 || dq <= 0) return 0; 
	if (dq > max_dist_y || dq > max_dist_x) return 0;
	dd = dr > dq? dr - dq : dq - dr;
	if (dd > bw) return 0;
	if (dr > max_dist_y) return 0;
	int min_d = dq < dr? dq : dr;
	sc = min_d > q_span? q_span : dq < dr? dq : dr;
	log_dd = dd? ilog2_32(dd) : 0;
	sc -= (int)(dd * avg_qspan) + (log_dd>>1);
	sc += f;

	return sc;
}

inline void chain(long n, int max_dist_x, 
					int max_dist_y, int bw, 
					float avg_qspan, long offset,
					__global const ulong2 *restrict a,
					__global int *restrict f, __global int *restrict p)
{
	unsigned long a_x_local[INNER_LOOP_TRIP_COUNT + 1] = {0};
	unsigned long a_y_local[INNER_LOOP_TRIP_COUNT + 1] = {0};
	int f_local[INNER_LOOP_TRIP_COUNT + 1] = {0};

	// fill the score and backtrack arrays
	for (long i = 0; i < n; ++i) {

        ulong2 a_local = __prefetching_load(&a[offset + i]);
		a_x_local[0] = a_local.x;
		a_y_local[0] = a_local.y;

		unsigned long ri = a_x_local[0];
		int qi = (int)a_y_local[0], q_span = a_y_local[0]>>32&0xff; 
		
		long max_j = -1;
		int max_f = q_span;

		#pragma unroll
		for (int j = INNER_LOOP_TRIP_COUNT; j > 0; j--) {
			int sc = compute_sc(a_x_local[j], a_y_local[j], f_local[j], ri, qi, q_span, avg_qspan, max_dist_x, max_dist_y, bw);
			if (sc >= max_f && sc != q_span) {
				max_f = sc, max_j = i - j;
			}
		}
		
		f[offset + i] = max_f;
        p[offset + i] = max_j;
		f_local[0] = max_f;

		#pragma unroll
		for (int reg = INNER_LOOP_TRIP_COUNT; reg > 0; reg--) {
			f_local[reg] = f_local[reg - 1];
		}

		#pragma unroll
		for (int reg = INNER_LOOP_TRIP_COUNT; reg > 0; reg--) {
			a_x_local[reg] = a_x_local[reg - 1];
		}

		#pragma unroll
		for (int reg = INNER_LOOP_TRIP_COUNT; reg > 0; reg--) {
			a_y_local[reg] = a_y_local[reg - 1];
		}
	}
}


__kernel void chain0(long n, int max_dist_x, 
						int max_dist_y, int bw, 
						float avg_qspan, long offset,
						__global const ulong2 *restrict a,
						__global int *restrict f, __global int *restrict p)
{
    chain(n, max_dist_x, max_dist_y, bw, avg_qspan, offset, a, f, p);
}

__kernel void chain1(long n, int max_dist_x, 
						int max_dist_y, int bw, 
						float avg_qspan, long offset,
						__global const ulong2 *restrict a,
						__global int *restrict f, __global int *restrict p)
{
    chain(n, max_dist_x, max_dist_y, bw, avg_qspan, offset, a, f, p);
}

__kernel void chain2(long n, int max_dist_x, 
						int max_dist_y, int bw, 
						float avg_qspan, long offset,
						__global const ulong2 *restrict a,
						__global int *restrict f, __global int *restrict p)
{
    chain(n, max_dist_x, max_dist_y, bw, avg_qspan, offset, a, f, p);
}

__kernel void chain3(long n, int max_dist_x, 
						int max_dist_y, int bw, 
						float avg_qspan, long offset,
						__global const ulong2 *restrict a,
						__global int *restrict f, __global int *restrict p)
{
    chain(n, max_dist_x, max_dist_y, bw, avg_qspan, offset, a, f, p);
}
