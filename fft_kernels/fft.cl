// Out of place 1024-point complex data radix-4 FFT designed to saturate two banks of
//  DDR3-1600 memory.  Will achieve ~70 GFLOPS.
// vim: set ts=2 sw=2 expandtab:

// The following design is hardcoded for a 1024-point transform.  Different
// sizes require modification to the number of stages.
#define FFT_POINTS  1024
#define LOG4N       5
#define RADIX       4

//#define TWIDDLE_FACTORS_ON_FPGA

// Complex multiplication
#define COMP_MULT(a, b, out) \
  do { \
    (out).x = (a).x*(b).x - (a).y*(b).y; \
    (out).y = (a).x*(b).y + (a).y*(b).x; \
  } while (0)

inline void fft_stage( bool first_iter, bool final_iter, const unsigned t, local float2 *in_buf0, local float2 *out_buf0, constant ushort *unordered_index_lut, constant float2* twiddle );

__kernel
__attribute__((reqd_work_group_size((FFT_POINTS/RADIX),1,1)))
void
fft(  __global float2 * restrict dataIn, __global float2 * restrict dataOut, __constant float2 * twiddle, __constant ushort * unordered_index_lut)
{
  // Manual vector lane 1
  __local float2 array0[FFT_POINTS];
  __local float2 array1[FFT_POINTS];
  __local float2 array2[FFT_POINTS];
  __local float2 array3[FFT_POINTS];
  __local float2 array4[FFT_POINTS];
  __local float2 array5[FFT_POINTS];

  // Manual vector lane 2
  __local float2 array0b[FFT_POINTS];
  __local float2 array1b[FFT_POINTS];
  __local float2 array2b[FFT_POINTS];
  __local float2 array3b[FFT_POINTS];
  __local float2 array4b[FFT_POINTS];
  __local float2 array5b[FFT_POINTS];

  const unsigned tid = get_local_id(0);
  unsigned lut_base;
  unsigned base;

  // Load input data
  {
    size_t gmem_base = 8 * get_global_id(0);
    unsigned lmem_base = 8 * (tid & 127);

#pragma unroll
    for (unsigned i=0; i<8; i++) {
      float2 tmp = dataIn[gmem_base + i];
      if (tid < 128) {
        array0[lmem_base + i] = tmp;
      } else {
        array0b[lmem_base + i] = tmp;
      }
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // STAGE 0 ==============================================================
  fft_stage( true, false, 0, array0, array1, unordered_index_lut, twiddle );
  fft_stage( true, false, 0, array0b, array1b, unordered_index_lut, twiddle );
  barrier(CLK_LOCAL_MEM_FENCE);

  // STAGE 1 ==============================================================
  fft_stage( false, false, 1, array1, array2, unordered_index_lut, twiddle );
  fft_stage( false, false, 1, array1b, array2b, unordered_index_lut, twiddle );
  barrier(CLK_LOCAL_MEM_FENCE);

  // STAGE 2 ==============================================================
  fft_stage( false, false, 2, array2, array3, unordered_index_lut, twiddle );
  fft_stage( false, false, 2, array2b, array3b, unordered_index_lut, twiddle );
  barrier(CLK_LOCAL_MEM_FENCE);

  // STAGE 3 ==============================================================
  fft_stage( false, false, 3, array3, array4, unordered_index_lut, twiddle );
  fft_stage( false, false, 3, array3b, array4b, unordered_index_lut, twiddle );
  barrier(CLK_LOCAL_MEM_FENCE);
  
  // STAGE 4 ==============================================================
  fft_stage( false, true, 4, array4, array5, unordered_index_lut, twiddle );
  fft_stage( false, true, 4, array4b, array5b, unordered_index_lut, twiddle );
  barrier(CLK_LOCAL_MEM_FENCE);
  
  // Store data - assuming N/4 threads
  {
    size_t gmem_base = 8 * get_global_id(0);
    lut_base = 4*1024; // Offset for this stage's LUT data
    float2 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    unsigned lut_ind = tid & 127;

    if (tid < 128) {
      tmp0 = array5[unordered_index_lut[lut_base + 8*lut_ind + 0]];;
      tmp1 = array5[unordered_index_lut[lut_base + 8*lut_ind + 1]];;
      tmp2 = array5[unordered_index_lut[lut_base + 8*lut_ind + 2]];;
      tmp3 = array5[unordered_index_lut[lut_base + 8*lut_ind + 3]];;
      tmp4 = array5[unordered_index_lut[lut_base + 8*lut_ind + 4]];;
      tmp5 = array5[unordered_index_lut[lut_base + 8*lut_ind + 5]];;
      tmp6 = array5[unordered_index_lut[lut_base + 8*lut_ind + 6]];;
      tmp7 = array5[unordered_index_lut[lut_base + 8*lut_ind + 7]];;
    } else {
      tmp0 = array5b[unordered_index_lut[lut_base + 8*lut_ind + 0]];;
      tmp1 = array5b[unordered_index_lut[lut_base + 8*lut_ind + 1]];;
      tmp2 = array5b[unordered_index_lut[lut_base + 8*lut_ind + 2]];;
      tmp3 = array5b[unordered_index_lut[lut_base + 8*lut_ind + 3]];;
      tmp4 = array5b[unordered_index_lut[lut_base + 8*lut_ind + 4]];;
      tmp5 = array5b[unordered_index_lut[lut_base + 8*lut_ind + 5]];;
      tmp6 = array5b[unordered_index_lut[lut_base + 8*lut_ind + 6]];;
      tmp7 = array5b[unordered_index_lut[lut_base + 8*lut_ind + 7]];;
    }

    dataOut[gmem_base + 0] = tmp0;
    dataOut[gmem_base + 1] = tmp1;
    dataOut[gmem_base + 2] = tmp2;
    dataOut[gmem_base + 3] = tmp3;
    dataOut[gmem_base + 4] = tmp4;
    dataOut[gmem_base + 5] = tmp5;
    dataOut[gmem_base + 6] = tmp6;
    dataOut[gmem_base + 7] = tmp7;
  }
}



inline void fft_stage( bool first_iter, bool final_iter, const unsigned t, local float2 *in_buf0, local float2 *out_buf0, constant ushort *unordered_index_lut, constant float2* twiddle ) {

  float2 c0, c1, c2, c3;
  float2 c0b, c1b, c2b, c3b;
  float2 c0c, c1c, c2c, c3c;
  unsigned outSubArraySize;
  unsigned outSubArrayID;
  unsigned quarterOutSubArraySize;
  unsigned posInQuarterOutSubArray;
  unsigned ind0, ind1, ind2, ind3;
  float2 c0_p_c2, c1_p_c3;
  float2 c0_m_c2, c1_m_c3;
  float2 c1_m_c3_times_j;
  float2 c0_p_c2b, c1_p_c3b;
  float2 c0_m_c2b, c1_m_c3b;
  float2 c1_m_c3_times_jb;
  float2 c0_p_c2c, c1_p_c3c;
  float2 c0_m_c2c, c1_m_c3c;
  float2 c1_m_c3_times_jc;
  float2 out0, out1, out2, out3, tmpF2;
  float2 out0b, out1b, out2b, out3b, tmpF2b;
  float2 out0c, out1c, out2c, out3c, tmpF2c;
  float2 trig2;
  float theta;

  const unsigned tid = get_local_id(0);
  unsigned lut_base;
  unsigned base;

  outSubArraySize = FFT_POINTS >> (t<<1);  // N >> (t*2)
  outSubArrayID = tid >> ( (2*(LOG4N-1)) - (2*t) );
  quarterOutSubArraySize = outSubArraySize >> 2;
  posInQuarterOutSubArray = tid & (quarterOutSubArraySize-1);

  ind0 = outSubArrayID * outSubArraySize + posInQuarterOutSubArray;
  ind1 = outSubArrayID * outSubArraySize + posInQuarterOutSubArray + quarterOutSubArraySize;
  ind2 = outSubArrayID * outSubArraySize + posInQuarterOutSubArray + 2*quarterOutSubArraySize;
  ind3 = outSubArrayID * outSubArraySize + posInQuarterOutSubArray + 3*quarterOutSubArraySize;

  if (first_iter) {
    c0 = in_buf0[ind0];
    c1 = in_buf0[ind1];
    c2 = in_buf0[ind2];
    c3 = in_buf0[ind3];
  } else {
    lut_base = (t-1)*1024; // Offset for this stage's LUT data
    c0 = in_buf0[unordered_index_lut[lut_base + ind0]];
    c1 = in_buf0[unordered_index_lut[lut_base + ind1]];
    c2 = in_buf0[unordered_index_lut[lut_base + ind2]];
    c3 = in_buf0[unordered_index_lut[lut_base + ind3]];
  }

  c0_p_c2 = c0 + c2;
  c1_p_c3 = c1 + c3;
  c0_m_c2 = c0 - c2;
  c1_m_c3 = c1 - c3;
  c1_m_c3_times_j.x = -1*c1_m_c3.y;
  c1_m_c3_times_j.y = c1_m_c3.x;

  // Output 0
  out0 = c0_p_c2 + c1_p_c3;

  // Output 1
  tmpF2 = c0_p_c2 - c1_p_c3;
#ifdef TWIDDLE_FACTORS_ON_FPGA
  theta = -1 * TWOPI * 2*posInQuarterOutSubArray / outSubArraySize;
  trig2.x = ( cos(theta) );
  trig2.y = ( sin(theta) );
  COMP_MULT(tmpF2, trig2, out1);
#else  // Use CPU-computed twiddle factors in __constant memory
  COMP_MULT(tmpF2, twiddle[(2*posInQuarterOutSubArray*FFT_POINTS)/outSubArraySize], out1);
#endif

  // Output 2
  tmpF2 = c0_m_c2 - c1_m_c3_times_j;
#ifdef TWIDDLE_FACTORS_ON_FPGA
  theta = -1 * TWOPI * 1*posInQuarterOutSubArray / outSubArraySize;
  trig2.x = ( cos(theta) );
  trig2.y = ( sin(theta) );
  COMP_MULT(tmpF2, trig2, out2);
#else  // Use CPU-computed twiddle factors in __constant memory
  COMP_MULT(tmpF2, twiddle[(posInQuarterOutSubArray*FFT_POINTS)/outSubArraySize], out2);
#endif

  // Output 3
  tmpF2 = c0_m_c2 + c1_m_c3_times_j;
#ifdef TWIDDLE_FACTORS_ON_FPGA
  theta = -1 * TWOPI * 3*posInQuarterOutSubArray / outSubArraySize;
  trig2.x = ( cos(theta) );
  trig2.y = ( sin(theta) );
  COMP_MULT(tmpF2, trig2, out3);
#else  // Use CPU-computed twiddle factors in __constant memory
  COMP_MULT(tmpF2, twiddle[(3*posInQuarterOutSubArray*FFT_POINTS)/outSubArraySize], out3);
#endif

  // Coalesced (but misordered) output
  base = tid*4;
  out_buf0[base] = out0;
  out_buf0[base + 1] = out1;
  out_buf0[base + 2] = out2;
  out_buf0[base + 3] = out3;
}




