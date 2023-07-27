.text
    .balign 4
    .global sgemm_nn_rvv_u8
# RV64IDV system
#
# void
# sgemm_nn(size_t n,
#          size_t m,
#          size_t k,
#          const int8_t*a,   // m * k matrix
#          size_t lda,
#          const int8_t*b,   // k * n matrix
#          size_t ldb,
#          int8_t*c,         // m * n matrix
#          size_t ldc)
#
#  c += a*b (alpha=1, no transpose on input matrices)
#  matrices stored in C row-major order

#define n a0
#define m a1
#define k a2
#define ap a3
#define astride a4
#define bp a5
#define bstride a6
#define cp a7
#define cstride t0
#define kt t1
#define nt t2
#define bnp t3
#define cnp t4
#define akp t5
#define bkp s0
#define nvl s1
#define ccp s2
#define amp s3

# Use args as additional temporaries
#define ft12 fa0
#define ft13 fa1
#define ft14 fa2
#define ft15 fa3

# This version holds a 16*VLMAX block of C matrix in vector registers
# in inner loop, but otherwise does not cache or TLB tiling.

sgemm_nn_rvv_u8:
    addi sp, sp, -104
    sd s0, 8(sp)
    sd s1, 16(sp)
    sd s2, 24(sp)
    sd s3, 32(sp)
    sd s4, 40(sp)
    sd s5, 48(sp)
    sd s6, 56(sp)
    sd s7, 64(sp)
    sd s8, 72(sp)
    sd s9, 80(sp)
    sd s10, 88(sp)
    sd s11, 96(sp)

    # Check for zero size matrices        
    beqz n, exit
    beqz m, exit
    beqz k, exit

    # Convert elements strides to byte strides.
    ld cstride, 104(sp)   # Get arg from stack frame
    #slli astride, astride, 2
    #slli bstride, bstride, 2
    #slli cstride, cstride, 2

    slti t6, m, 16
    bnez t6, end_rows

c_row_loop: # Loop across rows of C blocks

    mv nt, n  # Initialize n counter for next row of C blocks

    mv bnp, bp # Initialize B n-loop pointer to start
    mv cnp, cp # Initialize C n-loop pointer

c_col_loop: # Loop across one row of C blocks
    vsetvli nvl, nt, e8, m1, ta, ma  # 8-bit vectors, LMUL=1

    mv akp, ap   # reset pointer into A to beginning
    mv bkp, bnp # step to next column in B matrix

    # Initalize current C submatrix block from memory.
    vle8.v  v0, (cnp)
    add ccp, cnp, cstride
    vle8.v  v1, (ccp)
    add ccp, ccp, cstride
    vle8.v  v2, (ccp)
    add ccp, ccp, cstride
    vle8.v  v3, (ccp)
    add ccp, ccp, cstride
    vle8.v  v4, (ccp)
    add ccp, ccp, cstride
    vle8.v  v5, (ccp)
    add ccp, ccp, cstride
    vle8.v  v6, (ccp)
    add ccp, ccp, cstride
    vle8.v  v7, (ccp)
    add ccp, ccp, cstride
    vle8.v  v8, (ccp)
    add ccp, ccp, cstride
    vle8.v  v9, (ccp)
    add ccp, ccp, cstride
    vle8.v v10, (ccp)
    add ccp, ccp, cstride
    vle8.v v11, (ccp)
    add ccp, ccp, cstride
    vle8.v v12, (ccp)
    add ccp, ccp, cstride
    vle8.v v13, (ccp)
    add ccp, ccp, cstride
    vle8.v v14, (ccp)
    add ccp, ccp, cstride
    vle8.v v15, (ccp)


    mv kt, k # Initialize inner loop counter

    # Inner loop scheduled assuming 4-clock occupancy of vfmacc instruction and single-issue pipeline
    # Software pipeline loads
    lb s4, (akp)
    add amp, akp, astride
    lb s5, (amp)
    add amp, amp, astride
    lb s6, (amp)
    add amp, amp, astride
    lb s7, (amp)
    add amp, amp, astride
    # Get vector from B matrix
    vle8.v v16, (bkp)

    # Loop on inner dimension for current C block
 k_loop:
    vmacc.vx v0, s4, v16
    add bkp, bkp, bstride
    lb s8, (amp)
    add amp, amp, astride
    vmacc.vx v1, s5, v16
    addi kt, kt, -1    # Decrement k counter
    lb s9, (amp)
    add amp, amp, astride
    vmacc.vx v2, s6, v16
    lb s10, (amp)
    add amp, amp, astride
    lb s11, (amp)
    vmacc.vx v3, s7, v16
    add amp, amp, astride
    lb s4, (amp)
    add amp, amp, astride
    vmacc.vx v4, s8, v16
    lb s5, (amp)
    add amp, amp, astride
    vmacc.vx v5, s9, v16
    lb s6, (amp)
    add amp, amp, astride
    vmacc.vx v6, s10, v16
    lb s7, (amp)
    add amp, amp, astride
    vmacc.vx v7, s11, v16
    lb s8, (amp)
    add amp, amp, astride
    vmacc.vx v8, s4, v16
    lb s9, (amp)
    add amp, amp, astride
    vmacc.vx v9, s5, v16
    lb s10, (amp)
    add amp, amp, astride
    vmacc.vx v10, s6, v16
    lb s11, (amp)
    add amp, amp, astride
    addi akp, akp, 1            # Move to next column of a
    vmacc.vx v11, s7, v16
    beqz kt, 1f                 # Don't load past end of matrix
    lb s4, (akp)
    add amp, akp, astride
1:  vmacc.vx v12, s8, v16
    beqz kt, 1f
    lb s5, (amp)
    add amp, amp, astride
1:  vmacc.vx v13, s9, v16
    beqz kt, 1f
    lb s6, (amp)
    add amp, amp, astride
1:  vmacc.vx v14, s10, v16
    beqz kt, 1f                 # Exit out of loop
    lb s7, (amp)
    add amp, amp, astride
    vmacc.vx v15, s11, v16
    vle8.v v16, (bkp)            # Get next vector from B matrix, overlap loads with jump stalls
    j k_loop

1:  vmacc.vx v15, s11, v16
    
    # Save C matrix block back to memory
    vse8.v  v0, (cnp); add ccp, cnp, cstride;
    vse8.v  v1, (ccp); add ccp, ccp, cstride;
    vse8.v  v2, (ccp); add ccp, ccp, cstride;
    vse8.v  v3, (ccp); add ccp, ccp, cstride;
    vse8.v  v4, (ccp); add ccp, ccp, cstride;
    vse8.v  v5, (ccp); add ccp, ccp, cstride;
    vse8.v  v6, (ccp); add ccp, ccp, cstride;
    vse8.v  v7, (ccp); add ccp, ccp, cstride;
    vse8.v  v8, (ccp); add ccp, ccp, cstride;
    vse8.v  v9, (ccp); add ccp, ccp, cstride;
    vse8.v v10, (ccp); add ccp, ccp, cstride;
    vse8.v v11, (ccp); add ccp, ccp, cstride;
    vse8.v v12, (ccp); add ccp, ccp, cstride;
    vse8.v v13, (ccp); add ccp, ccp, cstride;
    vse8.v v14, (ccp); add ccp, ccp, cstride;
    vse8.v v15, (ccp)

    # Following tail instructions should be scheduled earlier in free slots during C block save.
    # Leaving here for clarity.

    # Bump pointers for loop across blocks in one row
    #slli t6, nvl, 1
    mv  t6, nvl
    add cnp, cnp, t6                         # Move C block pointer over
    add bnp, bnp, t6                         # Move B block pointer over
    sub nt, nt, nvl                          # Decrement element count in n dimension
    bnez nt, c_col_loop                      # Any more to do?

    # Move to next set of rows
    addi m, m, -16  # Did 16 rows above
    slli t6, astride, 4  # Multiply astride by 16
    add ap, ap, t6         # Move A matrix pointer down 16 rows
    slli t6, cstride, 4  # Multiply cstride by 16
    add cp, cp, t6         # Move C matrix pointer down 16 rows
    
    slti t6, m, 16
    beqz t6, c_row_loop

    # Handle end of matrix with fewer than 16 rows.
    # Can use smaller versions of above decreasing in powers-of-2 depending on code-size concerns.
end_rows:
    # Not done.

exit:
    ld s0, 8(sp)
    ld s1, 16(sp)
    ld s2, 24(sp)
    ld s3, 32(sp)
    ld s4, 40(sp)
    ld s5, 48(sp)
    ld s6, 56(sp)
    ld s7, 64(sp)
    ld s8, 72(sp)
    ld s9, 80(sp)
    ld s10, 88(sp)
    ld s11, 96(sp)
    addi sp, sp, 104
    ret