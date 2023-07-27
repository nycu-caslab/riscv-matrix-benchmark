.text
  .balign 4
  .global sgemm_nn_rvm_uh

# RV64RVM system
# 
# void
# sgemm_nn_rvm(size_t n,
#              size_t m,
#              size_t k,
#              const uint8_t* a,
#              size_t lda,
#              const uint8_t* b,
#              size_t ldb,
#              const uint8_t* c,
#              size_t ldc)

# c += a*b

#define n a0
#define mo a1
#define k a2
#define ap a3
#define astride a4
#define bp a5
#define bstride a6
#define cp a7
#define cstride t0
#define mtype t1
#define tilek t2
#define tilem t3
#define tilen t4
#define bmp   t5
#define cnp   s0
#define akp   s1
#define bkp   s2
#define ccp   s3
#define amp   s4
#define nt    s5
#define mt    s6
#define kt    s7



sgemm_nn_rvm_uh:
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

    # check for zero size matrices
    beqz n, exit
    beqz mo, exit
    beqz k, exit

    # Convert elements strides to byte strides
    ld cstride, 104(sp)

    # set mtype
    li mtype, 1
    li tilem, 0
    li tilen, 0
    li tilek, 0



    msettype x0, mtype # set mtype
    mv mt, mo
    mv amp, ap
    mv ccp, cp
c_row_loop:
    mul t6, tilem, astride
    add amp, amp, t6
    mul t6, tilem, cstride
    add ccp, ccp, t6
    msettilem tilem, mt

    mv nt, n
    mv bmp, bp
    mv cnp, ccp
    li tilen, 0
c_col_loop:
    add bmp, bmp, tilen
    add bkp, bmp, x0
    add akp, amp, x0
    add cnp, cnp, tilen
    msettilen tilen, nt
    
    mwemulc.mi acc0, acc1, 0
    mv kt, k
    li tilek, 0
inner_loop:
    slli t6, tilek, 1
    add akp, akp, tilek
    mul t6, tilek, bstride
    msettilek tilek, kt
    add bkp, bkp, t6
  
    mlae16.m tr0, (akp), astride
    mlbe16.m tr1, (bkp), bstride
    sub kt, kt, tilek
    mma.mm acc0, tr0, tr1

    bnez kt, inner_loop
    #beqz kt, c_col_loop_end
    #j inner_loop

c_col_loop_end:

    msce16.m acc0, (cnp), cstride
    sub nt, nt, tilen
    bnez nt, c_col_loop
    #beqz nt, c_row_loop_end
    #j c_col_loop

c_row_loop_end:
    sub mt, mt, tilem
    bnez mt, c_row_loop
    #beqz mt, exit
    #j c_row_loop

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

