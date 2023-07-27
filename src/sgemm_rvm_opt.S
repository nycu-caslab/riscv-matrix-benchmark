.text
  .balign 4
  .global sgemm_nn_rvm

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
#define m a1
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



sgemm_nn_rvm:
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
    beqz m, exit
    beqz k, exit

    # Convert elements strides to byte strides
    ld cstride, 104(sp)

    # set mtype
    li mtype, 1
    li tilem, 0
    li tilen, 0
    li tilek, 0



    msettype x0, mtype
    mv mt, m
    mv amp, ap
c_row_loop:
    mul t6, tilem, astride
    add amp, amp, t6
    msettilem tilem, mt

    mv nt, n
    mv bmp, bp
    li tilen, 0
c_col_loop:
    add bmp, bmp, tilen
    msettilen tilen, nt

    mv kt, k
    li tilek, 0
inner_loop:
    add akp, amp, tilek
    mul t6, tilek, bstride
    add bkp, bmp, t6
    msettilek tilek, kt
  
    mle16.tr.r.a tr0, (akp), astride
    mle16.tr.r.b tr1, (bkp), bstride
    sub kt, kt, tilek
    beqz kt, 1f
    
    

1:  mopa.mm acc0, tr0, tr1


    beqz kt, c_col_loop_end
    j inner_loop

c_col_loop_end:

    sub nt, nt, tilen
    beqz nt, c_row_loop_end
    j c_col_loop

c_row_loop_end:
    sub mt, mt, tilem
    beqz mt, exit
    j c_row_loop

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

