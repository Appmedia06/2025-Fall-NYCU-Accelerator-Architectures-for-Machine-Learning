M = int(input("input M: "))
K = int(input("input K: "))
N = int(input("input N: "))

M_block = int(( M + 3 ) / 4)
N_block = int(( N + 3 ) / 4)
MN_block = M_block * N_block

M_remainer = M % 4
if ( M_remainer == 0 ):
    M_sub = 0
else:
    M_sub =  4 - M_remainer


cycle_count = ( K + 4 ) * MN_block + ( MN_block * 4 - ( N_block * M_sub ))

print("cycle count is ", cycle_count)
