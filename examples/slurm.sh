cd $TMP
enroot start --rw --root tensorflow+2.11.0-gpu bash
cd

# fix for
# OSError: /usr/local/lib/python3.8/dist-packages/nvidia/cublas/lib/libcublas.so.11: undefined symbol: cublasLtGetStatusString, version libcublasLt.so.11
export LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH

srun --jobid=21625465 --overlap --pty /bin/bash