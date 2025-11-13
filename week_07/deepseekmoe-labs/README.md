# deepseekmoe-labs
Minimal DeepSeekMoE lab with deterministic tests.

## Generate a case
python3 tests/gen_cases.py --name case1 --N 2 --B 64 --d 32 --h 64 --E 8 --seed 1

## Single-thread C
make -C c_single
./c_single/moe_single --case cases/case1 --rank 0

## MPI all-to-all
make -C mpi_ep
mpirun -np 2 ./mpi_ep/moe_mpi --case cases/case1

## CUDA/NCCL
make -C cuda_nccl
mpirun -np 2 ./cuda_nccl/moe_nccl --case cases/case1
