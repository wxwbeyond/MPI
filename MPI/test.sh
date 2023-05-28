#test.sh
#!/bin/sh
#PBS -N test
#PBS -l nodes=8
pssh -h $PBS_NODEFILE mkdir ?p /home/s2012411 1>&2
pscp -h $PBS_NODEFILE /home/s2012411/test01 /home/s2000001 1>&2
mpiexec -np 8 -machinefile $PBS_NODEFILE /home/s2012411/test01 []