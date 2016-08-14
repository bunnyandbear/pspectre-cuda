#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

batchfile=$(mktemp)

echo "#!/bin/bash
#SBATCH -J batch_job_cuda
#SBATCH -A uoa00436
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH -C kepler
#SBATCH -D $DIR
#SBATCH --mail-user=cliu712@aucklanduni.ac.nz
#SBATCH --mail-type=ALL

module load GCC/4.9.2
module load CUDA/7.5.18
srun $DIR/a.out -L 6 -N 256 -t 0.00625
" > $batchfile
sbatch $batchfile
