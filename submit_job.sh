#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

batchfile=$(mktemp)

echo "#!/bin/bash
#SBATCH -J batch_job_cuda
#SBATCH -A uoa00436
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:1
#SBATCH -C kepler
#SBATCH -D $DIR
#SBATCH --mail-user=cliu712@aucklanduni.ac.nz
#SBATCH --mail-type=ALL

module load GCC/4.9.2
module load CUDA/7.5.18
#srun /bin/bash $DIR/build.sh
srun $DIR/pspectre @params.txt
#srun $DIR/pspectre
#srun $DIR/pspectre -L 6 -N 128 -t 0.00625
" > $batchfile
sbatch $batchfile
