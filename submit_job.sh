#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

batchfile=$(mktemp)

echo "#!/bin/bash
#SBATCH -J batch_job_cuda
#SBATCH -A uoa00436
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH -C kepler
#SBATCH -D $DIR
#SBATCH --mail-user=cliu712@aucklanduni.ac.nz
#SBATCH --mail-type=ALL

module load GCC/5.4.0
module load CUDA/8.0.61
#srun /bin/bash $DIR/build.sh
srun $DIR/pspectre @params.txt
#srun $DIR/pspectre
#srun $DIR/pspectre -L 6 -N 128 -t 0.00625
" > $batchfile
sbatch $batchfile
