#!/usr/bin/env bash
sbatch <<EOT
#!/usr/bin/env bash

#SBATCH --job-name=her_$1_$2
#SBATCH --output her_$1_$2.out
#SBATCH --ntasks=1
#SBATCH --time 24:00:00

python dqn_her.py -n $1 -r $2 -a $3 -b $4
exit 0
EOT


