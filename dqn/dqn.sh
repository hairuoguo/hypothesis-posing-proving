#!/usr/bin/env bash
sbatch <<EOT
#!/usr/bin/env bash

#SBATCH --job-name=her_$1_$2_$3
#SBATCH --output out/her_$1_$2_$3.out
#SBATCH --ntasks=1
#SBATCH --time 24:00:00
#SBATCH --mem 4000

python dqn_her.py -n $1 -r $2 -w 4096 -d $3
exit 0
EOT
