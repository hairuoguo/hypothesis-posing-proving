#!/usr/bin/env bash
sbatch <<EOT
#!/usr/bin/env bash

#SBATCH --job-name=$1
#SBATCH --output $1.out
#SBATCH --ntasks=1
#SBATCH --time 12:00:00

python dqn_her.py

exit 0
EOT


