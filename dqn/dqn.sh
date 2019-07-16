#!/usr/bin/env bash
sbatch <<EOT
#!/usr/bin/env bash

#SBATCH --job-name=cnn_$1_$2
#SBATCH --output out/cnn_$1_$2
#SBATCH --ntasks=1
#SBATCH --time 24:00:00
#SBATCH --mem 400

python dqn_her.py -n $1 -r $2
exit 0
EOT
