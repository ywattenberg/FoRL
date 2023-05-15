#!/bin/bash

algos="ppo a2c ppo_lstm a2c_lstm dqn ddpg"
envs="FoRLCartPole-v0 FoRLMountainCar-v0 FoRLPendulum-v0 FoRLAcrobot-v0 FoRLHopper-v0 FoRLHalfCheetah-v0"

for algo in $algos; do
    for env in $envs; do
        cp "templates/train_eval_template.sh" "gen/train_eval_${algo}_${env}.sh"
        sed -i "s/<ALGO>/${algo}/g" "gen/train_eval_${algo}_${env}.sh"
        sed -i "s/<ENV>/${env}/g" "gen/train_eval_${algo}_${env}.sh"
        echo "generated gen/train_eval_${algo}_${env}.sh"
    done
done

