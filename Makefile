.PHONY: train-cw

train-cw:
	python -m scripts.train --env dia.envs.causalworld_env:CausalWorldPushingEnv --steps 100000 --seed 0
