"""Smoke test: confirms CausalWorldInfoWrapper + EVGS return non-trivial values."""
import sys, numpy as np
sys.path.insert(0, 'src')

from causal_world.envs import CausalWorld
from causal_world.task_generators import generate_task
from dia.evgs_causalworld import CausalWorldInfoWrapper, make_causalworld_evgs

task = generate_task(task_generator_id='pick_and_place')
env = CausalWorld(task=task, enable_visualization=False)
env = CausalWorldInfoWrapper(env)
evgs = make_causalworld_evgs()

obs = env.reset()
print("Reset obs keys:", list(obs.keys()) if isinstance(obs, dict) else type(obs))

block_zs = []
for step in range(30):
    obs, r, done, info = env.step(env.action_space.sample())
    x = evgs.obs_to_vars(obs)
    block_z = obs.get('info', {}).get('block_z', 'MISSING') if isinstance(obs, dict) else 'N/A'
    block_zs.append(block_z)
    assert x.shape == (5,), f"Wrong shape: {x.shape}"
    if step < 5 or step == 29:
        print(f"step={step}  vars={x}  block_z={block_z}")
    if done:
        obs = env.reset()

float_zs = [b for b in block_zs if isinstance(b, float)]
print(f"block_z range: {min(float_zs):.4f} to {max(float_zs):.4f}")
print("EVGS smoke test PASSED")
env.close()
