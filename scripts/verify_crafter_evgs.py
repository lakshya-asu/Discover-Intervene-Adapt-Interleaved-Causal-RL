"""Smoke test: confirms CrafterInfoWrapper + EVGS produce correct variable vectors."""
import sys, crafter, numpy as np
sys.path.insert(0, 'src')
from dia.evgs_crafter import make_crafter_evgs

evgs = make_crafter_evgs()
env = crafter.Env()

# crafter.Env.reset() returns a raw numpy image; inventory is in info from step().
# On reset we get a zero vector (no inventory data yet).
raw_obs = env.reset()
wrapped_reset = {"obs": raw_obs, "info": {}}
x = evgs.obs_to_vars(wrapped_reset)
assert x.shape == (15,), f"Wrong shape: {x.shape}, expected (15,)"
print(f"EVGS shape OK: {x.shape}")
print(f"Values at reset (zeros expected): {x}")

# Step a few times: inventory arrives in info dict from step().
for i in range(5):
    raw_obs, _, _, info = env.step(env.action_space.sample())
    wrapped = {"obs": raw_obs, "info": info}
    x = evgs.obs_to_vars(wrapped)
    assert x.shape == (15,), f"Wrong shape at step {i}: {x.shape}"
    # Survival stats (health/food/drink) should be non-zero at start of episode
    health = x[12]
    food   = x[13]
    drink  = x[14]
    assert health >= 0, f"Negative health at step {i}: {health}"
print("All steps OK")

# Verify survival stats are populated from inventory
raw_obs, _, _, info = env.step(0)  # noop
wrapped = {"obs": raw_obs, "info": info}
x = evgs.obs_to_vars(wrapped)
health = x[12]
food   = x[13]
drink  = x[14]
print(f"Survival stats — health:{health:.1f}  food:{food:.1f}  drink:{drink:.1f}")
assert health > 0 or food > 0 or drink > 0, \
    "All survival stats are zero — health/food/drink not extracted from inventory"

print("verify_crafter_evgs PASSED")
