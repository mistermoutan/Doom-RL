from gdoom.gdoom_env import GDoomEnv
from gdoom.gdoom_env import WGDoomEnv

from gym.envs.registration import register

# Pretty sure this is the wrong way to go about this problem. How can we apply custom wrappers in the openai baseline
# framework?
for i in range(8):
    for frame_size in [64, 640]:
        register(
            id='doom_scenario%i_%i-v0'%(i, frame_size),
            entry_point='gdoom:WGDoomEnv',
            kwargs = {'level': i, 'frame_size': frame_size, 'mode': gdoom_env.CPU}
        )
    # human play
    register(
        id='doom_scenario%i_human-v0' % (i,),
        entry_point='gdoom:WGDoomEnv',
        kwargs={'level': i, 'mode': gdoom_env.HUMAN}
    )
    a = 234
