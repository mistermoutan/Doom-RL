from utils.args import parse_arguments

params = parse_arguments()

import numpy as np


if (params.scenario == "basic"):
    scenario = "doom_scenario0_640-v0"
elif (params.scenario == "deadly_corridor"):
    scenario = "doom_scenario1_640-v0"
elif (params.scenario == "defend_the_center"):
    scenario = "doom_scenario2_640-v0"
elif (params.scenario == "defend_the_line"):
    scenario = "doom_scenario3_640-v0"
elif (params.scenario == "health_gathering"):
    scenario = "doom_scenario4_640-v0"

if params.scenario == 'deadly_corridor':
    resize = (64,64)
    crop = (30,-35,1,-1)
    
    if params.actions=='all':
        action_size = 10
    elif params.actions=='single':
        action_size = 6
    
    state_size = np.prod(resize)
    
elif params.scenario == 'basic':
    resize = (84,84)
    crop = (10,-10,30,-30)
    
    if params.actions=='all':
        action_size = 6
    elif params.actions=='single':
        action_size = 3
    
    state_size = np.prod(resize)

        
elif params.scenario == 'defend_the_center':
    resize = (84,159)
    crop = (40,-32,1,-1)
    
    if params.actions=='all':
        action_size = 6
    elif params.actions=='single':
        action_size = 3
    
    state_size = np.prod(resize)
    
elif params.scenario == 'defend_the_line':
    resize = (84,159)
    crop = (40,-32,1,-1)
    
    if params.actions=='all':
        action_size = 6
    elif params.actions=='single':
        action_size = 3
    
    state_size = np.prod(resize)
    
elif params.scenario == 'my_way_home':
    resize = (84,112)
    crop = (1,-1,1,-1)
    
    if params.actions=='all':
        action_size = 5
    elif params.actions=='single':
        action_size = 3
    
    state_size = np.prod(resize)

elif params.scenario == 'health_gathering':
    resize = (84,159)
    crop = (40,-32,1,-1)
    
    if params.actions=='all':
        action_size = 6
    elif params.actions=='single':
        action_size = 3
    
    state_size = np.prod(resize)

beta = 0.2
lr_pred = 10.0
pred_bonus_coef = 0.01

