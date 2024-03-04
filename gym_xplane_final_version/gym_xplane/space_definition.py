import numpy as np

from gym import spaces
import gym




class xplane_space():
 
    def _action_space(self):
        
        '''
        return  spaces.Dict({"Latitudinal_Stick":  spaces.Box(low=-1, high=1, shape=()),
            "Longitudinal_Stick":  spaces.Box(low=-1, high=1, shape=()),
            "Rudder_Pedals":  spaces.Box(low=-1, high=1, shape=()),"Throttle":  spaces.Box(low=-1, high=1, shape=()),
            "Gear":  spaces.Discrete(2),"Flaps":  spaces.Box(low=0, high=1, shape=()),
            "Speedbrakes": spaces.Box(low=-0.5, high=1.5, shape=())})
        '''
        return spaces.Box(np.array([ -1, -1, -1,-1/4]),np.array([1,1,1,1]))

    def _observation_space(self):
        
        '''
        return spaces.Dict({1"Latitude":  spaces.Box(low=0, high=360, shape=()),
            2"Longitude":  spaces.Box(low=0, high=360, shape=()),
            3"Altitude":  spaces.Box(low=0, high=8500, shape=()),4"Pitch":  spaces.Box(low=-290, high=290, shape=()),5"Roll":  spaces.Box(low=-100, high=100, shape=()),6"Heading":  spaces.Box(low=0, high=360, shape=()),7"gear":  spaces.Discrete(2),8"yoke_pitch_ratio":  spaces.Box(low=-2.5, high=2.5, shape=()),9"yoke_roll_ratio":  spaces.Box(low=-300, high=300, shape=()),10"yoke_heading_ratio":  spaces.Box(low=-180, high=180,shape=()),11"alpha":  spaces.Box(low=-100, high=100,shape=()),
            "wing_sweep_ratio":  spaces.Box(low=-100, high=100, shape=()),"flap_ratio":  spaces.Box(low=-100, high=100, shape=()),
            12"speed": spaces.Box(low=-2205, high=2205, shape=())})
            
            state14
            (1 rollrate, 2 pitch rate, 3 altitude, 4 pitch, 5 roll, 6 velocity_x, 7 velocity_y, 8 velocity_z, 9 delta altitude, 10 delta heading, 11 yaw rate)
        '''
        # return spaces.Box(np.array([ -360, -360, 0 ,-290,-100,-360,-360,-1000,-1300,-1000,-1000]),np.array([360,360,8500,290,100,360,360,1000,1300,1000,1000]))
        return spaces.Box(np.array([ -180, -180, 0 ,-180,-180,-180,-180,-180,0,-360,-180]),np.array([180,180,2000,180,180,180,180,180,2000,360,180]))

      
