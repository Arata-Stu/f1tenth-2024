import numpy as np
from src.planner import PurePursuitPlanner

class ProgressReward:
    def __init__(self, map_manager, max_theta=100.0, reward_gain = 1.0):
        self.max_theta = max_theta
        self.map_manager = map_manager
        self.reward_gain = reward_gain
        
    def calc_reward(self, pre_obs, obs ,action=None):
        reward = 0

        if obs['lap_counts'] == 1:
            return 1  # complete
        if obs['collisions'][0]:
            return -1 # crash
        if abs(obs['poses_theta'][0]) > self.max_theta:
            return -1 #spin
        
        current_position = np.array([obs['poses_x'][0], obs['poses_y'][0]])
        prev_position = np.array([pre_obs['poses_x'][0], pre_obs['poses_y'][0]])

        current_progress = self.calc_progress(current_position)
        prev_progress = self.calc_progress(prev_position)

        progress = current_progress - prev_progress

        reward = progress
        reward *= self.reward_gain

        return reward
    


    def get_trackline_segment(self, point):
        wpts = self.map_manager.waypoints[:, :2]
        # Convert list of tuples to numpy array for efficient calculations
        wpts_array = np.array(wpts)
        point_array = np.array(point).reshape(1, 2)  # pointを2次元配列に変換して形状を合わせる
    
        # Calculate the distance from the point to each of the waypoints
        dists = np.linalg.norm(point_array - wpts_array, axis=1)
    
        # Find the segment that is closest to the point
        min_dist_segment = np.argmin(dists)
        if min_dist_segment == 0:
            return 0, dists
        elif min_dist_segment == len(dists)-1:
            return len(dists)-2, dists 

        if dists[min_dist_segment+1] < dists[min_dist_segment-1]:
            return min_dist_segment, dists
        else: 
            return min_dist_segment - 1, dists
        
    def interp_pts(self, idx, dists):
        """
        Returns the distance along the trackline and the height above the trackline
        Finds the reflected distance along the line joining wpt1 and wpt2
        Uses Herons formula for the area of a triangle
        
        """
        d_ss = self.map_manager.cum_dis[idx+1] - self.map_manager.cum_dis[idx]
        d1, d2 = dists[idx], dists[idx+1]

        if d1 < 0.01: # at the first point
            x = 0   
            h = 0
        elif d2 < 0.01: # at the second point
            x = dists[idx] # the distance to the previous point
            h = 0 # there is no distance
        else: 
            # if the point is somewhere along the line
            s = (d_ss + d1 + d2)/2
            Area_square = (s*(s-d1)*(s-d2)*(s-d_ss))
            Area = Area_square**0.5
            h = Area * 2/d_ss
            if np.isnan(h):
                h = 0
            x = (d1**2 - h**2)**0.5

        return x, h



    def calc_progress(self, point):
        idx, dists = self.get_trackline_segment(point)

        x, h = self.interp_pts(idx, dists)

        s = self.map_manager.cum_dis[idx] + x
        
        return s
    

class PPReward:
    def __init__(self, map_manager, steer_range, speed_range, max_theta=100.0, steer_w=0.4, speed_w=0.4, alpha= 0.25, reward_gain=1.0):
        self.max_theta = max_theta
        self.map_manager = map_manager
        self.steer_range = steer_range
        # self.speed_range = speed_range
        self.speed_range = 20.0
        self.steer_w = steer_w
        self.speed_w = speed_w
        self.alpha = alpha
        self.reward_gain = reward_gain

        wheelbase=(0.17145+0.15875)
        self.planner = PurePursuitPlanner(wheelbase=wheelbase, map_manager=map_manager, lookahead=0.6 ,max_reacquire=20.) 

    def calc_reward(self, pre_obs, obs ,action):
        reward = 0

        if obs['lap_counts'][0] == 2:
            return 1  # complete
        if obs['collisions'][0]:
            return -1 # crash
        if abs(obs['poses_theta'][0]) > self.max_theta:
            return -1 #spin

        pp_action = self.planner.plan(pre_obs, id=0, gain=0.2)
        
        
        steer_reward =  (abs(pp_action[0] - action[0]) / self.steer_range)  * self.steer_w
        throttle_reward =   (abs(pp_action[1] - action[1]) / self.speed_range) * self.speed_w

        reward = self.alpha - steer_reward - throttle_reward
        reward = max(reward, 0) # limit at 0

        reward *= self.reward_gain
        return reward