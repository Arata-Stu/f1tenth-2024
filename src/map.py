import numpy as np
import math


MAP_DICT = {
                0: 'Austin',
                1: 'BrandsHatch',
                2: 'Budapest', 
                3: 'Catalunya', 
                4: 'Hockenheim', 
                5: 'IMS', 
                6: 'Melbourne', 
                7: 'MexicoCity', 
                8: 'Monza', 
                9: 'MoscowRaceway', 
                10: 'Nuerburgring', 
                11: 'Oschersleben', 
                12: 'Sakhir', 
                13: 'SaoPaulo', 
                14: 'Sepang', 
                15: 'Silverstone', 
                16: 'Sochi', 
                17: 'Spa', 
                18: 'Spielberg', 
                19: 'YasMarina', 
                20: 'Zandvoort'
}

class MapManager:
    #no raceline :Montreal, MexicoCity, Shanghai
    #pure pursuit doesn`t work well : IMS, 
    mapno = ["Austin","BrandsHatch","Budapest","Catalunya","Hockenheim","IMS","Melbourne","Monza","MoscowRaceway",
         "Nuerburgring","Oschersleben","Sakhir","SaoPaulo","Sepang","Silverstone","Sochi","Spa","Spielberg","YasMarina","Zandvoort"]
    map_index = 0

    
    def __init__(self, map_name='', raceline='raceline', delimiter=',', speed = 8.0 ,dir_path='./../f1tenth_racetracks/'):
        self.map_name = map_name
        self.raceline = raceline
        self.dir_path = dir_path
        self.map_path, self.csv_path = self.get_map_path_from_name(map=self.map_name, line=self.raceline)
        self.speed = speed
        
        if raceline == 'raceline':
            self.waypoints = np.genfromtxt(self.csv_path, delimiter=delimiter, skip_header=3, usecols=(1, 2, 5))

            self.update_speeds()
            self.cum_dis = np.genfromtxt(self.csv_path, delimiter=delimiter, usecols=(0))
            self.total_dis = self.cum_dis[-1]
            
        elif raceline == 'centerline':
            self.waypoints = np.genfromtxt(self.csv_path, delimiter=delimiter, usecols=(0, 1,))
            self.waypoints = np.hstack((self.waypoints, speed * np.ones_like(self.waypoints[:, :1])))
            self.update_speeds()
            diffs = np.diff(self.waypoints[:, :2], axis=0)
            seg_lengths = np.linalg.norm(np.diff(self.waypoints[:, :2], axis=0), axis=1)
            seg_lengths = np.linalg.norm(diffs, axis=1)
            self.cum_dis = np.insert(np.cumsum(seg_lengths), 0, 0)
            self.total_dis = self.cum_dis[-1]

        elif raceline == None:
            self.waypoints = None


    def get_map_path_from_name(self, map='Austin', line='raceline'):

        self.map_name = map
        
        map_path = f"{self.dir_path}{self.map_name}/{self.map_name}_map"
        line_path = line
        csv_path = f"{self.dir_path}{self.map_name}/{self.map_name}_{line_path}.csv"

        self.map_path = map_path
        self.csv_path = csv_path

        return map_path, csv_path
    
    def get_trackline_segment(self, point):
        wpts = self.waypoints[:, :2]
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
        d_ss = self.cum_dis[idx+1] - self.cum_dis[idx]
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

        s = self.cum_dis[idx] + x

        s = s/self.total_dis * 100
        
        return s
    
    
    
    def calculate_angle_between_points(self, p1, p2):
    
        # Calculate the slope of the line
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
    
        # Calculate the angle in radians
        angle_in_radians = math.atan2(dy, dx)
    
        # Convert the angle to degrees
        angle_in_degrees = math.degrees(angle_in_radians)
    
        return angle_in_degrees
    
    def calculate_angle_between_vectors(self, p1, p2, p3):
        v1 = p2 - p1
        v2 = p3 - p2
        unit_v1 = v1 / np.linalg.norm(v1)
        unit_v2 = v2 / np.linalg.norm(v2)
        dot_product = np.dot(unit_v1, unit_v2)
        angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
        return angle
    
    def update_speeds(self):
        # 全ウェイポイントの速度をself.speedに統一する
        self.waypoints[:, 2] = self.speed

    
    def update_speeds_based_on_angle(self, lookahead_backward=1, lookahead_forward=0):
        angles = [
            self.calculate_angle_between_vectors(self.waypoints[i-1][:2], self.waypoints[i][:2], self.waypoints[i+1][:2])
            for i in range(1, len(self.waypoints) - 1)
        ]
        angles = [angles[0]] + angles + [angles[-1]]  # Extend angles to match waypoints length

        # Speed settings based on curve types
        max_speed_straight = self.speed  # m/s for straight
        speed_shallow_curve = 3.5  # m/s for shallow curves
        speed_sharp_curve = 2.0  # m/s for sharp curves
        angle_shallow_curve = 4.0  # degrees, threshold for shallow curve
        angle_sharp_curve = 10.0  # degrees, threshold for sharp curve

        # angle_shallow_curve = 3.0  # degrees, threshold for shallow curve
        # angle_sharp_curve = 10.0  # degrees, threshold for sharp curve

        # Create a temporary speeds list to modify based on lookahead
        speeds = [max_speed_straight] * len(self.waypoints)

        for i in range(1, len(self.waypoints) - 1):
            current_angle = angles[i]

            if current_angle < angle_shallow_curve:
                reduced_speed = max_speed_straight
            elif current_angle < angle_sharp_curve:
                reduced_speed = speed_shallow_curve
            else:
                reduced_speed = speed_sharp_curve

            # Apply speed reduction to the current, backward and forward waypoints
            start_idx = max(0, i - lookahead_backward)
            end_idx = min(i + lookahead_forward, len(speeds) - 1)
            for j in range(start_idx, end_idx + 1):
                speeds[j] = min(speeds[j], reduced_speed)

        # Update the waypoints speeds
        self.waypoints[:, 2] = speeds





        