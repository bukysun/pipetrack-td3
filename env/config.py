import math
import numpy as np

# The center coordnates of straight pipe
init_pos_set = [(0.0, 0.0, 1.57), (0.0, 2.0, 1.57), (0.0, 4.0, 1.57),
                (0.0, 6.0, 1.57), (0.0, -2.0, 1.57), (-1.28, 7.28, 0.0), 
                (-3.28, 7.28, 0.0), (-5.28, 7.28, 0.0), (-7.28, 7.28, 0.0), (-9.28, 7.28, 0.0),
                (-10.56, 6.0, 1.57), (-10.56, 4.0, 1.57), (-10.56, 2.0, 1.57),
                (-10.56, 0.0, 1.57), (-10.56, -2.0, 1.57), (-9.28, -3.28, 0.0), 
                (-7.28, -3.28, 0.0), (-5.28, -3.28, 0.0), (-3.28, -3.28, 0.0),
                (-1.28, -3.28, 0.0)]

# The center coordnates of turn pipe
turn_center_set = [(-0.28, 7.0, 0.0, 1.57), (-10.28, 7.0, 1.57, 3.14), (-10.28, -3.0, 3.14, 4.71), (-0.28, -3.0, 4.71, 6.28)]
             
class Tile(object):
    """This is a tile class for dividing experiment area and board."""
    def __init__(self, kind, coord):
        self.kind = kind
        self.coord = coord
 
    def is_in(self, position):
        x1, y1 = position
        if len(self.coord) == 3:
            x0, y0, theta0 = self.coord
        else:
            x0, y0, stheta0, etheta0 = self.coord
        pipe_len, pipe_wid = 1.0, 0.5

        if self.kind == "straight":
            if abs(theta0) < 1e-5:
                if abs(x1-x0) <= pipe_len and abs(y1-y0) <= pipe_wid:
                    return True
            else:
                if abs(x1-x0) <= pipe_wid and abs(y1-y0) <= pipe_len:
                    return True
        elif self.kind == "turn":
            radius = 0.28
            dvec = [x1-x0, y1-y0]
            ang = math.atan2(dvec[1], dvec[0])
            if ang < 0: ang += 2 * np.pi 
            dist = np.linalg.norm(dvec)
            if dist < radius + pipe_wid and ang >=stheta0 and ang < etheta0:
                return True
        return False

    def get_pos_lane(self, pos, ang1):
        x1, y1 = pos
        if len(self.coord) == 3:
            x0, y0, theta0 = self.coord
        else:
            x0, y0, stheta0, etheta0 = self.coord
        
        if self.kind == "straight":
            if abs(theta0) < 1e-5:
                dist = abs(y1-y0)
            else:
                dist = abs(x1-x0)    
            ang = abs(theta0 - ang1)
            ang = min(ang, abs(np.pi-ang))
            return dist, ang
        elif self.kind == "turn":
            radius = 0.28
            dvec = [x1-x0, y1-y0]
            ang = math.atan2(dvec[1], dvec[0]) + np.pi/2
            ang = ang - ang1
            ang = abs(math.atan2(math.sin(ang), math.cos(ang)))
            ang = min(ang, abs(np.pi-ang))
            dist = abs(np.linalg.norm(dvec) - radius)
            return dist, ang 
        else:
            raise NotImplementedError
            

straight_tiles = [Tile("straight", p) for p in init_pos_set]
circle_tiles = [Tile("turn", p) for p in turn_center_set]
map_tiles = straight_tiles + circle_tiles 
