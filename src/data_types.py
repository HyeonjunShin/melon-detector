from dataclasses import dataclass
import numpy as np
from tracker import MelonPosFilter
import time
import math

@dataclass
class Frame:
    image: np.uint8
    depth : np.uint16
    timestamp_image: int

class Melon:
    id : int
    cx_entry : float
    cy_entry : float
    tx_entry : float
    ty_entry : float
    timestamp_entry : float

    angle : float
    velocity : float
    melon_pos_filter : MelonPosFilter

    def __init__(self) -> None:
        self.cx_entry = 0.0
        self.cy_entry = 0.0
        self.tx_entry = 0.0
        self.ty_entry = 0.0
        self.timestamp_entry = 0.0

        self.cx_last = 0.0
        self.cy_last = 0.0
        self.tx_last = 0.0
        self.ty_last = 0.0
        self.timestamp_last = 0.0
        self.timestamp_comp = 0.0

        self.angle = 0.0
        self.velocity = 0.0
        self.z = 0.0
        self.melon_pos_filter = MelonPosFilter()

    def setEntry(self, cx_entry, cy_entry, tx_entry, ty_entry, timestamp_entry):
        self.cx_entry = cx_entry
        self.cy_entry = cy_entry
        self.tx_entry = tx_entry
        self.ty_entry = ty_entry
        self.timestamp_entry = timestamp_entry

        self.melon_pos_filter.update(cx_entry, cy_entry, tx_entry, ty_entry, timestamp_entry)

    def setLast(self, cx_last, cy_last, tx_last, ty_last, timestamp_last):
        self.cx_last = cx_last
        self.cy_last = cy_last
        self.tx_last = tx_last
        self.ty_last = ty_last
        self.timestamp_last = timestamp_last
        self.timestamp_comp = time.time()

        self.melon_pos_filter.update(cx_last, cy_last, tx_last, ty_last, timestamp_last)

    def setZ(self, z):
        self.z = z

    def updata(self, cx, cy, tx, ty, timestamp):
        self.melon_pos_filter.update(cx, cy, tx, ty, timestamp)
        if self.timestamp_entry == 0.0:
            self.timestamp_entry = timestamp
    
    def calcVelocity(self):
        total_time_sec = (self.timestamp_last - self.timestamp_entry) / 1_000_000.0
        total_dist_m = math.sqrt((self.cx_last - self.cx_entry)**2 + (self.cy_last -  self.cy_entry)**2)

        if total_time_sec > 0:
            self.velocity = total_dist_m / total_time_sec

    def calcAngle(self):
        state_center = self.melon_pos_filter.center.kf.statePost
        state_tail = self.melon_pos_filter.tail.kf.statePost

        cx, cy = state_center[0, 0], state_center[1, 0]
        tx, ty = state_tail[0, 0], state_tail[1, 0]

        dx = tx - cx
        dy = ty - cy
        self.angle = math.degrees(math.atan2(dy, dx))
        if self.angle < 0:
            self.angle += 360
        
    def getPos(self, timestamp = None):
        cx = self.cx_last + self.velocity * (timestamp - self.timestamp_comp) 

        return (cx, self.cy_last, self.angle, self.velocity)

