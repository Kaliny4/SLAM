import numpy as np
from custom_types import FrameID, CameraID

class TrackedCamera():
    def __init__(self, R, t, frame_id: FrameID, frame, camera_id: CameraID = CameraID(-1), fixed = False):
        self.R = R
        self.t = t
        self.frame_id: FrameID = frame_id
        self.camera_id: CameraID = camera_id
        self.frame = frame
        self.fixed = fixed

    def pose(self) -> np.ndarray:
        if self.t.shape == (3, 1):
            # TODO catch this earlier
            self.t = self.t.T[0]
        ret = np.eye(4)
        ret[:3, :3] = self.R
        ret[:3, 3] = self.t
        return ret

    def __repr__(self):
        return repr("Camera %d [%s] %s (%f %f %f) %s" % (self.camera_id,
            self.frame_id, 
            self.fixed,
            self.t[0],
            self.t[1],
            self.t[2],
            self.R))