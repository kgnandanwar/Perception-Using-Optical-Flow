#<=============================== Imports ===================================>#
import cv2
import numpy as np
from os import system
from video_handler import *
import matplotlib.pyplot as plt
from pandas import DataFrame
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# =========================== OpticalFlow Class =========================== #
class OpticalFlow:
    
    """
    Optical Flow Class Declaration
    
    args: Camera paremeters dict, left image, right image
    
    The Optical Flow class
    """
    
    def __init__(self, mesh_res=25, debug=False) -> None:
        self.prev_frame = None
        self.mesh_res = mesh_res

    def get_optical_flow(self, frame, args):
        # If it is the first iteration...
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Init mesh grid for full frame size
            self.hsv = np.zeros_like(frame)
            self.hsv[...,1] = 255
            
            h, w = self.prev_frame.shape[:2]
            x = np.arange(0, w, self.mesh_res)
            y = np.arange(0, h, self.mesh_res)
            self.X, self.Y = np.meshgrid(x, y)
            return None
        
        # Convert to gray, get optical flow
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(self.prev_frame, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Get vector field
        fig = plt.figure()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame_bgr)
        plt.quiver(self.X, self.Y, flow[self.Y, self.X, 0], -flow[self.Y, self.X, 1], scale=100)
        canvas = FigureCanvas(fig)
        canvas.draw()
        plt.close()
        mat = np.array(canvas.renderer._renderer)[105:-105, 80:-65, :]
        mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
        mat = cv2.resize(mat, (frame.shape[1], frame.shape[0]), interpolation = cv2.INTER_AREA)
        
        # Convert to polar, degrees, normalize magnitudes
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        self.hsv[..., 0] = ang * 180 / np.pi / 2
        self.hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # Mask img
        bgr_frame = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask_eroded = cv2.morphologyEx(mask, cv2.MORPH_ERODE, np.ones((11,11), np.uint8))
        cars = cv2.bitwise_and(frame, frame, mask=mask_eroded)
        self.prev_frame = frame_gray

        both = np.vstack((mat, cars))
        cv2.imshow("both", both)

        return both

#<=============================== Main ===================================>#
if __name__ == "__main__":
    system('cls')
    
    video_name = "Cars On Highway.mp4"
    handle = VideoHandler(video_name, "vector_field_cars", 40)
    opt_flow = OpticalFlow(debug=True)        
    
    handle.run(opt_flow.get_optical_flow, 0)