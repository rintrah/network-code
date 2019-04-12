import cv2
import numpy as np

def density_image(pts, shape, sigma):
    padding = int((sigma - .8) / .3 + 1)
    
    # Filter out points which are too far away
    mask = (-padding <= pts[:, 1]) & (pts[:, 1] < shape[0] + padding) \
         & (-padding <= pts[:, 0]) & (pts[:, 0] < shape[1] + padding)
    pts_x, pts_y = pts.T[:, mask, None, None]
    yg, xg = np.ogrid[: shape[0], : shape[1]]
    peaks = np.exp(-(  (pts_y - yg) ** 2 \
                     + (pts_x - xg) ** 2) \
                   / (2 * sigma ** 2)) \
            / (2 * np.pi * sigma ** 2) 
    
    # Compute density
    density = peaks.sum(axis=0)
    return density

