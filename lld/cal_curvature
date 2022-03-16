import cv2

def cal_curvature(hull):
    dx_dt = np.gradient(hull[:, 0])
    dy_dt = np.gradient(hull[:, 1])

    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5

    return curvature
