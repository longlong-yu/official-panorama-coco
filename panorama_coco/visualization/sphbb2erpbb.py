# encoding: utf-8
import numpy as np


class Sphbb2Erpbb(object):
    """
    Draw spherical rectangles on 360 degree images.
    """
    def __init__(self, sphereW, sphereH, view_angle_w=64, view_angle_h=64, long_side=640):
        """
        sphereW, sphereH:the width and height of spherical image in ERP format
        view_angle_w, view_angle_h:FOVs
        long_side:the longest side of ERP
        """
        self.sphereW = sphereW
        self.sphereH = sphereH
        fov_w, fov_h = view_angle_w, view_angle_h
        self._long_side = long_side

        if fov_w >= fov_h:
            self._imgW = long_side
            self._imgH = max(
                1, 
                int(np.tan(fov_h / 360 * np.pi) * self._imgW / float(np.tan(fov_w / 360 * np.pi)))
            )
        else:
            self._imgH = long_side
            self._imgW = max(
                1, 
                int(np.tan(fov_w / 360 * np.pi) * self._imgH / float(np.tan(fov_h / 360 * np.pi)))
            )

        TX, TY = self._meshgrid()
        R, ANGy = self._compute_radius(view_angle_w, TY)

        self._R = R
        self._ANGy = ANGy
        self._Z = TX

    def _meshgrid(self):
        """
        Construct mesh point
        :returns: TX, TY
        """
        if self._imgW >= self._imgH:
            offset = int((self._imgW - self._imgH)/2)
            TX, TY = np.meshgrid(range(self._imgW), range(
                offset, self._imgH + offset))
        else:
            offset = int((self._imgH - self._imgW)/2)
            TX, TY = np.meshgrid(
                range(offset, self._imgW + offset), range(self._imgH))

        TX = TX.astype(np.float64) - 0.5
        TX -= self._long_side/2

        TY = TY.astype(np.float64) - 0.5
        TY -= self._long_side/2
        return TX, TY

    def _compute_radius(self, view_angle, TY):
        '''
        '''
        _view_angle = np.pi * view_angle / 180.
        r = self._imgW/2 / np.tan(_view_angle/2)
        R = np.sqrt(np.power(TY, 2) + r**2)
        ANGy = np.arctan(-TY/r)
        return R, ANGy

    def _sample_points(self,boxes,erp_w,erp_h, border_only=False):
        """
        Sample necessary points.
        x, y: the coordinate of the center point
        """
        erp_boxes_feat=boxes.copy()
        x,y=erp_boxes_feat[0],erp_boxes_feat[1]
        angle_x, angle_y = self._direct_camera(x, y, border_only)
        erp_feat_Px = (angle_x + np.pi) / (2*np.pi) * self.sphereW + 0.5
        erp_feat_Py = (np.pi/2 - angle_y) / np.pi * self.sphereH + 0.5
        INDx = erp_feat_Px < 1
        erp_feat_Px[INDx] += self.sphereW
        erp_boxes_Px, erp_boxes_Py= ro_Shpbbox(boxes[0], boxes[1], erp_feat_Px, erp_feat_Py, boxes[4], erp_w=erp_w, erp_h=erp_h)
        return erp_boxes_Px, erp_boxes_Py

    def _direct_camera(self, rotate_x, rotate_y, border_only=False):
        """
        """
        if border_only:
            angle_y = np.hstack([self._ANGy[0, :], self._ANGy[-1, :],
                                 self._ANGy[:, 0], self._ANGy[:, -1]]) + rotate_y
            Z = np.hstack([self._Z[0, :], self._Z[-1, :],
                           self._Z[:, 0], self._Z[:, -1]])
            R = np.hstack([self._R[0, :], self._R[-1, :],
                           self._R[:, 0], self._R[:, -1]])
        else:
            angle_y = self._ANGy + rotate_y
            Z = self._Z
            R = self._R

        X = np.sin(angle_y) * R
        Y = - np.cos(angle_y) * R

        INDn = np.abs(angle_y) > np.pi/2

        angle_x = np.arctan(Z / -Y)
        RZY = np.linalg.norm(np.stack((Y, Z), axis=0), axis=0)
        angle_y = np.arctan(X / RZY)

        angle_x[INDn] += np.pi
        angle_x += rotate_x

        INDy = angle_y < -np.pi/2
        angle_y[INDy] = -np.pi - angle_y[INDy]
        angle_x[INDy] = angle_x[INDy] + np.pi

        INDx = angle_x <= -np.pi
        angle_x[INDx] += 2*np.pi
        INDx = angle_x > np.pi
        angle_x[INDx] -= 2*np.pi
        return angle_x, angle_y

class Tools(object):
    def __init__(self, erp_w=1024, erp_h=512):
        self.erp_w = erp_w
        self.erp_h = erp_h

    def pxpy_to_xyz(self, p):
        '''
        erp2sph
        '''
        theta, phi = self.pxpy_to_theta_phi(p[0],p[1])
        xyz = self.theta_phi_to_xyz(theta, phi)
        return xyz

    def xyz_to_pxpy(self, xyz):
        '''
        sph2erp
        '''
        theta, phi = self.xyz_to_theta_phi(xyz)
        px, py = self.theta_phi_to_px_py(theta, phi)
        return [px, py]

    def pxpy_to_theta_phi(self, px, py):
        theta = px / self.erp_w * (2 * np.pi) - np.pi
        phi = -py / self.erp_h * np.pi + np.pi / 2
        return theta, phi

    def theta_phi_to_px_py(self, theta, phi):
        px = (theta + np.pi) / (2 * np.pi) * self.erp_w
        py = -((phi + np.pi / 2) / np.pi * self.erp_h) + self.erp_h
        return px, py

    def theta_phi_to_xyz(self, theta, phi):
        sph_r = 1
        x_3d = sph_r * np.cos(phi) * np.sin(theta)
        y_3d = sph_r * np.sin(phi)
        z_3d = sph_r * np.cos(phi) * np.cos(theta)

        return np.array([x_3d, y_3d, z_3d])

    def xyz_to_theta_phi(self, xyz):
        theta = np.arctan2(xyz[0], xyz[2])
        phi = np.arctan2(xyz[1], np.sqrt(xyz[0] ** 2 + xyz[2] ** 2))
        return theta, phi

    def roll_T(self, n, xyz, gamma=0):
        '''
        '''
        n11 = (n[0] ** 2) * (1 - np.cos(gamma)) + np.cos(gamma)
        n12 = n[0] * n[1] * (1 - np.cos(gamma)) - n[2] * np.sin(gamma)
        n13 = n[0] * n[2] * (1 - np.cos(gamma)) + n[1] * np.sin(gamma)

        n21 = n[0] * n[1] * (1 - np.cos(gamma)) + n[2] * np.sin(gamma)
        n22 = (n[1] ** 2) * (1 - np.cos(gamma)) + np.cos(gamma)
        n23 = n[1] * n[2] * (1 - np.cos(gamma)) - n[0] * np.sin(gamma)

        n31 = n[0] * n[2] * (1 - np.cos(gamma)) - n[1] * np.sin(gamma)
        n32 = n[1] * n[2] * (1 - np.cos(gamma)) + n[0] * np.sin(gamma)
        n33 = (n[2] ** 2) * (1 - np.cos(gamma)) + np.cos(gamma)

        x, y, z = xyz[0], xyz[1], xyz[2]
        xx = n11 * x + n12 * y + n13 * z
        yy = n21 * x + n22 * y + n23 * z
        zz = n31 * x + n32 * y + n33 * z

        return [xx, yy, zz]

def roBbox(center, p, ang, erp_w, erp_h):
    t = Tools(erp_w, erp_h)
    cx, cy = t.theta_phi_to_px_py(center[0], center[1])
    c_xyz = t.pxpy_to_xyz([cx, cy])
    p_xyz = t.pxpy_to_xyz(p)
    pp_xyz = t.roll_T(c_xyz, p_xyz, ang)
    pp = t.xyz_to_pxpy(pp_xyz)
    return pp

def ro_Shpbbox(theta, phi, Px, Py, ang, erp_w=1920, erp_h=960):
    px = Px.copy()
    py = Py.copy()
    for i in range(len(Px)):
        p = roBbox([theta, phi], [Px[i], Py[i]], ang, erp_w, erp_h)
        px[i], py[i] = p[0], p[1]
    return px, py
