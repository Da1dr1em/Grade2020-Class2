import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize
import pandas as pd
from pandas import DataFrame
from openpyxl import load_workbook
import math
# 这是计算路程里的被积函数
def curvedif(theta):
    return (1 + theta ** 2) ** 0.5


# 这是计算龙头路程的函数，采用积分的方法
def S(theta0, theta1, b, fun):
    result, _ = integrate.quad(fun, theta0, theta1)
    return b * result


# 这是等距螺线的极坐标方程
def rot(theta, b):
    return b * theta


# 定义用于求解的函数
def Si(theta, i, theta1, b, fun):
    return S(theta, theta1, b, fun) - i

# 定义把手和把手距离函数并求解
def find_point_on_spiral(theta_head, r_head, L, b, tolerance=1e-10):
    """
    Find the polar coordinates of a point on the spiral r = b * theta that is a distance L away from the head.

    Parameters:
    - theta_head: float, the polar angle of the head.
    - r_head: float, the radial distance of the head (should be equal to b * theta_head if on the spiral).
    - L: float, the linear distance from the head to the desired point.
    - b: float, the parameter of the spiral r = b * theta.
    - tolerance: float, the tolerance for the convergence of fsolve (default is 1e-6).

    Returns:
    - theta_tail: float, the polar angle of the found point.
    - r_tail: float, the radial distance of the found point.
    """

    # Define a function that returns the difference between the actual distance and the target distance L
    def distance_diff(theta_tail):
        r_tail = b * theta_tail
        distance = np.sqrt(r_head ** 2 + r_tail ** 2 - 2 * r_head * r_tail * np.cos((theta_head - theta_tail)))
        return distance - L

        # Use fsolve to find the theta_tail that satisfies the distance condition

    theta_tail_initial_guess = theta_head  # Initial guess (may need adjustment based on the spiral)
    theta_tail, _, _, message = optimize.fsolve(distance_diff, theta_tail_initial_guess, xtol=tolerance, full_output=True)
    #print(theta_tail)
    # Check if fsolve converged successfully
    if message != 'The solution converged.':
        plt.show()
        raise ValueError("fsolve did not converge to a solution within the given tolerance.")

        # Calculate the corresponding r_tail value
    r_tail = b * theta_tail

    # Ensure theta_tail is in the expected range (theta_head < theta_tail)
    # Note: We do not check for theta_tail < 2*pi here because the spiral can extend beyond 2*pi
    #       If you want to restrict the solution to a single revolution, you can add this check.
    if not (theta_head < theta_tail):
        plt.show()
        raise ValueError("The found θ_tail is not greater than θ_head.")

    return theta_tail, r_tail
def angtoxy(theta,r):
    return [r*np.cos(theta),r*np.sin(theta)]

#定义了一个类，它能计算两个旋转矩形是否碰撞


class OBB:
    def __init__(self, center, angle,width, height):
        self.center = np.array(center)
        self.width = width
        self.height = height
        self.angle = angle
        self.half_width = width / 2
        self.half_height = height / 2
        self.orientation = np.array([np.cos(angle), np.sin(angle)])
        self.corners = self._get_corners()

    def _get_corners(self):
        cos_angle = np.cos(self.angle)
        sin_angle = np.sin(self.angle)
        dx = self.half_width * cos_angle
        dy = self.half_width * sin_angle
        hx = self.half_height * sin_angle
        hy = self.half_height * cos_angle

        corners = [
            self.center + np.array([dx - hx, dy + hy]),
            self.center + np.array([dx + hx, dy - hy]),
            #self.center + np.array([-dx - hx, -dy + hy]),
            self.center + np.array([-dx + hx, -dy - hy]),
            self.center + np.array([-dx - hx, -dy + hy])
        ]
        return np.array(corners)

    def _get_axes(self):
        edges = [self.corners[i] - self.corners[i - 1] for i in range(4)]
        return [np.array([-edge[1], edge[0]]) / np.linalg.norm(edge) for edge in edges]

    def _project_onto_axis(self, axis):
        axis = axis.flatten()
        projections = [np.dot(corner.flatten(), axis) for corner in self.corners]
        return [min(projections), max(projections)]

    def _overlap(self, projection1, projection2):
        return projection1[0] <= projection2[1] and projection2[0] <= projection1[1]

    def is_colliding(self, other):
        axes = self._get_axes() + other._get_axes()
        for axis in axes:
            if not self._overlap(self._project_onto_axis(axis), other._project_onto_axis(axis)):
                return False
        return True




if __name__ == '__main__':
    p = 0.55
    pi = np.pi
    b = p / (2 * pi)  # 修改b的计算
    theta = np.linspace(0, 32 * pi, 2000)  # 生成theta的值


    upperangle = 0
    headtheta = []
    search_initial = 0  # 初始猜测

    # 下面这些是为计算矩形位置的
    handtheta = []#记录每个把手的theta角度
    rect = [] #记录每个矩形的中心点位置
    rectangle = [] #记录每个矩形的旋转角度

    #定义特定点xy变量名

    # 下面这俩是用来求速度的
    bodyx = []
    bodyy = []

    b = 0.0662829
    while b > 0:
        iscounter = False
        iscollision = False
        isfind = False
        f = 412
        for i in range(f*10):  # 计算300个点
            # 使用fsolve找到满足条件的theta
            i = i/10
            tail = []
            ans, _, _, message = optimize.fsolve(Si, search_initial, args=(i, 32 * pi, b, curvedif), full_output=True)
            if message == 'The solution converged.':

                search_initial = ans  # 更新初始猜测为当前解
                upperangle = ans
                headx, heady = angtoxy(ans, rot(ans, b))
                # print("龙首坐标:", [headx, heady])
            else:
                print(f"Warning: fsolve did not, converge for i={i}")
            # ax.plot(ans,rot(ans,b),color='red')
            handtheta = [ans]
            count = 0
            numberinrow = 0
            if abs(rot(ans, b) - 4.5) < 1e-6 or rot(ans, b) < 4.5:
                iscounter = True

            for j in range(1, 225):

                if j == 1:
                    L = 2.86
                    tailtheta, _ = find_point_on_spiral(upperangle, rot(ans, b), L, b)
                else:
                    L = 1.65
                    tailtheta, _ = find_point_on_spiral(upperangle, rot(upperangle, b), L, b)
                    tail.append(tailtheta)

                upperangle = tailtheta
                handtheta.append(tailtheta)
                count = count + 1
                if tailtheta - ans < np.pi * 3:
                    numberinrow += 1
                # print(count)

            # print(numberinrow)
            # print(handtheta)
            # 接下来处理每个矩形的坐标、位置
            # 由于第i个矩形由第i和第i+1个把手的位置所确定，因此需要一些计算
            # 把手的位置来源于以上的head和tail
            # handtheta是每个把手的theta角度
            # 由于每个矩形的位置是由两个把手的位置所确定的，因此需要一些计算
            rect = []
            rectangle = []
            for j in range(len(handtheta) - 1):
                # 计算矩形的中心点位置
                # print(j)
                midposupper = angtoxy(handtheta[j], rot(handtheta[j], b))
                midposlater = angtoxy(handtheta[j + 1], rot(handtheta[j + 1], b))

                center_x = (midposupper[0] + midposlater[0]) / 2
                center_y = (midposupper[1] + midposlater[1]) / 2
                rect.append((center_x, center_y))

                # 计算矩形的旋转角度(弧度制？)
                turningangle = np.arctan2(midposlater[1] - midposupper[1], midposlater[0] - midposupper[0])
                rectangle.append(turningangle)
                # print((midposupper[0]+midposlater[0])/2)
            handtheta.clear()
            # 现在记录了每个矩形的位置和旋转角度
            # 接下来就是对是否碰撞进行检测
            # 由于之前已验证300s前不会碰撞，因此从300s后再开始检测
            # print(len(rect))

            if iscounter:
                for j in range(0, 3):
                    if j == 0:
                        rect1 = OBB(rect[j], rectangle[j], 3.41, 0.3)
                    else:
                        rect1 = OBB(rect[j], rectangle[j], 2.20, 0.3)

                    for k in range(j + 2, len(rect)):

                        if k == 0:
                            rect2 = OBB(rect[k], rectangle[k], 3.41, 0.3)
                        else:
                            rect2 = OBB(rect[k], rectangle[k], 2.20, 0.3)
                        if rect1.is_colliding(rect2):
                            print(f"Warning: Collision detected at i(s)={i}, j(head)={j}, k={k}")
                            print("The R = ", rot(ans, b))
                            iscollision = True
                            break
                    if iscollision == True:
                        break
                break
        if not(iscounter == True and iscollision == False):
            isfind = True

        # 0.001精度发现b = 0.06653521870054242
        # 0.0001 b = 0.06579999999999997
        # 0.00001 b= 0.06586000000000002
        # 0.000001 b = 0.06585499999999997
        # 0.0000001 b = 0.0679895
        print("b =", b)
        if isfind == True:
            print("临界点发现！")
            btrue = b + 0.0000001
            print("此时b=", btrue)
            output = np.pi * 2 * btrue
            print("螺距为",output)
            break
        b -= 0.0000001
