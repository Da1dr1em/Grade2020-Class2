import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize
import pandas as pd
from openpyxl import load_workbook


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
    theta_tail, _, _, message = optimize.fsolve(distance_diff, theta_tail_initial_guess, xtol=tolerance,
                                                full_output=True)
    # print(theta_tail)
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
        raise ValueError("The found θ_tail is not greater than θ_head.")

    return theta_tail, r_tail


def angtoxy(theta, r):
    return [r * np.cos(theta), r * np.sin(theta)]


# 定义了一个类，它能计算两个旋转矩形是否碰撞

class RotatedRectangle:
    def __init__(self, center, angle, width, height):
        self.center = center  # 中心坐标 (x, y)
        self.angle = angle  # 旋转角度（弧度）
        self.width = width  # 宽度
        self.height = height  # 高度

        # 预先计算矩形的顶点
        self.update_vertices()

    def update_vertices(self):
        # 定义矩形的轴
        axis_x = [np.cos(self.angle), np.sin(self.angle)]
        axis_y = [-np.sin(self.angle), np.cos(self.angle)]

        # 计算矩形的四个顶点
        self.vertices = [
            [self.center[0] + self.width / 2 * axis_x[0] + self.height / 2 * axis_y[0],
             self.center[1] + self.width / 2 * axis_x[1] + self.height / 2 * axis_y[1]],
            [self.center[0] - self.width / 2 * axis_x[0] + self.height / 2 * axis_y[0],
             self.center[1] - self.width / 2 * axis_x[1] + self.height / 2 * axis_y[1]],
            [self.center[0] - self.width / 2 * axis_x[0] - self.height / 2 * axis_y[0],
             self.center[1] - self.width / 2 * axis_x[1] - self.height / 2 * axis_y[1]],
            [self.center[0] + self.width / 2 * axis_x[0] - self.height / 2 * axis_y[0],
             self.center[1] + self.width / 2 * axis_x[1] - self.height / 2 * axis_y[1]]
        ]

    def dot(self, v, w):
        return v[0] * w[0] + v[1] * w[1]

    def project(self, a, b):
        return (self.dot(a, b) / self.dot(b, b)) * b

    def on_segment(self, p, q, r):
        return q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and \
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])

    def orientation(self, p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # collinear
        elif val > 0:
            return 1  # clock
        else:
            return 2  # counterclock

    def do_intersect(self, p1, q1, p2, q2):
        o1 = self.orientation(p1, q1, p2)
        o2 = self.orientation(p1, q1, q2)
        o3 = self.orientation(p2, q2, p1)
        o4 = self.orientation(p2, q2, q1)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and self.on_segment(p1, p2, q1):
            return True
        if o2 == 0 and self.on_segment(p1, q2, q1):
            return True
        if o3 == 0 and self.on_segment(p2, p1, q2):
            return True
        if o4 == 0 and self.on_segment(p2, q1, q2):
            return True

        return False

    def collides_with(self, other):
        # 检查边-边碰撞
        for i in range(4):
            for j in range(4):
                if self.do_intersect(self.vertices[i], self.vertices[(i + 1) % 4],
                                     other.vertices[j], other.vertices[(j + 1) % 4]):
                    return True

        return False

def velocitycal(velupper,theta,thetaplus1):   # 注意plus1意味着向龙尾靠近1
    xiplus1,yiplus1 = angtoxy(thetaplus1,rot(thetaplus1,b))
    xi,yi = angtoxy(theta,rot(theta,b))
    ki = ((xiplus1-xi)/(np.sqrt(np.square(xiplus1-xi)+np.square(yiplus1-yi))),(yiplus1-yi)/(np.sqrt(np.square(xiplus1-xi)+np.square(yiplus1-yi))))
    viplus1 = ((np.cos(thetaplus1)-thetaplus1*np.sin(thetaplus1))/np.sqrt(np.square(np.cos(thetaplus1)-thetaplus1*np.sin(thetaplus1))+np.square(np.sin(thetaplus1)+thetaplus1*np.cos(thetaplus1))),(np.sin(thetaplus1)+thetaplus1*np.cos(thetaplus1))/np.sqrt(np.square(np.cos(thetaplus1)-thetaplus1*np.sin(thetaplus1))+np.square(np.sin(thetaplus1)+thetaplus1*np.cos(thetaplus1))))
    vi = ((np.cos(theta)-theta*np.sin(theta))/np.sqrt(np.square(np.cos(theta)-theta*np.sin(theta))+np.square(np.sin(theta)+theta*np.cos(theta))),(np.sin(theta)+theta*np.cos(theta))/np.sqrt(np.square(np.cos(theta)-theta*np.sin(theta))+np.square(np.sin(theta)+theta*np.cos(theta))))

    resultup = 0
    resultdown = 0
    for i in range(2):
        resultup += ki[i]*vi[i]
        resultdown += ki[i]*viplus1[i]

    return abs(resultup*velupper/resultdown)

if __name__ == '__main__':
    p = 0.55
    pi = np.pi
    b = p / (2 * pi)  # 修改b的计算
    theta = np.linspace(0, 32 * pi, 2000)  # 生成theta的值



    upperangle = 0
    headtheta = []
    search_initial = 0  # 初始猜测

    # 定义特定点xy变量名
    headx, heady, bodyx, bodyy = 0, 0, 0, 0
    # 下面这俩是用来求速度的
    bodyx1 = []
    bodyy1 = []

    bodytheta1 = []
    bodyr1 = []
    bodytheta2 = []
    bodyr2 = []
    #  下面是保存数据的部分
    veloall = []
    savepath = 'C:\\Users\\31827\\Downloads\\result1.xlsx'
    book = load_workbook(savepath)
    writer = pd.ExcelWriter(savepath, engine='openpyxl')
    writer._book = book

    timemark = 0
    timeinterval = 0.7
    # 碰撞时刻：412.473838
    for i in range(301):  # 计算300个点
        timemark = i
        dt = np.linspace(timemark - timeinterval, timemark + timeinterval, 2)
        tail = []
        bodytheta1 = []
        bodyr1 = []
        bodytheta2 = []
        bodyr2 = []
        ans, _, _, message = optimize.fsolve(Si, search_initial, args=(i, 32 * pi, b, curvedif), full_output=True,xtol=1e-8)
        if message == 'The solution converged.':
            headtheta.append(ans)
            search_initial = ans  # 更新初始猜测为当前解
            upperangle = ans
            headx, heady = angtoxy(ans, rot(ans, b))
            #tail.append(ans)
            print("龙首坐标:", [headx, heady])
        else:
            print(f"Warning: fsolve did not, converge for i={i}")
        #对dt求速度
        for j in range(1, 225):
            if j == 1:
                L = 2.86
                tailtheta, _ = find_point_on_spiral(upperangle, rot(ans, b), L, b)
                tail.append(tailtheta)
            else:
                L = 1.65
                tailtheta, _ = find_point_on_spiral(upperangle, rot(upperangle, b), L, b)
                tail.append(tailtheta)
            upperangle = tailtheta

        print(len(tail))
        velocity = []
        velocity.append(1)
        velotem = 0
        inherit = 0
        for j in range(0,224):
            if j == 0:
                velotem = velocitycal(1,ans,tail[j])
            else:
                velotem = velocitycal(inherit,tail[j-1],tail[j])
            inherit = velotem
            velotem = np.format_float_scientific(velotem,precision=6)
            velocity.append(velotem)
        veloall.append(velocity)
    print('lens=',len(veloall))

    savedict = {f"{l} s":veloall[l] for l in range(301)}
    df = pd.DataFrame(savedict)
    print(df)
    df.to_excel(writer, sheet_name='速度', startcol=1, header=1,index=False)
    writer._save()
    writer.close()




