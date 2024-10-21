import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
xx=np.zeros(401)
yy =np.zeros(401)
#x_plt = np.zeros(601)
# 定义系统参数
Xu, Yv, Zw = -62.5, -62.5, -62.5  # 阻尼系数
Kp, Mvq, Nr = -30, -30, -30  # 刚度系数

# 二阶阻尼系数
Xuu, Yvv, Zww = -48, -48, -48
Kpp, Mvqq, Nrr = -80, -80, -80
m = 5454.54  # 总质量
a1, a2, a3, a4, a5, a6 = 0.15, 0.61, 0.61, 0.15, 0.3, 0.15  # 力臂
m1, m2, m3, m4, m5, m6 = 20, 80, 80, 20, 40, 20  # 各个物体的质量

Ix, Iy, Iz = 2038, 13587, 13587  # 惯性矩
Ixy, Iyz, Ixz = -13.58, -13.58, -13.58
rBB = np.zeros((3, 1))
rGB = np.array([[0], [0], [0.061]])

# 定义变量 W 和 B
W = 53400
B = 53400

# 定义矩阵 xita1
xita1 = np.array([[10, 26, 11],
                  [-12, -13, -5],
                  [0, 0, 0]])
xita2=np.zeros((3,3))
a = [6.019 * 10**3, 9.551 * 10**3, 2.332 * 10**4, 4.129 * 10**3, 4.913 * 10**4, 2.069 * 10**4]

# 创建对角矩阵 M
M = np.diag(a)

# 假设 Xu, Yv, Zw, Kp, Mvq, Nr 的值已经定义
# 例如：
Xu, Yv, Zw, Kp, Mvq, Nr = 1, 2, 3, 4, 5, 6  # 请根据您的需求进行修改

# 定义向量 E1 和 f1
E1 = np.array([Xu, Yv, Zw, Kp, Mvq, Nr])
f1 = np.array([Xuu, Yvv, Zww, Kpp, Mvqq, Nrr])
D = -np.diag(E1) - np.diag(f1)
fG1 = np.array([0, 0, W])
fB1 = np.array([0, 0, B])

# 计算 g 向量
g = np.zeros((6,1))
tao10 = np.zeros((6,1))
tao20 = np.zeros((6,1))
tao30 = np.zeros((6,1))
ts = 0.1

# 初始化速度向量
v1 = np.zeros((6,1))
v2 = np.zeros((6,1))
v3 = np.zeros((6,1))

# 切分速度向量
v11 = v1[:3]  # 选择 v1 的前 3 个元素
v21 = v1[3:]  # 选择 v1 的后 3 个元素
v12 = v2[:3]  # 选择 v2 的前 3 个元素
v22 = v2[3:]  # 选择 v2 的后 3 个元素
v13 = v3[:3]  # 选择 v3 的前 3 个元素
v23 = v3[3:]  # 选择 v3 的后 3 个元素
ke11 = 0.4

ke21 = 1
kev1 = 2000
ke12 = 0.5
ke22 = 1
kev2 = 2000

ke13 = 0.5
ke23 = 1
kev3 = 2000
end_time = 40
# xita1d=[]
# xita2d=[]
# dxita1d=[]
# dxita2d=[]
ttt = 0
# 生成时间序列
time_series = np.arange(0, end_time + ts, ts)
for t in time_series:
    xita1d = np.array([
        [100 * np.sin(0.1 * t + 0.5) + 10],  # First component
        [100 * np.cos(0.1 * t + 0.5) + 10],  # Second component
        [np.zeros_like(t)]  # Third component
    ])

    # xita2d
    xita2d = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0.03 * t - np.pi, (0.01 / 3) * np.pi * t - (10 * np.pi / 6), -np.pi / 4 * 10]
    ]) * 0.1

    # dxita1d
    dxita1d = np.array([
       [10 * np.cos(0.1 * t + 0.5)],  # First component derivative
        [-10 * np.sin(0.1 * t + 0.5)],  # Second component derivative
       [np.zeros_like(t)]  # Third component derivative
   ])

    # dxita2d
    dxita2d = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0.03 * np.ones_like(t), (0.01 / 3) * np.pi * np.ones_like(t), np.zeros_like(t)]
    ]) * 0.1
    d = np.array([[10, 10, 0], [10, -10, 0], [0, 0, 0]]).T * 0
    for j in range(3):  # Loop over rows
        for k in range(3):  # Loop over columns
            while xita2[j, k] > np.pi:
                xita2[j, k] -= 2 * np.pi
            while xita2[j, k] <= -np.pi:
                xita2[j, k] += 2 * np.pi

    # Replace specific values with 0.01 * pi
    for k in range(3):  # Loop over columns of the second row
        if xita2[1, k] == 0.5 * np.pi:
            xita2[1, k] = 0.01 * np.pi
        if xita2[1, k] == -0.5 * np.pi:
            xita2[1, k] = 0.01 * np.pi


    def rotation_matrix(angle):
        return np.array([[np.cos(angle), -np.sin(angle), 0],
                                 [np.sin(angle), np.cos(angle), 0],
                                 [0, 0, 1]])


            # Create the rotation matrices based on the values in xita2
    RBI1 = rotation_matrix(xita2[2, 0])  # corresponds to xita2(3,1)
    RBI2 = rotation_matrix(xita2[2, 1])  # corresponds to xita2(3,2)
    RBI3 = rotation_matrix(xita2[2, 2])


    T1 = np.array([[1, np.sin(xita2[0, 0]) * np.tan(xita2[1, 0]), np.cos(xita2[0, 0]) * np.tan(xita2[1, 0])],
                   [0, np.cos(xita2[0, 0]), -np.sin(xita2[0, 0])],
                   [0, np.sin(xita2[0, 0]) / np.cos(xita2[1, 0]), np.cos(xita2[0, 0]) / np.cos(xita2[1, 0])]])

    T2 = np.array([[1, np.sin(xita2[0, 1]) * np.tan(xita2[1, 1]), np.cos(xita2[0, 1]) * np.tan(xita2[1, 1])],
                   [0, np.cos(xita2[0, 1]), -np.sin(xita2[0, 1])],
                   [0, np.sin(xita2[0, 1]) / np.cos(xita2[1, 1]), np.cos(xita2[0, 1]) / np.cos(xita2[1, 1])]])

    T3 = np.array([[1, np.sin(xita2[0, 2]) * np.tan(xita2[1, 2]), np.cos(xita2[0, 2]) * np.tan(xita2[1, 2])],
                   [0, np.cos(xita2[0, 2]), -np.sin(xita2[0, 2])],
                   [0, np.sin(xita2[0, 2]) / np.cos(xita2[1, 2]), np.cos(xita2[0, 2]) / np.cos(xita2[1, 2])]])

    # Calculate transformation matrices t1, t2, t3
    t1 = np.array([[1, 0, -np.sin(xita2[1, 0])],
                   [0, np.cos(xita2[0, 0]), np.cos(xita2[1, 0]) * np.sin(xita2[0, 0])],
                   [0, -np.sin(xita2[0, 0]), np.cos(xita2[1, 0]) * np.cos(xita2[0, 0])]])

    t2 = np.array([[1, 0, -np.sin(xita2[1, 1])],
                   [0, np.cos(xita2[0, 1]), np.cos(xita2[1, 1]) * np.sin(xita2[0, 1])],
                   [0, -np.sin(xita2[0, 1]), np.cos(xita2[1, 1]) * np.cos(xita2[0, 1])]])

    t3 = np.array([[1, 0, -np.sin(xita2[1, 2])],
                   [0, np.cos(xita2[0, 2]), np.cos(xita2[1, 2]) * np.sin(xita2[0, 2])],
                   [0, -np.sin(xita2[0, 2]), np.cos(xita2[1, 2]) * np.cos(xita2[0, 2])]])
    e11 = [[xita1[0, 0]],[xita1[1, 0]],[xita1[2, 0]]] - xita1d
    e12 = [[xita1[0, 1]],[xita1[1, 1]],[xita1[2, 1]]] - xita1d
    e13 = [[xita1[0, 2]],[xita1[1, 2]],[xita1[2, 2]]]- xita1d

            # Calculate the differences for xita2
    e21 = (xita2[0:3, 0] - xita2d[0:3, 0])

    e22 = xita2[0:3, 1] - xita2d[0:3, 1]
    e23 = xita2[0:3, 2] - xita2d[0:3, 2]


    def normalize_angles(angles):
        """Normalize angles to the range [-pi, pi]."""
        angles = np.mod(angles + np.pi, 2 * np.pi) - np.pi
        return angles


    # Normalize each angle array
    e21 = normalize_angles(e21)
    e22 = normalize_angles(e22)
    e23 = normalize_angles(e23)

    e21=[[e21[0]],[e21[1]],[e21[2]]]
    e22 = [[e22[0]],[e22[1]],[e22[2]]]
    e23 = [[e23[0]],[e23[1]],[e23[2]]]
    u1 = v11[0]
    vv1 = v11[1]
    w1 = v11[2]
    p1 = v21[0]
    q1 = v21[1]
    r1 = v21[2]
    arfav11 = np.linalg.solve(RBI1, -ke11 * e11 + dxita1d)
    ev11 = v11 - arfav11
    delta11 = RBI1 @ev11  # Using @ for matrix multiplication in Python

    arfav21 = -ke21 * t1 @ e21 + t1 @ [[dxita2d[0, 0]],[dxita2d[1, 0]],[dxita2d[2, 0]]]
    ev21 = v21 - arfav21


    # Combine the results into a single array
    ev1 = np.concatenate((ev11, ev21))
    arfav1 = np.concatenate((arfav11, arfav21))

    C1 = np.array([[0, 0, 0, 0, m * w1, -m * vv1],
                   [0, 0, 0, -m * w1, 0, m * u1],
                   [0, 0, 0, m * vv1, -m * u1, 0],
                   [0, -m * w1, m * vv1, 0, -Iyz * q1 - Ixz * p1 + Iz * r1, Iyz * r1 + Ixy * p1 - Iy * q1],
                   [m * w1, 0, -m * u1, Iyz * q1 + Ixz * p1 - Iz * r1, 0, -Ixz * r1 - Ixy * q1 + Ix * p1],
                   [-m * vv1, m * u1, 0, -Iyz - Ixz * p1 + Iy * q1, Ixy * r1 + Iyz * q1 - Ix * p1, 0]],dtype=object)
    darfav1 = -ev1 / ts

# 计算力矩
# 计算矩阵的乘法和转置



    part1=np.concatenate((0* RBI1 @ e11,T1@  e21))
    part2=C1 @ v1
    part3 = D @ v1
    part4 = M @ darfav1
    part5 =kev1 * ev1
    tao1 = part2 + part3 + part4- part1-part5

# 计算力矩的变化率
    dtao1 = (tao1 - tao10) / ts
    tao10 = tao1
    qtao1 = tao1

# 计算加速度，使用 numpy.linalg.solve 代替 MATLAB 的左除运算符
    pppp=qtao1 - C1 @ v1 + D @ v1
    ppp=np.linalg.inv(M)
    dv1 =np.dot(ppp,pppp)

# 更新速度
    v1 = v1 + ts * dv1

# 分离速度分量
    v11 = v1[0:3]  # MATLAB 索引从 1 开始，Python 从 0 开始
    v21 = v1[3:6]

# 计算状态变化
    dxita21 = T1 @ v21
    dxita11 = RBI1@ v11# 假设 RBI1 是一个矩阵
    pppppp=np.zeros((3, 1))
# 更新状态变量
    xita1=xita1.astype(float)

    xita1[0, 0] += dxita11[0, 0] * ts
    xita1[1, 0] += dxita11[1, 0] * ts
    xita1[2, 0] += dxita11[2, 0] * ts
    xx[ttt]=xita1[0, 0]
    yy[ttt]=xita1[1, 0]
    ttt+=1
    xita2[0, 0] += dxita21[0, 0] * ts
    xita2[1, 0] += dxita21[1, 0] * ts
    xita2[2, 0] += dxita21[2, 0] * ts
    print(e11)
    print(t)

ts = 0.1  # 时间步长
t11 = np.arange(0, 40, ts)

# 计算小船的 X 和 Y 坐标
x20 = 100 * np.sin(0.1 * t11 + 0.5) + 10
y20 = 100 * np.cos(0.1 * t11 + 0.5) + 10


# 创建图形和坐标轴
fig, ax = plt.subplots()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('boat move')

# 绘制正弦曲线
ax.plot(x20, y20, 'b', linewidth=2)

# 初始化小船的点
boat, = ax.plot([], [], 'ro', markersize=10, markerfacecolor='r')

# 设置坐标轴的范围
ax.set_xlim(np.min(x20) - 20, np.max(x20) + 20)
ax.set_ylim(np.min(y20) - 20, np.max(y20) + 20)

# 初始化函数，用于创建动画
def init():
    boat.set_data([], [])
    return boat,

# 更新函数，在动画中更新小船的位置
def update(frame):
    boat.set_data(xx[frame],yy[frame])  # 更新小船位置
    return boat,

# 创建动画
ani = animation.FuncAnimation(fig, update, frames=len(t11), init_func=init, blit=True, interval=10)
ani.save('boat_animation.gif', writer='pillow', fps=50)
# 显示动画
plt.show()