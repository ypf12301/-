 clc
 clear
 tao3c=[];
 qtao3c=[];
u=[];
Xu=-62.5;Yv=-62.5;Zw=-62.5;
Kp=-30;Mvq=-30;Nr=-30;
Xuu=-48;Yvv=-48;Zww=-48;Kpp=-80;Mvqq=-80;Nrr=-80;
m=5454.54;   
a1=0.15;
a2=0.61;
a3=0.61;
a4=0.15;
a5=0.3;
a6=0.15;
m1=20;
m2=80;
m3=80;
m4=20;
m5=40;
m6=20;
g=9.8;
Ix=2038;
Iy=13587;
Iz=13587;   
Ixy=-13.58;
Iyz=-13.58;
Ixz=-13.58;
rBB=zeros(3,1);
rGB=[0;0;0.061];
W=53400;
B=53400;
xita1=[10 26 11;-12 -13 -5;0 0 0];
evt23=[];
xita2=zeros(3,3);
et1=[];
et2=[];
et3=[];
et=[];
ta=[];
ta1=[];
p11=[];
p22=[];
p33=[];
j11=[];
j22=[];
j33=[];
T=[];
U1=[];
U2=[];
a=[6.019*10^3,9.551*10^3,2.332*10^4,4.129*10^3,4.913*10^4,2.069*10^4];
M=diag(a);
E1=[Xu,Yv,Zw,Kp,Mvq,Nr];
f1=[Xuu,Yvv,Zww,Kpp,Mvqq,Nrr];

D=-diag(E1)-diag(f1);
fG1=[0;0;W];
fB1=[0;0;B];
g=[fG1+fB1;cross(rGB,fG1)+cross(rBB,fG1)]*0;
% g=zeros(6,1);
tao10=[0;0;0;0;0;0];
tao20=[0;0;0;0;0;0];
tao30=[0;0;0;0;0;0];
ts=0.1;
v1=zeros(6,1);
v2=zeros(6,1);
v3=zeros(6,1);
v11=v1(1:3);
v21=v1(4:6);
v12=v2(1:3);
v22=v2(4:6);
v13=v3(1:3);
v23=v3(4:6);



ke11=0.4;


ke21=1;
kev1=2000;
ke12=0.5;
ke22=1;
kev2=2000;

ke13=0.5;
ke23=1;
kev3=2000;





for t=0:ts:60
    t
xita1d=[100*sin(0.1*t+0.5)+10;100*cos(0.1*t+0.5)+10;0];
xita2d=[0 0 0;0 0 0;0.03*t-pi 0.01/3*pi*t-10*pi/6 -pi/4*10]*0.1;
dxita1d=[10*cos(0.1*t+0.5);-10*sin(0.1*t+0.5);0];
dxita2d=[0 0 0;0 0 0;0.03*t 0.01/3*pi*t 0]*0.1;

d=[10 10 0;10 -10 0;0 0 0]'*0;
for j=1:3
    for k=1:3
while xita2(j,k)>pi
    xita2(j,k)=xita2(j,k)-2*pi;
end
while xita2(j,k)<=-pi
    xita2(j,k)=xita2(j,k)+2*pi;
end
    end


end

for j=1:3
    if xita2(2,k)==0.5*pi
    xita2(2,k)=0.01*pi;
    end
    if xita2(2,k)==-0.5*pi
    xita2(2,k)=0.01*pi;
    end
end
RBI1= [cos(xita2(3,1)), -sin(xita2(3,1)), 0;
         sin(xita2(3,1)),  cos(xita2(3,1)), 0;
         0,         0,        1];
RBI2=[cos(xita2(3,2)), -sin(xita2(3,2)), 0;
         sin(xita2(3,2)),  cos(xita2(3,2)), 0;
         0,         0,        1];
RBI3=[cos(xita2(3,2)), -sin(xita2(3,2)), 0;
         sin(xita2(3,2)),  cos(xita2(3,2)), 0;
         0,         0,        1];

T1=[1,sin(xita2(1,1))*tan(xita2(2,1)),cos(xita2(1,1))*tan(xita2(2,1));...
    0,cos(xita2(1,1)),-sin(xita2(1,1));...
    0,sin(xita2(1,1))/cos(xita2(2,1)),cos(xita2(1,1))/cos(xita2(2,1))];
T2=[1,sin(xita2(1,2))*tan(xita2(2,2)),cos(xita2(1,2))*tan(xita2(2,2));...
    0,cos(xita2(1,2)),-sin(xita2(1,2));...
    0,sin(xita2(1,2))/cos(xita2(2,2)),cos(xita2(1,2))/cos(xita2(2,2))];
T3=[1,sin(xita2(1,3))*tan(xita2(2,3)),cos(xita2(1,3))*tan(xita2(2,3));...
    0,cos(xita2(1,3)),-sin(xita2(1,3));...
    0,sin(xita2(1,3))/cos(xita2(2,3)),cos(xita2(1,3))/cos(xita2(2,3))];
t1=[1,0,-sin(xita2(2,1));...
    0,cos(xita2(1,1)),cos(xita2(2,1))*sin(xita2(1,1));...
    0,-sin(xita2(1,1)),cos(xita2(2,1))*cos(xita2(1,1))];
t2=[1,0,-sin(xita2(2,2));...
    0,cos(xita2(1,2)),cos(xita2(2,2))*sin(xita2(1,2));...
    0,-sin(xita2(1,2)),cos(xita2(2,2))*cos(xita2(1,2))];
t3=[1,0,-sin(xita2(2,3));...
    0,cos(xita2(1,3)),cos(xita2(2,3))*sin(xita2(1,3));...
    0,-sin(xita2(1,3)),cos(xita2(2,3))*cos(xita2(1,3))];

e11=xita1(1:3,1)-xita1d-d(1:3,1);
e12=xita1(1:3,2)-xita1d-d(1:3,2);
e13=xita1(1:3,3)-xita1d-d(1:3,3);

e21=xita2(1:3,1)-xita2d(1:3,1);
e22=xita2(1:3,2)-xita2d(1:3,2);
e23=xita2(1:3,3)-xita2d(1:3,3);
for j=1:3
while e21(j)>pi
    e21(j)=e21(j)-2*pi;
end
while e21(j)<=-pi
    e21(j)=e21(j)+2*pi;
end
while e22(j)>pi
    e22(j)=e22(j)-2*pi;
end
while e22(j)<=-pi
    e22(j)=e22(j)+2*pi;
end
while e23(j)>pi
    e23(j)=e23(j)-2*pi;
end
while e23(j)<=-pi
    e23(j)=e23(j)+2*pi;
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u1=v11(1);
vv1=v11(2);
w1=v11(3);
p1=v21(1);
q1=v21(2);
r1=v21(3);
arfav11=RBI1\(-ke11*e11+dxita1d);
ev11=v11-arfav11;
delta11=RBI1*ev11;
arfav21=-ke21*t1*e21+t1*dxita2d(1:3,1);
ev21=v21-arfav21;

ev1=[ev11;ev21];

arfav1=[arfav11;arfav21];

C1=[0          0         0         0                   m*w1              -m*vv1;...
    0          0         0       -m*w1                   0                m*u1;...
    0          0         0         m*vv1                -m*u1                0 ;...
    0         -m*w1       m*vv1        0            -Iyz*q1-Ixz*p1+Iz*r1    Iyz*r1+Ixy*p1-Iy*q1;...
    m*w1        0        -m*u1   Iyz*q1+Ixz*p1-Iz*r1          0            -Ixz*r1-Ixy*q1+Ix*p1;...
    -m*vv1       m*u1         0     -Iyz-Ixz*p1+Iy*q1    Ixy*r1+Iyz*q1-Ix*p1      0];
darfav1=-ev1/ts;

tao1=C1*v1+D*v1+M*darfav1-[(e11)'*0*RBI1,e21'*T1]'-kev1*ev1+g; 
dtao1=(tao1-tao10)/ts;
tao10=tao1;
qtao1=tao1;
dv1=M\(qtao1-g-C1*v1+D*v1);
v1=v1+ts*dv1;
v11=v1(1:3);
v21=v1(4:6);
dxita21=T1*v21;
dxita11=RBI1*v11;
xita1(1:3,1)=xita1(1:3,1)+dxita11*ts;
xita2(1:3,1)=xita2(1:3,1)+dxita21*ts;



p33=[p33 e11];
p11=[p11 e12];
p22=[p22 e13];
j33=[j33 e21];
j11=[j11 e22];
j22=[j22 e23];
ta=[ta qtao1];
ta1=[ta1 tao1];
et1=[et1 xita1(1:3,1)];
et2=[et2 xita1(1:3,2)];
et3=[et3 xita1(1:3,3)];
et=[et xita1d];
T=[T,t];
evt23=[evt23 xita1(:,1)-xita1(:,2)];
U1=[U1 xita1(:,1)-xita1(:,3)];
U2=[U2 xita1(:,2)-xita1(:,3)];

end
t11=0:ts:60;
x20=100*sin(0.1*t11+0.5)+10;
y20=100*cos(0.1*t11+0.5)+10;
figure;
hold on; % 保持当前图形
xlabel('X轴');
ylabel('Y轴');
title('小船移动');

plot(x20, y20, 'b', 'LineWidth', 2); 
boat2 = plot(NaN, NaN, 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
boat1 = plot(NaN, NaN, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r'); 
% 创建小船的点

text(10, 30, '小船', 'Color', 'r', 'FontSize', 12); % 小船1的标注
text(0.5, -0.5, '小船在某一刻的理想位置', 'Color', 'b', 'FontSize', 12); % 小船2的标注
for frame = 1:601

    % 更新小船的位置
    set(boat2, 'XData', x20(frame), 'YData', y20(frame));
    set(boat1, 'XData', et1(1,frame), 'YData', et1(2,frame));
 
    % 暂停以控制动画速度
    pause(0.01);





   % 捕获当前帧并保存为GIF
    frameData = getframe(gcf); % 获取当前图形的帧数据
    A = frame2im(frameData); % 将帧数据转换为图像
    [A, map] = rgb2ind(A, 256); % 将RGB图像转换为索引图像
    
    if frame == 1 % 如果是第一帧，创建GIF
        imwrite(A, map,'myGray.gif', 'gif', 'LoopCount', Inf, 'DelayTime', 0.1);
    else % 后续帧添加到GIF
        imwrite(A, map,'myGray.gif' , 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
    end
end
hold off; % 释放图形