import pybullet as p, pybullet_data, time
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
p.resetDebugVisualizerCamera(3.0, 50, -35, [0,0,0.3])
# BOX TRAI (do): chi co hinh, khong co khoi va cham
vis1 = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3,0.3,0.3], rgbaColor=[1,0,0,1])
p.createMultiBody(0, -1, vis1, [-0.9,0,0.3])
# BOX PHAI (xanh la): co ca khoi va cham + hinh
col2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3,0.3,0.3])
vis2 = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3,0.3,0.3], rgbaColor=[0,1,0,1])
p.createMultiBody(0, col2, vis2, [0.9,0,0.3])
print(">>> BOX TRAI = do (chi hinh). BOX PHAI = xanh la (hinh + va cham).")
print(">>> Noi cho tao biet may THAY MAY BOX (ca hai / chi xanh la / khong cai nao).")
while True:
    p.stepSimulation(); time.sleep(1/240)
