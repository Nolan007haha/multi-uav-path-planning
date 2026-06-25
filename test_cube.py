import pybullet as p
import pybullet_data
import time

client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")

# Thu 3 cach tao cot khac nhau
# Cach 1: loadURDF cube
try:
    c1 = p.loadURDF("cube.urdf", [-2, 0, 1], globalScaling=2.0, useFixedBase=True)
    p.changeVisualShape(c1, -1, rgbaColor=[1,0,0,1])
    print("Cach 1 (cube.urdf): OK, id =", c1)
except Exception as e:
    print("Cach 1 FAIL:", e)

# Cach 2: createVisualShape BOX
try:
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3,0.3,1.0])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3,0.3,1.0], rgbaColor=[0,1,0,1])
    c2 = p.createMultiBody(0, col, vis, [0, 0, 1])
    print("Cach 2 (createVisualShape box): OK, id =", c2)
except Exception as e:
    print("Cach 2 FAIL:", e)

# Cach 3: createVisualShape SPHERE
try:
    vis3 = p.createVisualShape(p.GEOM_SPHERE, radius=0.5, rgbaColor=[0,0,1,1])
    c3 = p.createMultiBody(0, -1, vis3, [2, 0, 1])
    print("Cach 3 (sphere): OK, id =", c3)
except Exception as e:
    print("Cach 3 FAIL:", e)

print("Tong so vat the trong scene:", p.getNumBodies())

p.resetDebugVisualizerCamera(6, 50, -30, [0,0,1])

# giu cua so mo 30 giay
for i in range(7200):
    p.stepSimulation()
    time.sleep(1/240)
