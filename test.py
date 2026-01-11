import genesis as gs
import torch
from sliders_for_joints import JointControllerApp
import numpy as np


gs.init(backend=gs.cpu, logging_level = "warning")

scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())
go = scene.add_entity(
    gs.morphs.URDF(file='/home/yayy/My/Codeeeeee/Simulators/Genesis/genesis/assets/urdf/go2/urdf/go2.urdf'),
)

scene.build()



# links = go.links
# joints = [joints for joints in links]
# joints_name = [(joint.name, joint.idx_local) for joint in joints]
# dofs_idx = [joint.idx_local for joint in joints]

# print([joint for joint in joints])

# print(joints_name)

# # dofs_idx = [go.get_joint(name).dof_idx_local for name in joints_name[1:]]
# print(dofs_idx)

joints_local_idx = []
ranges = []
for links in go.links:
    joints = links.joints
    for joint in joints:
        joints_local_idx.append(joint.dof_idx_local)
        ranges.append(np.rad2deg(joint.dofs_limit))
        print(f"Joint Name : {joint.name}, DOF : {joint.n_dofs}, type : {type(joint)}, Axis : {torch.rad2deg(torch.tensor(joint.dofs_limit))}")


# print(joints_local_idx[1:])
total_joints = 12

# tensor([[-60.0001,  60.0001]], dtype=torch.float64)
# tensor([[-60.0001,  60.0001]], dtype=torch.float64)
# tensor([[-60.0001,  60.0001]], dtype=torch.float64)
# tensor([[-60.0001,  60.0001]], dtype=torch.float64)
# tensor([[-90.0002, 200.0024]], dtype=torch.float64)
# tensor([[-90.0002, 200.0024]], dtype=torch.float64)
# tensor([[-30.0001, 260.0025]], dtype=torch.float64)
# tensor([[-30.0001, 260.0025]], dtype=torch.float64)
# tensor([[-155.9992,  -48.0001]], dtype=torch.float64)
# tensor([[-155.9992,  -48.0001]], dtype=torch.float64)
# tensor([[-155.9992,  -48.0001]], dtype=torch.float64)
# tensor([[-155.9992,  -48.0001]], dtype=torch.float64)


print(ranges)
# for i in ranges:
#     print(i)


# i just have to get all the joints and actuators of the robot 

# and then i can maybe implement gait cycle as in TWN that is good!!!!

# First I have to make it stand not fall of gravity yk!!!


a, b, c, d = 0, 45, -100, -120
target_angles_deg = torch.tensor([a, a, a, a, b, b, b, b, d, d, c, c])
target_angles = torch.deg2rad(target_angles_deg)

print(type(target_angles))

angles2 = [a, a, a, a, b, b, b, b, -c, -c, -c, -c]
angles2 = torch.deg2rad(torch.tensor(angles2))
go.set_dofs_kp(
    torch.tensor([100] * 12),
    dofs_idx_local = joints_local_idx[1:]

)

go.set_dofs_kv(
    torch.tensor([10] * 12),
    dofs_idx_local = joints_local_idx[1:]

)
# -60 -90 -30 -156  Initital Angles
# 0, 45, 45, -80 Target angles
num = 0
i = 1

initial_values = torch.zeros(12)
target_values = target_angles

fall_time = 500
rise_time = 500


num = 0

# app = JointControllerApp(ranges[1:], [a, a, a, a, b, b, b, b, c, c, c, c])

while True:
    num += 1

    if num <= fall_time:
        
        initial_values = go.get_dofs_position(dofs_idx_local = joints_local_idx[1:])
        print(f"Okk Got initial values at num = {num}")

    elif num <= (fall_time + rise_time):
        print(f"Started to process target at {num}")
        progress = (num - fall_time)/rise_time
        current_pose = initial_values + progress * (target_values - initial_values)

    # else:
    #     current_pose = target_values
    


    # if num > 2:
    # initial_angles = torch.rad2deg(go.get_dofs_position(dofs_idx_local = joints_local_idx[1:]))
    # print(type(initial_angles))
    # difference = target_angles_deg - initial_angles
    # print(f"Initiatl_angles : {initial_angles}\nDifference : {difference}")
    # final_angles = [initial_angles[i] + difference[i]/(1 *abs(difference[i])) for i in range(12)]
    # print(f"Final Angles : {final_angles}")
    if num > fall_time:
        go.control_dofs_position(
            current_pose, 
            dofs_idx_local = joints_local_idx[1:]
        )
    # target_angles= np.deg2rad(app.read_values())

        # for diff in difference:
        #     initial_angles += diff

    # # angles[4] += 0.017 * i
    # # angles[5] += 0.017 * i
    # # angles[6] += 0.017 * i
    # # angles[7] += 0.017 * i

    # angles[8] += 0.017 * i
    # angles[9] += 0.017 * i
    # angles[10] += 0.017 * i
    # angles[11] += 0.017 * i


    num += 1
    # print(torch.rad2deg(go.get_dofs_position(dofs_idx_local = joints_local_idx[1:])))
    scene.step()
    # app.update()


