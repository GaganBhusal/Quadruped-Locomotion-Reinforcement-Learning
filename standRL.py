import genesis as gs
import torch
from sliders_for_joints import JointControllerApp
import numpy as np
import gymnasium as gym



gs.init(backend=gs.cpu, logging_level = "warning")

scene = gs.Scene(show_viewer=True)
print(scene)

plane = scene.add_entity(gs.morphs.Plane())
go = scene.add_entity(
    gs.morphs.URDF(file='/home/yayy/My/Codeeeeee/Simulators/Genesis/genesis/assets/urdf/go2/urdf/go2.urdf'),
)
print(go.idx)
imu = scene.add_sensor(
    gs.sensors.IMU(
        entity_idx = go.idx,
        link_idx_local = go.get_link("base").idx_local,
        interpolate = True,
        draw_debug = True
    )
)


class StandENV(gym.Env):

    def __init__(self, robot, scene):
        super().__init__()
        
        self.scene = scene
        self.robot = robot

        self.imu = scene.add_sensor(
            gs.sensors.IMU(
                entity_idx = go.idx,
                link_idx_local = go.get_link("base").idx_local,
                interpolate = True,
                draw_debug = True
            )
        )

        self._get_internal_info()
        self.scene.build()

        imu_values_range_low = [-torch.pi/4] * 3
        imu_values_range_high = [torch.pi/4] * 3

        obs_space_low = self.joints_limit_low + imu_values_range_low
        obs_space_high = self.joints_limit_low + imu_values_range_high
        print(obs_space_high)

        self.observation_space = gym.spaces.Box(
            low=np.array(obs_space_low),
            high=np.array(obs_space_high),
            shape=(15, ),
            dtype=float
        )

        self.action_space = gym.spaces.Box(
            low=np.ones_like(self.joints_limit_low) * -1,
            high=np.ones_like(self.joints_limit_high),
            shape=(12, ),
            dtype=float
        )

    def _get_imu_values(self):
        _linear_acc, _angular_acc = self.imu.read()
        print(_angular_acc)
        return _linear_acc, _angular_acc

    def _calculate_reward(self):
         
        terminated = False
        self.robot.control_dofs_position(
            self.new_joint_angles, 
            dofs_idx_local = self.joints_local_idx
        )

        z_coordinate_base = self.robot.get_link("base").get_pos()[2]
        angular_velocity_joints = self.robot.get_dofs_velocity(dofs_idx_local = self.joints_local_idx[1:])

        reward = z_coordinate_base * 0.5 - abs(torch.sum(angular_velocity_joints**2))
        
        roll, yaw, pitch = self._get_imu_values()[1]
        if roll > 1 or pitch > 1:
            terminated = True

        return reward.item(),  terminated

    
    def _get_obs(self):
        print(self.robot.get_dofs_position(dofs_idx_local = self.joints_local_idx[1:]).numpy())
        return self.robot.get_dofs_position(dofs_idx_local = self.joints_local_idx).tolist() + self._get_imu_values()[1].tolist()


    def _get_internal_info(self):
        self.joints_local_idx = []
        self.joints_limit_low = []
        self.joints_limit_high = []
        for links in self.robot.links[1:]:
            print(links)
            joints = links.joints
            for joint in joints:
                self.joints_local_idx.append(joint.dof_idx_local)
                low, high = np.rad2deg(joint.dofs_limit)[0]
                self.joints_limit_low.append(low.item())
                self.joints_limit_high.append(high.item())
                # self.ranges.append(torch.rad2deg(torch.tensor(joint.dofs_limit)))/
                # print(f"Joint Name : {joint.name}, DOF : {joint.n_dofs}, type : {type(joint)}, Axis : {torch.rad2deg(torch.tensor(joint.dofs_limit))}")
        # print(f"Ranges : {self.joints_limit_low}\n{self.joints_limit_high}")

    def reset(self):

        super().__init__()
        observation = self._get_obs()
        info = 2
        return observation, info
        
    def step(self, action):
        self.new_joint_angles = np.add(self.robot.get_dofs_position(dofs_idx_local = self.joints_local_idx), action)

        reward, terminated = self._calculate_reward()
        observation = self._get_obs()

        return observation, reward, terminated, True, 1
    

env = StandENV(go, scene)
print(env.reset())
print(env.step(action=[0.5] * 12))
