import math

import genesis as gs
import torch
import numpy as np
import gymnasium as gym
from scipy.spatial.transform import Rotation
from genesis.utils.geom import transform_by_quat, inv_quat
import tensordict

class WalkENV(gym.Env):

    def __init__(self, render = True, backend = gs.gpu, num_envs = 100, device="cuda", kwargs = {}):
        super().__init__()


        self.device = device
        self.num_envs = num_envs
        self.max_episode_length = 1000


        # Basic Parameters
        self.height_range = (0.2, 0.35)
        self.vx_range = (-0.5, 1.0)
        self.wz_range = (-1.2, 1.2)
        self.pitch_range = (-0.26, 0.26)
        self.action_scale = 0.25
        self.time_step = 0
        self.sigma = 0.25
        self.dt = 0.02
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        
        
        # Setting Up Scene and sensors
        self.scene = gs.Scene(sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=1,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                tolerance=1e-5,
                max_collision_pairs=20,
            ),
            show_viewer=render)
        plane = self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(file='/home/yayy/My/Codeeeeee/Simulators/Genesis/genesis/assets/urdf/go2/urdf/go2.urdf'),
        )

        self.imu = self.scene.add_sensor(
            gs.sensors.IMU(
                entity_idx = self.robot.idx,
                link_idx_local = self.robot.get_link("base").idx_local,
                interpolate = True,
                draw_debug = True
            )
        )
        self.scene.build(n_envs = self.num_envs, env_spacing = (5.0, 5.0))

        self._get_internal_info()
        self.robot.set_dofs_kp(torch.tensor([20] * 12),dofs_idx_local = self.joints_local_idx)
        self.robot.set_dofs_kv(torch.tensor([0.5] * 12),dofs_idx_local = self.joints_local_idx)

        # Observations and actions
        self.num_obs = 46
        self.obs_history_length = 3
        self.total_obs_len = self.num_obs * self.obs_history_length
        self.num_actions = 12
        self.cfg = {}

        custom_commands_list = ["Velocity forward", "Angular Velocity XY Plane", "Base Height", "Body Pitch"]
        self.custom_commands = torch.zeros((self.num_envs, len(custom_commands_list)), device=self.device)
        self.command_envs = torch.zeros(self.num_envs, device=self.device)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.total_obs_len, ),
            dtype=float
        )
        self.obs_histoy = torch.zeros((self.num_envs, self.total_obs_len), device = self.device)
        self.obs_buf = torch.zeros((self.num_envs, self.total_obs_len), device=self.device)

        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(12, ),
            dtype=float
        )

        self.__initial_positions = torch.deg2rad(
            torch.tensor([
                0, 0, 0, 0, 46, 46, 57, 57, -86, -86, -86, -86
            ])
        )

        self.actions = torch.zeros((self.num_envs, 12), device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.second_last_actions = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros((self.num_envs, 12))

        self.joint_limits = torch.deg2rad(
            torch.tensor([
                (-10, 10),
                (-10, 10),
                (-10, 10),
                (-10, 10),

                (25, 70),
                (25, 70),

                (40, 110),
                (40, 110),

                (-130, -70),
                (-130, -70),
                (-130, -70),
                (-130, -70)
            ])
        )

        self.__get_linear_velocity()
    
        self.extras = dict()        
        self.extras["observations"] = dict()
        self.extras["episode"] = dict()

        self.feet_idx = [
            self.robot.get_link("FL_calf").idx_local,
            self.robot.get_link("FR_calf").idx_local,
            self.robot.get_link("RL_calf").idx_local,
            self.robot.get_link("RR_calf").idx_local,
        ]
        
        self.feet_air_time = torch.zeros(self.num_envs, 4, device=self.device)

        self.episode_sums = {
            "reward_for_tracking_vx": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "reward_for_tracking_wz": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "height_penalty": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "pitch_penalty" : torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "lin_vel_z_penalty": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "roll_pitch_velocity_penalty": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "action_rate_penalty": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "second_order_action_rate": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "similar_to_default": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "torque_penalty": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "joint_vel_penalty" : torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "out_of_range_penalty": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "feet_air_time": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "Reward_Per_Environment": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
        }

    def __get_linear_velocity(self):
        vel_body = self.robot.get_link("base").get_vel()
        base_quat = self.robot.get_link("base").get_quat()
        vel_body = transform_by_quat(vel_body, inv_quat(base_quat))
        return torch.tensor(vel_body)

    def _get_internal_info(self):
        self.joints_local_idx = []
        for links in self.robot.links[1:]:
            joints = links.joints
            for joint in joints:
                print(f"Joint name: {joint.name}, local index: {joint.dof_idx_local}")
                self.joints_local_idx.append(joint.dof_idx_local)


    def _get_imu_values(self):
        _, _angular_vel = self.imu.read()
        return _angular_vel

    def _get_ypr(self):
        quat_wxyz = self.robot.get_link("base").get_quat()
        quat_xyzw = torch.stack([quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3], quat_wxyz[:, 0]], dim = 1)
        rotation = Rotation.from_quat(quat_xyzw.detach().cpu().numpy())
        euler_angles = rotation.as_euler("xyz")
        return torch.tensor(np.array(euler_angles), dtype=torch.float)

    def _calculate_projected_acceleration(self):
        quat = self.robot.get_link("base").get_quat()
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        gx = 2 * (x*z - w*y)
        gy = 2 * (y*z + w*x)
        gz = 1 - 2 * (x*x + y*y)
        return torch.stack([-gx, -gy, -gz], dim=1)
    

    def _calculate_reward(self):
        # Values for calculating Reward
        base_pos = self.robot.get_link("base").get_pos()
        base_vel = self.robot.get_link("base").get_vel()
        base_ang_vel = self._get_imu_values()
        dof_pos = self.robot.get_dofs_position(dofs_idx_local=self.joints_local_idx)
        dof_vel = self.robot.get_dofs_velocity(dofs_idx_local=self.joints_local_idx)
        torques = self.robot.get_dofs_control_force(dofs_idx_local=self.joints_local_idx)

        base_quat = self.robot.get_link("base").get_quat()
        base_lin_vel_body = self.__get_linear_velocity()

        vx = base_lin_vel_body[:, 0]
        vy = base_lin_vel_body[:, 1]
        wz = base_ang_vel[:, 2]
        vz = base_lin_vel_body[:, 2]
        base_height = base_pos[:, 2]
        current_pitch = self._get_ypr()[:, 1]

        target_air_time = 0.5
        contact_forces = self.robot.get_links_net_contact_force()
        foot_forces_z = contact_forces[:, self.feet_idx, 2]
        contact = foot_forces_z > 1.0
        first_contact = (self.feet_air_time > 0.0) & contact

        # Rewards and Penalty for Custom Commands
        reward_for_tracking_vx = torch.exp(-torch.square(self.custom_commands[:, 0] - vx) / self.sigma)
        reward_for_tracking_wz = torch.exp(-torch.square(self.custom_commands[:, 1] - wz) / self.sigma)
        height_penalty = torch.square(base_height - self.custom_commands[:, 2])
        pitch_penalty = torch.square(current_pitch - self.custom_commands[:, 3])

        # Additional Penaties
        lin_vel_z_penalty = torch.square(vz)
        roll_pitch_velocity_penalty = torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)
        action_rate_penalty = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        second_order_action_rate = torch.sum(torch.square(self.second_last_actions - 2 * self.last_actions + self.actions), dim=1)
        reward_similar_to_default = torch.sum(torch.square(dof_pos - self.__initial_positions), dim=1)
        torque_penalty = torch.sum(torch.square(torques), dim=1)
        joint_vel_penalty = torch.sum(torch.square(dof_vel), dim=1)
        out_of_range_penalty = torch.sum(torch.where((dof_pos < self.joint_limits[:, 0]) | (dof_pos > self.joint_limits[:, 1])).float(), dim=1)
        reward_feet_air_time = torch.sum(torch.clamp(self.feet_air_time, max=0.28) - target_air_time * first_contact, dim=1)

        self.feet_air_time += self.dt       
        self.feet_air_time[contact] = 0.0

        reward = (
            + 1.0 * reward_for_tracking_vx
            + 0.5 * reward_for_tracking_wz
            - 5.0 * height_penalty
            - 5.0 * pitch_penalty
            - 0.02 * lin_vel_z_penalty
            - 0.001 * roll_pitch_velocity_penalty
            - 0.1 * action_rate_penalty
            - 0.1 * second_order_action_rate
            - 0.1 * reward_similar_to_default
            - 0.001 * torque_penalty
            - 0.001 * joint_vel_penalty
            - 10.0 * out_of_range_penalty
            + 2.0 * reward_feet_air_time
        )
        
        self.episode_sums["reward_for_tracking_vx"] += reward_for_tracking_vx * 1.0
        self.episode_sums["reward_for_tracking_wz"] += reward_for_tracking_wz * 0.5
        self.episode_sums["height_penalty"] += height_penalty * -5.0
        self.episode_sums["pitch_penalty"] += pitch_penalty * -5.0
        self.episode_sums["lin_vel_z_penalty"] += lin_vel_z_penalty * -0.02
        self.episode_sums["roll_pitch_velocity_penalty"] += roll_pitch_velocity_penalty * -0.001
        self.episode_sums["action_rate_penalty"] += action_rate_penalty * -0.1
        self.episode_sums["second_order_action_rate"] += second_order_action_rate * -0.1
        self.episode_sums["similar_to_default"] += reward_similar_to_default * -0.1
        self.episode_sums["torque_penalty"] += torque_penalty * -0.001
        self.episode_sums["joint_vel_penalty"] += joint_vel_penalty * -0.001
        self.episode_sums["out_of_range_penalty"] += out_of_range_penalty * -10.0
        self.episode_sums["feet_air_time"] += reward_feet_air_time * 2.0
        self.episode_sums["Reward_Per_Environment"] += reward

        euler = self._quat_to_euler(base_quat)
        terminated = (
            (torch.abs(euler[:, 0]) > 0.35) |
            (torch.abs(euler[:, 1]) > 0.35) |
            (base_height < 0.15)
        )

        reward[terminated] -= 200
        reward *= self.dt

        return reward, terminated

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras
    
    def get_privileged_observations(self):
        return None

    def _get_obs(self):
        """
        https://arxiv.org/pdf/2406.04835v1
        I am using the obs space from here!!!
        """
                
        base_vel = self.robot.get_link("base").get_vel()
        base_ang_vel = self._get_imu_values()
        base_quat = self.robot.get_link("base").get_quat()
        base_lin_vel_body = self.__get_linear_velocity()
        projected_gravity = self._calculate_projected_acceleration()
        dof_pos = self.robot.get_dofs_position(dofs_idx_local=self.joints_local_idx) - self.__initial_positions
        dof_vel = self.robot.get_dofs_velocity(dofs_idx_local=self.joints_local_idx)
        
        scaled_commands = torch.stack([
            self.custom_commands[:, 0] * 1.0,   
            self.custom_commands[:, 1] * 0.5   
        ], dim=1)

        current_obs = torch.cat([
            base_ang_vel * 0.5,                     # 3
            projected_gravity ,                       # 3
            dof_pos,                                  # 12
            dof_vel * 0.05,                           # 12
            self.actions,                             # 12
            scaled_commands                            # 4
        ], dim=1)

        # Add history of observations
        self.obs_histoy[:, :-self.num_obs] = self.obs_histoy[:, self.num_obs:].clone()
        self.obs_histoy[:, -self.num_obs:] = current_obs
        return self.obs_histoy

    def _reset_idx(self, envs_idx):
        
        if len(envs_idx) == 0:
            return
        
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            avg_reward = torch.mean(value[envs_idx]) / self.max_episode_length
            self.extras["episode"]["rew_" + key] = avg_reward
            
            value[envs_idx] = 0.0

        self.robot.set_dofs_position(
            self.__initial_positions.repeat(len(envs_idx), 1),
            dofs_idx_local=self.joints_local_idx,
            envs_idx=envs_idx
        )
        self.robot.set_dofs_velocity(
            torch.zeros((len(envs_idx), 12), device=self.device),
            dofs_idx_local=self.joints_local_idx,
            envs_idx=envs_idx
        )

        self.episode_length_buf[envs_idx] = 0
        self.actions[envs_idx] = 0
        self.last_actions[envs_idx] = 0
        self.second_last_actions[envs_idx] = 0
        self.last_dof_vel[envs_idx] = 0
        self.obs_histoy[envs_idx] = 0

        base_quat = torch.tensor([1, 0, 0, 0], device=self.device).repeat(len(envs_idx), 1)
        self.robot.set_quat(quat=base_quat, envs_idx=envs_idx)
        base_pos = torch.tensor([0, 0, 0.42], device=self.device).repeat(len(envs_idx), 1)
        self.robot.set_pos(base_pos, envs_idx=envs_idx)

        self.update_commands(envs_idx)
        

    def reset(self):
        all_idx = torch.arange(self.num_envs, device=self.device)
        self._reset_idx(all_idx)
        self.scene.reset()
        self.obs_buf = self._get_obs()
        return self.obs_buf, {} 

    def update_commands(self, idx):
    
        self.custom_commands[idx, 0] = torch.rand(len(idx), device=self.device) * (self.vx_range[1] - self.vx_range[0]) + self.vx_range[0]
        self.custom_commands[idx, 1] = torch.rand(len(idx), device=self.device) * (self.wz_range[1] - self.wz_range[0]) + self.wz_range[0]
        self.custom_commands[idx, 2] = torch.rand(len(idx), device=self.device) * (self.height_range[1] - self.height_range[0]) + self.height_range[0]
        self.custom_commands[idx, 3] = torch.rand(len(idx), device=self.device) * (self.pitch_range[1] - self.pitch_range[0]) + self.pitch_range[0]

        self.command_envs[idx] = 0

    def step(self, action):
        self.actions = action.clone()

        target_dof_pos = self.__initial_positions + action * self.action_scale
        self.robot.control_dofs_position(target_dof_pos, dofs_idx_local = self.joints_local_idx)
        self.scene.step()
        self.command_envs += 1
    
        reward, terminated = self._calculate_reward()

        self.episode_length_buf += 1
        time_out = self.episode_length_buf >= self.max_episode_length
        self.extras["time_outs"] = time_out
        dones = terminated | time_out

        final_obs = self._get_obs().clone()
        
        self.second_last_actions = self.last_actions.clone()
        self.last_actions = self.actions.clone()

        self.previous_joint_velocity = self.robot.get_dofs_velocity(dofs_idx_local = self.joints_local_idx)

        reset_ids = torch.nonzero(dones).flatten()
        if len(reset_ids) > 0:
            self._reset_idx(reset_ids)

        observation = self._get_obs()
        
        info = {
            "final_obs": final_obs.detach().cpu().numpy(),
            "terminated": terminated.detach().cpu().numpy(),
        }

        self.extras["observations"]["critic"] = observation
    
        return observation, reward, dones, self.extras
    

    def _quat_to_euler(self, q):
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        pitch = torch.where(torch.abs(sinp) >= 1,
                            torch.sign(sinp) * torch.pi / 2,
                            torch.asin(sinp))
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return torch.stack([roll, pitch, yaw], dim=-1)
    

if __name__ == "__main__":
    gs.init(backend=gs.gpu, logging_level="warning")
    env = WalkENV(render=True, n_batch=1)
    
    state, info = env.reset()

    while True:
        action = env.action_space.sample()
        state, reward, dones, info = env.step(action)

