import genesis as gs
import torch
import numpy as np
import gymnasium as gym
from scipy.spatial.transform import Rotation
from domain_randomization import DomainRandomization 
import random
import cv2
import open3d as o3d

class WalkENV(gym.Env):

    def __init__(self, render = True, backend = gs.gpu, num_envs = 100, device="cuda", t_x = 20, t_y = 50, number_of_lanes=1, number_of_rows=10):
        super().__init__()


        self.device = device
        self.num_envs = num_envs
        self.max_episode_length = 9000

        self.target_base_height = 0.3
        self.target_vx = 0.6
        self.target_wz = 0.0
        self.action_scale = 0.25

        self.terrain_lenght = 103
        

        self.num_obs = 45
        self.num_actions = 12
        self.cfg = {}

        self.dt = 0.02
        terrain_x, self.terrain_y = t_x, t_y
        self.start_x, self.start_y, self.start_z = 10, 10, 0.42
        self.num_lanes = number_of_lanes
        self.num_of_rows = number_of_rows
        self.env_seperation = (0, (self.terrain_y * self.num_lanes-5)/self.num_envs)

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
        # plane = self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(file='/home/yayy/My/Codeeeeee/Simulators/Genesis/genesis/assets/urdf/go2/urdf/go2.urdf', 
                        #    convexify=True, 
                        #    decimate=True, 
                        #    decimate_face_num=1000
                           ),     
        )

        self.imu = self.scene.add_sensor(
            gs.sensors.IMU(
                entity_idx = self.robot.idx,
                link_idx_local = self.robot.get_link("base").idx_local,
                interpolate = True,
                draw_debug = True
            )
        )
        self.cam = self.scene.add_camera(
            res=(640, 480),
            pos=(0, 0.0, 0),
            lookat=(10, 0, 0),
            fov=100,
            GUI=True,
        )

        offset_T = np.eye(4)
        
        # Set the translation (Position offset: e.g., 0.3m forward, 0.05m up)
        offset_T[0, 3] = 1   # X offset
        offset_T[1, 3] = 0.0   # Y offset
        offset_T[2, 3] = 0.5  # Z offset
        r = Rotation.from_euler('z', -90, degrees=True)
        offset_T[:3, :3] = r.as_matrix()
        self.cam.attach(rigid_link=self.robot.get_link("base"), offset_T=offset_T)
        # depth_image = depth_cam.read_image()
        
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device)

        a = self.num_lanes

        curriculum_terrains = [
            "flat_terrain",
            "wave_terrain",
            "pyramid_sloped_terrain",
            "pyramid_stairs_terrain",
            "discrete_obstacles_terrain",
            "random_uniform_terrain",
        ]
        print(curriculum_terrains)
        print()
        curriculum_terrains = [[random.choice(curriculum_terrains) for _ in range(a)] for i in range(self.num_of_rows)]
        print(curriculum_terrains)

        curriculum_terrains = [
            ["flat_terrain"],
            ["random_uniform_terrain"],
            ["pyramid_sloped_terrain"],
            ["pyramid_stairs_terrain"],
            ["random_uniform_terrain"],
            ["discrete_obstacles_terrain"],
            ["random_uniform_terrain"],
        ]

        self.num_of_rows = len(curriculum_terrains)
        self.scene.add_entity(
            morph=gs.morphs.Terrain(
                n_subterrains=(self.num_of_rows, a),
                subterrain_size=(terrain_x, self.terrain_y),      
                horizontal_scale=0.1,            
                vertical_scale=0.005,            
                subterrain_types=curriculum_terrains,
                randomize=False,
                name="my_dog_curriculum"
            ),
        )
        self.terrain_lenght = self.num_of_rows * terrain_x - 1

        self._get_internal_info()
        self.scene.build(n_envs = self.num_envs, env_spacing = self.env_seperation, n_envs_per_row = self.num_envs)

        self.robot.set_dofs_kp(torch.tensor([40] * 12),dofs_idx_local = self.joints_local_idx)
        self.robot.set_dofs_kv(torch.tensor([1] * 12),dofs_idx_local = self.joints_local_idx)

        self.domainrandomizer = DomainRandomization(self.robot, self.num_envs, self.joints_local_idx)
        

        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(45, ),
            dtype=float
        )

        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(12, ),
            dtype=float
        )

        self.__initial_positions = torch.deg2rad(
            torch.tensor([
                0, 0, 0, 0, 45, 45, 57, 57, -85, -85, -85, -85
            ])
        )

        self.actions = torch.zeros((self.num_envs, 12), device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros((self.num_envs, 12))

        joint_ranges = torch.deg2rad(
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
        self.dof_min = joint_ranges[:, 0]
        self.dof_max = joint_ranges[:, 1]


        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)


        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)


        self.__get_linear_velocity()

        # For rsl_rl library
        self.extras = dict()        
        self.extras["observations"] = dict()
        import tkinter as tk
        
# --- GUI Setup for Full Camera Tuning ---
        self.root = tk.Tk()
        self.root.title("Full Camera Tuner")
        
        # Translation Variables (Position)
        self.cam_x = tk.DoubleVar(value=0.3)   # Forward/Back
        self.cam_y = tk.DoubleVar(value=0.0)   # Left/Right
        self.cam_z = tk.DoubleVar(value=0.05)  # Up/Down
        
        # Rotation Variables (Orientation)
        self.cam_roll = tk.DoubleVar(value=0.0)   # Tilt side-to-side
        self.cam_pitch = tk.DoubleVar(value=15.0) # Look Up/Down
        self.cam_yaw = tk.DoubleVar(value=0.0)    # Turn Left/Right
        
        # Intrinsic Variables (Lens)
        self.cam_fov = tk.DoubleVar(value=87.0)   # Field of View
        
        # Create Translation Sliders
        tk.Label(self.root, text="--- Position (Translation) ---").pack()
        tk.Scale(self.root, variable=self.cam_x, from_=-1.0, to=1.0, resolution=0.01, orient="horizontal", label="X Offset").pack(fill="x")
        self.cam_x.set(0.4)
        tk.Scale(self.root, variable=self.cam_y, from_=-1.0, to=1.0, resolution=0.01, orient="horizontal", label="Y Offset").pack(fill="x")
        self.cam_y.set(0.0)
        tk.Scale(self.root, variable=self.cam_z, from_=-1.0, to=1.0, resolution=0.01, orient="horizontal", label="Z Offset").pack(fill="x")
        self.cam_z.set(0.22)
        
        # Create Rotation Sliders
        tk.Label(self.root, text="--- Orientation (Rotation) ---").pack()
        tk.Scale(self.root, variable=self.cam_roll, from_=-180, to=180, resolution=1, orient="horizontal", label="Roll").pack(fill="x")
        self.cam_roll.set(37)
        tk.Scale(self.root, variable=self.cam_pitch, from_=-180, to=180, resolution=1, orient="horizontal", label="Pitch").pack(fill="x")
        self.cam_pitch.set(-21)
        tk.Scale(self.root, variable=self.cam_yaw, from_=-180, to=180, resolution=1, orient="horizontal", label="Yaw").pack(fill="x")
        self.cam_yaw.set(-62)
        
        # Create Intrinsic Sliders
        tk.Label(self.root, text="--- Intrinsics ---").pack()
        tk.Scale(self.root, variable=self.cam_fov, from_=30.0, to=150.0, resolution=1, orient="horizontal", label="Field of View (FOV)").pack(fill="x")


    def __get_linear_velocity(self):
        vel_body = self.robot.get_link("base").get_vel()
        return torch.tensor(vel_body)

    def _get_internal_info(self):
        self.joints_local_idx = []
        for links in self.robot.links[1:]:
            joints = links.joints
            for joint in joints:
                self.joints_local_idx.append(joint.dof_idx_local)

    def _get_imu_values(self):
        _, _angular_vel = self.imu.read()
        """ The arrangement of angular_vel is roll pitch yaw"""
        return _angular_vel
    
    def _show_depth_image(self):
        from scipy.spatial.transform import Rotation as R
        import cv2
        import numpy as np

        # 1. Keep the Tkinter window responsive
        self.root.update_idletasks()
        self.root.update()
        
        # 2. Build the offset_T matrix dynamically from Translation sliders
        offset_T = np.eye(4)
        offset_T[0, 3] = self.cam_x.get()
        offset_T[1, 3] = self.cam_y.get()
        offset_T[2, 3] = self.cam_z.get()
        
        # 3. Apply full 3D Rotation (Roll, Pitch, Yaw)
        # Using 'xyz' ordering. Adjust if Genesis expects a different order for your specific robot frame
        r = R.from_euler('xyz', [self.cam_roll.get(), self.cam_pitch.get(), self.cam_yaw.get()], degrees=True)
        offset_T[:3, :3] = r.as_matrix()
        
        # 4. Update Intrinsic Parameters (FOV)
        # Note: If your version of Genesis does not support dynamic FOV updates via a setter property, 
        # this parameter might only apply when instantiating the sensor in __init__.
        # if hasattr(self.cam, 'set_fov'):
        #     self.cam.set_fov(self.cam_fov.get())
        # elif hasattr(self.cam, 'fov'):
        #     self.cam.fov = self.cam_fov.get()
        
        # 5. Apply the new configuration
        base_link = self.robot.get_link("base")
        self.cam.attach(base_link, offset_T)
        self.cam.move_to_attach()

        depth_img, seg, col_seg, normal = self.cam.render(rgb=True, depth=False, segmentation=False, colorize_seg=False, normal=True) 
        

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
        base_pos = self.robot.get_link("base").get_pos()
        base_vel = self.robot.get_link("base").get_vel()
        base_ang_vel = self._get_imu_values()
        dof_pos = self.robot.get_dofs_position(dofs_idx_local=self.joints_local_idx)
        dof_vel = self.robot.get_dofs_velocity(dofs_idx_local=self.joints_local_idx)

        base_quat = self.robot.get_link("base").get_quat()
        base_lin_vel_body = self.__get_linear_velocity()
        projected_acceleration = self._calculate_projected_acceleration()

        vx = base_lin_vel_body[:, 0]
        vy = base_lin_vel_body[:, 1]
        vz = base_lin_vel_body[:, 2]
        base_height = base_pos[:, 2]


        lin_vel_error = torch.square(vx - self.target_vx)
        lin_vel_z_penalty = torch.square(vz)
        ang_vel_penalty = torch.sum(torch.square(base_ang_vel), dim=1)
        reward_for_straight_walking = torch.exp(-torch.square(base_ang_vel[:, 2])/0.25)
        # penalty_angular_velocity_pit = torch.exp(-torch.square(base_ang_vel[:, 2] - self.target_wz)/0.25)
        # penalty_angular_velocity_yaw = torch.exp(-torch.square(base_ang_vel[:, 2] - self.target_wz)/0.25)
        orientation_penalty = torch.sum(torch.square(projected_acceleration[:, :2]), dim=1)
        height_error = torch.square(base_height - self.target_base_height)
        action_rate_penalty = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        reward_similar_to_default = torch.sum(torch.abs(dof_pos - self.__initial_positions), dim=1)
        joint_vel_penalty = torch.sum(torch.square(dof_vel), dim=1)
        torques = self.robot.get_dofs_control_force(dofs_idx_local=self.joints_local_idx)
        torque_penalty = torch.sum(torch.square(torques), dim=1)
        action_rate_penalty = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        reward_tracking = torch.exp(-lin_vel_error / 0.25)
        penalty_y_velocity = torch.sum(torch.square(vy)/0.25)

        reward = (
            + 3.0 * reward_tracking
            + 0.5 * reward_for_straight_walking
            - 0.1 * orientation_penalty
            - 1.0 * ang_vel_penalty
            - 1.0 * lin_vel_z_penalty
            - 50.0 * height_error
            - 0.005 * action_rate_penalty
            - 0.1 * reward_similar_to_default
            - 1e-5 * torque_penalty
            - 5e-4 * penalty_y_velocity
            -0.001 * joint_vel_penalty
        )

        euler = self._quat_to_euler(base_quat)
        terminated = (
            (torch.abs(euler[:, 0]) > 0.5) |
            (torch.abs(euler[:, 1]) > 0.5)
        )
        trajectory_completed = self.robot.get_link('base').get_pos()[:, 0] >= self.terrain_lenght
        reward[terminated] -= 5
        reward[trajectory_completed] += 10

        return reward, terminated, trajectory_completed

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

        obs = torch.cat([
            base_lin_vel_body * 2.0,                # 3
            base_ang_vel * 0.25,                     # 3
            projected_gravity,                       # 3
            dof_pos,                                  # 12
            dof_vel * 0.05,                           # 12
            self.actions,                             # 12
        ], dim=1)
        return obs

    def _reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        base_pos = torch.zeros((len(envs_idx), 3), device=self.device)
        base_pos[:, 0] = self.start_x + torch.rand(len(envs_idx), device=self.device)
        base_pos[:, 1] = self.start_y + torch.rand(len(envs_idx), device=self.device)
        base_pos[:, 2] = self.start_z + torch.rand(len(envs_idx), device=self.device) * 0.1

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
        base_quat = torch.tensor([1, 0, 0, 0], device=self.device).repeat(len(envs_idx), 1)
        self.robot.set_quat(quat=base_quat, envs_idx=envs_idx)
        self.robot.set_pos(base_pos, envs_idx=envs_idx)

        self.domainrandomizer.randomize(envs_idx)

        self.episode_length_buf[envs_idx] = 0
        self.actions[envs_idx] = 0
        self.last_actions[envs_idx] = 0
        self.last_dof_vel[envs_idx] = 0
        

    def reset(self):
        all_idx = torch.arange(self.num_envs, device=self.device)
        self._reset_idx(all_idx)
        self.scene.reset()
        self.obs_buf = self._get_obs()
        return self.obs_buf, {} 
        
    def step(self, action):
        # action = torch.tensor(action, dtype=torch.float)
        self.actions = action.clone()

        target_dof_pos = self.__initial_positions + action * self.action_scale
        self.robot.control_dofs_position(target_dof_pos, dofs_idx_local = self.joints_local_idx)
        self.scene.step()
        # self.cam.move_to_attach()
       
        reward, terminated, trajectory_completed = self._calculate_reward()

        self.episode_length_buf += 1
        time_out = self.episode_length_buf >= self.max_episode_length
        self.extras["time_outs"] = time_out
        dones = terminated | time_out | trajectory_completed

        final_obs = self._get_obs().clone()

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
        self._show_depth_image()
    
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
    

