import argparse
import os
import pickle
import math
import genesis as gs
import pygame
from rsl_rl.runners import OnPolicyRunner
from Environments.walk_env_batch_custom_command_advanced import WalkENV
import torch
import re
from PIL import ImageDraw, Image
import numpy as np

# Import your VLM thread class here!
from Environments.vlm_test import AsyncVLM 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2_rsl_custom_command_terrain")
    parser.add_argument("--ckpt", type=int, default=3990)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.device == "cuda" else gs.cpu, logging_level="warning")

    log_dir = f"logs/{args.exp_name}"

    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        train_cfg = pickle.load(f)

    env = WalkENV(num_envs=1, 
                render=True, 
                device=gs.device, 
                t_x=8, 
                t_y=20, 
                number_of_lanes=1, 
                number_of_rows=5
                )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=args.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    print(f"Loading checkpoint from {resume_path}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=args.device)

    obs, info = env.reset()
    obs = obs.to(args.device)

    # --- SETUP CONTROLS ---
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("No controller detected! Please plug one in and restart.")
        return

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Controller connected: {joystick.get_name()}")
    
    # Boot up the VLM Brain
    robot_brain = AsyncVLM()

    # State variables for the AI's last known command
    vlm_v_x = 0.0
    vlm_w_z = 0.0

    def apply_deadzone(val, deadzone=0.1):
        if abs(val) < deadzone:
            return 0.0
        return val

    print("Starting simulation loop. Press 'B' (or Circle) to exit.")
    camera = info["camera"]

    with torch.no_grad():
        while True:
            pygame.event.pump()

            if joystick.get_button(1):
                print("Exit button pressed. Shutting down.")
                break

            # 1. Grab a frame from Genesis and send it to the VLM 
            # (Assuming you have a way to pull the camera obs here. 
            # If your camera is named 'rgb', pull it like this)
            # camera_frame = env.get_camera_image() 
            # print(camera)/
            # 1. Grab a frame from Genesis
            img_array = camera[0].cpu().numpy() if hasattr(camera[0], 'cpu') else camera[0]

            # FIX A: If shape is (3, H, W), swap it to (H, W, 3)
            if len(img_array.shape) == 3 and img_array.shape[0] == 3:
                img_array = np.transpose(img_array, (1, 2, 0))

            # FIX B: If values are 0.0 to 1.0, scale them to 0 to 255
            if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)

            # --- SANITY CHECK ---
            # Save exactly what we are sending to the AI.
            Image.fromarray(img_array).save("what_the_ai_sees.jpg")

            # Send the clean image to the VLM 
            robot_brain.send_image(img_array)

            # 2. Check if the VLM finished thinking
            # 2. Check if the VLM finished thinking
            sitrep = robot_brain.get_latest_output()
            if sitrep:
                try:
                    # THE BOUNCER: Now hunts for exactly 4 numbers (the bounding box)
                    match = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', sitrep)
                    
                    if match:
                        ymin = int(match.group(1))
                        xmin = int(match.group(2))
                        ymax = int(match.group(3))
                        xmax = int(match.group(4))
                        print(f"\n🎯 [VLM TARGET ACQUIRED]: Box at y1:{ymin}, x1:{xmin}, y2:{ymax}, x2:{xmax}")
                        
                        # --- THE VISUAL FLEX (Draw the Box) ---
                        try:
                            # 1. Grab the latest camera frame
                            # Use .cpu().numpy() if it's a PyTorch tensor!
                            img_array = camera[0].cpu().numpy() if hasattr(camera[0], 'cpu') else camera[0]
                            img = Image.fromarray(img_array)
                            draw = ImageDraw.Draw(img)
                            
                            # 2. Get actual image resolution (e.g., 640x480)
                            width, height = img.size
                            
                            # 3. Un-normalize the 0-1000 coordinates to actual pixels
                            left = (xmin / 1000.0) * width
                            top = (ymin / 1000.0) * height
                            right = (xmax / 1000.0) * width
                            bottom = (ymax / 1000.0) * height
                            
                            # 4. Draw a thick red rectangle
                            draw.rectangle([left, top, right, bottom], outline="red", width=5)
                            
                            # 5. Save the image so you can look at it
                            img.save("vlm_vision_check.jpg")
                            print("📸 Saved 'vlm_vision_check.jpg'. Go check what the AI is looking at!")
                            
                        except Exception as e:
                            print(f"⚠️ Couldn't draw the box: {e}")
                            
                        # Keep the robot still for now while we test vision
                        vlm_v_x = 0.0
                        vlm_w_z = 0.0
                    else:
                        print(f"\n⚠️ [VLM HALLUCINATION]: Ignored weird output -> {sitrep}")
                        
                except Exception as e:
                    print(f"\n❌ [CRITICAL PARSE ERROR]: {e}")

            # 3. Read the Human Controller
            raw_x = apply_deadzone(joystick.get_axis(0)) 
            raw_y = apply_deadzone(joystick.get_axis(1)) 
            
            # 4. THE OVERRIDE LOGIC
            # If the human touches the stick at all, human takes control
            human_active = abs(raw_x) > 0.0 or abs(raw_y) > 0.0

            if human_active:
                magnitude = math.sqrt(raw_x**2 + raw_y**2)
                if magnitude > 1.0:
                    raw_x /= magnitude
                    raw_y /= magnitude
                
                final_vx = -raw_y 
                final_wz = raw_x 
                # print(f"🎮 HUMAN: vx={final_vx:.2f}, wz={final_wz:.2f}", end='\r')
            else:
                # If hands are off the controller, the AI takes the wheel
                final_vx = vlm_v_x
                final_wz = vlm_w_z
                # print(f"🤖 AI: vx={final_vx:.2f}, wz={final_wz:.2f}    ", end='\r')

            # 5. Apply the winning commands to the environment
            env.custom_commands[0, 0] = final_vx
            env.custom_commands[0, 1] = final_wz

            obs[0, -2] = final_vx
            obs[0, -1] = -final_wz

            env.command_envs[:] = 0 

            # Step the policy
            actions = policy(obs)
            obs, reward, done, info = env.step(actions)
            camera = info["camera"]

if __name__ == "__main__":
    main()