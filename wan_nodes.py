import os
import numpy as np
import torch
import torch.distributed as dist
import sys
import datetime
from multiprocessing import Queue, Process


from PIL import Image
import folder_paths
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS
from wan.text2video import WanT2V
from wan.image2video import WanI2V
from wan.utils.utils import cache_video
import json
import uuid
import time
import subprocess

wan_configs = {
    'Wan-AI/Wan2.1-T2V-14B': 't2v-14B',
    'Wan-AI/Wan2.1-T2V-1.3B': 't2v-1.3B',
    'Wan-AI/Wan2.1-I2V-14B-480P': 'i2v-14B',
    'Wan-AI/Wan2.1-I2V-14B-720P': 'i2v-14B',
}
def model_load(rank, world_size, ckpt_dir, qins, qout, ulysses_size=1, ring_size=1, task='t2v-14B'):
    local_rank = rank
    device = local_rank
    qin = qins[rank]

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)

    if ulysses_size > 1 or ring_size > 1:
        assert ulysses_size * ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (initialize_model_parallel,
                                             init_distributed_environment)
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=ring_size,
            ulysses_degree=ulysses_size,
        )

    cfg = WAN_CONFIGS[task]
    if ulysses_size > 1:
        assert cfg.num_heads % ulysses_size == 0, f"`num_heads` must be divisible by `ulysses_size`."

    if "t2v" in task:
        wan_t2v = WanT2V(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=True if world_size>1 else False,
            dit_fsdp=True if world_size>1 else False,
            use_usp=(ulysses_size > 1 or ring_size > 1),
            t5_cpu=False,
        )
        while True:
            prompt, size, frame_num, sample_shift, sample_solver, sample_steps, sample_guide_scale, base_seed, img_path, n_prompt = qin.get()
            video = wan_t2v.generate(
                prompt,
                n_prompt=n_prompt,
                size=SIZE_CONFIGS[size],
                frame_num=frame_num,
                shift=sample_shift,
                sample_solver=sample_solver,
                sampling_steps=sample_steps,
                guide_scale=sample_guide_scale,
                seed=base_seed,
                offload_model=False)
            if rank == 0:
                formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                formatted_prompt = prompt.replace(" ", "_").replace("/",
                                                                        "_")[:50]
                suffix = '.png' if "t2i" in task else '.mp4'
                save_file = f"{folder_paths.output_directory}/{task}_{size.replace('*','x') if sys.platform=='win32' else size}_{ulysses_size}_{ring_size}_{formatted_prompt}_{formatted_time}" + suffix
                cache_video(
                    tensor=video[None],
                    save_file=save_file,
                    fps=cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1))
                qout.put(save_file)

    else:
        wan_i2v = WanI2V(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=True if world_size>1 else False,
            dit_fsdp=True if world_size>1 else False,
            use_usp=(ulysses_size > 1 or ring_size > 1),
            t5_cpu=False,
        )
        while True:
            prompt, size, frame_num, sample_shift, sample_solver, sample_steps, sample_guide_scale, base_seed, img_path, n_prompt = qin.get()
            video = wan_i2v.generate(
                prompt,
                img_path,
                n_prompt=n_prompt,
                max_area=MAX_AREA_CONFIGS[size],
                frame_num=frame_num,
                shift=sample_shift,
                sample_solver=sample_solver,
                sampling_steps=sample_steps,
                guide_scale=sample_guide_scale,
                seed=base_seed,
                offload_model=False)
            if rank == 0:
                formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                formatted_prompt = prompt.replace(" ", "_").replace("/",
                                                                        "_")[:50]
                suffix = '.mp4'
                save_file = f"{folder_paths.output_directory}/{task}_{size.replace('*','x') if sys.platform=='win32' else size}_{ulysses_size}_{ring_size}_{formatted_prompt}_{formatted_time}" + suffix
                cache_video(
                    tensor=video[None],
                    save_file=save_file,
                    fps=cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1))
                qout.put(save_file)

class PaiWanxModelLoad:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_type": (["Wan-AI/Wan2.1-T2V-1.3B", "Wan-AI/Wan2.1-I2V-14B-720P", "Wan-AI/Wan2.1-T2V-14B", "Wan-AI/Wan2.1-I2V-14B-480P"], ),
            }
        }
    RETURN_TYPES = ("WAN_MODEL",)
    RETURN_NAMES = ("wan_model",)
    FUNCTION = "model_load"
    CATEGORY = "pai_custom/pai_img_gen"

    def model_load(self, model_type):
        task = wan_configs[model_type]
        if os.path.exists(os.path.join(folder_paths.models_dir, model_type)):
            model_path = os.path.join(folder_paths.models_dir, model_type)
        elif os.path.exists(os.path.join(folder_paths.cache_dir, model_type)):
            model_path = os.path.join(folder_paths.cache_dir, model_type)
        else:
            print('no model find,')
        world_size = torch.cuda.device_count()
        cmd = f'nohup torchrun --nproc_per_node={world_size} dist_gen.py --task {task} --size 1280*720 --ckpt_dir {model_path} --dit_fsdp --t5_fsdp --ulysses_size {world_size} &'
        os.system(cmd)
        outs = {
            "task_json": f'{task}_{world_size}_1.json',
            "cmd": cmd
        }
        return (outs, )
        
        


class PaiWanxI2VXdit:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1}),
                "cfg_scale":("FLOAT", {"default": 5, "min": 0, "max": 100, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "size": (["720*1280", "480*832"], ),
                "wan_model": ("WAN_MODEL", ),
                "image": ("IMAGE", ),
                "frame_num": ("INT", {"default": 81, "min": 1, "max": 81, "step": 4}),
                "sample_shift": ("FLOAT", {"default": 5, "min": 0, "max": 100, "step": 0.1}),
                "sample_solver": (["unipc", "dpm++"],),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "encode"

    CATEGORY = "pai_custom/pai_img_gen"
    def encode(self, prompt, negative_prompt, steps, cfg_scale, seed, size, wan_model, image, frame_num, sample_shift, sample_solver):
        input_img_path = "/code/wanx_input_img.png"
        image = (image[0] * 255).numpy().astype(np.uint8)
        image = Image.fromarray(image)
        image.save(input_img_path)
        json_name = wan_model["task_json"]
        out_path =  os.path.join(folder_paths.output_directory, f"{uuid.uuid4()}.mp4")
        config = {
            "prompt": prompt,
            "image": input_img_path,
            "frame_num": frame_num,
            "sample_steps": steps,
            "cfg": cfg_scale,
            "seed": seed,
            "n_prompt": negative_prompt,
            "size": size,
            "sample_solver": sample_solver,
            "sample_shift": sample_shift,
            "save_file": out_path,
        }
        json.dump(config, open(json_name, "w"), indent=4)
        while True:
            if not os.path.exists(out_path):
                time.sleep(1)
                tmp = subprocess.Popen('ps -uax | grep dist_gen', shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if len(tmp.stdout.readlines())<3:
                    os.system(wan_model["cmd"])
                    if not os.path.exists(json_name):
                        json.dump(config, open(json_name, "w"), indent=4)
            else:
                break
        video_path = out_path
        return (video_path, )

NODE_CLASS_MAPPINGS = {
    "PaiWanxModelLoad": PaiWanxModelLoad,
    "PaiWanxI2VXdit": PaiWanxI2VXdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PaiWanxI2VXdit": "Pai Wanx I2V Xdit",
    "PaiWanxModelLoad": "Pai Wanx Model Load",
}


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']