import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal
import numpy as np
from modal import Volume, gpu, App, Secret, enter, method, web_endpoint, Function
from pydantic import BaseModel


dockerfile_image = modal.Image.from_dockerfile("Dockerfile")
GPU = gpu.A10G
# GPU = None
vol_path = "/data"

vol = Volume.from_name("data-volume", create_if_missing=True)

image = modal.Image.from_dockerfile("Dockerfile")

with image.imports():
    import logging
    import os
    import os.path as osp

    import firebase_admin
    import torch
    import torch.utils.checkpoint
    from firebase_admin import credentials
    from omegaconf import OmegaConf
    from PIL import Image
    from diffusers import AutoencoderKL, DDIMScheduler
    from diffusers import DPMSolverMultistepScheduler as DefaultDPMSolver
    from transformers import CLIPVisionModelWithProjection

    from firebase_utils import is_firebase_initialized, update_docs, upload_file_to_firebase, download_file_from_url, \
        download_folder_from_storage_in_parallel, upload_folder_to_storage


class DPMSolverMultistepScheduler(DefaultDPMSolver):
    def set_timesteps(
            self, num_inference_steps=None, device=None,
            timesteps=None
    ):
        if timesteps is None:
            super().set_timesteps(num_inference_steps, device)
            return

        all_sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        self.sigmas = torch.from_numpy(all_sigmas[timesteps])
        self.timesteps = torch.tensor(timesteps[:-1]).to(device=device, dtype=torch.int64)  # Ignore the last 0

        self.num_inference_steps = len(timesteps)

        self.model_outputs = [None, ] * self.config.solver_order
        self.lower_order_nums = 0

        # add an index counter for schedulers that allow duplicated timesteps
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication


app = App("animator")


@app.cls(
    gpu='A10G',
    image=image,
    retries=1,
    timeout=60 * 10,
    container_idle_timeout=60 * 1,
    secrets=[Secret.from_name("firebase_secret")],
    # volumes={vol_path: vol},
    allow_concurrent_inputs=1
)
class Animator:
    @enter()
    def enter(self):
        from inference import ChampModel, ReferenceAttentionControl, UNet2DConditionModel, UNet3DConditionModel, \
            setup_guidance_encoder, setup_savedir

        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        cred = eval(os.environ["firebase_json"])
        cred = credentials.Certificate(cred)
        if not is_firebase_initialized():
            firebase_admin.initialize_app(cred, {
                'storageBucket': 'twerkai.appspot.com'
            })
        if not is_firebase_initialized():
            firebase_admin.initialize_app(cred, {'storageBucket': 'twerkai.appspot.com'})
        logging.info("Firebase initialized")

        self.cfg = OmegaConf.load("configs/inference/inference.yaml")
        self.cfg.width = 512
        self.cfg.height = 512
        self.cfg.num_inference_steps = 10

        self.save_dir = setup_savedir(self.cfg)

        # setup pretrained models
        if self.cfg.weight_dtype == "fp16":
            self.weight_dtype = torch.float16
        else:
            self.weight_dtype = torch.float32

        sched_kwargs = OmegaConf.to_container(self.cfg.noise_scheduler_kwargs)
        if self.cfg.enable_zero_snr:
            sched_kwargs.update(
                rescale_betas_zero_snr=True,
                timestep_spacing="trailing",
                prediction_type="v_prediction",
            )
        self.noise_scheduler = DDIMScheduler(**sched_kwargs)
        self.noise_scheduler = DPMSolverMultistepScheduler.from_config(self.noise_scheduler.config,
                                                                       use_karras_sigmas=True, euler_at_final=True)
        sched_kwargs.update({"beta_schedule": "scaled_linear"})

        ckpt_dir = self.cfg.ckpt_dir
        guidance_encoder_group = setup_guidance_encoder(self.cfg)
        self.image_enc = CLIPVisionModelWithProjection.from_pretrained(self.cfg.image_encoder_path).to(
            dtype=self.weight_dtype, device="cuda")
        self.vae = AutoencoderKL.from_pretrained(self.cfg.vae_model_path).to(dtype=self.weight_dtype, device="cuda")
        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            self.cfg.base_model_path,
            self.cfg.motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=self.cfg.unet_additional_kwargs,
        ).to(device="cuda", dtype=self.weight_dtype)
        reference_unet = UNet2DConditionModel.from_pretrained(
            self.cfg.base_model_path,
            subfolder="unet",
        ).to(device="cuda", dtype=self.weight_dtype)

        logging.info("Loaded pretrained models")

        with ThreadPoolExecutor() as executor:
            guidance_dicts = {}
            for guidance_type, guidance_encoder_module in guidance_encoder_group.items():
                guidance_dicts[guidance_type] = executor.submit(torch.load, osp.join(
                    ckpt_dir, f"guidance_encoder_{guidance_type}.pth"), map_location="cpu")
            denoising_unet_future = executor.submit(torch.load, osp.join(ckpt_dir, f"denoising_unet.pth"),
                                                    map_location="cpu")
            reference_unet_future = executor.submit(torch.load, osp.join(ckpt_dir, f"reference_unet.pth"),
                                                    map_location="cpu")

            for guidance_type, guidance_encoder_module in guidance_encoder_group.items():
                guidance_encoder_module.load_state_dict(guidance_dicts[guidance_type].result(), strict=False)

            denoising_unet.load_state_dict(denoising_unet_future.result(), strict=False)
            reference_unet.load_state_dict(reference_unet_future.result(), strict=False)

        logging.info("Loaded model weights")

        reference_control_writer = ReferenceAttentionControl(
            reference_unet,
            do_classifier_free_guidance=False,
            mode="write",
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            denoising_unet,
            do_classifier_free_guidance=False,
            mode="read",
            fusion_blocks="full",
        )

        self.model = ChampModel(
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            reference_control_writer=reference_control_writer,
            reference_control_reader=reference_control_reader,
            guidance_encoder_group=guidance_encoder_group,
        ).to("cuda", dtype=self.weight_dtype)
        reference_unet.enable_xformers_memory_efficient_attention()
        denoising_unet.enable_xformers_memory_efficient_attention()
        logging.info("init completed")

    class Data(BaseModel):
        user_id: str = None
        image_url: str
        dance_id: str
        preprocess_folder: str

    @web_endpoint(method="POST")
    def infer(self, data: Data):
        self.process(data)
        return {"call_id": 0}

    def process(self, data: Data):  # img_file: UploadFile = File(...)):
        from inference import combine_guidance_data, inference, resize_tensor_frames, save_videos_grid
        job_id = next(tempfile._get_candidate_names())
        cred = eval(os.environ["firebase_json"])
        cred = credentials.Certificate(cred)
        if not is_firebase_initialized():
            firebase_admin.initialize_app(cred, {
                'storageBucket': 'twerkai.appspot.com'
            })
        job_path = os.path.join("temp", job_id)
        reference_image_loc = os.path.join(job_path, "reference_images")
        reference_imgs_folder = os.path.join(reference_image_loc, "images")
        reference_image_path = os.path.join(reference_imgs_folder, "reference.png")
        Path(job_path).mkdir(exist_ok=True, parents=True)
        Path(reference_imgs_folder).mkdir(exist_ok=True, parents=True)
        download_file_from_url(data.image_url, reference_image_path)

        Path(f"{job_path}/transferd_result").mkdir(exist_ok=True, parents=True)
        download_folder_from_storage_in_parallel(data.preprocess_folder, f"{job_path}/transferd_result")

        num_frames = min(300, len(os.listdir(f"{job_path}/transferd_result/dwpose")))
        self.cfg.data = OmegaConf.create({
            "ref_image_path": reference_image_path,
            "guidance_data_folder": f"{job_path}/transferd_result",
            "frame_range": [0, num_frames - 1]
        })
        # ref_image_pil = Image.open(io.BytesIO(img_file.file.read()))
        ref_image_pil = Image.open(reference_image_path)
        ref_image_pil.thumbnail((self.cfg.width, self.cfg.height))
        ref_image_w, ref_image_h = ref_image_pil.size

        guidance_pil_group, video_length = combine_guidance_data(self.cfg)

        result_video_tensor = inference(
            cfg=self.cfg,
            vae=self.vae,
            image_enc=self.image_enc,
            model=self.model,
            scheduler=self.noise_scheduler,
            ref_image_pil=ref_image_pil,
            guidance_pil_group=guidance_pil_group,
            video_length=video_length,
            width=self.cfg.width,
            height=self.cfg.height,
            device="cuda",
            dtype=self.weight_dtype,
        )  # (1, c, f, h, w)

        result_video_tensor = resize_tensor_frames(
            result_video_tensor, (ref_image_h, ref_image_w)
        )
        save_videos_grid(result_video_tensor, osp.join(self.save_dir, "animation.mp4"))
        logging.info(f"Inference completed, results saved in {self.save_dir}")
        output_url = upload_file_to_firebase(osp.join(self.save_dir, "animation.mp4"),
                                             f"inference_results/{data.user_id}/{data.dance_id}.mp4")
        update_docs(output_url, data.dance_id, data.user_id)
        logging.info(f"Uploaded to Firebase, URL: {output_url}")
        return output_url
