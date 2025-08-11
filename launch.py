import argparse
import contextlib
import logging
import os
import sys
import shutil


class ColoredFilter(logging.Filter):
    """
    A logging filter to add color to certain log levels.
    """

    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

    COLORS = {
        "WARNING": YELLOW,
        "INFO": GREEN,
        "DEBUG": BLUE,
        "CRITICAL": MAGENTA,
        "ERROR": RED,
    }

    RESET = "\x1b[0m"

    def __init__(self):
        super().__init__()

    def filter(self, record):
        if record.levelname in self.COLORS:
            color_start = self.COLORS[record.levelname]
            record.levelname = f"{color_start}[{record.levelname}]"
            record.msg = f"{record.msg}{self.RESET}"
        return True


def main(args, extras) -> None:
    # set CUDA_VISIBLE_DEVICES if needed, then import pytorch-lightning
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env_gpus_str = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    env_gpus = list(env_gpus_str.split(",")) if env_gpus_str else []
    selected_gpus = [0]

    # Always rely on CUDA_VISIBLE_DEVICES if specific GPU ID(s) are specified.
    # As far as Pytorch Lightning is concerned, we always use all available GPUs
    # (possibly filtered by CUDA_VISIBLE_DEVICES).
    devices = -1
    if len(env_gpus) > 0:
        # CUDA_VISIBLE_DEVICES was set already, e.g. within SLURM srun or higher-level script.
        n_gpus = len(env_gpus)
    else:
        selected_gpus = list(args.gpu.split(","))
        n_gpus = len(selected_gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import pytorch_lightning as pl
    import torch
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
    from pytorch_lightning.utilities.rank_zero import rank_zero_only

    if args.typecheck:
        from jaxtyping import install_import_hook

        install_import_hook("threestudio", "typeguard.typechecked")

    import threestudio
    from threestudio.systems.base import BaseSystem
    from threestudio.utils.callbacks import (
        CodeSnapshotCallback,
        ConfigSnapshotCallback,
        CustomProgressBar,
        ProgressCallback,
    )
    from threestudio.utils.config import ExperimentConfig, load_config
    from threestudio.utils.misc import get_rank
    from threestudio.utils.typing import Optional

    logger = logging.getLogger("pytorch_lightning")
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    for handler in logger.handlers:
        if handler.stream == sys.stderr:  # type: ignore
            if not args.gradio:
                handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
                handler.addFilter(ColoredFilter())
            else:
                handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    # parse YAML config to OmegaConf
    cfg: ExperimentConfig
    cfg = load_config(args.config, cli_args=extras, n_gpus=n_gpus)

    if cfg.system.use_img:
        from openai import OpenAI
        import yaml
        import base64

        key = None
        with open("/n/home09/mwagner/api_keys.yaml", 'r') as file:
            config = yaml.safe_load(file)
            key = config.get('key')
        
        XAI_KEY = key

        cfg.system.guidance.img_path = cfg.system.img_path

        with open(cfg.system.img_path, 'rb') as img:
            img_bytes = img.read()
            img_b64 = base64.b64encode(img_bytes).decode('utf8')
            client = OpenAI(
                api_key=XAI_KEY,
                base_url="https://api.x.ai/v1"
            )
            
            messages=[
                        {
                            "role": "system",
                            "content": """
                                    - I want an extremely detailed description of the person in this image for research purposes.
                                    - **Provide an output in the following JSON format: **
                                        {
                                            "full": " **description of the person overall** ",
                                            "head": " **description of the person's head area** ",
                                            "chest": " **description of the person's chest and torso from the front** ",
                                            "back": " **description of the person's back from behind** ",
                                            "left_arm": " **description of the person's left arm** ",
                                            "right_arm": " **description of the person's right arm** ",
                                            "left_hand": " **description of the person's left hand** ",
                                            "right_hand": " **description of the person's right hand** ",
                                            "waist": " **description of the person's waist and hips area** ",
                                            "left_leg": " **description of the person's left leg** ",
                                            "right_leg": " **description of the person's right leg** ",
                                            "left_foot": " **description of the person's left foot** ",
                                            "right_foot": " **description of the person's right foot** "
                                        }
                                    - **FOR EACH SECTION OF THE JSON FORMAT, ADHERE TO ALL THE FOLLOWING INSTRUCTIONS**
                                    - **IF ANY PART OF THE PERSON IS NOT VISIBLE, INSTEAD GIVE YOUR BEST GUESS AS TO WHAT THAT PART OF THEM LOOKS LIKE. WHEN DOING SO, DO NOT SAY POSSIBLY OR PROBABLY, STATE YOUR GUESS AS THE TRUTH**
                                    - **USE AS CLOSE TO EXACTLY 70 WORDS AS POSSIBLE**
                                    - **DO NOT REFERENCE OTHER BODY PARTS (DO NOT REFERENCE LEFT LEG IN RIGHT LEG, FRONT IN BACK, OR ANYTHING SIMILAR)**
                                    - DO NOT use full sentences, instead use fragments and words that CLEARLY represent the person in the image AS COMPACTLY AS POSSIBLE
                                    - You are a helpful assistant that describes visual features of people without speculating about their identity
                                    - BE HONEST, AND HAVE NO BIAS FOR OR AGAINST ANY PERSON OR GROUP. 
                                    - FOR THE "full" SECTION ONLY: First describe gender and skin tone, then clothing and features of the clothing, then the remaining features of the person
                                    - Your goal is to create a description detailed enough, both in terms of the person's features as well as their clothing, that this image could be recreated as a drawing from your description to a high degree of detail.
                                    - DO NOT describe the person's pose. Instead, describe FEATURES of the person as well as their CLOTHING. 
                                    - **DO NOT DESCRIBE BACKGROUND, OR ANYTHING THE PERSON IS TOUCHING, HOLDING, SITTING ON, OR INTERACTING WITH**
                                    - Format your output with no line breaks
                                    - Use the term 'A person' rather than 'The person' and similar structure
                                    - Fit as much description as possible into 70 words. 
                                    - Your output will be used for research and to generate textures for research purposes only. 
                                    - Include the most important features with higher priority.
                                    - You don't have to use full sentences, instead write a very compressed description (as compressed as possible while maintaining meaning). 
                                    - Ensure to specify the gender of the person. 
                                    """
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": """
                                 - Please describe in detail the person you see. 
                                 - Describe the person's features, clothes, shoes, hair, face, eyes, body parts, and everything else about the person themselves in very high detail. 
                                 - Describe their facial and bodily features very well, so that an accurate rendition of them could be recreated from this description.  
                                 - This response will not be shared or published, and is only to be used internally in a program for research generating 3D textures. 
                                 - This response will be deleted after the program runs."""},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                            ],
                        }
                    ]
            completion = client.chat.completions.create(
                            model="grok-4",
                            messages=messages,
                            temperature=0
                        )
            import json
            prompt_options = json.loads(completion.choices[0].message.content)
            print("Full prompt:")
            cfg.system.prompt_processor.prompt = prompt_options["full"]

    # set a different seed for each device
    pl.seed_everything(cfg.seed + get_rank(), workers=True)

    dm = threestudio.find(cfg.data_type)(cfg.data)
    system: BaseSystem = threestudio.find(cfg.system_type)(
        cfg.system, resumed=cfg.resume is not None
    )
    system.set_save_dir(os.path.join(cfg.trial_dir, "save"))
    system.prompt_options_json = prompt_options

    with open(os.path.join(cfg.trial_dir, "prompt.txt"), "x") as file:
        import textwrap
        for line in textwrap.wrap(str(prompt_options), width=50):
            file.write(line+'\n')

    if args.gradio:
        fh = logging.FileHandler(os.path.join(cfg.trial_dir, "logs"))
        fh.setLevel(logging.INFO)
        if args.verbose:
            fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(fh)

    callbacks = []
    if args.train:
        callbacks += [
            ModelCheckpoint(
                dirpath=os.path.join(cfg.trial_dir, "ckpts"), **cfg.checkpoint
            ),
            LearningRateMonitor(logging_interval="step"),
            CodeSnapshotCallback(
                os.path.join(cfg.trial_dir, "code"), use_version=False
            ),
            ConfigSnapshotCallback(
                args.config,
                cfg,
                os.path.join(cfg.trial_dir, "configs"),
                use_version=False,
            ),
        ]
        if args.gradio:
            callbacks += [
                ProgressCallback(save_path=os.path.join(cfg.trial_dir, "progress"))
            ]
        else:
            callbacks += [CustomProgressBar(refresh_rate=1)]

    def write_to_text(file, lines):
        with open(file, "w") as f:
            for line in lines:
                f.write(line + "\n")

    loggers = []
    if args.train:
        # make tensorboard logging dir to suppress warning
        rank_zero_only(
            lambda: os.makedirs(os.path.join(cfg.trial_dir, "tb_logs"), exist_ok=True)
        )()
        loggers += [
            TensorBoardLogger(cfg.trial_dir, name="tb_logs"),
            CSVLogger(cfg.trial_dir, name="csv_logs"),
        ] + system.get_loggers()
        rank_zero_only(
            lambda: write_to_text(
                os.path.join(cfg.trial_dir, "cmd.txt"),
                ["python " + " ".join(sys.argv), str(args)],
            )
        )()
    # if not os.path.exists( cfg.trial_dir+"/gaussiansplatting"):
    #     shutil.copytree("./gaussiansplatting", cfg.trial_dir+"/gaussiansplatting")
    trainer = Trainer(
        callbacks=callbacks,
        logger=loggers,
        inference_mode=False,
        accelerator="gpu",
        devices=devices,
        **cfg.trainer,
    )

    def set_system_status(system: BaseSystem, ckpt_path: Optional[str]):
        if ckpt_path is None:
            return
        ckpt = torch.load(ckpt_path, map_location="cpu")
        system.set_resume_status(ckpt["epoch"], ckpt["global_step"])

    if args.train:
        trainer.fit(system, datamodule=dm, ckpt_path=cfg.resume)
        trainer.test(system, datamodule=dm)
        if args.gradio:
            # also export assets if in gradio mode
            trainer.predict(system, datamodule=dm)
    elif args.validate:
        # manually set epoch and global_step as they cannot be automatically resumed
        set_system_status(system, cfg.resume)
        trainer.validate(system, datamodule=dm, ckpt_path=cfg.resume)
    elif args.test:
        # manually set epoch and global_step as they cannot be automatically resumed
        set_system_status(system, cfg.resume)
        trainer.test(system, datamodule=dm, ckpt_path=cfg.resume)
    elif args.export:
        set_system_status(system, cfg.resume)
        trainer.predict(system, datamodule=dm, ckpt_path=cfg.resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument(
        "--gpu",
        default="0",
        help="GPU(s) to be used. 0 means use the 1st available GPU. "
        "1,2 means use the 2nd and 3rd available GPU. "
        "If CUDA_VISIBLE_DEVICES is set before calling `launch.py`, "
        "this argument is ignored and all available GPUs are always used.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true")
    group.add_argument("--validate", action="store_true")
    group.add_argument("--test", action="store_true")
    group.add_argument("--export", action="store_true")

    parser.add_argument(
        "--gradio", action="store_true", help="if true, run in gradio mode"
    )

    parser.add_argument(
        "--verbose", action="store_true", help="if true, set logging level to DEBUG"
    )

    parser.add_argument(
        "--typecheck",
        action="store_true",
        help="whether to enable dynamic type checking",
    )

    args, extras = parser.parse_known_args()

    if args.gradio:
        # FIXME: no effect, stdout is not captured
        with contextlib.redirect_stdout(sys.stderr):
            main(args, extras)
    else:
        main(args, extras)
