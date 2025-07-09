# Based on https://github.com/facebookresearch/multiface/blob/main/download_dataset.py
# Main changes:
# - Added gamma correction for images
# - Added conversion of images to video using ffmpeg
# - Download only subset of cameras, subjects and expressions

import argparse
import json
import logging
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import subprocess
import tarfile
import shutil
import numpy as np
from multirex.scripts.utils import gammaCorrect
from PIL import Image


MAX_TRY = 50
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Dataset Download")


def gamma_correct_images(imgs_basepath, gamma_corrected_output_folder):
    # call gammaCorrect for each image in the folder
    for img_path in imgs_basepath.glob('*.png'):
        img = np.array(Image.open(img_path), dtype=np.float32)
        img_corrected = gammaCorrect(img / 255.0)
        img_corrected = (img_corrected * 255).astype(np.uint8)
        Image.fromarray(img_corrected).save(gamma_corrected_output_folder / img_path.name)


def download_tar(download_dest, entity, download_img, download_tex, download_mesh, download_audio, download_metadata, expression, cameras):
    # extract urls
    misc = set(["CHECKSUM", "index.html"])
    root = "https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/MugsyDataRelease/v0.0/identities/"
    entity_urls = []
    for e in entity:
        entity_urls.append(root + e + "/index.html")

    meta_dest = download_dest / 'metadata'
    gamma_corrected_folder = download_dest / 'videos_gamma_corrected'
    meta_dest.mkdir(exist_ok=True)
    gamma_corrected_folder.mkdir(exist_ok=True)

    for url in entity_urls:
        entity = url.split("/")[-2]
        logging.info("Start downloading entity %s...." % (entity))
        reqs = requests.get(url)
        soup = BeautifulSoup(reqs.text, "html.parser")

        links = [a.get("href") for a in soup.find_all("a")]
        with open(meta_dest / f"{entity}_links.txt", "w") as f:
            for link in links:
                f.write(f"{link}\n")
                
        used_links = []
        for download_url in links:
            file_name = download_url.split("/")[-1]
            if "unwrapped_uv" in file_name and not download_tex:
                continue
            if "tracked_mesh" in file_name and not download_mesh:
                continue
            if "images" in file_name and not download_img:
                continue
            if "audio" in file_name and not download_audio:
                continue
            if "metadata" in file_name and not download_metadata:
                continue

            included_file = False
            if file_name in misc or "metadata" in file_name or "audio" in file_name:
                included_file = True
            else:
                for exp in expression:
                    if exp in file_name:
                        included_file = True
                        break

            if included_file is False:
                continue

            if 'images' in file_name and len(cameras) > 0:
                included_file = False
                for camera in cameras:
                    if camera in file_name:
                        included_file = True
                        break

                if included_file is False:
                    continue

            used_links.append(download_url)

            dest = download_dest if not ("CHECKSUM" in file_name or "index.html" in file_name) else meta_dest
            file_path = dest / f'{entity}{file_name}'

            subprocess.run(["wget", "-O", file_path, download_url], check=True)

            if file_path.suffix == ".tar":
                subprocess.run(["tar", "-xvf", file_path, "-C", file_path.parent], check=True)

                with tarfile.open(file_path) as tar:
                    name = tar.getmembers()[0].name

                if name.endswith(".png") or name.endswith("*.txt"):
                    name = '/'.join(name.split("/")[:-1])

                assert not Path(name).is_absolute()  # making the rmtree safer

                if "images" in name:
                    imgs_basepath = Path(file_path).parent / name

                    save_name = '#'.join(name.split('/'))
                    gamma_corrected_output_folder = gamma_corrected_folder / f'{save_name}'
                    gamma_corrected_output_folder.mkdir(exist_ok=True, parents=True)
                    gamma_correct_images(imgs_basepath, gamma_corrected_output_folder)

                    save_path = gamma_corrected_folder / f'{save_name}.mp4'

                    convert_video(gamma_corrected_output_folder.as_posix(), save_path)

                    # Cleanup
                    shutil.rmtree(gamma_corrected_output_folder)
                    shutil.rmtree(imgs_basepath)

                file_path.unlink()

        with open(meta_dest / f"{entity}_used_links.txt", "w") as f:
            for link in used_links:
                f.write(f"{link}\n")
    return True

def convert_video(imgs_basepath, save_path):
    subprocess.run(["ffmpeg", "-nostdin", "-framerate", "30", "-pattern_type", "glob", "-i", f"{imgs_basepath}/*.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", f'{save_path}'], check=True)


def main(args):
    save_path = Path(args.base_installation_folder)
    save_path.mkdir(parents=True, exist_ok=True)
    f = open(args.download_config, "r")
    download_config = json.load(f)
    f.close()

    cameras = args.cameras.split()

    entity = download_config["entity"]
    download_img = download_config["image"]
    download_tex = download_config["texture"]
    download_mesh = download_config["mesh"]
    download_audio = download_config["audio"]
    download_metadata = download_config["metadata"]
    expression = download_config["expression"]

    if download_tar(
        download_dest=save_path,
        entity=entity,
        download_img=download_img,
        download_tex=download_tex,
        download_mesh=download_mesh,
        download_audio=download_audio,
        download_metadata=download_metadata,
        expression=expression,
        cameras=cameras
    ):
        logging.info("%s .tar extraction has completed" % (entity))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_installation_folder",
        metavar="Installation Destination",
        type=str,
        required=False,
        help="Directory of data to be downloaded",
        default="./",
    )
    parser.add_argument(
        "--download_config",
        metavar="Download Config File",
        type=str,
        required=False,
        help="File path of download_config file",
        default="./assets/download_config.json",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        help="Camera views to download",
        default="400017 400018 400030 400039 400042 400275 400291 400347 400436 400485",
    )

    args = parser.parse_args()
    main(args)
