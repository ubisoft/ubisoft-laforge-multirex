from pathlib import Path
import numpy as np
import argparse

def main(args):

    base_installation_folder = Path(args.base_installation_folder)
    output_multiface_gt_path = Path(args.output_multiface_gt_path)
    output_multiface_gt_path.mkdir(exist_ok=True, parents=True)

    bin_paths = [
        r"m--20180227--0000--6795937--GHS/tracked_mesh/EXP_ROM07_Facial_Expressions",
        r"m--20180406--0000--8870559--GHS/tracked_mesh/EXP_ROM07_Facial_Expressions",
        r"m--20180510--0000--5372021--GHS/tracked_mesh/EXP_ROM07_Facial_Expressions",
        r"m--20181017--0000--002914589--GHS/tracked_mesh/EXP_ROM07_Facial_Expressions",
        r"m--20190529--1300--002421669--GHS/tracked_mesh/EXP_free_face",
        r"m--20190828--1318--002645310--GHS/tracked_mesh/EXP_free_face",
        r"m--20180426--0000--002643814--GHS/tracked_mesh/EXP_ROM07_Facial_Expressions",
        r"m--20180927--0000--7889059--GHS/tracked_mesh/EXP_ROM07_Facial_Expressions"
    ]

    # Iterate over bin_base_path, get a list of all .bin sort them, then load them using np.fromfile and add them to a list
    # of numpy arrays
    for bin_path in bin_paths:
        bin_base_path = base_installation_folder / bin_path
        bin_files = sorted(list(bin_base_path.glob("*.bin")))
        if len(bin_files) == 0:
            print(f"No .bin files found in {bin_base_path}. Skipping...")
            continue
        bin_data = []
        for bin_file in bin_files:
            data = np.fromfile(bin_file, dtype=np.float32)
            bin_data.append(data)
        # get np array and reshape to (n_frames, n_vertices, 3)
        gt_sequence = np.array(bin_data).reshape(-1, 7306, 3)
        save_name = '#'.join(bin_path.split('/')).replace("tracked_mesh", "images")
        np.save(output_multiface_gt_path / f"{save_name}.npy", gt_sequence)
        print(f"gt sequence {save_name}.npy saved to to {output_multiface_gt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process multiface bin files.")
    parser.add_argument("--base_installation_folder", type=str, default="./", help="Path to the installation folder")
    parser.add_argument("--output_multiface_gt_path", type=str, default="./multiface_gt", help="Output path for multiface ground-truth sequences")
    args = parser.parse_args()
    main(args)