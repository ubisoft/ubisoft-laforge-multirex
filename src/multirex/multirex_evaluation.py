import os
import torch
import scipy
import argparse
import pickle
import trimesh
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from multirex.FLAME.FLAME import FLAME
def main():
    parser = argparse.ArgumentParser(description="Evaluate multiface sequences.")
    parser.add_argument('--input_folder', required=True, type=str, help='Path to pickle or npz flame components file')
    parser.add_argument('--output_folder', type=str, help='Output folder path', default='./results')

    parser.add_argument('--assets_path', type=str, help='Output folder path', default='./assets')

    parser.add_argument('--gt_path', type=str, help='Path to ground truth data (multiface)', default='./assets/multiface_gt')
    parser.add_argument('--n_exp_components', type=int, default=100, help='Nb of FLAME expression components')

    parser.add_argument('--metrics_from_csv_only', action="store_true", help='Only get metrics from precomputed csv file '
                                                                             '(if evaluation has been done already)')

    parser.add_argument('--save_multiface', action="store_true", help='Save meshes after multiface conversion for debug or sanity check')
    parser.add_argument('--save_rigid', action="store_true", help='Save meshes after rigid alignment for debug or sanity check')



    args = parser.parse_args()

    # Prepare main paths and load metadata
    output_folder_path = Path(args.output_folder)
    decoded_flame_mesh_path = output_folder_path / "decoded_flame_meshes"
    output_folder_path.mkdir(exist_ok=True, parents=True)
    decoded_flame_mesh_path.mkdir(exist_ok=True, parents=True)
    assets_path = Path(args.assets_path)

    with open(assets_path / "id_mesh_weights_metadata_flame.pickle", "rb") as f:
        id_metadata = pickle.load(f)

    # Run benchmark evaluation
    if not args.metrics_from_csv_only:
        # Decode flame parameters to meshes
        decode_flame_meshes(args, id_metadata, output_folder_path=decoded_flame_mesh_path, assets_path=assets_path)
        # Compute benchmark metrics from meshes and store in a csv file
        compute_benchmark_metrics(args, output_folder_path, id_metadata,
                                  multiface_gt_path = args.gt_path, generated_meshes_path = decoded_flame_mesh_path, assets_path=assets_path)

    # Compute avg metrics from csv file
    csv_file = Path(args.output_folder) / 'metrics.csv'
    assert csv_file.exists(), f"Metrics file not found - {csv_file}"
    metrics_from_csv(csv_file)

def decode_flame_meshes(args, id_metadata, output_folder_path, assets_path):

    input_folder = Path(args.input_folder)
    device = torch.device('cpu')

    flame = FLAME(flame_model_path= assets_path / "FLAME/generic_model.pkl",
                  flame_lmk_embedding_path= assets_path / "FLAME/landmark_embedding.npy",
                  n_shape=300,
                  n_exp=args.n_exp_components,
                  assets_path=assets_path)

    codes_to_decode = list(input_folder.glob("*.npz"))
    if len(codes_to_decode) == 0:
        codes_to_decode = list(input_folder.glob("*.pkl"))

    for idx, code_path in tqdm(enumerate(codes_to_decode)):
        code_id = code_path.stem.split("--")[3]
        try:
            neutral = assets_path / Path("Neutrals_FLAME") / id_metadata[code_id]["neutral_mesh_flame"]
        except KeyError:
            print(f"Neutral mesh not found for {code_id}")
            continue

        output_file = output_folder_path / (code_path.stem + ".npy")
        decode_mesh(code_path, output_file, neutral, flame, device)

    print(f"Decoded {len(codes_to_decode)} files - saved to {output_folder_path}")


def decode_mesh(code_path, output_file, neutral_obj_path, flame, device):

    flame_params = np.load(code_path, allow_pickle=True)

    neutral_mesh = trimesh.load(neutral_obj_path, merge_norm=True, merge_tex=True)
    neutral_vertices = neutral_mesh.vertices

    exp = flame_params['expressions']
    pose = flame_params['poses']
    eyelid_params = flame_params.get('eyelids', None)

    # Unpose
    pose[:, :3] = 0

    neutral_face = np.tile(neutral_vertices, (pose.shape[0], 1, 1))  # Adapt to nb frames

    input_dict = {
        "expression_params": torch.from_numpy(exp).float().to(device).squeeze(0),
        "pose_params": torch.from_numpy(pose[:, :3]).float().to(device).squeeze(0),
        "jaw_params": torch.from_numpy(pose[:, 3:]).float().to(device).squeeze(0),
        "eyelid_params": torch.from_numpy(eyelid_params).float().to(device).squeeze(0) if eyelid_params is not None else None,
    }

    output_dict = flame(input_dict, neutral_meshes=torch.from_numpy(neutral_face).float().to(device).squeeze(0))
    unposed_vertices = output_dict["vertices"].numpy()

    np.save(output_file, unposed_vertices)

def convert_sequence(weights_sparse, sequence, removed_vertices, base_head):
    """ Convert a sequence of meshes from one topology to another

    Parameters
    ----------
    weights_sparse: scipy.sparse.csr_matrix
        Weights that map vertices in the source topology to the target topology
    sequence: np.ndarray
        Sequence of meshes in the source topology
    removed_vertices: np.ndarray
        Indices of vertices that were removed from the target topology
    base_head: np.ndarray
        Vertices of the base head mesh in the target topology

    Returns
    -------
    np.ndarray
        Sequence of meshes in the target topology
    """
    result = []
    for i in range(sequence.shape[0]):
        new_verts = weights_sparse.dot(sequence[i].reshape(-1, 3))
        if removed_vertices is not None:
            new_verts[removed_vertices] = base_head[removed_vertices]
        result.append(new_verts)

    result = np.array(result)
    return result

def fit_icp_RT(source, target, with_scale=True):
    """

    Args:
        source: float vertices, shape: n1x3
        target: fixed vertices, shape: n1x3
        with_scale: whether use the scale factor, bool

    Returns: transformation matrix, scale, rotation matrix, translation matrix

    """

    assert source.shape[0] == 3

    npoint = source.shape[1]
    means = np.mean(source, 1)
    meant = np.mean(target, 1)
    s1 = source - np.tile(means, (npoint, 1)).transpose()
    t1 = target - np.tile(meant, (npoint, 1)).transpose()
    W = t1.dot(s1.transpose())
    U, sig, V = np.linalg.svd(W)
    rotation = U.dot(V)

    scale = np.sum(np.sum(abs(t1))) / np.sum(np.sum(abs(rotation.dot(s1)))) if with_scale else 1.0

    translation = target - scale * rotation.dot(source)
    translation = np.mean(translation, 1)

    trans = np.zeros((4, 4))
    trans[3, 3] = 1
    trans[:3, 0:3] = scale * rotation[:, 0:3]
    trans[:3, 3] = translation[:]

    return trans, scale, rotation, translation


def flame_to_multi_topo_and_scale(sequence, removed_vertices, base_head, weights_sparse_flame_to_multi):
    sequence_in_multi_topo = convert_sequence(weights_sparse_flame_to_multi, sequence, removed_vertices, base_head)
    sequence_in_multi_topo = sequence_in_multi_topo * 1000 # Move to m
    return sequence_in_multi_topo

def rigid_align(target_seq, source_seq, mask_to_align):
    for idx_frame, (vertices_source, vertices_target) in enumerate(
            zip(source_seq[:, mask_to_align], target_seq[:, mask_to_align])):
        transform, scale, rotation, translation = fit_icp_RT(vertices_source.T, vertices_target.T, with_scale=False)

        source_seq[idx_frame] = np.matmul(source_seq[idx_frame] * scale, rotation.T) + translation

    return source_seq

def compute_metrics_mask(gt_sequence, sequence_in_multi_topo, mask):
    gt_motion = gt_sequence[:, mask]
    pred_motion = sequence_in_multi_topo[:, mask]

    distances = np.linalg.norm(gt_motion - pred_motion, axis=2)

    l2_distances = np.array(distances)

    avg_per_frame = np.mean(l2_distances, axis=1)

    avg_l2_distance = np.mean(l2_distances)

    return avg_l2_distance, avg_per_frame

def prepare_masks(regions_path):
    # Masks
    mouth_cheek = np.load(regions_path / 'mouth_cheek_region.npy')
    eyes_forehead = np.load(regions_path / 'eyes_forehead_region.npy')
    nose = np.load(regions_path / 'nose_region.npy')

    # Masks for alignment
    rigid_align_masks = {"cheek_region": mouth_cheek,
                         "mouth_region": mouth_cheek,
                         "eyes_region": eyes_forehead,
                         "nose_region": nose,
                         "forehead_region": eyes_forehead,
                         "eyes_forehead_region": eyes_forehead
                         }

    # Names for files after rigid alignment
    output_file_name_rigid = {"cheek_region": "_rigid_jaw.npy",
                              "mouth_region": "_rigid_jaw.npy",
                              "eyes_region": "_rigid_eyes_forehead.npy",
                              "nose_region": "_rigid_nose.npy",
                              "forehead_region": "_rigid_eyes_forehead.npy",
                              "eyes_forehead_region": "_rigid_eyes_forehead.npy"}

    mask_names = ["mouth_region", "nose_region", "cheek_region", "eyes_forehead_region"]
    masks = [np.load(regions_path / f"{mask}.npy") for mask in mask_names]
    return [mask_names, masks, rigid_align_masks, output_file_name_rigid]
def evaluate_sequence(cam, identity, input_sequence, output_folder, output_csv_file, gt_sequence_path, masks_info,
                      weights_sparse_flame_to_multi, save_rigid, save_multiface, overwrite=True):

    output_file = output_folder / input_sequence.name
    output_folder_rigids = output_folder / "rigid_aligned"
    if save_rigid:
        output_folder_rigids.mkdir(exist_ok=True)

    mask_names, masks, rigid_align_masks, output_file_name_rigid = masks_info

    gt_sequence = np.load(gt_sequence_path)
    sequence = np.load(str(input_sequence))

    sequence_in_multi_topo = flame_to_multi_topo_and_scale(sequence, None, None, weights_sparse_flame_to_multi)
    if save_multiface:
        np.save(output_file, sequence_in_multi_topo)

    results_df = pd.read_csv(output_csv_file) if os.path.isfile(output_csv_file) else pd.DataFrame()

    results = {}
    results[input_sequence.name] = {}

    for mask_name, mask in zip(mask_names, masks):
        output_rigid_file = output_folder_rigids / (output_file.stem + output_file_name_rigid[mask_name])

        if output_rigid_file.exists() and not overwrite:
            print(f"Using existing aligned sequence_in_multi_topo {output_rigid_file}")
            aligned_seq = np.load(output_rigid_file)
        else:
            aligned_seq = rigid_align(gt_sequence, sequence_in_multi_topo, rigid_align_masks[mask_name])
            if save_rigid:
                np.save(output_rigid_file, aligned_seq)

        avg_l2_distance, avg_per_frame = compute_metrics_mask(gt_sequence, aligned_seq, mask)

        results[input_sequence.name][f"{mask_name}"] = avg_per_frame

    for region, frames_results in results[input_sequence.name].items():
        for frame_nb, frame_result in enumerate(frames_results):
            new_row = pd.DataFrame([{'source_video': input_sequence.name,
                                     'camera': cam, 'region': region, 'identity': str(identity), "frame_nb": frame_nb,
                                     'value': frame_result}])
            results_df = pd.concat([results_df, new_row], ignore_index=True)

    results_df.to_csv(output_csv_file, index=False)



def compute_benchmark_metrics(args, output_folder_path, id_metadata, multiface_gt_path, generated_meshes_path, assets_path):

    multiface_gt_path = Path(multiface_gt_path)
    base_regions_path = assets_path / "regions"
    base_conv_matrix_path = assets_path / "conversion_matrices"

    # Prepare output files
    csv_file = output_folder_path / 'metrics.csv'
    output_folder_multiface_meshes = output_folder_path / "multiface_meshes"

    # Overwrite the file if it exists
    if csv_file.exists():
        csv_file.unlink()
    if args.save_multiface:
        output_folder_multiface_meshes.mkdir(exist_ok=True, parents=True)

    # Prepare masks info
    masks_info = prepare_masks(base_regions_path)

    # For each individual, get the corresponding metadata
    for id_individual, related_metadata in tqdm(id_metadata.items()):
        print(f"==> Processing {id_individual}")

        gt_sequence = multiface_gt_path / id_metadata[id_individual]["gt_sequence_name"]
        flame_to_multi_weights_path = base_conv_matrix_path / related_metadata["conversion_weights_flame"]
        weights_sparse_flame_to_multi = scipy.sparse.load_npz(flame_to_multi_weights_path)

        # For each view: load the captured mesh sequence, and evaluate it
        for cam in id_metadata[id_individual]["cameras"]:
            input_sequence = list(generated_meshes_path.glob(f"*{id_individual}*{cam}*.npy"))

            assert len(input_sequence) != 0 and input_sequence[0].exists(), f"Input sequence not found for {id_individual} and {cam} in {generated_meshes_path}"

            evaluate_sequence(cam, id_individual, input_sequence[0], output_folder_multiface_meshes, csv_file, gt_sequence, masks_info,
                              weights_sparse_flame_to_multi, args.save_rigid, args.save_multiface)

    print(f"==> Results saved to {csv_file}")

def metrics_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    # For identity 002645310 remove a range of frames from the dataframe (a bunch of frames are corrupted in the source video for cam #400347 and cam #400275)
    df_rows_to_remove = df[(df['identity'] == int("002645310")) & (((df['frame_nb'] <= 214) & (df['camera'] == "#400347")) | \
                                                                   ((df['frame_nb'] >= 1117)) & (df['camera'] == "#400275")) ]

    # Remove the rows from the dataframe
    cleaned_df = df[~df.index.isin(df_rows_to_remove.index)].reset_index(drop=True)

    # Per region average
    avg_per_region_df = cleaned_df.groupby(['region'])['value'].mean()
    std_per_region_df = cleaned_df.groupby(['region'])['value'].std()

    print(avg_per_region_df.apply(lambda x: f"{x:.2f} ± {std_per_region_df.loc[avg_per_region_df.index[avg_per_region_df == x][0]]:.2f}"))
    print(f"Average over mean region performance: {avg_per_region_df.mean():.2f}")

    camera_groups = {
        "Frontal": "#400030|#400291",
        "Angled": "#400017|#400039|#400347|#400436",
        "Profile": "#400275|#400347|#400018|#400042"
    }
    # Get the average per Frontal, Angled and Profile group
    for view, cameras in camera_groups.items():
        df_view = cleaned_df[cleaned_df['camera'].str.contains(cameras)]
        avg_per_method_view = df_view["value"].mean()
        std_per_method_view = df_view["value"].std()
        print(f"Average performance for {view} view: {avg_per_method_view:.2f} ± {std_per_method_view:.2f}")

if __name__ == '__main__':
    main()
