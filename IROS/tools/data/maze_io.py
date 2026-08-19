#!/usr/bin/env python3

import os
import json
import pickle
import cv2

from iros.paths import ASSET_ROOT, DATA_ROOT

MASK_ALIGN_DIR = str(DATA_ROOT)
MAZE_PATH = str(ASSET_ROOT / "maze.png")

try:
    from iros.maze import MazeFeatureExtractor              
    MAZE_EXTRACTOR_AVAILABLE = True
except ImportError:
    MAZE_EXTRACTOR_AVAILABLE = False
    print("Warning: maze feature dependencies are unavailable; install iros[vision]")

def get_maze_shape():

    if not os.path.exists(MAZE_PATH): return None
    maze = cv2.imread(MAZE_PATH, cv2.IMREAD_UNCHANGED)
    if maze is None: return None
    return maze.shape[:2]

def load_maze_params(video_dir_name):

    params_path = os.path.join(MASK_ALIGN_DIR, video_dir_name, 
                              'registration_output', 'maze_params.json')
    if not os.path.exists(params_path):
        return None
    with open(params_path, 'r') as f:
        return json.load(f)

def load_maze_extractor(video_dir_name):

    pkl_path = os.path.join(MASK_ALIGN_DIR, video_dir_name,
                            'registration_output', 'maze_features.pkl')
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                extractor = pickle.load(f)
            return extractor
        except Exception as e:
            print(f"Error loading maze_features.pkl: {e}")
        return None

def query_point_features(extractor, cam_x, cam_y, maze_params, maze_shape, target_name='A'):

    if extractor is None or maze_params is None or maze_shape is None:
        return None

    my, mx = cam_to_maze(cam_x, cam_y, maze_params, maze_shape)
    point = (my, mx)

    try:
        features = extractor.query(point, target_name, normalize=False)
        return features
    except Exception as e:
        print(f"  query_point_features error at cam({cam_x:.1f},{cam_y:.1f}) "
              f"→ maze({my:.1f},{mx:.1f}): {e}")
        return None
