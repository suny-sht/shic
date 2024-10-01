import os
import pickle
import numpy as np
import sys
import torch
import trimesh
from spharapy import trimesh as tm
from spharapy import spharabasis as sb


CLASS_TO_OBJ = {
    'bear': 'bear.obj',
    'cow': 'cow.obj',
    'elephant': 'elephant.obj',
    'giraffe': 'giraffe.obj',
    'horse': 'horse.obj',
    'sheep': 'sheep.obj',
    'zebra': 'zebra.obj',
}

def load_obj_densepose(class_name):
    root = 'models/shapes'
    obj_path = os.path.join(root, CLASS_TO_OBJ[class_name])
    mesh = trimesh.load_mesh(obj_path)

    # Access vertices and faces
    verts = mesh.vertices
    faces = mesh.faces
    return torch.tensor(verts), torch.tensor(faces)

def center_verts(verts):
    # find center such that it is at max-min/2
    center = (verts.max(0)[0] + verts.min(0)[0]) / 2
    verts = verts - center
    return verts

def normalize_verts(verts):
    # find scale such that max-min = 1
    scale = (verts.max(0)[0] - verts.min(0)[0]).max()
    verts = verts / scale
    return verts


def load_shape_with_lbo(class_name='cat', topk=64, skip_first=True): 

    obj_raw = load_obj_densepose(class_name)

    verts_raw = obj_raw[0]
    verts = center_verts(verts_raw)
    verts = normalize_verts(verts)
    scale = (verts.max(0)[0] - verts.min(0)[0]).max()

    faces = obj_raw[1]

    trimesh = tm.TriMesh(faces, verts)

    eigenfunctions=None
    eigenvalues=None

    return {'eigenfunctions': eigenfunctions, 'eigenvalues': eigenvalues, 'faces': obj_raw[1], 'verts': verts, 'trimesh': trimesh, 'scale': scale}