from plyfile import PlyData
import torch
import numpy as np
ply = PlyData.read("05.ply")
vtx =ply['vertex']
pts = np.stack([vtx['x'], vtx['y'], vtx['z']], axis=-1)
pts = torch.from_numpy(pts)
print(len(pts))