from typing import Dict, Optional

import mujoco
import numpy as np

from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer


class EfficientOffScreenViewer(OffScreenViewer):
    """Return depth and RGB image without needing to render twice"""

    def render(
        self,
        render_mode: Optional[str],
        camera_id: Optional[int] = None,
        segmentation: bool = False,
    ):
        if camera_id is not None:
            if camera_id == -1:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = camera_id

        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.vopt,
            self.pert,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scn,
        )

        if segmentation:
            self.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
            self.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1

        for marker_params in self._markers:
            self._add_marker_to_scene(marker_params)

        mujoco.mjr_render(self.viewport, self.scn, self.con)

        for gridpos, (text1, text2) in self._overlays.items():
            mujoco.mjr_overlay(
                mujoco.mjtFontScale.mjFONTSCALE_150,
                gridpos,
                self.viewport,
                text1.encode(),
                text2.encode(),
                self.con,
            )

        if segmentation:
            self.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 0
            self.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 0

        rgb_arr = np.zeros(
            3 * self.viewport.width * self.viewport.height, dtype=np.uint8
        )
        depth_arr = np.zeros(
            self.viewport.width * self.viewport.height, dtype=np.float32
        )

        mujoco.mjr_readPixels(rgb_arr, depth_arr, self.viewport, self.con)

        if render_mode == 'all':
            depth_img = depth_arr.reshape(self.viewport.height, self.viewport.width)
            
            rgb_img = rgb_arr.reshape(self.viewport.height, self.viewport.width, 3)

            if segmentation:
                seg_img = (
                    rgb_img[:, :, 0]
                    + rgb_img[:, :, 1] * (2**8)
                    + rgb_img[:, :, 2] * (2**16)
                )
                seg_img[seg_img >= (self.scn.ngeom + 1)] = 0
                seg_ids = np.full(
                    (self.scn.ngeom + 1, 2), fill_value=-1, dtype=np.int32
                )

                for i in range(self.scn.ngeom):
                    geom = self.scn.geoms[i]
                    if geom.segid != -1:
                        seg_ids[geom.segid + 1, 0] = geom.objtype
                        seg_ids[geom.segid + 1, 1] = geom.objid
                rgb_img = seg_ids[seg_img]

            # original image is upside-down, so flip i
            return rgb_img[::-1, :, :], depth_img[::-1, :]            
        elif render_mode == "depth_array":
            depth_img = depth_arr.reshape(self.viewport.height, self.viewport.width)
            # original image is upside-down, so flip it
            return depth_img[::-1, :]
        else:
            rgb_img = rgb_arr.reshape(self.viewport.height, self.viewport.width, 3)

            if segmentation:
                seg_img = (
                    rgb_img[:, :, 0]
                    + rgb_img[:, :, 1] * (2**8)
                    + rgb_img[:, :, 2] * (2**16)
                )
                seg_img[seg_img >= (self.scn.ngeom + 1)] = 0
                seg_ids = np.full(
                    (self.scn.ngeom + 1, 2), fill_value=-1, dtype=np.int32
                )

                for i in range(self.scn.ngeom):
                    geom = self.scn.geoms[i]
                    if geom.segid != -1:
                        seg_ids[geom.segid + 1, 0] = geom.objtype
                        seg_ids[geom.segid + 1, 1] = geom.objid
                rgb_img = seg_ids[seg_img]

            # original image is upside-down, so flip i
            return rgb_img[::-1, :, :]