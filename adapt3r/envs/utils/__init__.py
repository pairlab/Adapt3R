from .efficient_offscreen_viewer import EfficientOffScreenViewer
from .frame_stack import FrameStackObservationFixed

# def image_key_to_pointcloud_key(image_key):
#     if "rgb" in image_key: #agentview_rgb
#         return f"{image_key[:-4]}_pointcloud_full"
#     elif "image" in image_key: #agentview_image
#         return f"{image_key[:-6]}_pointcloud_full"
#     else:
#         raise ValueError(f"Unknown image key: {image_key}")
    
def list_cameras(shape_meta):
    image_keys = shape_meta['observation']['rgb'].keys()
    depth_keys = shape_meta['observation']['depth'].keys()
    if image_keys:
        return [image_key_to_camera_name(x) for x in image_keys]
    elif depth_keys:
        return [depth_key_to_camera_name(x) for x in depth_keys]
    else:
        return []


def camera_name_to_image_key(camera_name):
    return f'{camera_name}_image'


def camera_name_to_depth_key(camera_name):
    return f'{camera_name}_depth'


def camera_name_to_intrinsic_key(camera_name):
    return f'{camera_name}_intrinsic'

def camera_name_to_extrinsic_key(camera_name):
    return f'{camera_name}_extrinsic'


def depth_key_to_camera_name(depth_key):
    if "depth" in depth_key:
        return depth_key[:-6]
    else:
        raise ValueError(f"Unknown depth key: {depth_key}")

def image_key_to_camera_name(image_key):
    if "rgb" in image_key: #agentview_rgb
        return image_key[:-4]
    elif "image" in image_key: #agentview_image
        return image_key[:-6]
    else:
        raise ValueError(f"Unknown image key: {image_key}")

