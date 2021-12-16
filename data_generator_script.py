import bpy
import os
import random
import mathutils
import numpy as np
from os import listdir
from os.path import isfile, join
from mathutils.bvhtree import BVHTree

# Whether we're running as an HPC batch job
RUN_FROM_CLI = True

# Enable GPU CUDA rendering from CLI
# See https://github.com/nytimes/rd-blender-docker/issues/3
if RUN_FROM_CLI:
    for scene in bpy.data.scenes:
        scene.cycles.device = 'GPU'
    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences
    # Calling this purges the device list so we need it
    cuda_devices, opencl_devices = cprefs.get_devices()
    # Attempt to set GPU device types if available
    for compute_device_type in ('CUDA', 'OPENCL'):
        try:
            cprefs.compute_device_type = compute_device_type
            break
        except TypeError:
            pass
    # Enable all CPU and GPU devices
    for device in cprefs.devices:
        device.use = True

# Find base directory to work from
base_dir = os.path.abspath(os.getcwd())
print("Running from dir " + base_dir)

# Update the scene resolution to match our desired output resolution
INPUT_RES = 224
bpy.context.scene.render.resolution_x = INPUT_RES
bpy.context.scene.render.resolution_y = INPUT_RES

# List the available HDRI background images
env_files_dir = base_dir + "/data_generator_assets/hdris"
env_img_files = [f for f in listdir(env_files_dir) if isfile(join(env_files_dir, f))]
env_tex_node = bpy.context.scene.world.node_tree.nodes[2]

# List the available surface texture images
text_files_dir = base_dir + "/data_generator_assets/texts"
text_img_folders = [f for f in listdir(text_files_dir) if not isfile(join(text_files_dir, f))]
text_mat_nodes = bpy.data.objects["Ground View"].material_slots[0].material.node_tree.nodes

def check_overlap(obj1, obj2):

    # Get the geometry in world coordinates
    vert1 = []
    poly1 = []
    vert1.extend([obj1.matrix_world @ v.co for v in obj1.data.vertices])
    poly1.extend([p.vertices for p in obj1.data.polygons])
    
    vert2 = [obj2.matrix_world @ v.co for v in obj2.data.vertices] 
    poly2 = [p.vertices for p in obj2.data.polygons]

    # Create the BVH trees
    bvh1 = BVHTree.FromPolygons(vert1, poly1)
    bvh2 = BVHTree.FromPolygons(vert2, poly2)
    
    # Test if overlap
    return not bvh1.overlap(bvh2) == []

def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))

def camera_view_bounds_2d(scene, cam_ob, me_ob):
    """
    From https://blender.stackexchange.com/questions/7198/save-the-2d-bounding-box-of-an-object-in-rendered-image-to-a-text-file
    
    Returns camera space bounding box of mesh object.

    Negative 'z' value means the point is behind the camera.

    Takes shift-x/y, lens angle and sensor size into account
    as well as perspective/ortho projections.

    :arg scene: Scene to use for frame size.
    :type scene: :class:`bpy.types.Scene`
    :arg obj: Camera object.
    :type obj: :class:`bpy.types.Object`
    :arg me: Untransformed Mesh.
    :type me: :class:`bpy.types.Mesh`
    :return: a Box object (call its to_tuple() method to get x, y, width and height)
    :rtype: :class:`Box`
    """

    mat = cam_ob.matrix_world.normalized().inverted()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh_eval = me_ob.evaluated_get(depsgraph)
    me = mesh_eval.to_mesh()
    me.transform(me_ob.matrix_world)
    me.transform(mat)

    camera = cam_ob.data
    frame = [-v for v in camera.view_frame(scene=scene)[:3]]
    camera_persp = camera.type != 'ORTHO'

    lx = []
    ly = []

    for v in me.vertices:
        co_local = v.co
        z = -co_local.z

        if camera_persp:
            if z == 0.0:
                lx.append(0.5)
                ly.append(0.5)
            # Does it make any sense to drop these?
            # if z <= 0.0:
            #    continue
            else:
                frame = [(v / (v.z / z)) for v in frame]

        min_x, max_x = frame[1].x, frame[2].x
        min_y, max_y = frame[0].y, frame[1].y

        x = (co_local.x - min_x) / (max_x - min_x)
        y = (co_local.y - min_y) / (max_y - min_y)

        lx.append(x)
        ly.append(y)

    min_x = clamp(min(lx), 0.0, 1.0)
    max_x = clamp(max(lx), 0.0, 1.0)
    min_y = clamp(min(ly), 0.0, 1.0)
    max_y = clamp(max(ly), 0.0, 1.0)

    mesh_eval.to_mesh_clear()

    r = scene.render
    fac = r.resolution_percentage * 0.01
    dim_x = r.resolution_x * fac
    dim_y = r.resolution_y * fac

    # Sanity check
    if round((max_x - min_x) * dim_x) == 0 or round((max_y - min_y) * dim_y) == 0:
        return (0, 0, 0, 0)

    return (
        min_x * dim_x,            # X
        dim_y - max_y * dim_y,    # Y
        (max_x - min_x) * dim_x,  # Width
        (max_y - min_y) * dim_y   # Height
    )

# Randomly place the camera within the scnee
def randomize_camera(camera):
    global img_meta
    focus_point = mathutils.Vector((0.0, 0.0, 0.0))
    focus_point[0] = (random.random() - 0.5) * 0.8
    focus_point[1] = (random.random() - 0.5) * 0.8
    img_meta = img_meta + "cam_focx_focy," + \
        str(focus_point[0]) + "," + \
        str(focus_point[1]) + "\n"

    camera.location[0] = random.random() * 1.0 - 0.5
    camera.location[1] = random.random() * 1.0 - 0.5
    camera.location[2] = random.random() * 0.4 + 0.2
    img_meta = img_meta + "cam_posx_posy_posz," + \
        str(camera.location[0]) + "," + \
        str(camera.location[1]) + "," + \
        str(camera.location[2]) + "\n"

    looking_direction = camera.location - focus_point
    rot_quat = looking_direction.to_track_quat('Z', 'Y')
    
    cam_dist = 0.3 + random.random() * 0.2
    camera.rotation_euler = rot_quat.to_euler()
    camera.location = rot_quat @ mathutils.Vector((0.0, 0.0, cam_dist))
    img_meta = img_meta + "cam_rotqx_rotqy_rotqz_rotqw," + \
        str(rot_quat[0]) + "," + \
        str(rot_quat[1]) + "," + \
        str(rot_quat[2]) + "," + \
        str(rot_quat[3]) + "\n"
    img_meta = img_meta + "cam_dist," + \
        str(cam_dist) + "\n"

# Recursively make the given object hidden/shown
# Does not affect the root node of the object
def setVisibility(should_hide, object):
    for child in object.children:
        child.hide_viewport = should_hide
        child.hide_render   = should_hide
        setVisibility(should_hide, child)

# Randomly place the given object within the scene
# Also shows/hides the object based on the 'should_hide" parameter
def randomize_object(should_hide, object):
    global img_meta
    object.hide_viewport = should_hide
    object.hide_render   = should_hide
    setVisibility(should_hide, object)
    
    # If this is the shown object
    if not should_hide:
        
        # Randomly position the object
        object.rotation_euler = (0, 0, random.random() * 2.0 * 3.14159265)
        object.location[0] = (random.random() - 0.5) * 0.1
        object.location[1] = (random.random() - 0.5) * 0.1
        
        # Get the evaluated mesh of the ground plane
        dg = bpy.context.evaluated_depsgraph_get()
        obj1 = object.evaluated_get(dg)
        obj2 = bpy.data.objects["Ground Mesh"].evaluated_get(dg)
        
        # Place the object on top of the ground
        object.location[2] = 0.1
        bpy.context.view_layer.update()
        overlap_prev = check_overlap(obj1, obj2)
        step_size = 0.02
        for i in range(5):
            step_size = step_size / 2.0
            for i in range(10):
                object.location[2] = object.location[2] + step_size * (1.0 if overlap_prev else -1.0)
                bpy.context.view_layer.update()
                if not overlap_prev == check_overlap(object, obj2):
                    overlap_prev = not overlap_prev
                    break
        
        # Doubly ensure that the positioning never fails
        if object.location[2] < -0.2:
            print("Bad location, retrying...")
            randomize_object(should_hide, object)
        
        # Save object's position and rotation to metadata
        img_meta = img_meta + "obj_posx_posy_posz," + \
            str(object.location[0]) + "," + \
            str(object.location[1]) + "," + \
            str(object.location[2]) + "\n"
        img_meta = img_meta + "obj_rotz," + \
            str(object.rotation_euler[2]) + "\n"
            
        # Save object's on-screen bounds to metadata
        img_meta = img_meta + "obj_bounds_2d," + str(camera_view_bounds_2d(
            bpy.context.scene, bpy.data.objects["Camera"], obj1)) \
            .replace(" ", "").replace("(", "").replace(")", "") + "\n"

# Actually randomize the entire scene, including objects + camera
# Could randomize any other desired parameters here
# Returns the class for the actually-shown object
# Shown object could be selected randomly, but is actually done in-order
# to ensure that all object classes are given an equal number of images
glob_index = 0
def randomize_scene():
    global glob_index, img_meta

    # Randomize environment HDRI
    env_image_idx = int(random.random() * len(env_img_files))
    env_tex_node.image = bpy.data.images.load(env_files_dir + "/" + env_img_files[env_image_idx])
    img_meta = img_meta + "env_image_idx," + str(env_image_idx) + "\n"
    
    # Randomize base object texture
    text_image_idx = int(random.random() * len(text_img_folders))
    bpy.data.images["ground_big_disp"].filepath = text_files_dir + "/" + text_img_folders[text_image_idx] + "/" + text_img_folders[text_image_idx] + "_disp_1k.jpg"
    text_mat_nodes["Displacement Image"].image = bpy.data.images.load(text_files_dir + "/" + text_img_folders[text_image_idx] + "/" + text_img_folders[text_image_idx] + "_disp_1k.jpg")
    text_mat_nodes["Displacement Image 2"].image = bpy.data.images.load(text_files_dir + "/" + text_img_folders[text_image_idx] + "/" + text_img_folders[text_image_idx] + "_disp_1k.jpg")
    text_mat_nodes["Base Image"].image = bpy.data.images.load(text_files_dir + "/" + text_img_folders[text_image_idx] + "/" + text_img_folders[text_image_idx] + "_diff_1k.jpg")
    text_mat_nodes["Roughness Image"].image = bpy.data.images.load(text_files_dir + "/" + text_img_folders[text_image_idx] + "/" + text_img_folders[text_image_idx] + "_roug_1k.jpg")
    text_mat_nodes["Normal Image"].image = bpy.data.images.load(text_files_dir + "/" + text_img_folders[text_image_idx] + "/" + text_img_folders[text_image_idx] + "_norm_1k.jpg")
    text_mat_nodes["Displacement Image"].image.colorspace_settings.name = "Non-Color"
    text_mat_nodes["Displacement Image 2"].image.colorspace_settings.name = "Non-Color"
    text_mat_nodes["Roughness Image"].image.colorspace_settings.name = "Non-Color"
    text_mat_nodes["Normal Image"].image.colorspace_settings.name = "Non-Color"
    img_meta = img_meta + "text_image_idx," + str(text_image_idx) + "\n"
    
    randomize_camera(bpy.data.objects['Camera'])
    index = glob_index
    glob_index = (glob_index + 1) % 8
    randomize_object(index != 0, bpy.data.objects['fork'])
    randomize_object(index != 1, bpy.data.objects['banana'])
    randomize_object(index != 2, bpy.data.objects['bulb'])
    randomize_object(index != 3, bpy.data.objects['glass'])
    randomize_object(index != 4, bpy.data.objects['plate'])
    randomize_object(index != 5, bpy.data.objects['teapot'])
    randomize_object(index != 6, bpy.data.objects['teacup'])
    randomize_object(index != 7, bpy.data.objects['spoon'])
    return index

# Next image number to write for the given class
writeIndexes = [-1, -1, -1, -1, -1, -1, -1, -1]

# Non-CLI: Just re-run the randomizer
if not RUN_FROM_CLI:
    img_meta = "";
    randomize_scene()
    print(img_meta)

# CLI: Write out N images for each of the 8 classes...
else:
    for i in range(8 * 64):
        img_meta = "";

        # Randomize the entire scene
        index = randomize_scene()

        # Select a path to save the image at
        while True:

            # Find or create the corresponding class folder
            writeFold = base_dir + "/data_generator_output/class" + str(index)
            if not os.path.exists(writeFold):
                os.makedirs(writeFold)

            # Find the next free sub-image path within that folder
            # This is done based on the 'writeIndexes' array
            writeIndexes[index] += 1
            writePath = writeFold + "/img" + str(writeIndexes[index]) + ".png"
            bpy.context.scene.render.filepath = os.path.join(writePath)
            if not os.path.isfile(writePath):
                break

        # Actually render (and write out) the image
        bpy.ops.render.render(write_still = True)
        
        # Save off the metadata, too
        writePathMeta = writeFold + "/meta" + str(writeIndexes[index]) + ".csv"
        with open(writePathMeta, "w") as f:
            f.write(img_meta)