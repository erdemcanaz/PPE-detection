import json
import Code.IP_camera.fetch_stream


#import the json file that contains the camera information password, username, ip address, and stream path
with open('secret_camera_info.json') as file:
    cameras = json.load(file)

which_camera = "koltuk_ambari"
username = cameras[which_camera]["username"]
password = cameras[which_camera]["password"]
ip_address = cameras[which_camera]["ip_address"]
stream_path = cameras[which_camera]["stream_path"]

#fetch single frame from camera
#Code.IP_camera.fetch_stream._capture_single_frame_from_ip_camera(camera_name = which_camera, username = username, password= password, ip_address = ip_address, stream_path = stream_path, VERBOSE = True)

#fetch video stream from camera
#Code.IP_camera.fetch_stream._display_single_ip_camera_stream(username = username, password= password, ip_address = ip_address, stream_path = stream_path)

