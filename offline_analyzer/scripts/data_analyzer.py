import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
from io import BytesIO
import matplotlib


def is_direct_passage(person_dict:dict, min_data_points: int, min_confidence: float, x_min_threshold:float, x_max_threshold:float):
    """
    This function checks if a person has directly passed through a restricted area
    """

    #sort the person_dict by timestamp
    person_tracking_data = sorted(person_dict, key = lambda x: float(x["timestamp"])) #list of dictionaries
    
    person_track_list = []
    for data_dict in person_tracking_data:
        confidence = float(data_dict["box_confidence"]) 
        person_x = float(data_dict["person_x"]) if data_dict["person_x"] != "" else None
        person_y = float(data_dict["person_y"]) if data_dict["person_y"] != "" else None
        person_z = float(data_dict["person_z"]) if data_dict["person_z"] != "" else None
        state = int(data_dict["state"])
        timestamp = float(data_dict["timestamp"])

        person_track_list.append([person_x, person_y, person_z, state, confidence, timestamp ])

    #calculate number of points in the restricted area and the number of points outside the restricted area
    restricted_area_points = 0
    mid_area_points = 0
    outside_area_points = 0


    starting_region = None
    for person_track in person_track_list:
        if person_track[0] is None:
            continue
        if person_track[4] < min_confidence:
            continue
        if person_track[0] > x_max_threshold:
            starting_region = "right"
            break
        elif person_track[0] < x_min_threshold:
            starting_region = "left"
            break
        else:
            starting_region = "mid"
            break
    ending_region = None
    for person_track in reversed(person_track_list):
        if person_track[0] is None:
            continue
        if person_track[4] < min_confidence:
            continue

        if person_track[0] > x_max_threshold:
            ending_region = "right"
            break
        elif person_track[0] < x_min_threshold:
            ending_region = "left"
            break
        else:
            ending_region = "mid"
            break
    
    start_and_end = f"Start: {starting_region}, End: {ending_region}"

    for person_track in person_track_list:
        if person_track[0] is None:
            continue
        if person_track[4] < min_confidence:
            continue

        if person_track[0] > x_max_threshold:
            restricted_area_points += 1
        elif person_track[0] < x_min_threshold:
            outside_area_points += 1
        else:
            mid_area_points += 1
    
    format_to_hh_mm_ss = lambda seconds: f"{int(seconds//3600):02d}:{int((seconds%3600)//60):02d}:{int(seconds%60):02d}"
    start_time = format_to_hh_mm_ss(person_track_list[0][5])
    end_time = format_to_hh_mm_ss(person_track_list[-1][5])

    if restricted_area_points<=min_data_points or outside_area_points<=min_data_points or mid_area_points<=min_data_points:
        return False
    
    print(f"\nID: {person_dict[0]['id']}  {start_time} - {end_time}")
    print(f"({(starting_region != ending_region)and(starting_region != 'mid' and ending_region != 'mid')})")
    print(f"Start time: {start_time}")
    print(f"Start and End: {start_and_end}")
    print(f"End time: {end_time}")
    print(f"No of points in restricted area: {restricted_area_points}")
    print(f"No of points in mid area: {mid_area_points}")
    print(f"No of points outside restricted area: {outside_area_points}")
    print(f"Total points: {len(person_track_list)}")

    return True

def calculate_direct_passage_likelyhood(person_dict:dict, min_data_points: int, min_confidence: float, x_border:float, x_border_width:float)-> float:
    """
    This function checks if a person has directly passed through a restricted area
    """

    #sort the person_dict by timestamp
    person_tracking_data = sorted(person_dict, key = lambda x: float(x["timestamp"])) #list of dictionaries
    
    person_track_list = []
    for data_dict in person_tracking_data:
        confidence = float(data_dict["box_confidence"]) 
        person_x = float(data_dict["person_x"]) if data_dict["person_x"] != "" else None
        person_y = float(data_dict["person_y"]) if data_dict["person_y"] != "" else None
        person_z = float(data_dict["person_z"]) if data_dict["person_z"] != "" else None
        state = int(data_dict["state"])
        timestamp = float(data_dict["timestamp"])

        person_track_list.append([person_x, person_y, person_z, state, confidence, timestamp ])


    x_max = -1e6 #a very small number
    x_min = 1e6 #a very large number

    for person_track in person_track_list:
        if person_track[0] is None:
            continue
        if person_track[4] < min_confidence:
            continue

        x_max = max(x_max, person_track[0])
        x_min = min(x_min, person_track[0])
    
    format_to_hh_mm_ss = lambda seconds: f"{int(seconds//3600):02d}:{int((seconds%3600)//60):02d}:{int(seconds%60):02d}"
    start_time = format_to_hh_mm_ss(person_track_list[0][5])
    end_time = format_to_hh_mm_ss(person_track_list[-1][5])

    x_max = min(x_max, x_border+x_border_width)
    x_min = max(x_min, x_border-x_border_width)

    x_right = x_max - x_border
    x_left = x_border-x_min

    likelihood = x_right*x_left


    quarter_border_width = x_border_width/4
    starting_region = None
    for person_track in person_track_list:
        if person_track[0] is None:
            continue
        if person_track[4] < min_confidence:
            continue
        if person_track[0] > x_border:
            starting_region = "right"
            break
        elif person_track[0] < x_border:
            starting_region = "left"
            break
        else:
            continue
    
    ending_region = None
    for person_track in reversed(person_track_list):
        if person_track[0] is None:
            continue
        if person_track[4] < min_confidence:
            continue
        if person_track[0] > x_border:
            ending_region = "right"
            break
        elif person_track[0] < x_border:
            ending_region = "left"
            break
        else:
            continue

    k = 1
    if starting_region == "left" and ending_region == "right":
        k = 1
    elif starting_region == "right" and ending_region == "left":
        k = 1
    else:
        k = 0.5

    likelihood = likelihood*k

    print(f"\nID: {person_dict[0]['id']}  {start_time} - {end_time}")
    print(f"Likelyhood: {likelihood} with k: {k}")
    print(f"X_min: {x_min}, X_max: {x_max}")
    print(f"Start time: {start_time}")
    print(f"End time: {end_time}")
    print(f"Total points: {len(person_track_list)}")

    #visualize moves 
    x_vals = []
    y_vals = []
    for person_track in person_track_list:
        if person_track[0] is None:
            continue
        if person_track[4] < min_confidence:
            continue

        x_vals.append(person_track[0])
        y_vals.append(person_track[1])

    return {
        "id":person_dict[0]['id'],
        "likelihood":likelihood,
        "k":k,
        "x_vals":x_vals,
        "y_vals":y_vals,
        "start_seconds":person_track_list[0][5],
        "end_seconds":person_track_list[-1][5],
        "start_time":start_time,
        "end_time":end_time
        }

def plot_xy_tracking_results(x_vals :list, y_vals :list, title:str):   
    x_lims = (0,10)
    y_lims = (0,10)
    # Load the background image
    background_img = mpimg.imread('C:\\Users\\Levovo20x\\Documents\\GitHub\\PPE-detection\\secret_2D_maps\\koltuk_ambari_2_map_rotated.png')

    # Plot the background image
    plt.imshow(background_img, extent=[0, 10, 0, 10])

    # Add circle at the first node
    plt.scatter(x_vals[0], y_vals[0], marker='o', color='r')

    # Add cross at the last node
    plt.scatter(x_vals[-1], y_vals[-1], marker='x', color='b')

    # Set the x and y limits
    plt.xlim(x_lims[0], x_lims[1])
    plt.ylim(y_lims[0], y_lims[1])

    # Plot the data points
    plt.plot(x_vals, y_vals)
    plt.title(title)
    plt.show()

def return_xy_tracking_results_frame(x_vals: list, y_vals: list, title: str):
    x_lims = (0, 10)
    y_lims = (0, 10)
    # Load the background image
    background_img = mpimg.imread('C:\\Users\\Levovo20x\\Documents\\GitHub\\PPE-detection\\secret_2D_maps\\koltuk_ambari_2_map_rotated.png')

    # Create a figure and axis to plot onto
    fig, ax = plt.subplots()

    # Plot the background image
    ax.imshow(background_img, extent=[0, 10, 0, 10])

    # Add circle at the first node
    ax.scatter(x_vals[0], y_vals[0], marker='o', color='r')

    # Add cross at the last node
    ax.scatter(x_vals[-1], y_vals[-1], marker='x', color='b')

    # Set the x and y limits
    ax.set_xlim(x_lims[0], x_lims[1])
    ax.set_ylim(y_lims[0], y_lims[1])

    # Plot the data points
    ax.plot(x_vals, y_vals)
    ax.set_title(title)

    # Save the plot to a BytesIO buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Convert buffer to OpenCV image
    img_buf_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(img_buf_arr, 1)

    # Clean up the plt figure to prevent memory leak
    plt.close(fig)

    return img

def apply_mean_filter(x_vals:list[float], window_size:int):
    # Calculate the half window size, for the number of elements on each side of the current element
    half_window = window_size // 2
    
    # Initialize the list for the filtered values
    filtered_vals = []
    
    # Loop through each element in the list
    for i in range(len(x_vals)):
        # Determine the start and end of the window for the current element
        start = max(0, i - half_window)
        end = min(len(x_vals), i + half_window + 1)
        
        # Calculate the mean of the window and append to the filtered list
        window_mean = sum(x_vals[start:end]) / (end - start)
        filtered_vals.append(window_mean)
    
    return filtered_vals

