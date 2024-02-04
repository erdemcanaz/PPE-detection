

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
    print(f"Start time: {start_time}")
    print(f"End time: {end_time}")
    print(f"\nID: {person_dict[0]['id']}  {start_time} - {end_time}")
    print(f"No of points in restricted area: {restricted_area_points}")
    print(f"No of points in mid area: {mid_area_points}")
    print(f"No of points outside restricted area: {outside_area_points}")
    print(f"Total points: {len(person_track_list)}")

    return True


