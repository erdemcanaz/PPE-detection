#INITIALIZATION
import json, os
import scripts.video_analyzer as video_analyzer
from datetime import datetime, timedelta, timezone
report_config = json.load(open('json/report_config.json', 'r'))

# Create a new folder to export the report and other related files. Note that it is created with a timestamp to avoid overwriting
new_folder_path = report_config["folder_to_export"] + datetime.now().strftime("_%Y_%m_%d_%H-%M-%S")
os.makedirs(new_folder_path)
print(f"Folder already exists, new folder created with timestamp: {new_folder_path}")
report_config["new_folder_path_dynamic_key"] = new_folder_path

video_analyzer_object = video_analyzer.videoAnalyzer()
video_start_date = datetime(
    year = report_config["video_start_date"]["year"],
    month = report_config["video_start_date"]["month"],
    day = report_config["video_start_date"]["day"],
    hour = report_config["video_start_date"]["hour"],
    second = report_config["video_start_date"]["second"],
    tzinfo= timezone(timedelta(hours=3))
    )
video_analyzer_object.import_video(report_config["video_path"], video_start_date = video_start_date)

#PRE-PROCESSING
from pre_process import pre_process
video_analyzer_object, pre_process_results = pre_process(video_analyzer_object, report_config)

#POST-PROCESSING
if report_config["check_restricted_area_violation"]:
    from post_process import post_process_restriced_area
    all_tracking_rows = post_process_restriced_area(
        pre_process_results = pre_process_results,
        report_config = report_config, 
        video_analyzer_object = video_analyzer_object,
    )
    
if report_config["check_hard_hat_violation"]:
    from post_process import post_process_hard_hat
    all_hard_hat_rows = post_process_hard_hat(
        pre_process_results = pre_process_results,
        report_config = report_config, 
        video_analyzer_object = video_analyzer_object,
    )
