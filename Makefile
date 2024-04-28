run_on_all_frames:
	python visual_slam.py input/frames/
	
run_on_limited_frames:
	python visual_slam.py input/frames_limited/
	
type_check:
	mypy visual_slam.py
