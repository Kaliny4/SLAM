import cv2 # type: ignore
import numpy as np
import glob
import OpenGL.GL as gl # type: ignore
import collections
import argparse
import ThreeDimViewer
from icecream import ic

#from display import Display3D
#import g2opy as g2o

from Datatypes import *
from TrackedPoint import TrackedPoint
from TrackedCamera import TrackedCamera
from Frame import Frame
from FrameGenerator import FrameGenerator
from Observation import Observation
from ImagePair import ImagePair
from Map import Map

from custom_types import FrameID, FeatureID

np.set_printoptions(precision=4, suppress=True)

        



class VisualSlam:
    def __init__(self, inputdirectory, feature = "ORB"):
        self.input_directory = inputdirectory

        # Use ORB features
        self.detector = cv2.ORB_create()
        #self.detector = cv2.SIFT_create()
        #self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        self.frame_generator = FrameGenerator(self.detector)
        self.list_of_frames = []
        self.map = Map()

        self.feature_mapper = {}
        self.feature_history = {}


    def set_camera_matrix(self):
        self.camera_matrix = np.array([[2676, 0., 3840 / 2 - 35.24], 
            [0.000000000000e+00, 2676., 2160 / 2 - 279],
            [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])
        #self.camera_matrix = np.array([[835, 0., 1008 / 2 + 61], 
        #    [0.000000000000e+00, 835, 756 / 2 - 9],
        #    [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])
        #self.camera_matrix = np.array([[1750, 0., 1920 / 2 - 10], 
        #    [0.000000000000e+00, 1750, 1080 / 2 + 32],
        #    [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])


        # Calibration from input/KITTY_sequence_1/calib.txt
        #self.camera_matrix = np.array([[7.070912e+02, 0.e+00, 6.018873e+02], 
        #    [0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02],
        #    [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])


        #self.scale_factor = 0.2
        self.scale_factor = 0.3
        #self.scale_factor = 1
        self.camera_matrix *= self.scale_factor
        self.camera_matrix[2, 2] = 1

    def add_to_list_of_frames(self, image):
        frame = self.frame_generator.make_frame(image)
        self.list_of_frames.append(frame)


    def initialize_map(self, image_pair):
        self.map.clean()
        self.map.camera_matrix = self.camera_matrix

        ip = image_pair

        for match in ip.matches_with_3d_information:
            self.feature_mapper[match.featureid2] = match.featureid1
            # Save all features matches in a dictionary for easy 
            # access later on.
            self.feature_history[match.featureid1] = match

        # We have only seen two frame, to initialize a map.
        first_camera = TrackedCamera(
                np.eye(3), 
                np.zeros((3)), 
                ip.frame1.id,
                image_pair.frame1.image,
                fixed = True)
        second_camera = TrackedCamera(
                ip.R, 
                ip.t.T[0],
                ip.frame2.id,
                image_pair.frame2.image,
                fixed = False)
        first_camera = self.map.add_camera(first_camera)
        second_camera = self.map.add_camera(second_camera)

        for match in ip.matches_with_3d_information:
            tp = TrackedPoint(
                    match.point, 
                    match.descriptor1, 
                    match.color, 
                    match.featureid1)
            tp = self.map.add_point(tp)

            observation1 = Observation(tp.point_id, 
                    first_camera.camera_id, 
                    match.keypoint1)
            observation2 = Observation(tp.point_id, 
                    second_camera.camera_id, 
                    match.keypoint2)

            self.map.observations.append(observation1)
            self.map.observations.append(observation2)

        #self.map.remove_observations_with_reprojection_errors_above_threshold(100)
        self.map.optimize_map()
        #print("Hej")
        #self.map.calculate_reprojection_error(0.1)
        #self.map.remove_observations_with_reprojection_errors_above_threshold(1)
        #print("Hej")


    def track_feature_back_in_time(self, featureid: FeatureID) -> FeatureID:
        while featureid in self.feature_mapper:
            featureid_old = featureid
            featureid = self.feature_mapper[featureid]
            #print(featureid, " <- ", featureid_old)
        return featureid


    def add_new_match_to_map(self, match, first_camera, second_camera):
        tp = self.map.add_point_from_match(match)

        observation1 = Observation(
                tp.point_id, 
                first_camera.camera_id, 
                match.keypoint1)
        observation2 = Observation(
                tp.point_id, 
                second_camera.camera_id, 
                match.keypoint2)

        self.map.observations.append(observation1)
        self.map.observations.append(observation2)
        #print(observation1)
        #print(observation2)


    def add_new_observation_of_existing_point(
            self, featureid, match, camera):

        try:
            tp = self.mappointdict[featureid]
            observation = Observation(tp.point_id, 
                    camera.camera_id, 
                    match.keypoint2)
            self.map.observations.append(observation)
            #print(observation)
        except Exception as e:
            print("Exception in add_new_observation_of_existing_point")
            print(e)
            print(self.mappointdict)
            print("-----------")


    def add_point_observation_to_map(self, match, first_camera, second_camera):
        try:
            # Check to see if the point is in the map already
            featureid = self.track_feature_back_in_time(match.featureid2)

            if featureid in self.mappointdict:
                self.add_new_observation_of_existing_point(
                        featureid, match, second_camera)
            else:
                pass
                self.add_new_match_to_map(match, first_camera, second_camera)
        except Exception as e:
            print("Error in add_point_observation_to_map")
            print(match)
            print(e)
###################################################################
    def calculate_reprojection_errors(self):
        """Calculate and report the reprojection errors for each observation in the map."""
        errors = []
        for obs in self.map.observations:
            point_3d = self.map.points[obs.point_id]
            cam = self.map.cameras[obs.camera_id]
            projected_point, _ = cv2.projectPoints(np.array([point_3d.coords]), 
                                                cam.R, cam.t, self.camera_matrix, None)
            original_point = obs.keypoint.pt
            error = cv2.norm(projected_point, original_point, cv2.NORM_L2)
            errors.append(error)
        
        mean_error = np.mean(errors)
        print(f"Mean reprojection error: {mean_error}")
        return mean_error
####################################################################


    def add_information_to_map(self):
        self.reset_mappoint_dict()
        self.reset_camera_dict()
        ip = self.triangulate_points_in_current_image_pair()
        self.add_triangulated_points_to_map(ip)
#############################################################3
    def update_map_with_new_data(self):
        # Assume some process here to update the map
        self.add_information_to_map()

        # New Integration Point for Reprojection Error Calculation
        reprojection_error = self.calculate_reprojection_errors()
        print(f"Current mean reprojection error: {reprojection_error}")

        # Optional: Perform bundle adjustment if the error is above a threshold
        #if reprojection_error > some_threshold:
            #self.optimize_with_g2o()
########################################################

    def reset_mappoint_dict(self):
        # Make dict with all points in the current map
        self.mappointdict: dict[FeatureID, TrackedPoint] = {}
        for point in self.map.points:
            self.mappointdict[point.feature_id] = point


    def reset_camera_dict(self):
        # Update map with more points
        self.camera_dict: dict[FrameID, TrackedCamera] = {}
        for item in self.map.cameras:
            self.camera_dict[item.frame_id] = item
            print(item)


    def triangulate_points_in_current_image_pair(self):
        ip = self.current_image_pair
        essential_matches = ip.determine_essential_matrix(ip.filtered_matches)
        projection_matrix_one = self.camera_dict[ip.frame2.id].pose()
        projection_matrix_two = self.camera_dict[ip.frame1.id].pose()

        ip.reconstruct_3d_points(
                essential_matches, 
                projection_matrix_one[0:3, :], 
                projection_matrix_two[0:3, :])
                
        return ip


    def add_triangulated_points_to_map(self, ip: ImagePair):
        first_camera = self.camera_dict[ip.frame1.id]
        second_camera = self.camera_dict[ip.frame2.id]
        for match in ip.matches_with_3d_information:
            self.add_point_observation_to_map(match, first_camera, second_camera)


    def update_feature_mapper(self):
        for match in self.current_image_pair.matches_with_3d_information:
            self.feature_mapper[match.featureid2] = match.featureid1
            # Save all features matches in a dictionary for easy 
            # access later on.
            self.feature_history[match.featureid1] = match


    def estimate_current_camera_position(self, current_frame: Frame):
        self.update_feature_mapper()
        self.reset_mappoint_dict()

        if len(self.list_of_frames) < 3:
            return

        try: 
            retval, rvec, tvec = self.estimate_camera_position_in_map()

            if retval:
                self.reset_mappoint_dict()
                self.add_new_camera_to_map(current_frame, rvec, tvec)
                self.add_information_to_map()
            else:
                print("Failed to estimate the camera position")
                print("####################")


        except Exception as e:
            print("Position estimation failed")
            print(e)
            pass

        self.freeze_nonlast_cameras()
        #self.print_camera_details()
        self.map.optimize_map()
        self.map.remove_observations_with_reprojection_errors_above_threshold(1)
        #self.unfreeze_cameras()
        #self.map.optimize_map()


    def add_new_camera_to_map(self, current_frame: Frame, rvec: np.ndarray, tvec: np.ndarray):
        R, _ = cv2.Rodrigues(rvec)
        camera = TrackedCamera(
                        R,
                        tvec,
                        FrameID(current_frame.id),
                        current_frame.image,
                        fixed = False)
        camera = self.map.add_camera(camera)


    def estimate_camera_position_in_map(self):
        matches_with_map = self.find_matches_between_current_image_pair_and_map()
        #matches_with_map = self.find_matches_between_current_frame_and_map()

        print("Matches with map")
        print(len(matches_with_map))
        image_coords = self.get_image_coords_from_matches(matches_with_map)
        map_coords = self.get_map_coords_from_matches(matches_with_map)

        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                    np.array(map_coords),
                    np.array(image_coords),
                    self.camera_matrix,
                    np.zeros(4))

        return retval, rvec, tvec


    def find_matches_between_current_image_pair_and_map(self) -> list[MatchWithMap]:
        matches_with_map = []
        for match in self.current_image_pair.matches_with_3d_information:
            feature_id = self.track_feature_back_in_time(match.featureid2)

            if feature_id in self.mappointdict:
                # ic(feature_id)
                image_feature = self.feature_history[match.featureid1]
                map_feature = self.mappointdict[feature_id]
                t = MatchWithMap(image_feature.featureid2, 
                        map_feature.feature_id, 
                        image_feature.keypoint2,
                        map_feature.point, 
                        image_feature.descriptor2, 
                        map_feature.descriptor, 
                        0)   # TODO: Set to the proper feature distance
                matches_with_map.append(t)
        return matches_with_map


    def find_matches_between_current_frame_and_map(self) -> list[MatchWithMap]:
        matches_with_map = []
        map_descriptors = np.array([point.descriptor for point in self.map.points])
        map_feature_ids = [point.feature_id for point in self.map.points]
        temp = self.bf.match(self.current_image_pair.frame2.descriptors,
                map_descriptors)

        try:
            for match in temp:
                feature_in_image = self.current_image_pair.frame2.features[match.queryIdx]
                feature_id_in_map = map_feature_ids[match.trainIdx]
                feature_id_map_resolved = self.track_feature_back_in_time(feature_id_in_map)

                if feature_id_map_resolved in self.mappointdict:
                    image_feature = feature_in_image
                    map_feature = self.mappointdict[feature_id_map_resolved]
                    t = MatchWithMap(image_feature.feature_id,
                            map_feature.feature_id,
                            image_feature.keypoint.pt,
                            map_feature.point,
                            image_feature.descriptor,
                            map_feature.descriptor,
                            match.distance)
                    # TODO: Consider to do Lowe's filtering here
                    # TODO: Adjust this threshold
                    if(match.distance < 2000):
                        matches_with_map.append(t)
        except Exception as e:
            print(e)
            print("Failed to match features with map")
        return matches_with_map


    def get_image_coords_from_matches(self, matches_with_map: list[MatchWithMap]):
        return [match.imagecoord 
                for match
                in matches_with_map]

        
    def get_map_coords_from_matches(self, matches_with_map):
        return [match.mapcoord
                for match 
                in matches_with_map]


    def freeze_nonlast_cameras(self):
        for idx, camera in enumerate(self.map.cameras):
            self.map.cameras[idx].fixed = True
        self.map.cameras[-1].fixed = False
        if len(self.map.cameras) > 2:
            self.map.cameras[-2].fixed = False


    def print_camera_details(self):
        for camera in self.map.cameras:
            print(camera)


    def unfreeze_cameras(self, number_of_fixed_cameras = 5):
        for idx, camera in enumerate(self.map.cameras):
            if idx > number_of_fixed_cameras:
                self.map.cameras[idx].fixed = False
            else:
                self.map.cameras[idx].fixed = True
##########################################################################

    def calculate_epipolar_distances(self, matches, F):
        """Calculate distances from points to their corresponding epipolar lines."""
        points1 = np.array([self.frame_generator.make_frame(m.img1).keypoints[m.queryIdx].pt for m in matches])
        points2 = np.array([self.frame_generator.make_frame(m.img2).keypoints[m.trainIdx].pt for m in matches])
        
        # Calculate the lines on the second image
        lines = cv2.computeCorrespondEpilines(points1.reshape(-1, 1, 2), 1, F)
        lines = lines.reshape(-1, 3)
        
        # Calculate distances from points to the lines
        distances = np.abs((lines[:, 0]*points2[:, 0] + lines[:, 1]*points2[:, 1] + lines[:, 2]) /
                        np.sqrt(lines[:, 0]**2 + lines[:, 1]**2))
        
        mean_distance = np.mean(distances)
        std_dev = np.std(distances)
        
        print(f"Mean epipolar distance: {mean_distance}, Std dev: {std_dev}")
        return mean_distance, std_dev
#####################################################################################

    def match_current_and_previous_frame(self):
        if len(self.list_of_frames) < 2:
            return

        frame1 = self.list_of_frames[-2]
        frame2 = self.list_of_frames[-1]
        self.current_image_pair = ImagePair(frame1, frame2, self.bf, self.camera_matrix)
        self.current_image_pair.match_features()
        essential_matrix = self.current_image_pair.determine_essential_matrix(self.current_image_pair.filtered_matches)

        # New Integration Point for Epipolar Distance Calculation
        _, epipolar_distances = self.calculate_epipolar_distances(self.current_image_pair.filtered_matches, essential_matrix)

        self.current_image_pair.estimate_camera_movement(essential_matrix)
        self.current_image_pair.reconstruct_3d_points(essential_matrix)

        if len(self.list_of_frames) == 2:
            self.initialize_map(self.current_image_pair)

        image_to_show = self.current_image_pair.visualize_matches(essential_matrix)
        cv2.imshow("matches", image_to_show)
###############################################################################

        self.estimate_current_camera_position(frame2)

        self.freeze_nonlast_cameras()
        #self.print_camera_details()

        self.map.limit_number_of_camera_in_map(18)
        cv2.waitKey(100)


    def show_3d_visualization(self):
        viewport = ThreeDimViewer.ThreeDimViewer()
        viewport.vertices = [point.point
                    for point
                    in self.map.points]
        viewport.colors = [point.color
                    for point 
                    in self.map.points]

        viewport.cameras = [camera
                    for camera
                    in self.map.cameras]
        viewport.main()


    def show_map_points(self, message):
        print(message)
        for element in self.map.points:
            print(element)


    def process_frame(self, frame):
        self.add_to_list_of_frames(frame)
        self.match_current_and_previous_frame()
        self.show_3d_visualization()
        return frame
#################################################################################
    def match_features_with_map(self, new_frame):
        """Match features from a new frame with the existing map."""
        new_descriptors = self.frame_generator.make_frame(new_frame).descriptors
        # Assume map_descriptors aggregates descriptors from all points in the map
        matches = self.bf.match(new_descriptors, self.map_descriptors)
        # Filter matches and estimate the camera pose
        self.estimate_camera_pose_from_matches(matches)
        
        
    def process_new_frame(self, new_frame):
        self.add_to_list_of_frames(new_frame)
        if len(self.list_of_frames) > 1:
            self.match_current_and_previous_frame()
        self.show_3d_visualization()

        # New Integration for Continuous Mapping
        self.match_features_with_map(new_frame)

        # After matching, possibly update the map
        self.update_map_with_new_data()

#####################################################################################
# Bundle Adjustment with g2o ????

    def optimize_map(self, postfix = ""):
            optimizer = g2o.SparseOptimizer()
            solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
            solver = g2o.OptimizationAlgorithmLevenberg(solver)
            optimizer.set_algorithm(solver)

            # Define camera parameters
            print(self.camera_matrix)
            #focal_length = 1000
            focal_length = self.camera_matrix[0, 0]
            #principal_point = (320, 240)
            principal_point = (self.camera_matrix[0, 2], self.camera_matrix[1, 2])
            baseline = 0
            cam = g2o.CameraParameters(focal_length, principal_point, baseline)
            cam.set_id(0)
            optimizer.add_parameter(cam)

            camera_vertices = {}
            for camera in self.cameras:
                # Use the estimated pose of the second camera based on the 
                # essential matrix.
                pose = g2o.SE3Quat(camera.R, camera.t)

                # Set the poses that should be optimized.
                # Define their initial value to be the true pose
                # keep in mind that there is added noise to the observations afterwards.
                v_se3 = g2o.VertexSE3Expmap()
                v_se3.set_id(camera.camera_id)
                v_se3.set_estimate(pose)
                v_se3.set_fixed(camera.fixed)
                optimizer.add_vertex(v_se3)
                camera_vertices[camera.camera_id] = v_se3
                #print("camera id: %d" % camera.camera_id)

            point_vertices = {}
            for point in self.points:
                # Add 3d location of point to the graph
                vp = g2o.VertexPointXYZ()
                vp.set_id(point.point_id)
                vp.set_marginalized(True)
                # Use positions of 3D points from the triangulation
                point_temp = np.array(point.point, dtype=np.float64)
                vp.set_estimate(point_temp)
                optimizer.add_vertex(vp)
                point_vertices[point.point_id]= vp


            for observation in self.observations:
                # Add edge from first camera to the point
                edge = g2o.EdgeProjectXYZ2UV()

                # 3D point
                edge.set_vertex(0, point_vertices[observation.point_id]) 
                # Pose of first camera
                edge.set_vertex(1, camera_vertices[observation.camera_id]) 
                
                edge.set_measurement(observation.image_coordinates)
                edge.set_information(np.identity(2))
                edge.set_robust_kernel(g2o.RobustKernelHuber())

                edge.set_parameter_id(0, 0)
                optimizer.add_edge(edge)

            print('num vertices:', len(optimizer.vertices()))
            print('num edges:', len(optimizer.edges()))

            print('Performing full BA:')
            optimizer.initialize_optimization()
            optimizer.set_verbose(True)
            #optimizer.save("ba0%s.g2o" % postfix);
            optimizer.optimize(140)

            for idx, camera in enumerate(self.cameras):
                t = camera_vertices[camera.camera_id].estimate().translation()
                self.cameras[idx].t = t
                q = camera_vertices[camera.camera_id].estimate().rotation()
                self.cameras[idx].R = quarternion_to_rotation_matrix(q)

            for idx, point in enumerate(self.points):
                p = point_vertices[point.point_id].estimate()
                # It is important to copy the point estimates.
                # Otherwise I end up with some memory issues.
                # self.points[idx].point = p
                self.points[idx].point = np.copy(p)

##################################################################
    def run(self):
        list_of_files = glob.glob("%s/*.jpg" % self.input_directory)
        list_of_files.sort()
        if len(list_of_files) == 0:
            print("No images found in the specified directory")
            return
        for idx, filename in enumerate(list_of_files):
            print(filename)
            img = cv2.imread(filename)
            cv2.imshow("test", img)

            scale_percent = 30 # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)

            img = cv2.resize(img, dim)
            frame = self.process_frame(img)
            self.map.show_map_statistics()

            #cv2.imshow("test", frame);
            k = cv2.waitKey(400000)
            if k == ord('q'):
                break
            if k == ord('p'):
                k = cv2.waitKey(100000)
            if k == ord('b'):
                # Perform bundle adjusment
                self.unfreeze_cameras(5)
                #self.print_camera_details()
                self.map.optimize_map()
                self.freeze_nonlast_cameras()
                #self.print_camera_details()

        while True:
            k = cv2.waitKey(100)
            if k == ord('q'):
                break


parser = argparse.ArgumentParser(description='Visual Slam.')
parser.add_argument('directory', type=str, 
                    help='directory with frames')
args = parser.parse_args()

vs = VisualSlam(args.directory)
vs.set_camera_matrix()
vs.run()