import copy
import os

ENABLE_6DOF = True
NUM_FIELDS = 3 + (6 if ENABLE_6DOF else 2)

TEST_SIZE_WIDTH = 1024
TEST_SIZE_HEIGHT = 1024

USE_DEPTH = False

IN_CHANNELS = 3
SEGMENTATION_CHANNELS = 1

if USE_DEPTH:
	IN_CHANNELS = 4

BACKBONE = 'tu-convnext_base'

args = dict(

    threshold = 0.1,

	use_segmnetation = False,
	seg_threshold = 0.99,

	use_depth = USE_DEPTH,
	size=[(TEST_SIZE_HEIGHT,TEST_SIZE_WIDTH)],
	cuda=True,
	#top_k_predictions=1,

	enable_6dof_from_eulers=True,

	postprocessing_kwargs=dict(
		robot_distance_thr=None,		# robot reachable points (min,max - in meters) - not used! THIS IS IMPLEMENTED IN ROBOT CONTROLER INSTEAD
		pregrasp_aproach_depth=0.025, 	# 2.5cm additional depth of grasp (in meters)
	),

	model=dict(
		name='fpn',
		kwargs={
			'backbone': BACKBONE,
			'use_depth': USE_DEPTH,
			'num_classes': [NUM_FIELDS, 1, SEGMENTATION_CHANNELS],
			'use_custom_fpn': True,
			'add_output_exp': False,
			'in_channels': IN_CHANNELS,
			'fpn_args': {
				# 'decoder_dropout':0
				'decoder_segmentation_head_channels': 64,
				'classes_grouping': [(0,1,2,9), (3,6), (4,7), (5,8), (10,)] if ENABLE_6DOF else [(0, 1, 2, 5), (3, 4), (6,)],
				# mean synthetic+real
				'depth_mean': 0.96, 'depth_std': 0.075,
			},
			'init_decoder_gain': 0.1,
			'pretrained': False
		},
	),
	center_model=dict(
		name='CenterOrientationEstimator',
		use_learnable_center_estimation=True,

		kwargs=dict(
			use_centerdir_radii = False,

			# use vector magnitude as mask instead of regressed mask
			use_magnitude_as_mask=True,
			# thresholds for conv2d processing
			local_max_thr=0.01,
			mask_thr=0.01, hough_thr=5000, local_max_thr_use_abs=True,
			suppression_by_mask=False,
			### dilated neural net as head for center detection
			ignore_centerdir_magnitude=True,
			ignore_cls_prediction=True,
			use_dilated_nn=True,
			dilated_nn_args=dict(
				return_sigmoid=False,
				# single scale version (nn6)
				inner_ch=16,
				inner_kernel=3,
				dilations=[1, 4, 8, 12],
				use_centerdir_radii=False,
				use_centerdir_magnitude=False,
				use_cls_mask=False
			),
			scale_r=1.0,  # 1024
			scale_r_gt=1,  # 1
			use_log_r=False,
			use_log_r_base='10',
			enable_6dof=ENABLE_6DOF,
		),

	),
	num_vector_fields=NUM_FIELDS,
)

def get_args():
	return copy.deepcopy(args)
