import copy
import os

import torch
from utils import transforms as my_transforms

CLOTH_COMPETITION_ICRA2024_DATASET = os.environ.get('CLOTH_COMPETITION_ICRA2024_DATASET', default='./dataset')

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', default='./exp')
BACKBONE = os.environ.get('CLOTH_COMPETITION_BACKBONE', 'tu-convnext_base')
TRAIN_SIZE_HEIGHT = 768
TRAIN_SIZE_WIDTH = 768
TEST_SIZE_HEIGHT = 1024
TEST_SIZE_WIDTH = 1024
NUM_FIELDS = 9
SEGMENTATION_CHANNELS = 1
RETURN_DATASET_DEPTH = True

DEPTH_ENCODING = os.environ.get('DEPTH_ENCODING', 'off').lower()
if DEPTH_ENCODING not in ['off', '3d_arm_plane']:
    raise ValueError(f'Unsupported DEPTH_ENCODING={DEPTH_ENCODING!r}, expected one of: off, 3d_arm_plane')

DEPTH_STR = DEPTH_ENCODING
USE_ARM_BASE_PLANE_DEPTH = DEPTH_ENCODING == '3d_arm_plane'
USE_DEPTH = DEPTH_ENCODING != 'off'
DATASET_DEPTH_ENCODING = 'arm_bases_plane' if USE_ARM_BASE_PLANE_DEPTH else ('raw' if USE_DEPTH else False)

CROP_MODE = os.environ.get('CROP_MODE', os.environ.get('CLOTH_COMPETITION_CROP_MODE', 'robot')).lower()
if CROP_MODE in ['0', 'false', 'none', 'no', 'off']:
    CROP_MODE = 'off'
elif CROP_MODE in ['1', 'true', 'yes', 'robot', 'workspace']:
    CROP_MODE = 'robot'
else:
    raise ValueError(f'Unsupported CROP_MODE={CROP_MODE!r}, expected robot/off')
CROP_BY_ROBOT_POSITIONS = (TEST_SIZE_WIDTH, TEST_SIZE_HEIGHT) if CROP_MODE == 'robot' else False

model_dir = os.path.join(
	OUTPUT_DIR,
	'cloth_6dof',
	'6dof-euler',
	'backbone={args[model][kwargs][backbone]}',
	'depth={args[depth_str]}',
	'crop={args[crop_mode]}',
)


DATASET_SUBSET0001 = {
	'reference_00': ['sample_000000', 'sample_000001', 'sample_000002', 'sample_000003', 'sample_000004', 'sample_000005', 'sample_000006', 'sample_000007', 'sample_000008', 'sample_000009', 'sample_000010', 'sample_000011', 'sample_000012'], 'reference_01': ['sample_000013', 'sample_000014', 'sample_000015', 'sample_000016', 'sample_000017', 'sample_000018', 'sample_000019', 'sample_000020', 'sample_000021', 'sample_000022', 'sample_000023'],
	'reference_02': ['sample_000024', 'sample_000025', 'sample_000026', 'sample_000027', 'sample_000028', 'sample_000029', 'sample_000030', 'sample_000031', 'sample_000032', 'sample_000033'],
	'reference_03': ['sample_000034', 'sample_000035', 'sample_000036', 'sample_000037', 'sample_000038', 'sample_000039', 'sample_000040', 'sample_000041', 'sample_000042', 'sample_000043'],
	'reference_04': ['sample_000044', 'sample_000045', 'sample_000046', 'sample_000047', 'sample_000048', 'sample_000049', 'sample_000050', 'sample_000051', 'sample_000052', 'sample_000053', 'sample_000054', 'sample_000055'],
	'reference_05': ['sample_000056', 'sample_000057', 'sample_000058', 'sample_000059', 'sample_000060', 'sample_000061', 'sample_000062', 'sample_000063', 'sample_000064', 'sample_000065'],
	'reference_06': ['sample_000066', 'sample_000067', 'sample_000068', 'sample_000069', 'sample_000070', 'sample_000071', 'sample_000072', 'sample_000073', 'sample_000074', 'sample_000075'],
	'reference_07': ['sample_000076', 'sample_000077', 'sample_000078', 'sample_000079', 'sample_000080', 'sample_000081', 'sample_000082', 'sample_000083', 'sample_000084', 'sample_000085', 'sample_000086'],
	'reference_08': ['sample_000087', 'sample_000088', 'sample_000089', 'sample_000090', 'sample_000091', 'sample_000092', 'sample_000093', 'sample_000094', 'sample_000095', 'sample_000096', 'sample_000097', 'sample_000098'],
	'reference_09': ['sample_000099', 'sample_000100', 'sample_000101', 'sample_000102', 'sample_000103', 'sample_000104', 'sample_000105', 'sample_000106', 'sample_000107', 'sample_000108', 'sample_000109', 'sample_000110', 'sample_000111', 'sample_000112'],
	'reference_10': ['sample_000113', 'sample_000114', 'sample_000115', 'sample_000116', 'sample_000117', 'sample_000118', 'sample_000119', 'sample_000120', 'sample_000121', 'sample_000122', 'sample_000123', 'sample_000124', 'sample_000125'],
	'reference_11': ['sample_000126', 'sample_000127', 'sample_000128', 'sample_000129', 'sample_000130', 'sample_000131', 'sample_000132', 'sample_000133', 'sample_000134', 'sample_000135', 'sample_000136', 'sample_000137'],
	'reference_12': ['sample_000138', 'sample_000139', 'sample_000140', 'sample_000141', 'sample_000142', 'sample_000143', 'sample_000144', 'sample_000145', 'sample_000146', 'sample_000147'],
	'reference_13': ['sample_000148', 'sample_000149', 'sample_000150', 'sample_000151', 'sample_000152', 'sample_000153', 'sample_000154', 'sample_000155', 'sample_000156', 'sample_000157', 'sample_000158'],
	'reference_14': ['sample_000159', 'sample_000160', 'sample_000161', 'sample_000162', 'sample_000163', 'sample_000164', 'sample_000165', 'sample_000166', 'sample_000167', 'sample_000168', 'sample_000169', 'sample_000170', 'sample_000171', 'sample_000172', 'sample_000173'],
	'reference_15': ['sample_000174', 'sample_000175', 'sample_000176', 'sample_000177', 'sample_000178', 'sample_000179', 'sample_000180', 'sample_000181', 'sample_000182', 'sample_000183', 'sample_000184', 'sample_000185'],
	'reference_16': ['sample_000186', 'sample_000187', 'sample_000188', 'sample_000189', 'sample_000190', 'sample_000191', 'sample_000192', 'sample_000193', 'sample_000194', 'sample_000195', 'sample_000196', 'sample_000197', 'sample_000198', 'sample_000199'],
	'reference_17': ['sample_000200', 'sample_000201', 'sample_000202', 'sample_000203', 'sample_000204', 'sample_000205', 'sample_000206', 'sample_000207', 'sample_000208', 'sample_000209', 'sample_000210', 'sample_000211'],
	'reference_18': ['sample_000212', 'sample_000213', 'sample_000214', 'sample_000215', 'sample_000216', 'sample_000217', 'sample_000218', 'sample_000219', 'sample_000220', 'sample_000221', 'sample_000222', 'sample_000223', 'sample_000224', 'sample_000225'],
	'reference_19': ['sample_000226', 'sample_000227', 'sample_000228', 'sample_000229', 'sample_000230', 'sample_000231', 'sample_000232', 'sample_000233', 'sample_000234', 'sample_000235'],
	'reference_20': ['sample_000236', 'sample_000237', 'sample_000238', 'sample_000239', 'sample_000240', 'sample_000241', 'sample_000242', 'sample_000243', 'sample_000244'],
	'reference_21': ['sample_000245', 'sample_000246', 'sample_000247', 'sample_000248', 'sample_000249', 'sample_000250', 'sample_000251', 'sample_000252', 'sample_000253', 'sample_000254', 'sample_000255', 'sample_000256', 'sample_000257', 'sample_000258', 'sample_000259', 'sample_000260', 'sample_000261', 'sample_000262', 'sample_000263', 'sample_000264', 'sample_000265', 'sample_000266', 'sample_000267', 'sample_000268'],
	'reference_22': [],
	'reference_23': ['sample_000269', 'sample_000270', 'sample_000271', 'sample_000272', 'sample_000273', 'sample_000274', 'sample_000275', 'sample_000276', 'sample_000277', 'sample_000278', 'sample_000279', 'sample_000280', 'sample_000281', 'sample_000282', 'sample_000283', 'sample_000284', 'sample_000285', 'sample_000286', 'sample_000287', 'sample_000288'],
	'reference_24': ['sample_000289', 'sample_000290', 'sample_000291', 'sample_000292', 'sample_000293', 'sample_000294', 'sample_000295', 'sample_000296', 'sample_000297', 'sample_000298', 'sample_000299', 'sample_000300', 'sample_000301', 'sample_000302'],
	'reference_25': ['sample_000303', 'sample_000304', 'sample_000305', 'sample_000306', 'sample_000307', 'sample_000308', 'sample_000309', 'sample_000310', 'sample_000311', 'sample_000312', 'sample_000313'],
	'reference_26': ['sample_000314', 'sample_000315', 'sample_000316', 'sample_000317', 'sample_000318', 'sample_000319', 'sample_000320', 'sample_000321', 'sample_000322', 'sample_000323', 'sample_000324', 'sample_000325', 'sample_000326', 'sample_000327'],
	'reference_27': ['sample_000328', 'sample_000329', 'sample_000330', 'sample_000331', 'sample_000332', 'sample_000333', 'sample_000334', 'sample_000335', 'sample_000336', 'sample_000337', 'sample_000338', 'sample_000339', 'sample_000340', 'sample_000341', 'sample_000342', 'sample_000343', 'sample_000344', 'sample_000345', 'sample_000346'],
	'reference_28': ['sample_000347', 'sample_000348', 'sample_000349', 'sample_000350', 'sample_000351', 'sample_000352', 'sample_000353', 'sample_000354', 'sample_000355', 'sample_000356', 'sample_000357', 'sample_000358', 'sample_000359', 'sample_000360', 'sample_000361', 'sample_000362', 'sample_000363', 'sample_000364', 'sample_000365'],
	'reference_29': ['sample_000366', 'sample_000367', 'sample_000368', 'sample_000369', 'sample_000370', 'sample_000371', 'sample_000372', 'sample_000373', 'sample_000374', 'sample_000375', 'sample_000376', 'sample_000377']
}

TEST_SAMPLES = [sample for f in ['reference_08','reference_14','reference_25'] for sample in DATASET_SUBSET0001[f]]


args = dict(

	cuda=True,
	display=False,
	autoadjust_figure_size=True,
	# display to file only
	display_to_file_only=True,
	groundtruth_loading = True,

	depth_str=DEPTH_STR,
	crop_mode=CROP_MODE,
	save_dir=os.path.join(model_dir,'{args[dataset][kwargs][type]}_results{args[eval_epoch]}'),
	checkpoint_path=os.path.join(model_dir,'checkpoint{args[eval_epoch]}.pth'),

	eval_epoch='',

	eval=dict(
		score_combination_and_thr=[
			{
				'center': [0.1, 0.01, 0.05, 0.15, 0.2, 0.25, 0.3, 0.35, 0.40, 0.45, 0.5, 0.55, 0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.94, 0.99],
			},
		],
		score_thr_final=[0.01],
		top_k_predictions=1,
		skip_center_eval=True,
		orientation=dict(
			display_best_threshold=False,
			tau_thr=[20]
		),
	),
	visualizer=dict(
		name='OrientationVisualizeTest',
		opts=dict(
			show_rot_axis=(True, True, True),
			axis_from_eulers=True,
			show_regressed_maps=False,
			show_gt_maps=False,
			plot_rank_number=True,
			bgr_axis=True,
			impath2name_fn=lambda x: ".".join(x.split('/')[-4:]).replace('.jpg', ''),
		),
	),
	dataset={
		'name': 'cloth_competition_icra2024',
		'kwargs': {
			'normalize': False,
			'root_dir': os.path.abspath(CLOTH_COMPETITION_ICRA2024_DATASET),

			'subfolders': ['cloth_competition_dataset_0000','cloth_competition_dataset_0001'],
			'type': 'test',
            'valid_sample_names': [f'cloth_competition_dataset_0000/sample_{i:06}' for i in range(100,125)] +
								  [f'cloth_competition_dataset_0001/{s}' for s in TEST_SAMPLES],
			'fixed_bbox_size': 5,
			'resize_factor': 1,
			'remove_pregrasp_depth': True,
			'use_depth': DATASET_DEPTH_ENCODING if USE_DEPTH else 'raw',
			'integrate_depth_with_image': RETURN_DATASET_DEPTH == (True if USE_DEPTH else False),
			'use_mean_for_depth_nan': True,
			'check_consistency': False,
			'use_6dof_pose': True,
			'orientation_as_euler_angles': True,
			'segment_cloth': True,
            'segment_cloth_optional': False, # evaluation must not silently use fabricated empty masks
			'crop_by_robot_positions': CROP_BY_ROBOT_POSITIONS,
			'transform': my_transforms.get_transform([
				{
					'name': 'ToTensor',
					'opts': {
						'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask', 'segmentation_mask') + (('depth',) if RETURN_DATASET_DEPTH else ()),
						'type': (
							torch.FloatTensor,
							torch.ShortTensor,
							torch.ByteTensor,
							torch.ByteTensor,
							torch.FloatTensor,
							torch.ByteTensor,
							torch.FloatTensor,
						) + ((torch.FloatTensor,) if RETURN_DATASET_DEPTH else ()),
					}
				},
			]),
			'MAX_NUM_CENTERS':2*1024,
		},
		'centerdir_gt_opts': dict(
			ignore_instance_mask_and_use_closest_center=True,

			MAX_NUM_CENTERS=2*1024,
		),

		'batch_size': 1,
		'workers': 0,
	},

	model=dict(
		name='fpn',
		kwargs={
			'backbone': BACKBONE,
			'use_depth': True if USE_DEPTH else False,
			'use_only_depth': False,
			'num_classes': [NUM_FIELDS, 1, SEGMENTATION_CHANNELS],
			'use_custom_fpn': True,
			'fpn_args': {
				'classes_grouping': [(0, 1, 2, 9), (3, 6), (4, 7), (5, 8), (10,)],
				'upsampling': 4,
				'depth_mean': 0 if USE_ARM_BASE_PLANE_DEPTH else 0.96,
				'depth_std': 1.0 if USE_ARM_BASE_PLANE_DEPTH else 0.075,
			},
		},
	),
	center_model=dict(
		name='CenterOrientationEstimator',
		use_learnable_center_estimation=True,

		kwargs=dict(
			# use vector magnitude as mask instead of regressed mask
			use_magnitude_as_mask=True,
			# thresholds for conv2d processing
			local_max_thr=0.01,
			suppression_by_mask=False,
			### dilated neural net as head for center detection
			ignore_centerdir_magnitude=True,
			ignore_cls_prediction=True,
			use_dilated_nn=True,
			dilated_nn_args=dict(
				inner_kernel=3,
				dilations=[1, 4, 8, 12],
				use_centerdir_magnitude=False,
				use_cls_mask=False
				),
			augmentation_name='CenterAugmentator',
			augmentation_kwargs=dict(
				occlusion_probability=0,
				occlusion_type='circle',
				occlusion_distance_type='larger',  # 'fixed',
				occlusion_center_jitter_px=0,
				gaussian_noise_probability=0,
				gaussian_noise_blur_sigma=3,
				gaussian_noise_std_polar=[0.1, 2.0],
				gaussian_noise_std_mask=[0.1, 2.0]

			),
			scale_r=1.0,  # 1024
			use_log_r=False,
			use_log_r_base='10',
			enable_6dof=True,
			enable_world_pose=False,
		),

	),
	num_vector_fields=NUM_FIELDS
)

def get_args():
	return copy.deepcopy(args)
