
from .center import CentersVisualizeTest, CentersVisualizeTrain
from .center_instance import CenterInstanceVisualizeTest
from .orientation import OrientationVisualizeTest, OrientationVisualizeTrain
from .segmentation import SegmentationVisualizeTest

def get_visualizer(name, opts):
    if name == 'CentersVisualizeTest':
        return CentersVisualizeTest(**opts)
    elif name == 'CenterInstanceVisualizeTest':
        return CenterInstanceVisualizeTest(**opts)
    elif name == 'CentersVisualizeTrain':
        return CentersVisualizeTrain(**opts)
    elif name == 'OrientationVisualizeTest':
        return OrientationVisualizeTest(**opts)
    elif name == 'OrientationVisualizeTrain':
        return OrientationVisualizeTrain(**opts)
    elif name == 'SegmentationVisualizeTest':
        return SegmentationVisualizeTest(**opts)
    else:
        raise Exception("Unknown visualizer: '%s'" % name)
