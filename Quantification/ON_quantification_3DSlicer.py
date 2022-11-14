
# -*- coding: utf-8 -*-
"""
Script to perform ON quantification, providing diameter and cross-sectional area measurements along the length of the model. 
For each subject, a CSV-file is created containing the diameter and cross-sectional measurements

Modules used
* ExtractCenterline
* CrossSectionAnalysis 
from: https://github.com/vmtk/SlicerExtension-VMTK

To make sure that the centerline extends until the endpoints, change the following in the source code of extractCenterline (l706):
centerlineFilter.SetAppendEndPointsToCenterlines(0)--> centerlineFilter.SetAppendEndPointsToCenterlines(1)

"""
import glob
import os 
import json
import sys

path = "PATH\TO\SEGMENTATIONS\\"
savePath = "PATH\TO\SAVE\MEASUREMENT\FILES\\"
subjects = [os.path.basename(x) for x in glob.glob(path +"\*-image.nii.gz")]


for subject in subjects:
    # Load subject scan and segmentation
    print('Load subject: ', subject)
    loadedVolumeNode=slicer.util.loadVolume(path + "\\" + subject + "-image.nii.gz")    # Input image
    LabelMap = slicer.util.loadSegmentation(path + "\\" + subject +"-label.nii.gz")     # Segmentation
    
    # Create segmentation node
    segmentationName=subject +'-label.nii.gz'
    segmentName='Segment_1'
    segmentationNode = slicer.util.getNode(segmentationName)
    segmentID = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segmentName)
    
    # 1. Centerline extraction
    print('Find endpoints')
    import ExtractCenterline
    extractLogic = ExtractCenterline.ExtractCenterlineLogic()
    inputSurfacePolyData = extractLogic.polyDataFromNode(segmentationNode, segmentID)
    
    # Preprocess the surface
    targetNumberOfPoints = 5000.0
    decimationAggressiveness = 4 
    subdivideInputSurface = False
    inputSurfacePolyData = extractLogic.preprocess(inputSurfacePolyData, targetNumberOfPoints, decimationAggressiveness, subdivideInputSurface)
    
    # Find endpoints
    endPointsMarkupsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "Centerline endpoints")
    networkPolyData = extractLogic.extractNetwork(inputSurfacePolyData, endPointsMarkupsNode)
    
    if 'OS' in subject:
        startPointPosition=[-100,100,-100] #OR None         # Start point 
    else:
        startPointPosition=[100,100,-100]                   # Start point OD opposite to OS
    endpointPositions = extractLogic.getEndPoints(networkPolyData, startPointPosition)
    endpointPositions = list((endpointPositions[0], endpointPositions[-1]))         # Only allow two endpoints: start and end
    endPointsMarkupsNode.RemoveAllMarkups()
    for position in endpointPositions:
    	endPointsMarkupsNode.AddControlPoint(vtk.vtkVector3d(position))
    
    # If necessary, manually adjust the endponts
    segmentationNode.GetDisplayNode().SetOpacity(0.4)
    print(" Manually adjust endpoints now if required")
    input(" Press spacebar+enter to continue…")
    print('Continue')
    centerlineEndPoints=array('Centerline endpoints')
    if (endpointPositions!=centerlineEndPoints).any():
    	print('Manual adjustment - update endpoints!')
    	endPointsMarkupsNode.RemoveAllMarkups()
    	for position in centerlineEndPoints:
    		endPointsMarkupsNode.AddControlPoint(vtk.vtkVector3d(position))
    
    # Create centerline between endpoints
    print('Create centerline')
    centerlineCurveNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", "Centerline curve")
    centerlinePolyData, voronoiDiagramPolyData = extractLogic.extractCenterline(inputSurfacePolyData, endPointsMarkupsNode, curveSamplingDistance=0.1)
    centerlinePropertiesTableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "Table_test") #None
    extractLogic.createCurveTreeFromCenterline(centerlinePolyData, centerlineCurveNode, centerlinePropertiesTableNode, curveSamplingDistance=0.1)
    
    # 2. Cross section analysis
    print('Start cross-section analysis')
    import CrossSectionAnalysis
    csLogic = CrossSectionAnalysis.CrossSectionAnalysisLogic()
    
    # Set input parameters
    csLogic.inputCenterlineNode=centerlineCurveNode
    radiusTableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "Radius Table")
    radiusPlotNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", "Radius Plot")
    csLogic.outputTableNode=radiusTableNode
    csLogic.outputPlotSeries = radiusPlotNode
    csLogic.crossSectionModelNode =slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "Cross-section")
    csLogic.maximumInscribedSphereModelNode=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "Maximum inscriped sphere")
    csLogic.showMaximumInscribedSphere=True
    csLogic.showCrossSection=True
    csLogic.lumenSurfaceNode=segmentationNode
    csLogic.currentSegmentID='Segment_1'
    csLogic.run()
    csLogic.updatePlot(radiusPlotNode, radiusTableNode)
    csLogic.updateMaximumInscribedSphereModel(0)
    csLogic.updateCrossSection(0)
    #csLogic.maximumInscribedSphereModelNode.GetDisplayNode().SetVisibility(csLogic.showMaximumInscribedSphere)
    #csLogic.crossSectionModelNode.GetDisplayNode().SetVisibility(csLogic.showCrossSection)
    
    #Save diameter measurements as dataframe
    subject_table=dataframeFromTable(radiusTableNode).to_dict()
    subject_table=dataframeFromTable(radiusTableNode)
    subject_table.to_csv(savePath + subject+'_measurements.csv', index=False)
    slicer.mrmlScene.Clear(0)