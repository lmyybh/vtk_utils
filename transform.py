import os
import re
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


def create_polydata(points, cells, mode='verts'):
    assert mode in ['verts', 'lines', 'polys'], "mode must be 'verts', 'lines' or 'polys'"
    polydata = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    for i in range(points.shape[0]):
        vtk_points.InsertNextPoint(points[i, :])
    vtk_cells = vtk.vtkCellArray()
    for cell in cells:
        vtk_cells.InsertNextCell(len(cell), cell)
    polydata.SetPoints(vtk_points)

    if mode == 'verts':
        polydata.SetVerts(vtk_cells)
    elif mode == 'lines':
        polydata.SetLines(vtk_cells)
    elif mode =='polys':
        polydata.SetPolys(vtk_cells)
    
    return polydata


def stl2polydata(filename):
    stlReader = vtk.vtkSTLReader()
    stlReader.SetFileName(filename)
    stlReader.Update()
    return stlReader.GetOutput()


def read_series(series_dir):
    """Read series directory, for dicom or mask(exported by mimics)

    Args:
        series_dir (str): directory path

    Returns:
        np.ndarray: 3D array of series
        np.ndarray: series spacing
        np.ndarray: series origin
    """      

    assert os.path.exists(series_dir), '不存在该路径：{}'.format(series_dir)
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(series_dir)
    reader.Update()

    spacing = np.array(reader.GetPixelSpacing()[::-1])
    origin = np.array(reader.GetImagePositionPatient()[::-1])
    volume = vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars()).reshape(
        reader.GetOutput().GetDimensions())
    volume = np.reshape(volume, volume.shape[::-1])
    volume = volume[::-1, ::-1, :]
    return volume, spacing, origin


def merge_masks(tl_dir, fl_dir, br_dir):
    tl_mask, _, _ = read_series(tl_dir)
    mask = np.zeros_like(tl_mask)
    mask[tl_mask == 1] = 1

    fl_mask, _, _ = read_series(fl_dir)
    mask[fl_mask == 1] = 2

    br_mask, _, _ = read_series(br_dir)
    mask[br_mask == 1] = 3

    return mask


def merge_ad_masks(ad_dir, tl_dir, fl_dir=None):
    ad_mask, _, _ = read_series(ad_dir)
    mask = np.zeros_like(ad_mask)

    tl_mask, _, _ = read_series(tl_dir)
    mask[tl_mask == 1] = 1

    br_mask = ad_mask - tl_mask
    if fl_dir is not None:
        fl_mask, _, _ = read_series(fl_dir)
        mask[fl_mask == 1] = 2
        br_mask = br_mask - fl_mask

    mask[br_mask == 1] = 3

    return mask


def polydata2stl(polydata, filename):
    writer = vtk.vtkSTLWriter()
    writer.SetInputData(polydata)
    writer.SetFileName(filename)
    writer.SetFileTypeToBinary()
    writer.Update()


def polydata2vtp(polydata, vtp_path):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(polydata)
    writer.SetFileName(vtp_path)
    writer.SetCompressorTypeToNone()
    writer.SetDataModeToAppended()
    writer.SetIdTypeToInt32()
    writer.SetEncodeAppendedData(False)
    writer.Write()


def vtp2polydata(vtp_path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    return reader.GetOutput()


def volume2imagedata(volume, spacing, origin):
    # reshape into a shape consistent with dicom, such as 512*512*800
    volume_rev = volume.reshape(volume.shape[::-1])

    # convert to vtkimagedata
    imagedata = vtk.vtkImageData()
    imagedata.SetDimensions(volume_rev.shape)
    imagedata.SetSpacing(spacing[::-1])  # set spacing, such as [0.5, 0.5, 1.2]
    imagedata.SetOrigin(origin[::-1])
    imagedata.GetPointData().SetScalars(numpy_to_vtk(volume_rev.flatten('C')))
    return imagedata


def imagedata2volume(imagedata):
    volume_rev = vtk_to_numpy(imagedata.GetPointData().GetScalars()).reshape(
        imagedata.GetDimensions(), order='C')
    return volume_rev.reshape(volume_rev.shape[::-1])


def imagedata2polydata(imagedata, range_begin=1, range_end=1, smooth=True):
    assert range_end >= range_begin, 'range_end(={}) cannot less than range_begin(={})'.format(
        range_end, range_begin)

    # threshold
    threshold = vtk.vtkImageThreshold()
    threshold.SetInputData(imagedata)
    threshold.ThresholdBetween(range_begin, range_end)  # inclusive
    threshold.ReplaceInOn()
    threshold.SetInValue(1)
    threshold.ReplaceOutOn()
    threshold.SetOutValue(0)
    threshold.Update()

    # MarchingCubes
    dmc = vtk.vtkDiscreteMarchingCubes()
    dmc.SetInputData(threshold.GetOutput())
    dmc.GenerateValues(1, 1, 1)
    dmc.Update()
    polydata = dmc.GetOutput()

    if smooth:
        RElAXATIONFACTOR = 0.03
        ITERATIONS = 100
        smoothFilter = vtk.vtkSmoothPolyDataFilter()
        smoothFilter.SetInputData(polydata)
        smoothFilter.SetRelaxationFactor(RElAXATIONFACTOR)
        smoothFilter.SetNumberOfIterations(ITERATIONS)
        smoothFilter.Update()
        polydata = smoothFilter.GetOutput()

    return polydata


def volume2polydata(volume, spacing, origin, range_begin=1, range_end=1, smooth=True):
    return imagedata2polydata(volume2imagedata(volume, spacing, origin),
                              range_begin=range_begin, range_end=range_end, smooth=smooth)


def imagedata2vti(imagedata, save_path):
    save_dir = os.path.split(save_path)[0]
    if save_dir != '' and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(save_path)
    writer.SetInputData(imagedata)
    writer.SetEncodeAppendedData(False)
    writer.SetCompressorTypeToNone()
    writer.Write()

def vti2imagedata(vti_path):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(vti_path)
    reader.Update()
    return reader.GetOutput()

def vti2volume(vti_path):
    return imagedata2volume(vti2imagedata(vti_path))

def volume2vti(volume, spacing, origin, save_path):
    imagedata2vti(volume2imagedata(volume, spacing, origin), save_path)


def dicom2vti(dicom_dir, save_path):
    volume, spacing, origin = read_series(dicom_dir)
    volume2vti(volume, spacing, origin, save_path)


def cgal2polydata(cgal_file_path, dimension=3):
    with open(cgal_file_path, 'r') as f:
        lines = f.readlines()
        lines = [re.split(r"[ ]+", line.strip('\n'))[1:] for line in lines]

    lines = [list(map(float, line)) for line in lines]

    vtk_points = vtk.vtkPoints()
    vtk_cells = vtk.vtkCellArray()
    for line in lines:
        points = np.array(line).reshape(-1, dimension)
        vtk_cells.InsertNextCell(points.shape[0], np.arange(
            vtk_points.GetNumberOfPoints(), vtk_points.GetNumberOfPoints()+points.shape[0]))
        for i in range(points.shape[0]):
            vtk_points.InsertNextPoint(points[i, :])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetLines(vtk_cells)
    return polydata


def cgal2vtp(cgal_file_path, vtp_file_path):
    polydata2vtp(cgal2polydata(cgal_file_path), vtp_file_path)


def points2polydata(points):
    polydata = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_cells = vtk.vtkCellArray()
    for i in range(points.shape[0]):
        vtk_points.InsertNextPoint(points[i])
        vtk_cells.InsertNextCell(1, [i])
    polydata.SetPoints(vtk_points)
    polydata.SetVerts(vtk_cells)
    return polydata


def points2vtp(points, vtp_file_path):
    polydata2vtp(points2polydata(points), vtp_file_path)


def polydata2imagedata(polydata, ref_imagedata, outval=0, inval=1):
    pol2stencil = vtk.vtkPolyDataToImageStencil()
    pol2stencil.SetInputData(polydata)
    pol2stencil.SetOutputOrigin(ref_imagedata.GetOrigin())
    pol2stencil.SetOutputSpacing(ref_imagedata.GetSpacing())
    pol2stencil.SetOutputWholeExtent(ref_imagedata.GetExtent())
    pol2stencil.SetTolerance(0)
    pol2stencil.Update()

    stencil2imagedata = vtk.vtkImageStencilToImage()
    stencil2imagedata.SetInputData(pol2stencil.GetOutput())
    stencil2imagedata.SetOutsideValue(outval)
    stencil2imagedata.SetInsideValue(inval)
    stencil2imagedata.Update()

    return stencil2imagedata.GetOutput()


def polydata2volume(polydata, ref_imagedata, outval=0, inval=1):
    return imagedata2volume(polydata2imagedata(polydata, ref_imagedata, outval=outval, inval=inval))

def cycle2polydata(cycle):
    polydata = vtk.vtkPolyData()

    vtk_points = vtk.vtkPoints()
    for i in range(cycle.shape[0]):
        vtk_points.InsertNextPoint(cycle[i, :])

    vtk_cells = vtk.vtkCellArray()
    cycle_cell = list(range(cycle.shape[0])) + [0]
    vtk_cells.InsertNextCell(len(cycle_cell), cycle_cell)

    polydata.SetPoints(vtk_points)
    polydata.SetPolys(vtk_cells)

    # triangulation
    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(polydata)
    triangle_filter.Update()

    return triangle_filter.GetOutput()


def polydata2xmlstr(polydata):   
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(polydata)
    writer.WriteToOutputStringOn()
    writer.Write()
    xmlstr = writer.GetOutputString()
    return xmlstr

def xmlstr2polydata(string):
    reader = vtk.vtkXMLPolyDataReader()
    reader.ReadFromInputStringOn()
    reader.SetInputString(string)
    reader.Update()
    return reader.GetOutput()

def imagedata2xmlstr(imagedata):   
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetInputData(imagedata)
    writer.WriteToOutputStringOn()
    writer.Write()
    xmlstr = writer.GetOutputString()
    return xmlstr

def xmlstr2imagedata(string):
    reader = vtk.vtkXMLImageDataReader()
    reader.ReadFromInputStringOn()
    reader.SetInputString(string)
    reader.Update()
    return reader.GetOutput()

def polydata_subdivide_to_imagedata(polydata, subdivide=1):
    points = vtk_to_numpy(polydata.GetPoints().GetData())

    min_x, min_y, min_z = np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2])
    max_x, max_y, max_z = np.max(points[:, 0]), np.max(points[:, 1]), np.max(points[:, 2])

    extent = (0, np.ceil(max_x) - np.floor(min_x), 0, np.ceil(max_y) - np.floor(min_y), 0, np.ceil(max_z) - np.floor(min_z))
    extent = tuple(int(v*subdivide) for v in extent)
    spacing = tuple([1/subdivide]*3)[::-1]
    origin = (min_x, min_y, min_z)[::-1]

    pol2stencil = vtk.vtkPolyDataToImageStencil()
    pol2stencil.SetInputData(polydata)
    pol2stencil.SetOutputOrigin(origin[::-1])
    pol2stencil.SetOutputSpacing(spacing[::-1])
    pol2stencil.SetOutputWholeExtent(extent)
    pol2stencil.SetTolerance(1e-12)
    pol2stencil.Update()

    stencil2imagedata = vtk.vtkImageStencilToImage()
    stencil2imagedata.SetInputData(pol2stencil.GetOutput())
    stencil2imagedata.SetOutsideValue(0)
    stencil2imagedata.SetInsideValue(1)
    stencil2imagedata.Update()

    return stencil2imagedata.GetOutput(), np.array(spacing), np.array(origin)

def stl_subdivide_to_imagedata(stl_path, subdivide=1):
    return polydata_subdivide_to_imagedata(stl2polydata(stl_path), subdivide=subdivide)

