from numpy.lib.polynomial import poly
import vtk
import numpy as np
import networkx as nx
from vtk.util.numpy_support import vtk_to_numpy
from ..utils import BSpline_lsq_fit
import trimesh
import networkx as nx


def simplify_polydata(polydata, targetReduction=0.95):
    # simplify mesh
    DP = vtk.vtkDecimatePro()
    DP.SetInputData(polydata)
    DP.PreserveTopologyOn()  # preserve the topology of the original mesh
    DP.SetTargetReduction(targetReduction)
    DP.Update()
    return DP.GetOutput()


def window_sinc_smooth_polydata(polydata, iterations=10):
    wndSincSmoothFilter = vtk.vtkWindowedSincPolyDataFilter()
    wndSincSmoothFilter.SetInputData(polydata)
    wndSincSmoothFilter.SetNumberOfIterations(iterations)
    wndSincSmoothFilter.Update()
    return wndSincSmoothFilter.GetOutput()


def loop_subdivide_polydata(polydata, number=1):
    loop = vtk.vtkLoopSubdivisionFilter()
    loop.SetInputData(polydata)
    # number代表四倍细分的次数，最终面数 = 初始面数 * 4 * number
    loop.SetNumberOfSubdivisions(number)
    loop.Update()
    return loop.GetOutput()


def fill_polydata_holes(polydata, size=30):
    filter = vtk.vtkFillHolesFilter()
    filter.SetInputData(polydata)
    filter.SetHoleSize(size)
    filter.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(filter.GetOutput())
    normals.ConsistencyOn()
    normals.SplittingOff()
    normals.Update()

    return normals.GetOutput()


def remove_duplicate_cells(polydata):  # 不能保证效果的稳定性
    # remove duplicate cells
    cells = vtk_to_numpy(polydata.GetPolys().GetData()).reshape(-1, 4)[:, 1:]

    # sort to find duplicate cell id
    sortCells = np.sort(cells, axis=1)
    _, unIndices = np.unique(sortCells, return_index=True, axis=0)
    sortUnIndices = np.sort(unIndices)
    sortUnIndicesDiff = np.diff(sortUnIndices)
    duplicateCells = []
    # not equal to 1 means the cell was removed
    for idx in np.where(sortUnIndicesDiff != 1)[0]:
        for i in range(1, sortUnIndicesDiff[idx]):
            duplicateCells.append(sortUnIndices[idx] + i)
    removeCellIds = []
    for idx in duplicateCells:
        removeCellIds += list(np.where((sortCells == sortCells[idx]).all(1))[0])
    cells = np.delete(cells, removeCellIds, axis=0)

    # create a new polydata
    points = vtk_to_numpy(polydata.GetPoints().GetData())
    vtkPoints = vtk.vtkPoints()
    for i in range(points.shape[0]):
        vtkPoints.InsertNextPoint(points[i, 0], points[i, 1], points[i, 2])
    vtkCells = vtk.vtkCellArray()
    for i in range(cells.shape[0]):
        vtkCells.InsertNextCell(3, cells[i, :])

    vtkData = vtk.vtkPolyData()
    vtkData.SetPoints(vtkPoints)
    vtkData.SetPolys(vtkCells)

    # clean polydata, remove unused points
    CP = vtk.vtkCleanPolyData()
    CP.SetInputData(vtkData)
    CP.Update()

    return CP.GetOutput()


def append_polydatas(polydatas):
    if len(polydatas) <= 0:
        return vtk.vtkPolyData()
    append = vtk.vtkAppendPolyData()
    for polydata in polydatas:
        append.AddInputData(polydata)
    append.Update()
    return append.GetOutput()


def cut_polydata_by_plane(polydata, origin, normal):
    # 构建切割平面
    vp = vtk.vtkPlane()
    vp.SetOrigin(origin)
    vp.SetNormal(normal)

    # cut
    vcutter = vtk.vtkCutter()
    vcutter.SetInputData(polydata)
    vcutter.SetCutFunction(vp)
    vcutter.Update()
    return vcutter.GetOutput()


def extract_cycles_in_lines_polydata(polydata):
    lines = vtk_to_numpy(polydata.GetLines().GetData()).reshape(-1, 3)[:, 1:]
    if lines.shape[0] <= 0:
        return []

    points = vtk_to_numpy(polydata.GetPoints().GetData())

    # construct graph
    graph = nx.Graph()
    graph.add_nodes_from(range(points.shape[0]))
    graph.add_edges_from(lines)

    # get cycles
    cycles = [points[cycle, :] for cycle in list(nx.cycle_basis(graph))]

    return cycles


def cycle2polydata(cycle):
    polydata = vtk.vtkPolyData()

    if cycle is None or cycle.shape[0] <= 0:
        return polydata

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


def remove_redundant_regions_of_polydata(polydata, regions_size_threshold):
    PCF = vtk.vtkPolyDataConnectivityFilter()
    PCF.SetExtractionModeToAllRegions()
    PCF.SetInputData(polydata)
    PCF.Update()

    regions = np.argwhere(
        vtk_to_numpy(PCF.GetRegionSizes()) > regions_size_threshold
    ).flatten()
    PCF.SetExtractionModeToSpecifiedRegions()
    PCF.InitializeSpecifiedRegionList()
    for rid in regions:
        PCF.AddSpecifiedRegion(rid)
    PCF.Update()

    return PCF.GetOutput()


def intersect_with_line(obbTree, p1, p2):
    dist = np.linalg.norm(p1 - p2)
    direction_vec = (p2 - p1) / dist

    points = vtk.vtkPoints()
    cells = vtk.vtkIdList()
    obbTree.IntersectWithLine(p1, p1 + dist * direction_vec, points, cells)

    # convert to numpy
    points = vtk_to_numpy(points.GetData())
    cells = np.array([cells.GetId(i) for i in range(cells.GetNumberOfIds())])

    return points, cells


def cut_polydata_and_extract_first_cycle(polydata, origin, normal):
    normal = normal / np.linalg.norm(normal)
    cutter_polydata = cut_polydata_by_plane(polydata, origin, normal)

    cycles = extract_cycles_in_lines_polydata(cutter_polydata)
    if len(cycles) <= 0:
        return None

    # find cycle that intersect with plane normal
    first_cycle = None
    for cycle in cycles:
        cycle_polydata = cycle2polydata(cycle)
        tol = 1e-12
        obbTree = vtk.vtkOBBTree()
        obbTree.SetTolerance(tol)
        obbTree.SetDataSet(cycle_polydata)
        obbTree.BuildLocator()
        intersect_points, _ = intersect_with_line(
            obbTree, origin - 0.1 * normal, origin + 0.1 * normal
        )

        if intersect_points.shape[0] > 0:
            dists = np.linalg.norm(intersect_points - origin, axis=1)
            if np.any(dists < 0.01):
                first_cycle = cycle
                break

    return first_cycle


def cut_polydata_and_extract_N_nearest_cycles(polydata, origin, normal, N, distance=10):
    normal = normal / np.linalg.norm(normal)
    cutter_polydata = cut_polydata_by_plane(polydata, origin, normal)

    cycles = extract_cycles_in_lines_polydata(cutter_polydata)
    if len(cycles) <= 0:
        return None

    dists = [np.linalg.norm(origin - np.mean(c, axis=0)) for c in cycles]

    n_cycles = [cycles[i] for i in np.argsort(dists)[:N] if dists[i] < distance]

    return n_cycles if len(n_cycles) > 0 else None


def cut_polydata_and_extract_cycles_within_distance(polydata, origin, normal, distance):
    normal = normal / np.linalg.norm(normal)
    cutter_polydata = cut_polydata_by_plane(polydata, origin, normal)

    cycles = extract_cycles_in_lines_polydata(cutter_polydata)
    if len(cycles) <= 0:
        return None

    dists = np.array([np.linalg.norm(origin - np.mean(c, axis=0)) for c in cycles])

    return [cycles[i] for i in np.where(dists <= distance)[0]]


def polydata_surface_area(polydata):
    if polydata.GetNumberOfPoints() < 3 or polydata.GetNumberOfCells() <= 0:
        return 0
    mass = vtk.vtkMassProperties()
    mass.SetInputData(polydata)
    mass.Update()

    return mass.GetSurfaceArea()


def polydata_volume(polydata):
    if polydata.GetNumberOfPoints() < 3 or polydata.GetNumberOfCells() <= 0:
        return 0
    mass = vtk.vtkMassProperties()
    mass.SetInputData(polydata)
    mass.Update()

    return mass.GetVolume()


def clip_polydata(polydata, origin, normal):
    vp = vtk.vtkPlane()
    vp.SetOrigin(origin)
    vp.SetNormal(normal)

    # 裁剪
    vcp = vtk.vtkClipPolyData()
    vcp.SetInputData(polydata)
    vcp.SetClipFunction(vp)
    vcp.GetClippedOutput()
    vcp.Update()

    if vcp.GetOutput().GetNumberOfPoints() <= 0:
        return vtk.vtkPolyData()

    # 保留所需区域
    CF = vtk.vtkPolyDataConnectivityFilter()
    CF.SetInputData(vcp.GetOutput())
    CF.SetExtractionModeToClosestPointRegion()
    CF.SetClosestPoint(origin)
    CF.Update()

    if CF.GetOutput().GetNumberOfPoints() <= 0:
        return vtk.vtkPolyData()

    # 判断该区域是否距离原点很远
    """ _, point = find_closest_point_on_polydata(CF.GetOutput(), origin)
    if np.linalg.norm(point-origin) > 50:
        return vtk.vtkPolyData() """

    # 填充截面
    fillholesfilter = vtk.vtkFillHolesFilter()
    fillholesfilter.SetHoleSize(1e9)
    fillholesfilter.SetInputData(CF.GetOutput())
    fillholesfilter.Update()

    # 调整法向量
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(fillholesfilter.GetOutput())
    normals.ConsistencyOn()  # 很重要，根据其他单元点的顺序调整补充点的顺序
    normals.SplittingOff()
    normals.Update()
    return normals.GetOutput()


def threshold_points(polydata, thresholds=[]):
    assert (
        len(thresholds) == 0 or len(thresholds) == 2
    ), "The number of thresholds must be 0 or 2."

    if len(thresholds) == 0 or (thresholds[0] is None and thresholds[1] is None):
        return vtk_to_numpy(polydata.GetPoints().GetData())
    else:
        threshold = vtk.vtkThresholdPoints()
        threshold.SetInputData(polydata)
        if thresholds[0] is None:
            threshold.ThresholdByLower(thresholds[1])
        elif thresholds[1] is None:
            threshold.ThresholdByUpper(thresholds[0])
        else:
            threshold.ThresholdBetween(thresholds[0], thresholds[1])
        threshold.Update()

        return vtk_to_numpy(threshold.GetOutput().GetPoints().GetData())


def cut_imagedata_with_plane_source(
    imagedata,
    plane_origin,
    plane_normal,
    plane_side=80,
    resolution_ratio=2,
    thresholds=None,
):
    # plane
    plane = vtk.vtkPlaneSource()
    plane.SetOrigin(0, 0, 0)
    plane.SetPoint1(plane_side, 0, 0)
    plane.SetPoint2(0, plane_side, 0)
    plane.SetResolution(plane_side * resolution_ratio, plane_side * resolution_ratio)
    plane.SetCenter(plane_origin)
    plane.SetNormal(plane_normal)
    plane.Update()

    # probe
    probeDem = vtk.vtkProbeFilter()
    probeDem.SetSourceData(imagedata)
    probeDem.SetInputConnection(plane.GetOutputPort())
    probeDem.Update()

    if thresholds is not None:
        return threshold_points(probeDem.GetOutput(), thresholds=thresholds)
    else:
        return probeDem.GetOutput()


def find_closest_point_on_polydata(polydata, point):
    if polydata.GetNumberOfPoints() <= 0:
        return None, None
    kd_tree = vtk.vtkKdTreePointLocator()
    kd_tree.SetDataSet(polydata)
    kd_tree.BuildLocator()
    pid = kd_tree.FindClosestPoint(point)
    return pid, np.array(polydata.GetPoint(pid))


def dijk_path(polydata, start_point, end_point, smooth=False, num_t=None):
    # find closest point id
    kd_tree = vtk.vtkKdTreePointLocator()
    kd_tree.SetDataSet(polydata)
    kd_tree.BuildLocator()

    start_pid = kd_tree.FindClosestPoint(start_point)
    end_pid = kd_tree.FindClosestPoint(end_point)

    dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
    dijkstra.SetInputData(polydata)
    dijkstra.SetStartVertex(start_pid)
    dijkstra.SetEndVertex(end_pid)
    dijkstra.Update()

    path = vtk_to_numpy(dijkstra.GetOutput().GetPoints().GetData())

    if smooth:
        path = BSpline_lsq_fit(path, num_t=num_t)

    return path


def get_points_and_cells_of_polydata(polydata):
    points = vtk_to_numpy(polydata.GetPoints().GetData())
    cells = vtk_to_numpy(polydata.GetPolys().GetData())
    cells = cells.reshape(-1, cells[0] + 1)[:, 1:]

    return points, cells


def polydata2graph(polydata):
    vertices, triangles = get_points_and_cells_of_polydata(polydata)
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    graph = nx.Graph()
    graph.add_nodes_from(range(triangles.shape[0]))
    graph.add_edges_from(mesh.face_adjacency)
    return graph


def PointData2CellData(polydata):
    PTC = vtk.vtkPointDataToCellData()
    PTC.SetInputData(polydata)
    PTC.Update()
    return PTC.GetOutput()


def CellData2PointData(polydata):
    CTP = vtk.vtkCellDataToPointData()
    CTP.SetInputData(polydata)
    CTP.Update()
    return CTP.GetOutput()
