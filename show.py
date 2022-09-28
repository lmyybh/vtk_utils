import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from .process import polydata2graph
import networkx as nx


def show_polydata(polydata, modelColor='Red', scalarVisibility=False, bgColor='Silver', representation=0):
    colors = vtk.vtkNamedColors()
    Mapper = vtk.vtkPolyDataMapper()
    Mapper.SetInputData(polydata)
    Mapper.SetScalarVisibility(scalarVisibility)

    Actor = vtk.vtkActor()
    Actor.SetMapper(Mapper)
    Actor.GetProperty().SetDiffuseColor(colors.GetColor3d(modelColor))  # 模型颜色
    Actor.GetProperty().SetRepresentation(representation)

    renderer = vtk.vtkRenderer()
    renderer.AddViewProp(Actor)
    renderer.SetBackground(colors.GetColor3d(bgColor))  # 背景颜色
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(1000, 800)  # 窗口大小
    renderWindow.SetPosition(500, 100)  # 窗口左上角在屏幕上的位置

    renWinInteractor = vtk.vtkRenderWindowInteractor()
    renWinInteractor.SetRenderWindow(renderWindow)
    renWinInteractor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    renderWindow.Render()
    renWinInteractor.Start()


def show_two_polydatas(polydata1, polydata2, options1={}, options2={}, bgColor='Silver'):
    defaultOptions1 = {'color': 'White',
                       'opacity': 0.5, 'scalar': False, 'pointSize': 3}
    defaultOptions2 = {'color': 'Red', 'opacity': 1,
                       'scalar': False, 'pointSize': 3}

    defaultOptions1.update(options1)
    defaultOptions2.update(options2)

    colors = vtk.vtkNamedColors()
    # polydata 1
    Mapper1 = vtk.vtkPolyDataMapper()
    Mapper1.SetInputData(polydata1)
    Mapper1.SetScalarVisibility(defaultOptions1['scalar'])

    Actor1 = vtk.vtkActor()
    Actor1.SetMapper(Mapper1)
    Actor1.GetProperty().SetDiffuseColor(
        colors.GetColor3d(defaultOptions1['color']))  # 模型颜色
    Actor1.GetProperty().SetOpacity(defaultOptions1['opacity'])  # 透明度
    Actor1.GetProperty().SetPointSize(defaultOptions1['pointSize'])

    # polydata 2
    Mapper2 = vtk.vtkPolyDataMapper()
    Mapper2.SetInputData(polydata2)
    Mapper2.SetScalarVisibility(defaultOptions2['scalar'])

    Actor2 = vtk.vtkActor()
    Actor2.SetMapper(Mapper2)
    Actor2.GetProperty().SetDiffuseColor(
        colors.GetColor3d(defaultOptions2['color']))  # 模型颜色
    Actor2.GetProperty().SetOpacity(defaultOptions2['opacity'])  # 透明度
    Actor2.GetProperty().SetPointSize(defaultOptions2['pointSize'])

    renderer = vtk.vtkRenderer()
    renderer.AddViewProp(Actor1)
    renderer.AddViewProp(Actor2)
    renderer.SetBackground(colors.GetColor3d(bgColor))  # 背景颜色
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(1000, 800)  # 窗口大小
    renderWindow.SetPosition(500, 100)  # 窗口左上角在屏幕上的位置

    renWinInteractor = vtk.vtkRenderWindowInteractor()
    renWinInteractor.SetRenderWindow(renderWindow)
    renWinInteractor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    renderWindow.Render()
    renWinInteractor.Start()

def select_cells(polydata, other_polydatas=[]):
    class Select(vtk.vtkInteractorStyleTrackballCamera):
        def __init__(self, polydata):
            self.AddObserver('LeftButtonPressEvent', self.left_button_press_event)
            self.AddObserver('KeyPressEvent', self.key_press_event)
            self.polydata = polydata
            self.graph = polydata2graph(polydata)
            self.cells = []
            self.paths = set()
            self.reset_celldata()
        
        def reset_celldata(self):
            self.polydata.GetCellData().SetScalars(numpy_to_vtk(np.zeros(self.polydata.GetNumberOfCells())))

        def change_celldata(self, cells_idxs, value):
            scalars = vtk_to_numpy(self.polydata.GetCellData().GetScalars())
            scalars[cells_idxs] = value
            self.polydata.GetCellData().SetScalars(numpy_to_vtk(scalars))

        def left_button_press_event(self, obj, event):
            pos = self.GetInteractor().GetEventPosition()
            picker = vtk.vtkCellPicker()
            picker.SetTolerance(1e-6)
            # Pick from this location.
            picker.Pick(pos[0], pos[1], 0, self.GetDefaultRenderer())
            if picker.GetCellId() != -1:
                if len(self.cells) >= 1:
                    source, target = self.cells[-1], picker.GetCellId()
                    path = nx.shortest_path(self.graph, source, target)[1:-1]
                    self.paths.update(path)
                
                if picker.GetCellId() not in self.cells:
                    self.cells.append(picker.GetCellId())

                self.reset_celldata()
                self.change_celldata(self.cells, 0.5)
                self.change_celldata(list(self.paths), 1.5)

            # Forward events
            self.OnLeftButtonDown()
        
        def key_press_event(self, obj, event):
            key = self.GetInteractor().GetKeySym()
            if key == 'a':
                if len(self.cells) >= 2:
                    source, target = self.cells[-1], self.cells[0]
                    self.cells[0], self.cells[-1] = self.cells[-1], self.cells[0]
                    path = nx.shortest_path(self.graph, source, target)[1:-1]
                    self.paths.update(path)

                self.reset_celldata()
                self.change_celldata(self.cells, 0.5)
                self.change_celldata(list(self.paths), 1.5)
                self.GetInteractor().GetRenderWindow().Render()
                
            # Forward events
            self.OnKeyPress()

    colors = vtk.vtkNamedColors()

    # table
    pColorTable = vtk.vtkLookupTable()
    # 设置颜色表中的颜色
    pColorTable.SetNumberOfColors(3)
    pColorTable.SetTableRange(0, 2)
    pColorTable.SetTableValue(0, colors.GetColor4d('White'))
    pColorTable.SetTableValue(1, colors.GetColor4d('OrangeRed'))
    pColorTable.SetTableValue(2, colors.GetColor4d('MediumSlateBlue'))
    pColorTable.Build()


    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.SetLookupTable(pColorTable)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(1)

    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(800, 600)
    renderWindow.SetPosition(350, 100)

    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderer.AddActor(actor)
    renderer.SetBackground(colors.GetColor3d("White"))

    # others
    for p in other_polydatas:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(p)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        renderer.AddActor(actor)
        actor.GetProperty().SetDiffuseColor(colors.GetColor3d('Blue'))  # 模型颜色
        actor.GetProperty().SetRepresentation(2)
        actor.GetProperty().SetLineWidth(5)

    select = Select(polydata)
    select.SetDefaultRenderer(renderer)
    renderWindowInteractor.SetInteractorStyle(select)

    renderWindow.Render()
    renderWindowInteractor.Start()

    return list(np.unique(select.cells + list(select.paths)))

class PolydatasShow:
    def __init__(self, bgColor='Silver', window_options={'size':[1000, 800], 'position':[500, 100]}):
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(vtk.vtkNamedColors().GetColor3d(bgColor))
        self.renderWindow = vtk.vtkRenderWindow()
        self.renderWindow.AddRenderer(self.renderer)
        self.renWinInteractor = vtk.vtkRenderWindowInteractor()
        self.renWinInteractor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.renWinInteractor.SetRenderWindow(self.renderWindow)
        self.renderWindow.SetSize(window_options['size'][0], window_options['size'][1])  # 窗口大小
        self.renderWindow.SetPosition(window_options['position'][0], window_options['position'][1])  # 窗口左上角在屏幕上的位置

    def add_polydata(self, polydata, representation=3, options={'color': 'Red', 'opacity': 1, 'line_width': 1}):
        colors = vtk.vtkNamedColors()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.SetScalarVisibility(False)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetDiffuseColor(colors.GetColor3d(options['color']))  # 模型颜色
        actor.GetProperty().SetOpacity(options['opacity'])  # 透明度
        actor.GetProperty().SetRepresentation(representation)
        if representation == 2:
            actor.GetProperty().SetLineWidth(options['line_width'])

        self.renderer.AddViewProp(actor)
    
    def show(self):
        self.renderWindow.Render()
        self.renWinInteractor.Start()


def show_two_polydatas_on_different_windows(polydata1, polydata2, options1={}, options2={}, bgColor='Silver'):
    defaultOptions1 = {'color': 'White', 'opacity': 0.1, 'scalar': False}
    defaultOptions2 = {'color': 'Red', 'opacity': 1, 'scalar': False}

    defaultOptions1.update(options1)
    defaultOptions2.update(options2)

    colors = vtk.vtkNamedColors()
    # polydata 1
    Mapper1 = vtk.vtkPolyDataMapper()
    Mapper1.SetInputData(polydata1)
    Mapper1.SetScalarVisibility(defaultOptions1['scalar'])

    Actor1 = vtk.vtkActor()
    Actor1.SetMapper(Mapper1)
    Actor1.GetProperty().SetDiffuseColor(
        colors.GetColor3d(defaultOptions1['color']))  # 模型颜色
    Actor1.GetProperty().SetOpacity(defaultOptions1['opacity'])  # 透明度

    # polydata 2
    Mapper2 = vtk.vtkPolyDataMapper()
    Mapper2.SetInputData(polydata2)
    Mapper2.SetScalarVisibility(defaultOptions2['scalar'])

    Actor2 = vtk.vtkActor()
    Actor2.SetMapper(Mapper2)
    Actor2.GetProperty().SetDiffuseColor(
        colors.GetColor3d(defaultOptions2['color']))  # 模型颜色
    Actor2.GetProperty().SetOpacity(defaultOptions2['opacity'])  # 透明度

    renderer1 = vtk.vtkRenderer()
    renderer1.AddViewProp(Actor1)
    renderer1.SetBackground(colors.GetColor3d(bgColor))  # 背景颜色
    renderer1.SetViewport(0.0, 0.0, 0.5, 1.0)

    renderer2 = vtk.vtkRenderer()
    renderer2.AddViewProp(Actor2)
    renderer2.SetBackground(colors.GetColor3d(bgColor))  # 背景颜色
    renderer2.SetViewport(0.5, 0.0, 1.0, 1.0)

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer1)
    renderWindow.AddRenderer(renderer2)
    renderWindow.SetSize(1000, 800)  # 窗口大小
    renderWindow.SetPosition(500, 100)  # 窗口左上角在屏幕上的位置

    renWinInteractor = vtk.vtkRenderWindowInteractor()
    renWinInteractor.SetRenderWindow(renderWindow)
    renWinInteractor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    renderWindow.Render()
    renWinInteractor.Start()


# 给定顶点顺序与vtk读取顺序可以不一致
def polydata_rgb_render_use_vertices_color(_polydata, vertices, vertices_color, scalar_bar_title='', scalar_bar_size={'width': 0.1, 'height': 0.5}, labels_num=6):
    assert _polydata.GetNumberOfPoints(
    ) == vertices.shape[0], '给定顶点数必须与模型顶点数一致'

    polydata = vtk.vtkPolyData()
    polydata.DeepCopy(_polydata)

    kd_tree = vtk.vtkKdTreePointLocator()
    kd_tree.SetDataSet(polydata)
    kd_tree.BuildLocator()

    # 存储标量值
    scalars = vtk.vtkFloatArray()
    # 设定每个顶点的标量值
    for i in range(vertices.shape[0]):
        closest_id = kd_tree.FindClosestPoint(vertices[i, :])  # 寻找对应点
        scalars.InsertTuple1(closest_id, vertices_color[i])

    # 设定每个顶点的标量值
    polydata.GetPointData().SetScalars(scalars)

    # render
    polydata_rgb_render(polydata, scalar_range=polydata.GetScalarRange(),
                        scalar_bar_title=scalar_bar_title, scalar_bar_size=scalar_bar_size, labels_num=labels_num)


# 给定顶点顺序与vtk读取顺序可以不一致
def polydata_grey_render_use_vertices_color(_polydata, vertices, vertices_color, scalar_bar_title='', scalar_bar_size={'width': 0.1, 'height': 0.5}, labels_num=6):
    assert _polydata.GetNumberOfPoints(
    ) == vertices.shape[0], '给定顶点数必须与模型顶点数一致'

    polydata = vtk.vtkPolyData()
    polydata.DeepCopy(_polydata)

    kd_tree = vtk.vtkKdTreePointLocator()
    kd_tree.SetDataSet(polydata)
    kd_tree.BuildLocator()

    # 存储标量值
    scalars = vtk.vtkFloatArray()
    # 设定每个顶点的标量值
    for i in range(vertices.shape[0]):
        closest_id = kd_tree.FindClosestPoint(vertices[i, :])  # 寻找对应点
        scalars.InsertTuple1(closest_id, vertices_color[i])

    # 设定每个顶点的标量值
    polydata.GetPointData().SetScalars(scalars)

    # 定义颜色映射表
    pColorTable = vtk.vtkLookupTable()
    # 设置颜色表中的颜色
    pColorTable.SetNumberOfColors(101)
    for i in range(101):
        pColorTable.SetTableValue(i, [i/100, i/100, i/100, 1])
    pColorTable.Build()

    # 数据映射
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.SetScalarRange(min(vertices_color), max(vertices_color))
    mapper.SetLookupTable(pColorTable)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # scalar bar
    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetLookupTable(pColorTable)
    scalar_bar.SetTitle(scalar_bar_title)  # 设置标题
    scalar_bar.SetNumberOfLabels(labels_num)  # 设置颜色范围的分段数目
    # 设置尺寸
    scalar_bar.SetWidth(scalar_bar_size['width'])
    scalar_bar.SetHeight(scalar_bar_size['height'])

    renderer = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(renderer)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    renderer.AddActor(actor)
    renderer.AddActor2D(scalar_bar)
    renderer.SetBackground(1, 1, 1)
    renWin.SetSize(800, 800)
    renWin.SetPosition(500, 100)  # 窗口左上角在屏幕上的位置
    renWin.Render()
    iren.Start()


def select_polydata_points(polydata, sphere_radius=2):
    class MyInteractor(vtk.vtkInteractorStyleTrackballCamera):
        def __init__(self, polydata, radius=2, parent=None):
            self.polydata = polydata
            self.radius = radius
            self.points = []

            self.AddObserver('RightButtonPressEvent',
                             self.right_button_press_event)
            self.AddObserver("KeyPressEvent", self.key_press_event)

        def right_button_press_event(self, obj, event):
            click_pos = self.GetInteractor().GetEventPosition()
            picker = self.GetInteractor().GetPicker()
            picker.Pick(click_pos[0], click_pos[1],
                        0, self.GetDefaultRenderer())

            # If CellId = -1, nothing was picked
            if(picker.GetCellId() != -1):
                point = self.polydata.GetPoint(picker.GetPointId())
                self.points.append(point)

                # Create a sphere
                sphere = vtk.vtkSphereSource()
                sphere.SetCenter(point)
                sphere.SetRadius(self.radius)

                # Create a mapper and actor
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(sphere.GetOutputPort())

                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(1.0, 0.0, 0.0)

                self.GetDefaultRenderer().AddActor(actor)

            # Forward events
            self.OnRightButtonDown()
            return

        def key_press_event(self, obj, event):
            key = self.GetInteractor().GetKeySym()
            if key == 'z':
                last_actor = self.GetDefaultRenderer().GetActors().GetLastActor()
                if (not isinstance(last_actor, vtk.vtkLODActor)):
                    self.GetDefaultRenderer().RemoveActor(last_actor)
                    self.points.pop()
                    self.GetDefaultRenderer().GetRenderWindow().Render()
            self.OnKeyPress()
            return

    ren = vtk.vtkRenderer()
    ren.SetBackground(.8, .8, .8)

    mapper = vtk.vtkPolyDataMapper()
    # maps polygonal data to graphics primitives
    mapper.SetInputData(polydata)
    actor = vtk.vtkLODActor()
    actor.SetMapper(mapper)
    actor.GetProperty().EdgeVisibilityOn()
    actor.GetProperty().SetLineWidth(0.3)
    ren.AddActor(actor)

    style = MyInteractor(polydata, radius=sphere_radius)
    style.SetDefaultRenderer(ren)
    cellPicker = vtk.vtkCellPicker()

    # Create a rendering window and renderer
    renWin = vtk.vtkRenderWindow()
    renWin.SetSize(800, 800)
    renWin.SetPosition(500, 100)
    renWin.AddRenderer(ren)

    # Create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.SetInteractorStyle(style)
    iren.SetPicker(cellPicker)
    iren.Initialize()
    iren.Start()

    return style.points


def polydata_rgb_render(polydata, scalar_range: tuple = None, scalar_bar_title='', scalar_bar_size={'width': 0.1, 'height': 0.5}, labels_num=6):
    if scalar_range is None:
        scalar_range = polydata.GetScalarRange()

    # 定义颜色映射表
    pColorTable = vtk.vtkLookupTable()
    # 设置颜色表中的颜色
    pColorTable.SetNumberOfColors(256)
    pColorTable.SetHueRange(0.67, 0.0)  # 色调范围从红色到蓝色
    pColorTable.Build()

    # 数据映射
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.SetScalarRange(scalar_range)
    mapper.SetLookupTable(pColorTable)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # scalar bar
    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetLookupTable(pColorTable)
    scalar_bar.SetTitle(scalar_bar_title)  # 设置标题
    scalar_bar.SetNumberOfLabels(labels_num)  # 设置颜色范围的分段数目
    # 设置尺寸
    scalar_bar.SetWidth(scalar_bar_size['width'])
    scalar_bar.SetHeight(scalar_bar_size['height'])

    renderer = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(renderer)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    renderer.AddActor(actor)
    renderer.AddActor2D(scalar_bar)
    renderer.SetBackground(1, 1, 1)
    renWin.SetSize(800, 800)
    renWin.SetPosition(500, 100)  # 窗口左上角在屏幕上的位置
    renWin.Render()
    iren.Start()


def show_imagedata(imagedata, bgColor='Silver'):
    volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
    volumeMapper.SetInputData(imagedata)

    scalars = vtk_to_numpy(imagedata.GetPointData().GetScalars())

    # 设置透明度传输函数
    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddSegment(np.min(scalars), 0, np.max(scalars),
                       1 if np.min(scalars) < np.max(scalars) else 0)

    # 设置颜色传输函数
    color = vtk.vtkColorTransferFunction()
    color.SetColorSpaceToDiverging()
    color.AddRGBSegment(np.min(scalars), 0.23, 0.3, 0.75,
                        np.max(scalars), 0.71, 0.02, 0.15)

    # 配置传输函数
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetScalarOpacity(opacity)
    volumeProperty.SetColor(color)
    volumeProperty.SetInterpolationTypeToLinear()
    # volumeProperty.ShadeOn()
    volumeProperty.SetAmbient(0.8)  # 环境光系数
    volumeProperty.SetDiffuse(0.2)  # 散射光系数
    volumeProperty.SetSpecular(0.2)  # 反射光系数
    # volumeMapper.SetBlendModeToMaximumIntensity() # 最大密度投影

    # 实例化体绘制
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    renderer = vtk.vtkRenderer()
    renderer.AddViewProp(volume)
    renderer.SetBackground(vtk.vtkNamedColors().GetColor3d(bgColor))  # 背景颜色

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(1000, 800)  # 窗口大小
    renderWindow.SetPosition(500, 100)  # 窗口左上角在屏幕上的位置

    renWinInteractor = vtk.vtkRenderWindowInteractor()
    renWinInteractor.SetRenderWindow(renderWindow)
    renWinInteractor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    renderWindow.Render()
    renWinInteractor.Start()
