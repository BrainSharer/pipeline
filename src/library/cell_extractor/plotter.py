import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from matplotlib.widgets import Slider
#from vtkplotter import v


class Plotter:
    def __init__(self):
        # self.vtk = VtkPlotter()
        self.junk = ""

    def set_show_as_true(self):
        self.show = True

    def set_show_as_false(self):
        self.show = False

    def get_contour_data(self, contours, down_sample_factor=10):
        data = []
        for sectioni in contours:
            datai = np.array(contours[sectioni])
            datai = datai[::down_sample_factor, :]
            npoints = datai.shape[0]
            datai = np.hstack(
                (datai, np.ones(npoints).reshape(npoints, 1)*int(sectioni)))
            data.append(datai)
        data = np.vstack(data)
        return data

    def plot_3d_scatter(self, data, marker={}, title=''):
        fig = go.Figure(data=[go.Scatter3d(
            x=data[:, 0], y=data[:, 1], z=data[:, 2], mode='markers', marker=marker)])
        fig.update_layout(title=title)
        fig.show()

    def plot_contours(self, contours, down_sample_factor=10, marker={}):
        data = self.get_contour_data(contours, down_sample_factor)
        self.plot_3d_scatter(data, marker=marker)

    def plot_3d_boolean_array(self, boolean_array, down_sampling_factor=10):
        downsampled_array = boolean_array[::down_sampling_factor,
                                          ::down_sampling_factor, ::down_sampling_factor]
        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(downsampled_array, edgecolor='k')
        self.show_according_to_setting()

    def plot_3d_image_stack_on_axes(self, fig_and_axes, stack, axis=0, vmin_and_max=None):
        fig, ax = fig_and_axes
        if axis != 0:
            stack = np.swapaxes(stack, 0, axis)

        def _update_image(val):
            current_section = slider.val
            plot.set_data(stack[int(current_section)])
            fig.canvas.draw_idle()
        mid_point = int(len(stack)/2)
        if vmin_and_max is not None:
            vmin, vmax = vmin_and_max
            plot = ax.imshow(stack[mid_point], vmin=vmin, vmax=vmax)
        else:
            plot = ax.imshow(stack[mid_point],
                             vmin=stack.min(), vmax=stack.max())
        ax = plt.axes([0.25, 0.15, 0.65, 0.03])
        slider = Slider(
            ax, 'index', 0, stack.shape[0]-1, valinit=mid_point, valfmt='%d')
        slider.on_changed(_update_image)

    def plot_3d_image_stack(self, stack, axis=0, vmin_and_max=None):
        fig_and_axes = plt.subplots()
        self.plot_3d_image_stack_on_axes(
            fig_and_axes, stack, axis=axis, vmin_and_max=vmin_and_max)
        self.show_according_to_setting()

    def plot_3d_and_scatter(self, stack, annotation, axis=0, vmin_and_max=None):
        fig, ax = plt.subplots()
        if axis != 0:
            stack = np.swapaxes(stack, 0, axis)

        def _update_image(val):
            current_section = slider.val
            volume_plot.set_data(stack[int(current_section)])
            stact_plot.set_offsets(annotation[int(current_section)].T)
            fig.canvas.draw_idle()
        mid_point = int(len(stack)/2)
        if vmin_and_max is not None:
            vmin, vmax = vmin_and_max
            volume_plot = ax.imshow(stack[mid_point], vmin=vmin, vmax=vmax)
            stact_plot = ax.scatter(
                annotation[mid_point][0], annotation[mid_point][1])
        else:
            volume_plot = ax.imshow(
                stack[mid_point], vmin=stack.min(), vmax=stack.max())
            stact_plot = ax.scatter(
                annotation[mid_point][0], annotation[mid_point][1])
        ax = plt.axes([0.25, 0.15, 0.65, 0.03])
        slider = Slider(
            ax, 'index', 0, stack.shape[0]-1, valinit=mid_point, valfmt='%d')
        slider.on_changed(_update_image)
        plt.show()

    def compare_contour_and_stack(self, contour, stack, axis=2):
        if axis != 0:
            stack = np.swapaxes(stack, 0, axis)

        def get_contouri(sectioni):
            _min = np.min([np.min(contouri, 0)
                          for contouri in contour], axis=0)
            contouri = np.array(contour[sectioni])
            contouri = contouri - _min
            return contouri

        def _update_image(val):
            current_section = slider.val
            sectioni = int(current_section)
            image.set_data(stack[sectioni])
            contouri = get_contouri(sectioni)
            scatter.set_offsets(contouri)
            fig.canvas.draw_idle()
        fig, ax = plt.subplots()
        mid_point = int(len(stack)/2)
        image = ax.imshow(stack[mid_point])
        contouri = get_contouri(mid_point)
        scatter = ax.scatter(contouri[:, 0], contouri[:, 1])
        ax = plt.axes([0.25, 0.15, 0.65, 0.03])
        slider = Slider(
            ax, 'index', 0, stack.shape[0]-1, valinit=mid_point, valfmt='%d')
        slider.on_changed(_update_image)
        self.show_according_to_setting()

    def imshow(self, img):
        plt.imshow(img)
        self.show_according_to_setting()

    def plot_contour(self, contour):
        if contour.shape[0] == 2:
            contour = contour.T
        plt.scatter(contour[:, 0], contour[:, 1])
        self.show_according_to_setting()

    def show_according_to_setting(self):
        if hasattr(self, 'show'):
            if self.show:
                plt.show()
        else:
            plt.show()

    def show(self):
        plt.show()

    def compare_point_dictionaries(self, point_dicts, names=None):
        point_set_3d = []
        for point_dict in point_dicts:
            values = np.array(list(point_dict.values()))
            point_set_3d.append(values)
        self.compare_3d_point_sets(point_set_3d, names=names)

    def compare_3d_point_sets(self, point_sets_3d, names=None):
        if names == None:
            names = [f'scatter{i}' for i in range(len(point_sets_3d))]
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
        for point_set, name in zip(point_sets_3d, names):
            fig.add_trace(go.Scatter3d(x=point_set[:, 0], y=point_set[:, 1], z=point_set[:, 2],
                                       mode='markers', name=name), row=1, col=1)
        print('ready to show')
        fig.show(renderer="browser")

    def batch_plotter(self, plot_objects, plotting_function, **kwargs):
        nplots = len(plot_objects)
        if 'nrow' in kwargs:
            nrow = kwargs['nrow']
            ncol = nplots//nrow
        if 'ncol' in kwargs:
            ncol = kwargs['ncol']
            nrow = nplots//ncol
        fig, ax = plt.subplots(ncols=ncol, nrows=nrow)
        for coli in range(ncol):
            for rowi in range(nrow):
                if ncol == 1:
                    fig_and_axes = (fig, ax[rowi])
                elif nrow == 1:
                    fig_and_axes = (fig, ax[coli])
                else:
                    fig_and_axes = (fig, ax[rowi, coli])
                plotting_function(
                    fig_and_axes, plot_objects[int(coli*nrow+rowi)])
        plt.show()

    def plot_3d_volume(self, volume_3d):
        raise NotImplementedError()
        volume_3d = np.swapaxes(volume_3d, 2, 1)
        shape = volume_3d.shape
        x = np.linspace(-5, 5, shape[0])
        y = np.linspace(-5, 5, shape[1])
        z = np.linspace(-5, 5, shape[2])
        X, Y, Z = np.meshgrid(x, y, z)

        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=volume_3d.flatten(),
            isomin=0.1,
            isomax=0.8,
            opacity=0.1,
            surface_count=17,
        ))
        fig.show()

        fig = go.Figure(data=go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=volume_3d.astype(int).flatten()*5,
            isomin=0.5,
            isomax=10,
        ))

        fig.show()
