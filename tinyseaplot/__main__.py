import typing
import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from bokeh.models import (
    ColumnDataSource,
    Slider,
    Button,
    PreText,
    HoverTool,
    CheckboxGroup,
    TapTool,
    ColorBar,
    FixedTicker,
)
from bokeh.palettes import Set1_3
from bokeh.plotting import figure
from bokeh.layouts import gridplot, layout, column
from bokeh.tile_providers import CARTODBPOSITRON, get_provider

from pytinysea import (
    latitude_t,
    longitude_t,
    time_t,
    WorldMapGrid,
    TimeWorldMap,
    BoatVelocityTable,
)
from pytinysea.gsp import CloseList, State

from tinyseaplot.transform import lat_to_web_mercator, lon_to_web_mercator
from tinyseaplot.world_map_loader import (
    build_time_world_map_from_grib,
    world_map_to_data_source,
)
from tinyseaplot.boat_velocity_table_loader import build_boat_velocity_table_from_csv
from tinyseaplot.compute_shortest_path import (
    compute_shortest_path,
    result_to_state_list,
    state_to_state_list,
)
from tinyseaplot.gpx_export import gpx_export


WORLD_MAP_DISCRET_STEP = 0.1

RESULT_STATE_POSITION_NAME = "result_position"
CLOSE_STATE_POSITION_NAME = "close_position"

SUCCESS_MESSAGE = """Solution found
Computation time: {comp_time}
Close state number: {close_state_number}
Open state number: {open_state_number}
Total distance: {total_dist:.2f}km
Totat time: {total_time}
"""

FAILURE_MESSAGE = """Solution not found
Computation time: {comp_time}s
Close state number: {close_state_number}
Open state number: {open_state_number}
"""


def check_path(str_path: str) -> Path:
    path = Path(str_path)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"{str_path} doesn't exist")
    return path


class BokehApp(object):
    def __init__(
        self, time_world_map: TimeWorldMap, boat_velocity_table: BoatVelocityTable
    ):
        self.time_world_map = time_world_map
        self.boat_velocity_table = boat_velocity_table
        self.close_list: typing.Optional[CloseList] = None
        self.shortest_path_state_list: typing.Optional[typing.List[State]] = None

    def setup(self, doc):
        world_map_t0: WorldMapGrid = self.time_world_map[time_t(0)].world_grid()
        lat_rad_space = world_map_t0.x_space()
        lon_rad_space = world_map_t0.y_space()
        lat_space = np.rad2deg((lat_rad_space.start().t, lat_rad_space.stop().t))
        lon_space = np.rad2deg((lon_rad_space.start().t, lon_rad_space.stop().t))

        tile_provider = get_provider(CARTODBPOSITRON)

        start_pos = (43.27, 3.53)
        # end_pos = (43.38, 3.68)
        end_pos = (43.27, 3.60)

        # World map figure
        p = figure(
            x_range=(
                lon_to_web_mercator(lon_space[0]),
                lon_to_web_mercator(lon_space[1]),
            ),
            y_range=(
                lat_to_web_mercator(lat_space[0]),
                lat_to_web_mercator(lat_space[1]),
            ),
            x_axis_type="mercator",
            y_axis_type="mercator",
            width=1280,
            tools="pan,wheel_zoom",
            active_drag="pan",
            active_scroll="wheel_zoom",
        )
        p.add_tile(tile_provider)

        # Path figure
        p_path_vel = figure(
            x_axis_type="datetime",
            width=1280,
            height=128,
            tools="xpan,xwheel_zoom,reset",
            active_drag="xpan",
            active_scroll="xwheel_zoom",
        )
        p_path_dist = figure(
            x_axis_type="datetime",
            width=1280,
            height=128,
            tools="xpan,xwheel_zoom,reset",
            active_drag="xpan",
            active_scroll="xwheel_zoom",
        )
        p_path_score = figure(
            x_axis_type="datetime",
            width=1280,
            height=128,
            tools="xpan,xwheel_zoom,reset",
            active_drag="xpan",
            active_scroll="xwheel_zoom",
        )

        # Setup wind display
        self._setup_wind(p, p.x_range, p.y_range)

        # Setup time slider
        self._setup_time_slider(self.time_world_map.x_space().nr_points())
        self._update_wind(self.time_world_map.x_space().value(0))

        # Setup start and end selection display
        self._setup_start_end(p, start_pos, end_pos, lat_space, lon_space)

        # Setup result state display
        self._setup_result_state(p)

        # Setup close state display
        self._setup_close_state(p)

        # Setup path display
        self._setup_path(p, p_path_vel, p_path_dist, p_path_score)

        # Setup Hover tool
        self._setup_state_hover_tool(p)

        # Setup Tap tool
        self._setup_state_tap_tool(p)

        # Setup checkbox
        self._setup_state_display_checkbox(p)

        # Setup compute button
        compute_button = Button(label="Compute")
        compute_button.on_click(self._compute)

        # Setup text resume
        self.compute_resume = PreText()

        # Setup layout
        widgets = column(
            [
                self.time_slider,
                self.start_x_slider,
                self.start_y_slider,
                self.end_x_slider,
                self.end_y_slider,
                compute_button,
                self.state_display_checkbox,
                self.compute_resume,
            ]
        )
        doc.add_root(
            layout(
                [
                    [p, widgets],
                    gridplot(
                        [[p_path_vel], [p_path_dist], [p_path_score]],
                        merge_tools=True,
                        toolbar_location="right",
                    ),
                ]
            )
        )

    def _setup_wind(self, p, x_range, y_range):
        """Setup wind intensity background image and arrow"""
        img_data_source = ColumnDataSource(data={"wind_intensity": []})
        wind_data_source = ColumnDataSource(data={"wind_xy": [], "wind_ys": []})

        a = p.image(
            image="wind_intensity",
            x=p.x_range.start,
            y=p.y_range.start,
            dw=(p.x_range.end - p.x_range.start),
            dh=(p.y_range.end - p.y_range.start),
            global_alpha=0.6,
            palette="Turbo256",
            source=img_data_source,
        )
        a.glyph.color_mapper.low = 0.0
        a.glyph.color_mapper.high = (
            self.boat_velocity_table.velocity_table()[0]
            .wind_velocity_to_boat_velocity.x_space()
            .stop()
            .t
        )
        p.patches(
            xs="wind_xs", ys="wind_ys", source=wind_data_source,
        )

        color_bar = ColorBar(
            color_mapper=a.glyph.color_mapper,
            ticker=FixedTicker(ticks=np.arange(0.0, a.glyph.color_mapper.high, 1.0)),
            location=(0, 0),
        )
        p.add_layout(color_bar, "left")

        self.img_data_source = img_data_source
        self.wind_data_source = wind_data_source

    def _update_wind(self, time: time_t):
        """Update wind intensity background image and arrow"""
        self.wind_index = self.time_world_map.x_space().index(time)
        world_map_grid = self.time_world_map[self.wind_index].world_grid()
        img_data_source_t, wind_data_source_t = world_map_to_data_source(
            world_map_grid, WORLD_MAP_DISCRET_STEP
        )
        self.img_data_source.data = dict(img_data_source_t.data)
        self.wind_data_source.data = dict(wind_data_source_t.data)

        if self.time_slider.value != self.wind_index:
            self.time_slider.value = self.wind_index

    def _setup_time_slider(self, nr_time: int):
        """Setup time slider"""
        time_slider = Slider(start=0, end=nr_time - 1, value=0, step=1, title="Time",)
        time_slider.on_change("value", self._update_wind_slider_cb)

        self.time_slider = time_slider

    def _update_wind_slider_cb(self, attr, old, new):
        """Update wind from time slider new position"""
        if self.wind_index != new:
            self._update_wind(self.time_world_map.x_space().value(new))

    def _setup_start_end(
        self,
        p,
        start_pos: typing.Tuple[float, float],
        end_pos: typing.Tuple[float, float],
        lat_space: typing.Tuple[float, float],
        lon_space: typing.Tuple[float, float],
    ):
        """Setup start and end position display"""

        start_end_source = ColumnDataSource(
            data={
                "x": list(map(lon_to_web_mercator, [start_pos[1], end_pos[1]])),
                "y": list(map(lat_to_web_mercator, [start_pos[0], end_pos[0]])),
            }
        )

        p.cross(
            x="x",
            y="y",
            color="red",
            line_width=2,
            size=lon_to_web_mercator(0.0002),
            source=start_end_source,
        )

        start_x_slider = Slider(
            start=lon_space[0],
            end=lon_space[1],
            value=start_pos[1],
            step=0.001,
            title="Start Lon",
        )
        start_x_slider.on_change(
            "value",
            lambda attr, old, new: self._update_start_end_cb(
                lon_to_web_mercator, ("x", 0), attr, old, new
            ),
        )

        start_y_slider = Slider(
            start=lat_space[0],
            end=lat_space[1],
            value=start_pos[0],
            step=0.001,
            title="Start Lat",
        )
        start_y_slider.on_change(
            "value",
            lambda attr, old, new: self._update_start_end_cb(
                lat_to_web_mercator, ("y", 0), attr, old, new
            ),
        )

        end_x_slider = Slider(
            start=lon_space[0],
            end=lon_space[1],
            value=end_pos[1],
            step=0.001,
            title="End Lon",
        )
        end_x_slider.on_change(
            "value",
            lambda attr, old, new: self._update_start_end_cb(
                lon_to_web_mercator, ("x", 1), attr, old, new
            ),
        )

        end_y_slider = Slider(
            start=lat_space[0],
            end=lat_space[1],
            value=end_pos[0],
            step=0.001,
            title="End Lat",
        )
        end_y_slider.on_change(
            "value",
            lambda attr, old, new: self._update_start_end_cb(
                lat_to_web_mercator, ("y", 1), attr, old, new
            ),
        )

        self.start_end_source = start_end_source

        self.start_x_slider = start_x_slider
        self.start_y_slider = start_y_slider
        self.end_x_slider = end_x_slider
        self.end_y_slider = end_y_slider

    def _update_start_end_cb(self, tr, pos, attr, old, new):
        """Update start stop position based on slider motion"""
        self.start_end_source.patch({pos[0]: [(pos[1], tr(new))]})

    def _setup_state_source_data(self):
        return {
            "x": [],
            "y": [],
            "time": [],
            "raw_time": [],
            "g": [],
            "h": [],
            "f": [],
            "discret_state": [],
        }

    def _setup_state_source(self):
        return ColumnDataSource(self._setup_state_source_data())

    def _setup_result_state(self, p):
        """Setup result state display"""
        result_source = self._setup_state_source()
        result_source.selected.on_change("indices", self._result_state_selected_cb)

        result_position = p.circle(
            x="x",
            y="y",
            color="green",
            size=lon_to_web_mercator(0.00005),
            source=result_source,
            name=RESULT_STATE_POSITION_NAME,
        )

        self.result_state_glyph = result_position
        self.result_state_source = result_source

    def _result_state_selected_cb(self, attr, old, new):
        if new:
            if self.shortest_path_state_list:
                self._update_path(self.shortest_path_state_list)

    def _setup_close_state(self, p):
        """Setup close state display"""
        close_source = self._setup_state_source()
        close_source.selected.on_change("indices", self._close_state_selected_cb)

        close_position = p.circle(
            x="x",
            y="y",
            color="yellow",
            size=lon_to_web_mercator(0.00005),
            source=close_source,
            name=CLOSE_STATE_POSITION_NAME,
        )
        close_position.visible = False

        self.close_state_glyph = close_position
        self.close_state_source = close_source

    def _close_state_selected_cb(self, attr, old, new):
        if new:
            discret_state = tuple(
                map(int, self.close_state_source.data["discret_state"][new[0]].split())
            )
            state = self.close_list[discret_state]
            state_list = state_to_state_list(state, self.close_list)
            self._update_path(state_list)

    def _compute_close_state_cb(self):
        """Compute close list data source if it's not already computed"""
        if self.close_list and not self.close_state_source.data["x"]:
            self.close_state_source.data = self._to_state_data(self.close_list)

    def _setup_path(self, p, p_path_vel, p_path_dist, p_path_score):
        """Setup path diplay"""
        path_source = ColumnDataSource(
            data={
                "x": [],
                "y": [],
                "time": [],
                "raw_time": [],
                "wind_vel": [],
                "boat_vel": [],
                "distance": [],
                "g": [],
                "h": [],
                "f": [],
            }
        )
        path_source.selected.on_change("indices", self._path_selected_cb)

        p.line(x="x", y="y", width=2, color=Set1_3[0], source=path_source)
        p.circle(x="x", y="y", width=2, color=Set1_3[1], source=path_source)

        def plot_line_circle(plot, y_name: str, legend: str, color):
            plot.circle(
                x="time",
                y=y_name,
                legend_label=legend,
                source=path_source,
                width=2,
                color=color,
            )
            renderer = plot.line(
                x="time",
                y=y_name,
                legend_label=legend,
                source=path_source,
                width=2,
                color=color,
            )
            return renderer

        def setup_tools(plot, tooltips, renderer=None):
            hover = HoverTool(
                tooltips=tooltips,
                formatters={"@time": "datetime"},
                mode="vline",
                point_policy="follow_mouse",
            )
            if renderer:
                hover.renderers = [renderer]
            plot.add_tools(hover)
            plot.add_tools(TapTool())

        # Plot velocity
        vel_main_renderer = plot_line_circle(
            p_path_vel, "wind_vel", "Wind velocity", Set1_3[0]
        )
        plot_line_circle(p_path_vel, "boat_vel", "Boat velocity", Set1_3[1])
        p_path_vel.yaxis.axis_label = "Velocity (m/s)"
        setup_tools(
            p_path_vel,
            [
                ("time", "@time{%d-%T}"),
                ("wind_vel", "@wind_vel"),
                ("boat_vel", "@boat_vel"),
            ],
            vel_main_renderer,
        )

        # Plot distance
        plot_line_circle(p_path_dist, "distance", "Distance", Set1_3[0])
        p_path_dist.yaxis.axis_label = "distance (km)"
        setup_tools(
            p_path_dist, [("time", "@time{%d-%T}"), ("distance", "@distance"),],
        )

        # Plot score
        plot_line_circle(p_path_score, "g", "g", Set1_3[0])
        plot_line_circle(p_path_score, "h", "h", Set1_3[1])
        score_main_renderer = plot_line_circle(p_path_score, "f", "f", Set1_3[2])
        p_path_score.yaxis.axis_label = "Score"
        setup_tools(
            p_path_score,
            [("time", "@time{%d-%T}"), ("g", "@g"), ("h", "@h"), ("f", "@f")],
            score_main_renderer,
        )

        # Configure x_range
        p_path_vel.x_range.range_padding = 0.1
        p_path_dist.x_range = p_path_vel.x_range
        p_path_score.x_range = p_path_vel.x_range

        # Configure y_range
        p_path_vel.y_range.range_padding = 0.1
        p_path_dist.y_range.range_padding = 0.1
        p_path_score.y_range.range_padding = 0.1

        self.path_vel_figure = p_path_vel
        self.path_dist_figure = p_path_dist
        self.path_score_figure = p_path_score
        self.path_source = path_source

    def _path_selected_cb(self, attr, old, new):
        if new:
            time = time_t(self.path_source.data["raw_time"][new[0]])
        else:
            time = time_t(self.path_source.data["raw_time"][0])
        self._update_wind(time)

    def _update_path(self, state_list: typing.List[State]):
        """Update path display"""
        self.path_source.selected.indices = []
        lat_lon = [s.position().to_lat_lon() for s in state_list]
        times = [s.time().t for s in state_list]
        distances = np.array(
            [
                s1.position().distance(s2.position()).t
                for s1, s2 in zip(state_list[:-1], state_list[1:])
            ]
        )
        cum_distances = np.concatenate(([0], np.cumsum(distances)))
        boat_vel = np.concatenate(([0], distances / np.diff(times)))

        self.path_source.data = {
            "x": [lon_to_web_mercator(np.rad2deg(ll[1].t)) for ll in lat_lon],
            "y": [lat_to_web_mercator(np.rad2deg(ll[0].t)) for ll in lat_lon],
            "raw_time": times,
            "time": [datetime.fromtimestamp(t, timezone.utc) for t in times],
            "wind_vel": [
                self.time_world_map[s.time()]
                .world_grid()
                .interpolated(*s.position().to_lat_lon())
                .wind_velocity.t
                for s in state_list
            ],
            "boat_vel": boat_vel,
            "distance": cum_distances / 1000.0,
            "g": [s.g().t for s in state_list],
            "h": [s.h().t for s in state_list],
            "f": [s.f().t for s in state_list],
        }
        self._update_wind(time_t(times[0]))

    def _setup_state_hover_tool(self, p):
        """Setup hover tool"""
        result_hover = HoverTool(
            tooltips=[
                ("(lat,lon)", "(@lat, @lon)"),
                ("time", "@time{%d-%T}"),
                ("g", "@g"),
                ("h", "@h"),
                ("f", "@f"),
                ("discret state", "@discret_state"),
            ],
            formatters={"@time": "datetime"},
            names=[RESULT_STATE_POSITION_NAME, CLOSE_STATE_POSITION_NAME],
        )
        p.add_tools(result_hover)

    def _setup_state_tap_tool(self, p):
        """Setup tap tool"""
        result_tap = TapTool(
            names=[RESULT_STATE_POSITION_NAME, CLOSE_STATE_POSITION_NAME]
        )
        p.add_tools(result_tap)

    def _setup_state_display_checkbox(self, p):
        """Setup checkbox to display result and/or close list state"""
        checkbox_group = CheckboxGroup(labels=["Result", "CloseList"], active=[0])
        checkbox_group.on_change("active", self._update_state_display_checkbox_cb)
        self.checkbox_group_show = [
            (self.result_state_glyph, None),
            (self.close_state_glyph, self._compute_close_state_cb),
        ]

        self.state_display_checkbox = checkbox_group
        self.checkbox_group = checkbox_group

    def _update_state_display_checkbox_cb(self, attr, old, new):
        """Update result/close state display"""
        for i, (glyph, cb) in enumerate(self.checkbox_group_show):
            if cb:
                cb()
            glyph.visible = i in new

    def _compute(self, new):
        """Compute shortest path"""
        start_pos = (
            latitude_t(np.deg2rad(self.start_y_slider.value)),
            longitude_t(np.deg2rad(self.start_x_slider.value)),
        )
        end_pos = (
            latitude_t(np.deg2rad(self.end_y_slider.value)),
            longitude_t(np.deg2rad(self.end_x_slider.value)),
        )
        start_time = self.time_world_map.x_space().value(self.wind_index)

        before_comp_time = datetime.now()
        ret, open_list, close_list = compute_shortest_path(
            self.time_world_map,
            self.boat_velocity_table,
            start_pos,
            end_pos,
            start_time,
        )
        total_comp_time = datetime.now() - before_comp_time
        self.close_list = close_list

        self.result_state_source.selected.indices = []
        self.close_state_source.selected.indices = []

        state_list = result_to_state_list(ret, close_list)
        if state_list:
            self.result_state_source.data = self._to_state_data(state_list)
            if 1 in self.checkbox_group.active:
                self.close_state_source.data = self._to_state_data(self.close_list)
            else:
                self.close_state_source.data = self._setup_state_source_data()
            self._update_path(state_list)
            self.shortest_path_state_list = state_list
            self.compute_resume.text = SUCCESS_MESSAGE.format(
                comp_time=total_comp_time,
                close_state_number=close_list.size(),
                open_state_number=open_list.size(),
                total_dist=self.path_source.data["distance"][-1],
                total_time=(
                    self.path_source.data["time"][-1] - self.path_source.data["time"][0]
                ),
            )
            gpx = gpx_export(state_list)
            with Path("/tmp/path.gpx").open("w") as f:
                f.write(gpx.to_xml())
        else:
            self._update_path([])
            self.shortest_path_state_list = None
            self.compute_resume.text = SUCCESS_MESSAGE.format(
                comp_time=total_comp_time,
                close_state_number=close_list.size(),
                open_state_number=open_list.size(),
            )

    def _to_state_data(
        self, state_list: typing.Union[typing.List[State], typing.Iterable[State]]
    ):
        """Fill result/close state data"""
        lat_lon = [s.position().to_lat_lon() for s in state_list]
        return {
            "x": [lon_to_web_mercator(np.rad2deg(ll[1].t)) for ll in lat_lon],
            "y": [lat_to_web_mercator(np.rad2deg(ll[0].t)) for ll in lat_lon],
            "lat": [np.rad2deg(ll[0].t) for ll in lat_lon],
            "lon": [np.rad2deg(ll[1].t) for ll in lat_lon],
            "time": [
                datetime.fromtimestamp(s.time().t, timezone.utc) for s in state_list
            ],
            "raw_time": [s.time().t for s in state_list],
            "g": [s.g().t for s in state_list],
            "h": [s.h().t for s in state_list],
            "f": [s.f().t for s in state_list],
            "discret_state": [
                " ".join(map(str, s.discret_state())) for s in state_list
            ],
        }


def main():
    parser = argparse.ArgumentParser(description="Plot grib wind data")
    parser.add_argument(
        "grib_file", type=check_path, help="Grib file path",
    )
    parser.add_argument(
        "polar_file", type=check_path, help="Polar file path",
    )
    args = parser.parse_args()

    time_world_map = build_time_world_map_from_grib(args.grib_file)
    boat_velocity_table = build_boat_velocity_table_from_csv(args.polar_file)

    bokeh_app = BokehApp(time_world_map, boat_velocity_table)

    from bokeh.server.server import Server

    server = Server({"/": bokeh_app.setup}, num_procs=1)
    server.start()
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
    server.io_loop.start()


if __name__ == "__main__":
    main()
