"""Run the deterministic demo and export a trajectory figure plus measurements."""

from __future__ import annotations

import argparse
from html import escape
import json
import math
from pathlib import Path

from navigation import RobotConfig, demo_map, metrics, plan_astar, simulate


class Figure:
    """Tiny shared SVG / optional Pillow renderer; no plotting framework needed."""

    def __init__(self, width: int, height: int, png: bool):
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                    '<title>2D differential-drive navigation simulation</title>',
                    '<desc>A synthetic occupancy map, A-star route, tracked robot trajectory and wheel-speed traces.</desc>',
                    f'<rect width="{width}" height="{height}" fill="#ffffff"/>']
        self.image = None
        self.draw = None
        self.fonts = {}
        if png:
            try:
                from PIL import Image, ImageDraw
            except ImportError as error:
                raise SystemExit("PNG export needs Pillow. Run python -m pip install -r requirements.txt or omit --png.") from error
            self.image = Image.new("RGB", (width, height), "white")
            self.draw = ImageDraw.Draw(self.image)

    def rect(self, bounds, fill, stroke=None, width=1):
        x1, y1, x2, y2 = bounds
        self.svg.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="{fill}" stroke="{stroke or "none"}" stroke-width="{width}"/>')
        if self.draw:
            self.draw.rectangle(bounds, fill=fill, outline=stroke, width=width)

    def line(self, points, fill, width=1, dashed=False):
        coords = " ".join(f"{x:.3f},{y:.3f}" for x, y in points)
        dash = ' stroke-dasharray="8 5"' if dashed else ""
        self.svg.append(f'<polyline points="{coords}" fill="none" stroke="{fill}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{dash}/>')
        if self.draw and len(points) >= 2:
            if dashed:
                # Segment the line in screen coordinates so the PNG has the
                # same visual distinction as the SVG, including short edges.
                for a, b in zip(points, points[1:]):
                    length = math.dist(a, b)
                    for position in range(0, math.ceil(length), 13):
                        first = position / length if length else 0
                        last = min(position + 8, length) / length if length else 0
                        self.draw.line([(a[0] + (b[0]-a[0])*first, a[1] + (b[1]-a[1])*first),
                                        (a[0] + (b[0]-a[0])*last, a[1] + (b[1]-a[1])*last)], fill=fill, width=width)
            else:
                self.draw.line(points, fill=fill, width=width, joint="curve")

    def circle(self, x, y, radius, fill, stroke=None, width=1):
        self.svg.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}" stroke="{stroke or "none"}" stroke-width="{width}"/>')
        if self.draw:
            self.draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=fill, outline=stroke, width=width)

    def text(self, x, y, value, size=20, fill="#17273B", bold=False, anchor="start"):
        weight = "700" if bold else "400"
        self.svg.append(f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{escape(str(value))}</text>')
        if self.draw:
            from PIL import ImageFont
            key = (size, bold)
            if key not in self.fonts:
                names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
                         "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"]
                for name in names:
                    try:
                        self.fonts[key] = ImageFont.truetype(name, size)
                        break
                    except OSError:
                        pass
                else:
                    self.fonts[key] = ImageFont.load_default(size=size)
            font = self.fonts[key]
            # Pillow and SVG both use the text baseline for alignment.
            self.draw.text((x, y), str(value), font=font, fill=fill,
                           anchor={"start": "ls", "middle": "ms", "end": "rs"}[anchor])

    def save(self, output: Path):
        output.with_suffix(".svg").write_text("\n".join(self.svg + ["</svg>"]), encoding="utf-8")
        if self.image:
            self.image.save(output.with_suffix(".png"))


def draw_demo(grid, path, result, measurements, output: Path, png: bool):
    figure = Figure(1440, 1120, png)
    navy, muted, teal, blue, orange = "#17273B", "#5C6E82", "#007F78", "#78A9E2", "#BA5C12"
    figure.text(88, 58, "Robotics Navigation Lab", size=35, bold=True)
    figure.text(88, 94, "A* planning + differential-drive feedback control | deterministic simulation", size=21, fill=muted)
    figure.text(88, 132, "SYNTHETIC WAREHOUSE MAP", size=15, bold=True, fill=muted)
    plot_x, plot_y, cell = 88, 160, 36
    plot_w, plot_h = grid.width * cell, grid.height * cell

    def map_point(point):
        return (plot_x + point[0] / grid.resolution * cell,
                plot_y + plot_h - point[1] / grid.resolution * cell)

    figure.rect((plot_x, plot_y, plot_x+plot_w, plot_y+plot_h), "#F7F9FC")
    for x in range(grid.width + 1):
        figure.line([(plot_x+x*cell, plot_y), (plot_x+x*cell, plot_y+plot_h)], "#E2E9F0")
    for y in range(grid.height + 1):
        figure.line([(plot_x, plot_y+y*cell), (plot_x+plot_w, plot_y+y*cell)], "#E2E9F0")
    for x, y in sorted(grid.occupied):
        left = plot_x + x * cell
        top = plot_y + (grid.height-y-1) * cell
        figure.rect((left, top, left+cell, top+cell), "#34465A")

    planned = [map_point(grid.centre(position)) for position in path]
    tracked = [map_point(pose.point) for pose in result.poses]
    figure.line(planned, blue, width=7, dashed=True)
    figure.line(tracked, teal, width=3)
    start, goal = planned[0], planned[-1]
    figure.circle(*start, 13, "#FFFFFF", teal, width=3)
    figure.text(start[0], start[1]+5, "S", size=14, fill=teal, bold=True, anchor="middle")
    figure.circle(*goal, 14, "#FFF4E9", orange, width=3)
    figure.text(goal[0]+24, goal[1]-12, "G", size=14, fill=orange, bold=True, anchor="middle")
    final = result.poses[-1]
    rx, ry = tracked[-1]
    robot_pixels = measurements["robot_radius_m"] / grid.resolution * cell
    figure.circle(rx, ry, robot_pixels, "#FFFFFF", teal, width=2)
    figure.line([(rx, ry), (rx+robot_pixels*math.cos(final.heading), ry-robot_pixels*math.sin(final.heading))], teal, width=2)

    for metre in range(0, int(grid.width * grid.resolution)+1, 2):
        px, py = map_point((metre, 0))
        figure.text(px, py+27, str(metre), size=16, fill=muted, anchor="middle")
    figure.text(plot_x+plot_w+39, plot_y+plot_h+27, "x (m)", size=15, fill=muted)
    for metre in range(0, int(grid.height * grid.resolution)+1):
        px, py = map_point((0, metre))
        figure.text(px-18, py+6, str(metre), size=16, fill=muted, anchor="end")
    figure.text(42, 143, "y (m)", size=15, fill=muted)

    sx = 1008
    figure.text(sx, 184, "MEASURED RESULT", size=15, bold=True, fill=muted)
    figure.text(sx, 224, "Goal reached" if result.reached_goal else "Goal not reached", size=28, fill=teal if result.reached_goal else orange, bold=True)
    items = [
        (f'{measurements["planned_path_length_m"]:.2f} m', "Planned route length"),
        (f'{measurements["sampled_trajectory_length_m"]:.2f} m', "Sampled trajectory length"),
        (f'{measurements["final_goal_error_m"] * 100:.1f} cm', "Final distance to goal"),
        (f'{measurements["simulated_time_s"]:.1f} s', "Simulated elapsed time"),
        (str(measurements["control_steps"]), "Control updates"),
    ]
    for index, (value, label) in enumerate(items):
        y = 280 + index * 83
        figure.text(sx, y, value, size=29, bold=True)
        figure.text(sx, y+27, label, size=17, fill=muted)
    figure.line([(sx, 678), (1360, 678)], "#DFE6EE")
    radius_cm = measurements["robot_radius_m"] * 100
    grid_cm = measurements["grid_resolution_m"] * 100
    control_hz = 1 / measurements["dt_s"]
    checks_pass = (measurements["collision_free_recorded_poses"]
                   and measurements["collision_free_motion_samples"])
    figure.text(sx, 710, f"Robot radius: {radius_cm:g} cm", size=18)
    figure.text(sx, 740, f"Grid: {grid_cm:g} cm | control: {control_hz:g} Hz", size=18)
    figure.text(sx, 770, f"Footprint collision checks: {'pass' if checks_pass else 'FAIL'}",
                size=18, fill=teal if checks_pass else orange)
    figure.text(sx, 800, "Static map; ideal pose feedback", size=16, fill=muted)

    figure.line([(sx, 863), (sx+46, 863)], blue, width=6, dashed=True)
    figure.text(sx+61, 870, "A* route", size=18)
    figure.line([(sx, 905), (sx+46, 905)], teal, width=3)
    figure.text(sx+61, 912, "Executed trajectory", size=18)
    figure.rect((sx, 935, sx+44, 955), "#34465A")
    figure.text(sx+61, 954, "Occupied cells", size=18)
    figure.text(sx, 1004, "Simulation evidence only.", size=16, fill=muted)
    figure.text(sx, 1030, "No hardware or ROS claims.", size=16, fill=muted)

    figure.text(88, 885, "WHEEL COMMANDS", size=15, bold=True, fill=muted)
    wx, wy, ww, wh = 88, 918, 864, 126
    wheel_limit = 16
    duration = max(measurements["simulated_time_s"], result.dt)
    for value in (-16, 0, 16):
        y = wy+wh/2 - value / wheel_limit * wh/2
        figure.line([(wx, y), (wx+ww, y)], "#DFE6EE")
        figure.text(wx-14, y+5, str(value), size=14, fill=muted, anchor="end")
    for index, color in ((0, teal), (1, orange)):
        points = [(wx + step*result.dt/duration*ww, wy+wh/2 - command[index]/wheel_limit*wh/2)
                  for step, command in enumerate(result.wheel_commands)]
        figure.line(points, color, width=2)
    figure.text(wx+ww, wy+wh+27, f"{duration:.1f} s", size=15, fill=muted, anchor="end")
    figure.text(wx, wy+wh+27, "0", size=15, fill=muted)
    figure.text(88, 909, "rad/s", size=13, fill=muted)
    figure.line([(650, 879), (676, 879)], teal, width=2)
    figure.text(685, 885, "Left", size=16, fill=muted)
    figure.line([(770, 879), (796, 879)], orange, width=2)
    figure.text(805, 885, "Right", size=16, fill=muted)
    figure.save(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    parser.add_argument("--png", action="store_true", help="Also export a PNG (requires Pillow).")
    args = parser.parse_args()
    config = RobotConfig()
    grid, start, goal = demo_map()
    path = plan_astar(grid, start, goal, config.radius)
    result = simulate(grid, path, config)
    measurements = metrics(grid, path, result, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(measurements, indent=2) + "\n", encoding="utf-8")
    draw_demo(grid, path, result, measurements, args.output_dir / "trajectory", args.png)
    print(json.dumps(measurements, indent=2))
    print(f"Artifacts saved to: {args.output_dir.resolve()}")
    return 0 if measurements["reached_goal"] and measurements["collision_free_motion_samples"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
