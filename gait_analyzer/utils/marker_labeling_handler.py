import numpy as np
import plotly.graph_objects as go
import ezc3d


class MarkerLabelingHandler:
    def __init__(self, c3d_path: str):
        self.c3d = ezc3d.c3d(c3d_path)
        self.markers = self.c3d["data"]["points"]
        self.marker_names = self.c3d["parameters"]["POINT"]["LABELS"]["value"]

    def show_marker_labeling_plot(self):

        fig = go.Figure()
        x_vector = np.arange(self.markers.shape[2])
        for i_marker, marker_name in enumerate(self.marker_names):
            fig.add_trace(
                go.Scatter(
                    x=x_vector,
                    y=np.linalg.norm(self.markers[:3, i_marker, :], axis=0),
                    mode="lines",
                    name=marker_name,
                    line=dict(width=2),
                )
            )
        fig.update_layout(xaxis_title="Frames", yaxis_title="Markers", template="plotly_white")

        fig.write_html("markers.html")
        fig.show(renderer="browser")

    def invert_marker_labeling(self, marker_names: list, frame_start: int, frame_end: int):
        """
        Invert the marker labeling by swapping two markers.
        """

        if not isinstance(marker_names, list):
            raise TypeError("marker_names should be a list of two marker names to invert.")
        if len(marker_names) != 2:
            raise ValueError("marker_names should contain exactly two marker names to invert.")
        if not isinstance(frame_start, int) or not isinstance(frame_end, int):
            raise TypeError("frame_start and frame_end should be integers.")
        if frame_start < 0 or frame_end >= self.markers.shape[2] or frame_end <= frame_start:
            raise ValueError(f"Invalid frame range specified [{frame_start}, {frame_end}].")

        marker_indices = [self.marker_names.index(name) for name in marker_names]

        # Keep a copy of the old marker data
        old_first_marker_data = self.markers[:, marker_indices[0], frame_start : frame_end + 1].copy()

        # Make the modifications
        self.markers[:, marker_indices[0], frame_start : frame_end + 1] = self.markers[
            :, marker_indices[1], frame_start : frame_end + 1
        ].copy()
        self.markers[:, marker_indices[1], frame_start : frame_end + 1] = old_first_marker_data
        self.c3d["data"]["points"] = self.markers

        # If the first frame is included, change the marker names also
        if frame_start == 0:
            old_marker_name = self.marker_names[marker_indices[0]]
            self.marker_names[marker_indices[0]] = self.marker_names[marker_indices[1]]
            self.marker_names[marker_indices[1]] = old_marker_name
            self.c3d["parameters"]["POINT"]["LABELS"]["value"] = self.marker_names
        return

    def save_c3d(self, output_c3d_path: str):
        """
        Save the modified c3d file with the new marker labeling.
        """
        self.c3d.write(output_c3d_path)
        print(f"C3D file saved to {output_c3d_path}")
