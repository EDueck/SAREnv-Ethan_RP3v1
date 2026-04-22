# sarenv/core/geometries.py
import math
import geojson
import pyproj
import shapely
import shapely.plotting as shplt
from shapely.geometry.polygon import orient
import random

from ..utils.logging_setup import get_logger

log = get_logger()


class GeoData:
    """
    Foundational base class for all simulated geospatial objects.
    Manages internal geometry states, Coordinate Reference System (CRS) transformations, 
    and standardizes GeoJSON serialization across all inherited spatial entities.
    """
    def __init__(self, geometry, crs="WGS84"):
        """
        Initialises the base geospatial data structure.

        Args:
            geometry: The underlying Shapely geometric object.
            crs (str): The initial Coordinate Reference System constraint. Defaults to "WGS84".
        """
        self.geometry = geometry
        self.crs = crs

    def set_crs(self, crs):
        """
        Transitions the geometrical data to a new Coordinate Reference System.
        Triggers coordinate reprojection if the requested CRS differs from the active state.
        """
        if not isinstance(crs, str):
            msg = "Target CRS must be defined as a valid string identifier."
            raise ValueError(msg)

        if crs != self.crs:
            self._convert_to_crs(crs)
        self.crs = crs
        return self

    def _convert_to_crs(self, crs):
        """Internal mathematical transformation protocol. Must be overridden by inherited classes."""
        msg = "_convert_to_crs(crs) must be implemented within specific topological data classes."
        raise NotImplementedError(msg)

    def is_geometry_of_type(self, geometry, expected_class):
        """Validates spatial object inheritance to ensure geometrical integrity."""
        if expected_class and not isinstance(geometry, expected_class):
            msg = f"Geometry constraint violation: Expected {expected_class.__name__}."
            raise ValueError(msg)

    def get_geometry(self):
        """Returns the active Shapely geometry object."""
        return self.geometry

    def buffer(self, distance, quad_segs=1, cap_style="square", join_style="bevel"):
        """
        Expands the geometrical boundary by a specified spatial distance to calculate 
        sensor footprints or topographical margins.
        """
        self.geometry = self.geometry.buffer(distance, quad_segs, cap_style, join_style)
        return self

    def __str__(self):
        """Returns the serialized topological string representation."""
        return f"Geometry in CRS: {self.crs}\nGeometry: {self.geometry}"

    def __geo_interface__(self):
        """Provides internal compatibility with standard GeoJSON mapping protocols."""
        return self.geometry.__geo_interface__

    def to_geojson(self, id_val=None, name=None, properties=None):
        """
        Serializes the spatial geometry into a standardized GeoJSON Feature dictionary.
        """
        if id_val is None:
            id_val = str(random.randint(0, 1000000000)) 
        if name is None:
            name = str(id_val)
        if properties is None:
            properties = {}

        properties["crs"] = self.crs
        properties["name"] = name
        return geojson.Feature(id=id_val, geometry=self.geometry, properties=properties)


class GeoTrajectory(GeoData):
    """
    Represents a continuous operational flight path or calculated agent trajectory.
    Constrained strictly to Shapely LineString topologies.
    """
    def __init__(self, geometry, crs="WGS84"):
        self.is_geometry_of_type(geometry, shapely.LineString)
        super().__init__(geometry, crs)

    def plot(self, ax=None, add_points=True, color=None, linewidth=2, **kwargs):
        """Renders the trajectory path on a Cartesian coordinate plane."""
        if self.crs == "WGS84":
            log.warning(
                "Geospatial visualization warning: Plotting native WGS84 coordinates on a Cartesian plane induces topological distortion."
            )
        shplt.plot_line(self.geometry, ax, add_points, color, linewidth, **kwargs)

    def _convert_to_crs(self, crs):
        """Translates the LineString coordinates into the target projection plane."""
        transformer = pyproj.Transformer.from_crs(self.crs, crs, always_xy=True)

        converted_coords = [
            transformer.transform(x, y) for x, y in list(self.geometry.coords)
        ]
        self.geometry = shapely.LineString(converted_coords)


class GeoMultiTrajectory(GeoData):
    """
    Manages aggregated arrays of discrete flight paths, facilitating multi-agent 
    swarm routing within a singular topological object.
    """
    def __init__(
        self,
        geometry: (
            shapely.MultiLineString
            | list[shapely.LineString]
            | list[GeoTrajectory]
            | shapely.LineString
        ),
        crs="WGS84",
    ):
        processed_geometry = None
        if isinstance(geometry, list):
            line_geoms = []
            for line_item in geometry:
                if isinstance(line_item, GeoTrajectory):
                     self.is_geometry_of_type(line_item.geometry, shapely.LineString)
                     line_geoms.append(line_item.geometry)
                elif isinstance(line_item, shapely.LineString):
                    self.is_geometry_of_type(line_item, shapely.LineString)
                    line_geoms.append(line_item)
                else:
                    msg = "Trajectory aggregations mandate Shapely LineString or GeoTrajectory inputs."
                    raise ValueError(msg)
            processed_geometry = shapely.MultiLineString(line_geoms)
        elif isinstance(geometry, shapely.LineString):
            self.is_geometry_of_type(geometry, shapely.LineString)
            processed_geometry = shapely.MultiLineString([geometry])
        elif isinstance(geometry, GeoTrajectory):
            self.is_geometry_of_type(geometry.geometry, shapely.LineString)
            processed_geometry = shapely.MultiLineString([geometry.geometry])
        elif isinstance(geometry, shapely.MultiLineString):
            self.is_geometry_of_type(geometry, shapely.MultiLineString)
            processed_geometry = geometry
        else:
            msg = f"Unsupported topological architecture for GeoMultiTrajectory: {type(geometry)}"
            raise ValueError(msg)
        super().__init__(processed_geometry, crs)

    def plot(self, ax=None, add_points=False, color=None, linewidth=2, **kwargs):
        """Renders the aggregate multi-agent flight paths simultaneously."""
        if self.crs == "WGS84":
            log.warning(
                "Geospatial visualization warning: Plotting native WGS84 coordinates on a Cartesian plane induces topological distortion."
            )
        for line in self.geometry.geoms:
            shplt.plot_line(line, ax, add_points, color, linewidth, **kwargs)

    def _convert_to_crs(self, crs):
        """Iterates and translates all nested LineString coordinates into the target projection plane."""
        transformer = pyproj.Transformer.from_crs(self.crs, crs, always_xy=True)
        converted_lines = []
        for line in self.geometry.geoms:
            converted_coords = [
                transformer.transform(x, y) for x, y in list(line.coords)
            ]
            converted_lines.append(shapely.LineString(converted_coords))
        self.geometry = shapely.MultiLineString(converted_lines)


class GeoPoint(GeoData):
    """
    Represents discrete geospatial coordinates, defining exact casualty locations 
    or operational deployment anchors (e.g., Last Known Point).
    """
    def __init__(self, geometry: shapely.Point, crs="WGS84"):
        self.is_geometry_of_type(geometry, shapely.Point)
        super().__init__(geometry, crs)

    def plot(self, ax=None, add_points=True, color=None, linewidth=2, **kwargs):
        """Renders the discrete spatial point."""
        if self.crs == "WGS84":
            log.warning(
                "Geospatial visualization warning: Plotting native WGS84 coordinates on a Cartesian plane induces topological distortion."
            )
        shplt.plot_points(self.geometry, ax, color=color, **kwargs)

    def _convert_to_crs(self, crs):
        """Translates the single coordinate datum into the target projection plane."""
        transformer = pyproj.Transformer.from_crs(self.crs, crs, always_xy=True)
        x, y = transformer.transform(self.geometry.x, self.geometry.y)
        self.geometry = shapely.Point(x, y)


class GeoPolygon(GeoData):
    """
    Defines closed spatial boundaries, establishing the operational search perimeters 
    and encapsulating localized topographical features.
    """
    def __init__(self, geometry: shapely.Polygon | shapely.LineString, crs="WGS84"):
        if isinstance(geometry, shapely.LineString):
            if geometry.is_ring:
                geometry = shapely.Polygon(geometry)
            else:
                msg = "Topological conversion failure: LineString must form a closed coordinate ring to instantiate a GeoPolygon."
                raise ValueError(msg)
        self.is_geometry_of_type(geometry, shapely.Polygon)
        super().__init__(geometry, crs)

    def _convert_to_crs(self, crs):
        """Translates both the exterior bounding perimeter and interior exclusions to the target CRS."""
        transformer = pyproj.Transformer.from_crs(self.crs, crs, always_xy=True)
        exterior = [
            transformer.transform(x, y) for x, y in self.geometry.exterior.coords
        ]
        interiors = [
            [transformer.transform(x, y) for x, y in interior.coords]
            for interior in self.geometry.interiors
        ]
        self.geometry = shapely.Polygon(exterior, interiors)

    def plot(
        self,
        ax=None,
        add_points=False,
        color=None, 
        facecolor=None,
        edgecolor=None,
        linewidth=2,
        **kwargs,
    ):
        """Renders the complete topological boundary and its area of influence."""
        if self.crs == "WGS84":
            log.warning(
                "Geospatial visualization warning: Plotting native WGS84 coordinates on a Cartesian plane induces topological distortion."
            )

        final_facecolor = facecolor if facecolor is not None else (color if color is not None else 'blue') 
        final_edgecolor = edgecolor if edgecolor is not None else (color if color is not None else 'black')

        shplt.plot_polygon(
            polygon=self.geometry,
            ax=ax,
            add_points=add_points,
            facecolor=final_facecolor, 
            edgecolor=final_edgecolor, 
            linewidth=linewidth,
            **kwargs,
        )


class GeoMultiPolygon(GeoData):
    """
    Manages fragmented or disjointed search boundaries, allowing for operations 
    across separated landmasses or non-contiguous probability zones.
    """
    def __init__(self, geometry, crs="WGS84"):
        processed_geometry = None
        if isinstance(geometry, list):
            poly_geoms = []
            for geom_item in geometry:
                if isinstance(geom_item, GeoPolygon):
                    self.is_geometry_of_type(geom_item.geometry, shapely.Polygon)
                    poly_geoms.append(geom_item.geometry)
                elif isinstance(geom_item, shapely.Polygon):
                    self.is_geometry_of_type(geom_item, shapely.Polygon)
                    poly_geoms.append(geom_item)
                else:
                    msg = "Polygon aggregations mandate Shapely Polygon or GeoPolygon inputs."
                    raise ValueError(msg)
            processed_geometry = shapely.MultiPolygon(poly_geoms)
        elif isinstance(geometry, shapely.Polygon): 
            processed_geometry = shapely.MultiPolygon([geometry])
        elif isinstance(geometry, shapely.MultiPolygon):
            self.is_geometry_of_type(geometry, shapely.MultiPolygon)
            processed_geometry = geometry
        else:
            msg = f"Unsupported topological architecture for GeoMultiPolygon: {type(geometry)}"
            raise ValueError(msg)
        super().__init__(processed_geometry, crs)

    def _convert_to_crs(self, crs):
        """Iterates and translates all non-contiguous boundaries into the target projection plane."""
        transformer = pyproj.Transformer.from_crs(self.crs, crs, always_xy=True)
        polygon_list = []
        for polygon in list(self.geometry.geoms):
            exterior = [transformer.transform(x, y) for x, y in polygon.exterior.coords]
            interiors = [
                [transformer.transform(x, y) for x, y in interior.coords]
                for interior in polygon.interiors
            ]
            polygon_list.append(shapely.Polygon(exterior, interiors))
        self.geometry = shapely.MultiPolygon(polygon_list)

    def plot(
        self,
        ax=None,
        add_points=False,
        color=None,
        facecolor=None,
        edgecolor=None,
        linewidth=2,
        **kwargs,
    ):
        """Renders the disjointed regional boundaries simultaneously."""
        if self.crs == "WGS84":
            log.warning(
                "Geospatial visualization warning: Plotting native WGS84 coordinates on a Cartesian plane induces topological distortion."
            )

        final_facecolor = facecolor if facecolor is not None else (color if color is not None else 'blue')
        final_edgecolor = edgecolor if edgecolor is not None else (color if color is not None else 'black')

        for polygon in self.geometry.geoms: 
            shplt.plot_polygon(
                polygon=polygon, 
                ax=ax,
                add_points=add_points,
                facecolor=final_facecolor,
                edgecolor=final_edgecolor,
                linewidth=linewidth,
                **kwargs,
            )
