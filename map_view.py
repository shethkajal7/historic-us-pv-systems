from __future__ import annotations

import html
import pandas as pd
import folium
from folium.plugins import MarkerCluster


def make_cluster_map(plant: pd.DataFrame) -> folium.Map:
    df = plant.dropna(subset=["Lat", "Long"]).copy()
    center = [df["Lat"].mean(), df["Long"].mean()] if len(df) else [39.5, -98.35]
    m = folium.Map(location=center, zoom_start=4, tiles=None)

    folium.TileLayer("CartoDB positron", name="Light map", control=True).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Satellite imagery",
        overlay=False,
        control=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attr="Labels © Esri and contributors",
        name="Satellite place labels",
        overlay=True,
        control=True,
        show=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}",
        attr="Road labels © Esri and contributors",
        name="Satellite road labels",
        overlay=True,
        control=True,
        show=False,
    ).add_to(m)

    cluster = MarkerCluster(name="PV site clusters", disable_clustering_at_zoom=10).add_to(m)

    for _, row in df.iterrows():
        name = html.escape(str(row.get("Name", "Unknown")))
        state = html.escape(str(row.get("State", "")))
        popup = f"""
        <div style='width:260px'>
          <b>{name}</b><br>
          State: {state}<br>
          EIA#: {row.get('EIA#', '')}<br>
          Type: {row.get('Type', row.get('(F)ix,(T)rkr', ''))}<br>
          Module: {row.get('Module type', '')}<br>
          MWp: {row.get('MWp', float('nan')):.2f}<br>
          MWac: {row.get('MWac', float('nan')):.2f}<br>
          Lifetime PI: {row.get('Lifetime PI', float('nan')):.3f}<br>
          Degradation: {row.get('Lifetime Degr', float('nan')):.2%}<br>
          First full year: {row.get('1st full yr', '')}
        </div>
        """
        folium.CircleMarker(
            location=[row["Lat"], row["Long"]], radius=6,
            fill=True, fill_opacity=0.75, weight=1,
            tooltip=name, popup=folium.Popup(popup, max_width=320)
        ).add_to(cluster)
    folium.LayerControl(collapsed=False).add_to(m)
    return m
