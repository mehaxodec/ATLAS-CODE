# This AI system uses Folium to generate an interactive and detail map (the risk calculation is similar to our AI_Ground_System_ctp&mpl file

import folium
from folium.plugins import HeatMap
import random
import numpy as np
import webbrowser
import os
from scipy.spatial import ConvexHull

# Fixed seed for consistency with Cartopy
random.seed(42)

# Historical buffers (identical to Cartopy)
HIST_SIZE = 100
pressure_hist = np.zeros(HIST_SIZE)
ndvi_hist = np.zeros(HIST_SIZE)
hist_idx = 0

# Zones (identical)
zones = [
    {"lat_min": 20.0, "lat_max": 23.0, "lon_min": 104.0, "lon_max": 106.0, "weight": 0.2},
    {"lat_min": 17.0, "lat_max": 19.0, "lon_min": 105.0, "lon_max": 107.0, "weight": 0.15}
]

# Risk functions (identical to Cartopy)
def calculate_landslide_risk(moisture, pressure, gnss_r, vibration, zone_w, ndvi_hist_avg, pressure_trend):
    saturation = (np.mean(moisture) * 0.7) + (gnss_r * 0.3)
    p_norm = min(pressure / 80.0, 1.0)
    features = np.array([saturation, p_norm, gnss_r, vibration])
    weights = np.array([0.4, 0.4, 0.3, 0.2])
    score = np.dot(features, weights) - 0.1 + zone_w
    
    if ndvi_hist_avg > 0.5:
        score -= 0.2
    
    forecast_adjust = pressure_trend * 0.15
    score += forecast_adjust
    
    return min(max(score, 0.0), 1.0)

def calculate_wildfire_risk(temp, humidity, ndvi):
    temp_norm = min(max((temp - 25) / 15, 0.0), 1.0)
    humidity_norm = 1.0 - (humidity / 100.0)
    ndvi_norm = 1.0 - ndvi
    score = (temp_norm * 0.5) + (humidity_norm * 0.3) + (ndvi_norm * 0.2)
    return min(max(score, 0.0), 1.0)

def calculate_flood_risk(moisture, pressure_trend, rain_flag):
    moisture_norm = np.mean(moisture)
    trend_norm = abs(pressure_trend) if pressure_trend < 0 else 0
    rain_norm = 1.0 if rain_flag else 0.0
    score = (moisture_norm * 0.5) + (trend_norm * 0.3) + (rain_norm * 0.2)
    return min(max(score, 0.0), 1.0)

def get_zone_weight(lat, lon):
    for z in zones:
        if z["lat_min"] <= lat <= z["lat_max"] and z["lon_min"] <= lon <= z["lon_max"]:
            return z["weight"]
    return 0.0

# Read file (strict)
nodes = []
if not os.path.exists("nodes_data.txt"):
    print("ERROR: nodes_data.txt not found! Run sim_nodes.py first.")
    exit()

with open("nodes_data.txt", "r") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) >= 11 and parts[0] == "NODE":
            node_id = int(parts[1])
            moisture = [float(parts[2]), float(parts[3]), float(parts[4])]
            pressure = float(parts[5])
            vibration = float(parts[6])
            temp = float(parts[7])
            humidity = float(parts[8])
            lat = float(parts[9])
            lon = float(parts[10])
            nodes.append({"id": node_id, "moisture": moisture, "pressure": pressure, "vibration": vibration,
                          "temp": temp, "humidity": humidity, "lat": lat, "lon": lon})

if len(nodes) == 0:
    print("ERROR: nodes_data.txt empty or invalid! Run sim_nodes.py first.")
    exit()

print(f"Folium using {len(nodes)} nodes from nodes_data.txt.")

# Satellite + trend (fixed seed)
gnss_r = random.uniform(0.4, 0.9)
hist_ndvi_avg = random.uniform(0.3, 0.8)
pressure_trend = random.uniform(-0.3, 0.3)
rain_flag = random.choice([True, False])

# Update historical buffers
avg_pressure = np.mean([n["pressure"] for n in nodes])
pressure_hist[hist_idx % HIST_SIZE] = avg_pressure
ndvi_hist[hist_idx % HIST_SIZE] = hist_ndvi_avg
hist_idx += 1

hist_pressure_avg = np.mean(pressure_hist[:min(hist_idx, HIST_SIZE)])
hist_ndvi_avg = np.mean(ndvi_hist[:min(hist_idx, HIST_SIZE)])

# Print node 3 & 8
for n in nodes:
    if n["id"] in [3, 8]:
        landslide_n = calculate_landslide_risk(n["moisture"], n["pressure"], gnss_r, n["vibration"], get_zone_weight(n["lat"], n["lon"]), hist_ndvi_avg, pressure_trend)
        flood_n = calculate_flood_risk(n["moisture"], pressure_trend, rain_flag)
        wildfire_n = calculate_wildfire_risk(n["temp"], n["humidity"], hist_ndvi_avg)
        print(f"Folium Node {n['id']}: Landslide {landslide_n:.2f}, Flood {flood_n:.2f}, Wildfire {wildfire_n:.2f}")

# Create map
center_lat = np.mean([n["lat"] for n in nodes])
center_lon = np.mean([n["lon"] for n in nodes])
m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='OpenStreetMap')

# Heatmap wildfire (rõ hơn)
heat_data = [[n["lat"], n["lon"], calculate_wildfire_risk(n["temp"], n["humidity"], hist_ndvi_avg)] for n in nodes]
HeatMap(heat_data, radius=30, blur=15, gradient={0.0: 'green', 0.2: 'lime', 0.4: 'yellow', 0.6: 'orange', 0.8: 'red', 1.0: 'darkred'}).add_to(m)

# Filled zones high risk
landslide_points = [[n["lat"], n["lon"]] for n in nodes if calculate_landslide_risk(n["moisture"], n["pressure"], gnss_r, n["vibration"], get_zone_weight(n["lat"], n["lon"]), hist_ndvi_avg, pressure_trend) > 0.6]
if len(landslide_points) > 2:
    folium.Polygon(landslide_points, color="darkred", weight=3, fill=True, fill_color="red", fill_opacity=0.4, tooltip="Landslide High Risk Zone").add_to(m)

flood_points = [[n["lat"], n["lon"]] for n in nodes if calculate_flood_risk(n["moisture"], pressure_trend, rain_flag) > 0.6]
if len(flood_points) > 2:
    folium.Polygon(flood_points, color="darkblue", weight=3, fill=True, fill_color="blue", fill_opacity=0.4, tooltip="Flood High Risk Zone").add_to(m)

# Hull outline
hull_points = [[n["lat"], n["lon"]] for n in nodes]
if len(hull_points) > 2:
    hull = ConvexHull(hull_points)
    hull_coords = [hull_points[i] for i in hull.vertices]
    folium.PolyLine(hull_coords, color="black", weight=2, dash_array='10', tooltip="Nodes Coverage Outline").add_to(m)

# Circles + popup
for n in nodes:
    landslide_n = calculate_landslide_risk(n["moisture"], n["pressure"], gnss_r, n["vibration"], get_zone_weight(n["lat"], n["lon"]), hist_ndvi_avg, pressure_trend)
    flood_n = calculate_flood_risk(n["moisture"], pressure_trend, rain_flag)
    wildfire_n = calculate_wildfire_risk(n["temp"], n["humidity"], hist_ndvi_avg)
    
    dominant_risk = max(landslide_n, flood_n, wildfire_n)
    color = 'green' if dominant_risk < 0.6 else 'orange' if dominant_risk < 0.8 else 'red'
    
    folium.CircleMarker(
        location=[n["lat"], n["lon"]],
        radius=10 + dominant_risk * 20,
        color='black',
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=folium.Popup(f"<b>Node {n['id']}</b><br>"
                           f"Landslide Risk: {landslide_n:.2f}<br>"
                           f"Flood Risk: {flood_n:.2f}<br>"
                           f"Wildfire Risk: {wildfire_n:.2f}<br>"
                           f"Moisture Avg: {np.mean(n['moisture']):.2f}<br>"
                           f"Pressure: {n['pressure']:.1f} kPa", max_width=300)
    ).add_to(m)

# Title + notes
title_html = '<h3 align="center" style="font-size:20px"><b>ASCEND Multi-Disaster Risk Map (12 Nodes - 12h Forecast)</b></h3>'
m.get_root().html.add_child(folium.Element(title_html))

notes_html = '<p align="center" style="font-size:12px">Notes: Filled red = Landslide high risk, Filled blue = Flood high risk, Heatmap = Wildfire (green-yellow-red). Circles per node multi-risk. Overlaps darker. Satellite data random.</p>'
m.get_root().html.add_child(folium.Element(notes_html))

# Save + open
map_file = 'ascend_multi_risk_map.html'
m.save(map_file)
print(f"Interactive map exported: {map_file}")
webbrowser.open('file://' + os.path.realpath(map_file))
