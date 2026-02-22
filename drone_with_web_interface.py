from vpython import *
import random
import heapq
import threading
import time

#Importing Flask server functions
try:
    from flask_server import run_flask, get_next_order, update_current_order_status, order_queue
    FLASK_ENABLED = True
    print("Flask server integration enabled!")
except ImportError:
    FLASK_ENABLED = False
    print("⚠️  Flask not available. Running in standalone mode.")

# SCENE SETUP

scene = canvas(title="🚁SkyLogix : An Autonomous Urban Drone Delivery  ",
               width=1400, height=800,
               center=vector(0,25,0),
               background=vector(0.6,0.85,1))
scene.forward = vector(-1,-0.3,-1)

is_night = False

# WEATHER SYSTEM VARIABLES

weather = "Clear"
wind_speed = 0
wind_dir = vector(1,0,0)
rain_particles = []

WEATHER_THRESHOLDS = {
    'wind_max': 12,
    'rain_max': 8,
}

weather_safe = True
protection_bubble = None
wind_particles = []
lightning_timer = 0
lightning_flash_active = False

# STAR SYSTEM

stars=[]
for _ in range(200):
    x=random.uniform(-300,300)
    y=random.uniform(80,160)
    z=random.uniform(-300,300)
    star=sphere(pos=vector(x,y,z),
                radius=0.3,
                color=color.white,
                emissive=True,
                visible=False)
    stars.append(star)

# GROUND & ROADS (EXPANDED)

box(pos=vector(0,-0.5,0), size=vector(320,1,320),
    color=vector(0.3,0.6,0.3))

for i in range(-150,160,20):
    box(pos=vector(i,0.01,0), size=vector(8,0.02,320),
        color=color.gray(0.35))
    box(pos=vector(0,0.01,i), size=vector(320,0.02,8),
        color=color.gray(0.35))

# BUILDING POSITIONS 

building_positions = {
    "A": vector(-60,0,-60),   # Home/Charging
    "B": vector(60,0,-60),    # Pizza Palace
    "C": vector(-60,0,60),    # HealthCare Pharmacy
    "D": vector(60,0,60),     # Fresh Mart
    "E": vector(0,0,-80),
    "F": vector(-80,0,0),     # Customer home
    "G": vector(80,0,0),      # Customer home
    "H": vector(0,0,80),      # Customer home
    "I": vector(-30,0,30),    # Customer home
    "J": vector(30,0,-30),    # Customer home
    "Food2": vector(-110,0,-40),    # Burger House
    "Food3": vector(110,0,40),      # Asian Kitchen
    "Pharmacy2": vector(-40,0,-110), # MedPlus Store
    "Grocery2": vector(40,0,110),   # SuperSave Grocery
    "B1": vector(-120,0,-90),
    "B2": vector(-90,0,-120),
    "B3": vector(90,0,-120),
    "B4": vector(120,0,-90),
    "B5": vector(-120,0,90),
    "B6": vector(-90,0,120),
    "B7": vector(90,0,120),
    "B8": vector(120,0,90),
    "B9": vector(-140,0,0),
    "B10": vector(140,0,0),
    "B11": vector(0,0,-140),
    "B12": vector(0,0,140),
    "B13": vector(-85,0,-85),
    "B14": vector(85,0,-85),
    "B15": vector(-85,0,85),
    "B16": vector(85,0,85),
    "B17": vector(-50,0,-50),
    "B18": vector(50,0,-50),
    "B19": vector(-50,0,50),
    "B20": vector(50,0,50),
    "B21": vector(0,0,-40),
}

buildings={}
building_heights={}
windows=[]

# Store information
STORES = {
    "B": {"type": "food", "label": "🍕 Pizza Palace", "color": vector(1, 0.5, 0)},
    "Food2": {"type": "food", "label": "🍔 Burger House", "color": vector(1, 0.6, 0.1)},
    "Food3": {"type": "food", "label": "🍜 Asian Kitchen", "color": vector(1, 0.4, 0)},
    "C": {"type": "pharmacy", "label": "💊 HealthCare", "color": vector(0.2, 0.8, 0.2)},
    "Pharmacy2": {"type": "pharmacy", "label": "💊 MedPlus", "color": vector(0.3, 0.9, 0.3)},
    "D": {"type": "grocery", "label": "🛒 Fresh Mart", "color": vector(0.3, 0.5, 1)},
    "Grocery2": {"type": "grocery", "label": "🛒 SuperSave", "color": vector(0.4, 0.6, 1)},
}

# Create all buildings
for name, pos in building_positions.items():
    if name == "A":
        height = 45  # Taller warehouse
        col = vector(1, 0.5, 0)
    elif name in STORES:
        height = 35 
        col = STORES[name]["color"]
    else:
        height = random.choice([20, 25, 30, 35, 40])
        col = vector(random.uniform(0.3,0.9), random.uniform(0.3,0.9), random.uniform(0.3,0.9))
    
    b = box(pos=pos+vector(0,height/2,0),
            size=vector(15,height,15),
            color=col)
    
    buildings[name] = b
    building_heights[name] = height
    
    # Building name label
    label(pos=pos+vector(0,height+3,0),
          text=name, box=False, height=10, color=color.white)
    
    # Store label
    if name in STORES:
        label(pos=pos+vector(0,height+8,0),
              text=STORES[name]["label"],
              box=False, height=8, color=color.yellow)
    
    # Windows
    for x in range(-6,7,4):
        for y in range(5, int(height), 8):
            for z in [-7.6, 7.6]:
                if random.random() > 0.4:
                    window = box(pos=pos+vector(x,y,z),
                                size=vector(3,5,0.1),
                                color=color.white, opacity=0.3)
                    windows.append(window)

# CHARGING STATION

charging_pad = cylinder(pos=building_positions["A"]+vector(0,building_heights["A"]+0.2,0),
                        axis=vector(0,0.4,0),
                        radius=4,
                        color=color.cyan,
                        opacity=0.7)

for angle in [0, 90, 180, 270]:
    rad = radians(angle)
    offset = vector(3*cos(rad), 0, 3*sin(rad))
    cylinder(pos=building_positions["A"]+offset,
             axis=vector(0,building_heights["A"],0),
             radius=0.3,
             color=color.gray(0.7))
    sphere(pos=building_positions["A"]+offset+vector(0,building_heights["A"],0),
           radius=0.6,
           color=color.cyan,
           emissive=True)

charging_label = label(pos=building_positions["A"]+vector(0,building_heights["A"]+8,0),
                      text="⚡ CHARGING STATION",
                      box=False,
                      height=12,
                      color=color.black)

# TREES

tree_positions = [
    vector(-40,0,-15), vector(-15,0,-40), vector(40,0,15), vector(15,0,40),
    vector(-110,0,-15), vector(-15,0,-110), vector(110,0,15), vector(15,0,110),
    vector(-70,0,0), vector(70,0,0), vector(0,0,-70), vector(0,0,70),
    vector(-95,0,-50), vector(95,0,50), vector(-50,0,95), vector(50,0,-95)
]

for pos in tree_positions:
    cylinder(pos=pos,axis=vector(0,4,0),radius=0.5,color=color.orange)
    sphere(pos=pos+vector(0,6,0),radius=2.5,color=color.green)

# STREET LIGHTS

wire_height=18
for z in [-120, -60, 60, 120]:
    poles=[]
    for x in range(-140, 141, 50):
        pole=cylinder(pos=vector(x,0,z),
                      axis=vector(0,wire_height,0),
                      radius=0.4,color=color.gray(0.6))
        lamp=sphere(pos=vector(x,wire_height,z),
                    radius=0.8,color=color.yellow,emissive=True)
        poles.append(pole)

# DRONE CREATION

def create_drone(pos):
    body=box(pos=pos,size=vector(3,1,3),color=color.white)
    arms=[];motors=[];props=[]
    dirs=[vector(2,0,2),vector(2,0,-2),
          vector(-2,0,2),vector(-2,0,-2)]
    
    for d in dirs:
        arms.append(cylinder(pos=pos,axis=d,radius=0.15))
        mpos=pos+d
        motors.append(cylinder(pos=mpos,axis=vector(0,0.5,0),radius=0.3))
        props.append(box(pos=mpos+vector(0,0.6,0),
                         size=vector(1.6,0.1,0.3),color=color.black))
    
    package=box(pos=pos-vector(0,1.5,0),
                size=vector(1,1,1),color=color.orange)
    
    return {"pos":pos,"body":body,"arms":arms,
            "motors":motors,"props":props,
            "package":package}

drone=create_drone(vector(-60,30,-60))
nav_light=sphere(pos=drone["body"].pos+vector(0,1.2,0),
                 radius=0.3,color=color.red,emissive=True)

drone["velocity"]=vector(0,0,0)
drone["battery"]=100.0

# WEATHER FUNCTIONS

def set_weather(new_weather):
    global weather, wind_speed, wind_dir, weather_safe, wind_particles, rain_particles
    
    weather = new_weather
    
    for r in rain_particles:
        r.visible = False
    rain_particles.clear()
    
    for w in wind_particles:
        w.visible = False
    wind_particles.clear()
    
    if weather == "Clear":
        scene.background = vector(0.6,0.85,1) if not is_night else vector(0.05,0.05,0.2)
        wind_speed = random.uniform(0, 3)
        weather_safe = True
        # Show birds in clear weather (if not night)
        if not is_night:
            for bird in birds:
                bird.set_visible(True)
        print("Weather: Clear")
    
    elif weather == "Rain":
        scene.background = vector(0.5,0.5,0.6) if not is_night else vector(0.03,0.03,0.15)
        wind_speed = random.uniform(3, 8)
        
        # Hide birds during rain
        for bird in birds:
            bird.set_visible(False)
        
        if wind_speed > WEATHER_THRESHOLDS['rain_max']:
            weather_safe = False
            print("⚠️ HEAVY RAIN - Unsafe!")
        else:
            weather_safe = True
            print("🌧️ Light Rain")
        
        for _ in range(150):
            x = random.uniform(-140, 140)
            z = random.uniform(-140, 140)
            y = random.uniform(60, 120)
            drop = cylinder(pos=vector(x,y,z), axis=vector(0,-3,0),
                          radius=0.05, color=color.cyan, opacity=0.4)
            rain_particles.append(drop)
    
    elif weather == "Wind":
        scene.background = vector(0.7,0.9,1) if not is_night else vector(0.05,0.05,0.2)
        wind_speed = random.uniform(6, 12)
        
        # Hide birds during wind
        for bird in birds:
            bird.set_visible(False)
        
        if wind_speed > WEATHER_THRESHOLDS['wind_max']:
            weather_safe = False
            print("⚠️ STRONG WIND - Unsafe!")
        else:
            weather_safe = True
            print("💨 Moderate Wind")
        
        for _ in range(60):
            x = random.uniform(-140, 140)
            z = random.uniform(-140, 140)
            y = random.uniform(5, 40)
            particle = sphere(pos=vector(x,y,z), radius=0.3,
                            color=vector(0.6,0.4,0.2), opacity=0.6)
            wind_particles.append(particle)
    
    elif weather == "Storm":
        scene.background = vector(0.2,0.2,0.3)
        wind_speed = random.uniform(14, 20)
        weather_safe = False
        
        # Hide birds during storm
        for bird in birds:
            bird.set_visible(False)
        
        print("⚠️ STORM - All unsafe!")
        
        for _ in range(200):
            x = random.uniform(-140, 140)
            z = random.uniform(-140, 140)
            y = random.uniform(60, 120)
            drop = cylinder(pos=vector(x,y,z), axis=vector(0,-4,0),
                          radius=0.08, color=color.cyan, opacity=0.5)
            rain_particles.append(drop)
        
        for _ in range(100):
            x = random.uniform(-140, 140)
            z = random.uniform(-140, 140)
            y = random.uniform(5, 50)
            particle = sphere(pos=vector(x,y,z), radius=0.4,
                            color=vector(0.5,0.5,0.5), opacity=0.7)
            wind_particles.append(particle)
    
    wind_dir = norm(vector(random.uniform(-1,1), 0, random.uniform(-1,1)))
    
    if 'delivery' in globals() and delivery is not None:
        delivery.check_weather_safety()

def create_protection_bubble():
    global protection_bubble
    if protection_bubble is None:
        protection_bubble = sphere(pos=drone["body"].pos, radius=4,
                                  color=vector(0.3, 0.7, 1), opacity=0.15, emissive=True)
    protection_bubble.visible = True

def update_protection_bubble():
    global protection_bubble
    if protection_bubble and protection_bubble.visible:
        protection_bubble.pos = drone["body"].pos

def hide_protection_bubble():
    global protection_bubble
    if protection_bubble:
        protection_bubble.visible = False

def create_lightning():
    global lightning_flash_active
    scene.background = vector(1, 1, 1)
    lightning_flash_active = True

# BIRD OBSTACLES

class Bird:
    def __init__(self, pos):
        self.body = sphere(pos=pos, radius=1.2, color=vector(0.9,0.4,0.2))
        self.wing1 = box(pos=pos+vector(1.5,0,0), size=vector(2,0.2,1), color=vector(0.3,0.3,0.3))
        self.wing2 = box(pos=pos+vector(-1.5,0,0), size=vector(2,0.2,1), color=vector(0.3,0.3,0.3))
        self.beak = cone(pos=pos+vector(0,0,1.2), axis=vector(0,0,0.6), radius=0.3, color=color.yellow)
        self.eye1 = sphere(pos=pos+vector(0.3,0.3,0.9), radius=0.15, color=color.white)
        self.eye2 = sphere(pos=pos+vector(-0.3,0.3,0.9), radius=0.15, color=color.white)
        self.warning_zone = sphere(pos=pos, radius=12, color=color.red, opacity=0.0)
        self.vel = vector(random.choice([-6,6]),random.uniform(-1,1),random.uniform(-2,2))
        self.flap = 0
        self.active = True

    def update(self):
        if not self.active:
            return
            
        self.body.pos += self.vel*(1/60)
        self.wing1.pos = self.body.pos + vector(1.5,0,0)
        self.wing2.pos = self.body.pos + vector(-1.5,0,0)
        self.beak.pos = self.body.pos + vector(0,0,1.2)
        self.eye1.pos = self.body.pos + vector(0.3,0.3,0.9)
        self.eye2.pos = self.body.pos + vector(-0.3,0.3,0.9)
        self.warning_zone.pos = self.body.pos

        angle = 0.6*sin(self.flap)
        self.wing1.rotate(angle=angle,axis=vector(0,0,1),origin=self.body.pos)
        self.wing2.rotate(angle=-angle,axis=vector(0,0,1),origin=self.body.pos)
        self.flap += 0.3

        if abs(self.body.pos.x) > 160:
            self.body.pos = vector(-160*self.vel.x/abs(self.vel.x),
                                    random.uniform(25,60),
                                    random.uniform(-120,120))
            self.beak.axis = vector(0,0,0.6) if self.vel.x > 0 else vector(0,0,-0.6)
    
    def show_warning(self, show=True):
        self.warning_zone.opacity = 0.15 if show else 0.0
    
    def set_visible(self, visible):
        self.active = visible
        self.body.visible = visible
        self.wing1.visible = visible
        self.wing2.visible = visible
        self.beak.visible = visible
        self.eye1.visible = visible
        self.eye2.visible = visible
        self.warning_zone.visible = visible

birds = []
for _ in range(8):
    birds.append(Bird(vector(random.uniform(-120,120),
                              random.uniform(25,60),
                              random.uniform(-120,120))))

# Set initial weather after birds are created
set_weather("Clear")

# A* PATHFINDING

class AStarNode:
    def __init__(self, pos, g=0, h=0, parent=None):
        self.pos = pos
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent
    
    def __lt__(self, other):
        return self.f < other.f

def heuristic(pos1, pos2):
    return mag(pos2 - pos1)

def get_neighbors(pos, grid_size=15):
    neighbors = []
    for dx in [-grid_size, 0, grid_size]:
        for dy in [-grid_size, 0, grid_size]:
            for dz in [-grid_size, 0, grid_size]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                neighbor = vector(pos.x + dx, pos.y + dy, pos.z + dz)
                neighbors.append(neighbor)
    return neighbors

def is_collision_free(pos, buildings_dict, bird_list, safety_margin=10):
    for building in buildings_dict.values():
        b_pos = building.pos
        b_size = building.size
        if (abs(pos.x - b_pos.x) < b_size.x/2 + safety_margin and
            abs(pos.y - b_pos.y) < b_size.y/2 + safety_margin and
            abs(pos.z - b_pos.z) < b_size.z/2 + safety_margin):
            return False
    
    for bird in bird_list:
        if bird.active and mag(pos - bird.body.pos) < 15:
            return False
    
    if abs(pos.x) > 150 or abs(pos.z) > 150 or pos.y < 5 or pos.y > 80:
        return False
    
    return True

def astar_pathfind(start, goal, buildings_dict, bird_list):
    # Check if start and goal are valid
    if not is_collision_free(start, buildings_dict, bird_list, safety_margin=5):
        print(f"⚠️ A* Warning: Start position too close to obstacle!")
    if not is_collision_free(goal, buildings_dict, bird_list, safety_margin=5):
        print(f"⚠️ A* Warning: Goal position too close to obstacle!")
    
    distance = mag(goal - start)
    print(f"🔍 A* Pathfinding: {distance:.1f} units")
    
    open_list = []
    closed_set = set()
    
    start_node = AStarNode(start, 0, heuristic(start, goal))
    heapq.heappush(open_list, start_node)
    
    iterations = 0
    max_iterations = 1500
    
    while open_list and iterations < max_iterations:
        iterations += 1
        current = heapq.heappop(open_list)
        
        # Use moderate rounding for good balance
        pos_key = (round(current.pos.x/3)*3, round(current.pos.y/3)*3, round(current.pos.z/3)*3)
        
        if pos_key in closed_set:
            continue
        
        closed_set.add(pos_key)
        
        # Goal tolerance
        if mag(current.pos - goal) < 18:
            path = []
            node = current
            while node:
                path.append(node.pos)
                node = node.parent
            print(f"A* Success: Found path in {iterations} iterations, {len(path)} waypoints")
            return list(reversed(path))
        
        for neighbor_pos in get_neighbors(current.pos):
            if not is_collision_free(neighbor_pos, buildings_dict, bird_list):
                continue
            
            neighbor_key = (round(neighbor_pos.x/3)*3, round(neighbor_pos.y/3)*3, round(neighbor_pos.z/3)*3)
            if neighbor_key in closed_set:
                continue
            
            g_cost = current.g + mag(neighbor_pos - current.pos)
            h_cost = heuristic(neighbor_pos, goal)
            neighbor_node = AStarNode(neighbor_pos, g_cost, h_cost, current)
            
            heapq.heappush(open_list, neighbor_node)
    
    # Determining why it failed
    if iterations >= max_iterations:
        print(f"⚠️ A* Failed: Max iterations ({max_iterations}) reached. Distance: {distance:.1f} units")
        print(f"   Checked {len(closed_set)} positions")
    else:
        print(f"⚠️ A* Failed: No valid path found (open list empty)")
    
    # Smart fallback: create a simple high-altitude path
    print("   → Using smart high-altitude fallback path")
    mid_height = 60  # Fly high to avoid obstacles
    path = [
        start,
        vector(start.x, mid_height, start.z),  # Rise up
        vector(goal.x, mid_height, goal.z),    # Move horizontally at height
        goal                                    # Descend to goal
    ]
    return path

# DRONE PHYSICS

def move_drone(drone, target, speed=9, slow_zone=False):
    if weather == "Rain":
        speed *= 0.7
    elif weather == "Wind" and wind_speed > 8:
        speed *= 0.6
    elif weather == "Storm":
        speed *= 0.4
    
    pos = drone["pos"]
    to_target = target - pos
    dist = mag(to_target)

    if dist < 0.6:
        drone["velocity"] *= 0.5
        return True

    if slow_zone and dist < 8:
        speed = 3

    desired = norm(to_target) * speed
    wind_effect = wind_dir * wind_speed * 0.3

    avoid = vector(0,0,0)
    for b in birds:
        if not b.active:
            continue
        d = mag(pos - b.body.pos)
        if d < 15:
            b.show_warning(True)
            avoid += norm(pos - b.body.pos) * (20 / max(d, 1))
        else:
            b.show_warning(False)

    accel = (desired + avoid - wind_effect - drone["velocity"]) * 0.15
    drone["velocity"] += accel

    step = drone["velocity"] * (1/60)
    drone["pos"] += step

    parts = [drone["body"], drone["package"]] + drone["arms"] + drone["motors"] + drone["props"]
    for p in parts:
        p.pos += step

    for p in drone["props"]:
        p.rotate(angle=1.2, axis=vector(0,1,0), origin=p.pos)

    drain_rate = 0.0009
    if len([b for b in birds if b.active and mag(pos - b.body.pos) < 15]) > 0:
        drain_rate = 0.0015
    if weather in ["Rain", "Wind"]:
        drain_rate *= 1.2
    elif weather == "Storm":
        drain_rate *= 1.5
    
    drone["battery"] -= drain_rate
    if drone["battery"] < 0:
        drone["battery"] = 0

    return False

# PATH VISUALIZATION

path_markers = []

def visualize_path(path):
    global path_markers
    
    for marker in path_markers:
        marker.visible = False
    path_markers.clear()
    
    for i, pos in enumerate(path):
        marker = sphere(pos=pos, radius=1, color=color.green, opacity=0.4)
        path_markers.append(marker)
        
        if i > 0:
            line = curve([path[i-1], pos], color=color.green, radius=0.2)
            path_markers.append(line)

def clear_path_markers():
    for marker in path_markers:
        marker.visible = False
    path_markers.clear()

# DELIVERY SYSTEM

class DeliverySystem:
    def __init__(self):
        self.state = "idle"
        self.path = []
        self.wp = 0
        self.from_s = None
        self.to_s = None
        self.emergency_triggered = False
        self.charge_timer = 0
        self.hover_timer = 0
        self.landing_height = 0
        self.replan_cooldown = 0
        self.current_goal = None
        self.web_order_mode = False
        self.current_web_order = None
        self.is_package_delivery = False
        self.manual_delivery_mode = False
        self.order_queue = []  # Queue for multiple orders
    
    def add_to_queue(self, from_building, to_building, is_manual=False, web_order=None, is_package=False):
        """Add order to queue and immediately sort by priority"""
        order_info = {
            'from': from_building,
            'to': to_building,
            'is_manual': is_manual,
            'web_order': web_order,
            'is_package': is_package
        }
        self.order_queue.append(order_info)
        
        # Immediately sort queue after adding
        self.sort_queue_by_priority()
        
        print(f"📋 Order added to queue: {from_building} → {to_building}")
        print(f"   Queue size: {len(self.order_queue)}")
        self.print_queue_order()
    
    def sort_queue_by_priority(self):
        """Sort queue by estimated delivery time from current drone position"""
        if len(self.order_queue) > 0:
            self.order_queue.sort(key=lambda x: self.get_delivery_time_estimate(x))
    
    def print_queue_order(self):
        """Print the current queue order with estimates"""
        if len(self.order_queue) > 0:
            print("\n   📊 Current Queue Priority:")
            for i, order in enumerate(self.order_queue):
                est_time = self.get_delivery_time_estimate(order)
                print(f"      {i+1}. {order['from']} → {order['to']} (Est: {est_time:.1f} units)")
    
    def get_delivery_time_estimate(self, order):
        """Calculate estimated delivery time based on current drone position"""
        current_pos = drone["pos"]
        
        # If it's a manual delivery (direct from current position to destination)
        if order['is_manual']:
            # Direct flight: current position → destination → warehouse
            delivery_dist = mag(current_pos - building_positions[order['to']])
            return_dist = mag(building_positions[order['to']] - building_positions["A"])
            total_time = delivery_dist + return_dist
        else:
            # Store order: current position → store → customer → warehouse
            pickup_dist = mag(current_pos - building_positions[order['from']])
            delivery_dist = mag(building_positions[order['from']] - building_positions[order['to']])
            return_dist = mag(building_positions[order['to']] - building_positions["A"])
            total_time = pickup_dist + delivery_dist + return_dist
        
        return total_time
    
    def process_next_order(self):
        """Process the next order from the queue"""
        if len(self.order_queue) > 0 and self.state == "idle":
            # Re-sort one final time before processing (in case drone moved)
            self.sort_queue_by_priority()
            
            next_order = self.order_queue.pop(0)
            est_time = self.get_delivery_time_estimate(next_order)
            
            print(f"\n🚁 PROCESSING NEXT ORDER:")
            print(f"   Route: {next_order['from']} → {next_order['to']}")
            print(f"   Estimated time: {est_time:.1f} units")
            print(f"   Type: {'Manual/Direct' if next_order['is_manual'] else 'Store Order'}")
            
            self.manual_delivery_mode = next_order['is_manual']
            self.web_order_mode = next_order['web_order'] is not None
            self.current_web_order = next_order['web_order']
            self.is_package_delivery = next_order['is_package']
            
            self.start_delivery(next_order['from'], next_order['to'])
            return True
        return False
    
    def check_weather_safety(self):
        global weather_safe
        
        if not weather_safe:
            if self.state not in ["idle", "charging"]:
                print(f"⚠️ WEATHER UNSAFE: Cancelling delivery")
                
                if self.web_order_mode and self.current_web_order and FLASK_ENABLED:
                    update_current_order_status('failed', f'Cancelled - {weather} too severe (Wind: {wind_speed:.1f}). We apologize for the inconvenience.')
                
                self.trigger_emergency()
    
    def check_web_orders(self):
        if not FLASK_ENABLED or self.state != "idle":
            return
        
        order = get_next_order()
        if order:
            if not weather_safe:
                print(f"⚠️ Order received but weather unsafe - cancelling")
                if FLASK_ENABLED:
                    from flask_server import active_orders
                    if order in active_orders:
                        idx = active_orders.index(order)
                        active_orders[idx]['status'] = 'failed'
                        active_orders[idx]['state'] = f'Cancelled - {weather} too severe. We apologize.'
                return
            
            # Check if it's a package delivery (building to building)
            if order['product'].get('isPackage', False):
                store_building = order['product']['store']
                customer_building = order['product']['destination']
                
                print(f"\n📦 PACKAGE DELIVERY ORDER!")
                print(f"   From: Building {store_building}")
                print(f"   To: Building {customer_building}")
                
                is_package = True
            else:
                store_building = order['product']['store']
                customer_building = order['location']
                
                print(f"\n🌐 WEB ORDER RECEIVED!")
                print(f"   Item: {order['product']['name']}")
                print(f"   From: {store_building}")
                print(f"   To: {customer_building}")
                
                is_package = False
            
            # Add web orders to queue with full order info
            self.add_to_queue(store_building, customer_building, is_manual=False, 
                            web_order=order, is_package=is_package)
            
            if FLASK_ENABLED:
                if is_package:
                    update_current_order_status('processing', f'📦 Queued for delivery')
                else:
                    update_current_order_status('processing', 'Queued for delivery')
    
    def start_delivery(self, fs, ts):
        if self.state == "charging":
            print("Cannot start while charging!")
            return
        self.from_s = fs
        self.to_s = ts
        self.emergency_triggered = False
        
        # For manual deliveries, go directly to destination
        if self.manual_delivery_mode:
            print(f"Direct delivery: {fs} → {ts}")
            self.state = "plan_deliver"
        else:
            self.state = "plan_pickup"
    
    def trigger_emergency(self):
        if self.state != "idle" and self.state != "charging":
            self.emergency_triggered = True
            
            if self.web_order_mode and self.current_web_order and FLASK_ENABLED:
                update_current_order_status('failed', '⚠️ Emergency - Order cancelled')
            
            self.state = "emergency_return"
            print("⚠️ EMERGENCY!")
    
    def start_charging(self):
        home = building_positions["A"] + vector(0, building_heights["A"]+5, 0)
        if mag(drone["pos"] - home) < 5 and self.state == "idle":
            self.state = "charging"
            self.charge_timer = 0
            print("🔋 Charging...")
    
    def check_path_blocked(self):
        if len(self.path) == 0 or self.wp >= len(self.path):
            return False
        
        check_distance = min(3, len(self.path) - self.wp)
        
        for i in range(check_distance):
            waypoint = self.path[self.wp + i]
            for bird in birds:
                if not bird.active:
                    continue
                if mag(waypoint - bird.body.pos) < 12:
                    return True
        
        return False
    
    def replan_path(self, goal):
        print("REROUTING!")
        self.path = astar_pathfind(drone["pos"], goal, buildings, birds)
        visualize_path(self.path)
        self.wp = 0
        self.replan_cooldown = 120
    
    def update(self):
        if self.replan_cooldown > 0:
            self.replan_cooldown -= 1
        
        if drone["battery"] < 15 and self.state not in ["idle", "emergency_return", "emergency_descend", "charging"]:
            self.trigger_emergency()

        if self.state == "idle":
            return
        
        if self.state in ["fly_pickup", "fly_deliver", "return_home", "emergency_return"]:
            if self.replan_cooldown == 0:
                if self.check_path_blocked():
                    if self.current_goal:
                        self.replan_path(self.current_goal)
        
        if self.state == "charging":
            self.charge_timer += 1
            if drone["battery"] < 100:
                drone["battery"] += 0.08
                if drone["battery"] > 100:
                    drone["battery"] = 100
            else:
                if self.charge_timer > 120:
                    print("Charged!")
                    self.state = "idle"
                    drone["package"].color = color.orange
                    if self.web_order_mode and FLASK_ENABLED:
                        self.web_order_mode = False
                        self.current_web_order = None
                        self.is_package_delivery = False
            return

        if self.state == "plan_pickup":
            self.current_goal = building_positions[self.from_s] + vector(0, building_heights[self.from_s]+15, 0)
            if self.web_order_mode and FLASK_ENABLED:
                if self.is_package_delivery:
                    update_current_order_status('processing', f'📦 Flying to {self.from_s}')
                else:
                    update_current_order_status('processing', f'Flying to {self.from_s}')
            self.path = astar_pathfind(drone["pos"], self.current_goal, buildings, birds)
            visualize_path(self.path)
            self.wp = 0
            self.state = "fly_pickup"
        
        elif self.state == "fly_pickup":
            if self.wp < len(self.path):
                if move_drone(drone, self.path[self.wp]):
                    self.wp += 1
            else:
                clear_path_markers()
                if self.web_order_mode and FLASK_ENABLED:
                    if self.is_package_delivery:
                        update_current_order_status('processing', f'📦 Landing at {self.from_s}')
                    else:
                        update_current_order_status('processing', 'Landing at store')
                self.landing_height = building_heights[self.from_s] + 2
                self.current_goal = None
                self.state = "descend_pickup"
        
        elif self.state == "descend_pickup":
            target_pos = building_positions[self.from_s] + vector(0, self.landing_height, 0)
            if move_drone(drone, target_pos, speed=3):
                if self.web_order_mode and FLASK_ENABLED:
                    if self.is_package_delivery:
                        update_current_order_status('processing', f'📦 Picking up package from {self.from_s}')
                    else:
                        item_name = self.current_web_order['product']['name'] if self.current_web_order else 'package'
                        update_current_order_status('processing', f'Picking up {item_name}')
                self.hover_timer = 90
                self.state = "hover_pickup"
        
        elif self.state == "hover_pickup":
            self.hover_timer -= 1
            if self.hover_timer <= 0:
                if self.web_order_mode and FLASK_ENABLED:
                    if self.is_package_delivery:
                        update_current_order_status('processing', '📦 Package secured')
                    else:
                        update_current_order_status('processing', 'Package loaded')
                self.state = "ascend_pickup"
        
        elif self.state == "ascend_pickup":
            target_pos = building_positions[self.from_s] + vector(0, building_heights[self.from_s]+15, 0)
            if move_drone(drone, target_pos, speed=5):
                if self.web_order_mode and FLASK_ENABLED:
                    if self.is_package_delivery:
                        update_current_order_status('processing', f'📦 En route to {self.to_s}')
                    else:
                        update_current_order_status('processing', f'En route to {self.to_s}')
                self.state = "plan_deliver"

        elif self.state == "plan_deliver":
            self.current_goal = building_positions[self.to_s] + vector(0, building_heights[self.to_s]+15, 0)
            if self.web_order_mode and FLASK_ENABLED:
                if self.is_package_delivery:
                    update_current_order_status('processing', f'📦 Flying to {self.to_s}')
                else:
                    update_current_order_status('processing', f'Flying to {self.to_s}')
            
            # For manual deliveries, set from_s to current drone position building
            if self.manual_delivery_mode:
                # Find closest building to drone's current position (usually warehouse A)
                current_pos = drone["pos"]
                min_dist = float('inf')
                closest_building = "A"
                for bname, bpos in building_positions.items():
                    dist = mag(current_pos - bpos)
                    if dist < min_dist:
                        min_dist = dist
                        closest_building = bname
                self.from_s = closest_building
            
            self.path = astar_pathfind(drone["pos"], self.current_goal, buildings, birds)
            visualize_path(self.path)
            self.wp = 0
            self.state = "fly_deliver"

        elif self.state == "fly_deliver":
            if self.wp < len(self.path):
                if move_drone(drone, self.path[self.wp], slow_zone=(self.wp == len(self.path)-1)):
                    self.wp += 1
            else:
                clear_path_markers()
                if self.web_order_mode and FLASK_ENABLED:
                    if self.is_package_delivery:
                        update_current_order_status('processing', f'📦 Landing at {self.to_s}')
                    else:
                        update_current_order_status('processing', 'Landing')
                self.landing_height = building_heights[self.to_s] + 2
                self.current_goal = None
                self.state = "descend_deliver"
        
        elif self.state == "descend_deliver":
            target_pos = building_positions[self.to_s] + vector(0, self.landing_height, 0)
            if move_drone(drone, target_pos, speed=3):
                if self.web_order_mode and FLASK_ENABLED:
                    if self.is_package_delivery:
                        update_current_order_status('processing', f'📦 Delivering to {self.to_s}')
                    else:
                        update_current_order_status('processing', 'Delivering')
                self.hover_timer = 90
                self.state = "drop_package"
        
        elif self.state == "drop_package":
            self.hover_timer -= 1
            if self.hover_timer == 60:
                drone["package"].color = vector(0.8, 0.6, 0.3)
            elif self.hover_timer == 30:
                drone["package"].color = color.gray(0.7)
            elif self.hover_timer <= 0:
                drone["package"].color = color.gray(0.5)
                if self.web_order_mode and FLASK_ENABLED:
                    if self.is_package_delivery:
                        update_current_order_status('processing', f'Package delivered to {self.to_s}!')
                    else:
                        update_current_order_status('processing', 'Delivered!')
                self.state = "ascend_deliver"
        
        elif self.state == "ascend_deliver":
            target_pos = building_positions[self.to_s] + vector(0, building_heights[self.to_s]+15, 0)
            if move_drone(drone, target_pos, speed=5):
                if self.web_order_mode and FLASK_ENABLED:
                    update_current_order_status('processing', 'Returning')
                self.state = "return_home"
                self.wp = 0

        elif self.state == "return_home":
            self.current_goal = building_positions["A"] + vector(0, building_heights["A"]+15, 0)
            if self.wp == 0:
                self.path = astar_pathfind(drone["pos"], self.current_goal, buildings, birds)
                visualize_path(self.path)
                self.wp = 1
            
            if self.wp > 0 and self.wp <= len(self.path):
                if move_drone(drone, self.path[self.wp-1], slow_zone=(self.wp == len(self.path))):
                    self.wp += 1
            
            if self.wp > len(self.path):
                clear_path_markers()
                self.landing_height = building_heights["A"] + 5
                self.current_goal = None
                self.state = "descend_home"
        
        elif self.state == "descend_home":
            target_pos = building_positions["A"] + vector(0, self.landing_height, 0)
            if move_drone(drone, target_pos, speed=3):
                drone["package"].color = color.orange
                self.wp = 0
                
                if self.web_order_mode and FLASK_ENABLED:
                    if self.is_package_delivery:
                        update_current_order_status('completed', 'Package delivered successfully!')
                    else:
                        update_current_order_status('completed', 'Delivered!')
                
                # Reset all flags before processing next order
                self.web_order_mode = False
                self.current_web_order = None
                self.is_package_delivery = False
                self.manual_delivery_mode = False
                
                print("✅ Complete!")
                self.state = "idle"
                
                # Process next order from queue if available
                if not self.process_next_order():
                    print(f"📭 Queue empty")

        elif self.state == "emergency_return":
            self.current_goal = building_positions["A"] + vector(0, building_heights["A"]+15, 0)
            if self.wp == 0:
                if self.web_order_mode and FLASK_ENABLED:
                    update_current_order_status('failed', 'Emergency return')
                self.path = astar_pathfind(drone["pos"], self.current_goal, buildings, birds)
                visualize_path(self.path)
                self.wp = 1
            
            if self.wp > 0 and self.wp <= len(self.path):
                if move_drone(drone, self.path[self.wp-1], speed=12, slow_zone=(self.wp == len(self.path))):
                    self.wp += 1
            
            if self.wp > len(self.path):
                clear_path_markers()
                self.landing_height = building_heights["A"] + 5
                self.current_goal = None
                self.state = "emergency_descend"
        
        elif self.state == "emergency_descend":
            target_pos = building_positions["A"] + vector(0, self.landing_height, 0)
            if move_drone(drone, target_pos, speed=4):
                self.wp = 0
                
                if self.web_order_mode and FLASK_ENABLED:
                    update_current_order_status('failed', 'Emergency complete')
                    self.web_order_mode = False
                    self.current_web_order = None
                    self.is_package_delivery = False
                
                self.state = "charging"
                self.charge_timer = 0

delivery = DeliverySystem()

# DAY/NIGHT TOGGLE

def toggle_daynight(b):
    global is_night
    is_night = not is_night
    
    if is_night:
        scene.background = vector(0.05,0.05,0.2)
        for w in windows:
            w.color = color.yellow
            w.opacity = 0.9
        for s in stars:
            s.visible = True
        for bird in birds:
            bird.set_visible(False)
    else:
        set_weather(weather)
        for w in windows:
            w.color = color.white
            w.opacity = 0.3
        for s in stars:
            s.visible = False
        for bird in birds:
            bird.set_visible(True)

# UI CONTROLS

scene.append_to_caption("\n=== CONTROL PANEL ===\n")

scene.append_to_caption("From: ")
from_choice = "B"
def set_from(m): 
    global from_choice
    from_choice = m.selected
menu(choices=list(building_positions.keys()), bind=set_from)

scene.append_to_caption(" To: ")
to_choice = "F"
def set_to(m): 
    global to_choice
    to_choice = m.selected
menu(choices=list(building_positions.keys()), bind=set_to)

scene.append_to_caption("\n")

button(text="🚁 Start Delivery", bind=lambda b: delivery.add_to_queue(from_choice, to_choice, is_manual=True, web_order=None, is_package=False))
button(text="⚠️ Emergency", bind=lambda b: delivery.trigger_emergency())
button(text="🔋 Charge", bind=lambda b: delivery.start_charging())
button(text="🌙 Day/Night", bind=toggle_daynight)

scene.append_to_caption("\n\n")

def add_test_bird(b):
    if len(birds) < 15:
        test_pos = drone["pos"] + vector(20, random.uniform(-5,5), random.uniform(-10,10))
        new_bird = Bird(test_pos)
        birds.append(new_bird)
        if is_night:
            new_bird.set_visible(False)
        print(f"🐦 Bird added! Total: {len(birds)}")

button(text="🐦 Add Bird", bind=add_test_bird)

scene.append_to_caption("\n")
scene.append_to_caption("Weather: ")
button(text="☀️ Clear", bind=lambda b: set_weather("Clear"))
button(text="🌧️ Rain", bind=lambda b: set_weather("Rain"))
button(text="💨 Wind", bind=lambda b: set_weather("Wind"))
button(text="⛈️ Storm", bind=lambda b: set_weather("Storm"))

scene.append_to_caption("\n\n")
status_label = label(pos=vector(0,85,0), box=False, height=14, color=color.white)
battery_warning = label(pos=vector(0,80,0), box=False, height=12, color=color.red, visible=False)

# STARTING FLASK SERVER

if FLASK_ENABLED:
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    time.sleep(2)
    print("\n" + "="*60)
    print("🌐 WEB INTERFACE READY!")
    print("📱 Open: http://localhost:5000")
    print("="*60 + "\n")

# MAIN LOOP

blink = 0
while True:
    rate(60)

    if FLASK_ENABLED and delivery.state == "idle":
        delivery.check_web_orders()
    
    # Process queued orders when idle
    if delivery.state == "idle" and len(delivery.order_queue) > 0:
        delivery.process_next_order()

    # Protection bubble
    if weather in ["Rain", "Storm"]:
        if delivery.state not in ["idle", "charging"]:
            create_protection_bubble()
            update_protection_bubble()
        else:
            hide_protection_bubble()
    else:
        hide_protection_bubble()
    
    # Wind particles
    for particle in wind_particles:
        particle.pos += wind_dir * wind_speed * 0.3
        if abs(particle.pos.x) > 170 or abs(particle.pos.z) > 170:
            particle.pos = vector(random.uniform(-140, 140), random.uniform(5, 40), random.uniform(-140, 140))
    
    # Lightning
    if weather == "Storm":
        lightning_timer += 1
        if lightning_flash_active and lightning_timer > 5:
            scene.background = vector(0.2,0.2,0.3)
            lightning_flash_active = False
        elif not lightning_flash_active and lightning_timer > random.randint(120, 300):
            create_lightning()
            lightning_timer = 0
    
    # Rain
    for r in rain_particles:
        r.pos.y -= 3 if weather == "Storm" else 2
        if r.pos.y < 0:
            r.pos.y = random.uniform(60, 120)
            if weather in ["Storm", "Wind"]:
                r.pos.x += wind_dir.x * wind_speed * 0.1
                r.pos.z += wind_dir.z * wind_speed * 0.1

    # Birds 
    for b in birds:
        if b.active and b.body.visible:
            b.update()

    delivery.update()

    # Battery warning
    if drone["battery"] < 20 and drone["battery"] > 0:
        battery_warning.visible = True
        battery_warning.text = "⚠️ LOW BATTERY!"
    else:
        battery_warning.visible = False

    # Status
    state_emoji = {
        "idle": "⏸️", "charging": "🔋", "plan_pickup": "📍", "fly_pickup": "🛫",
        "descend_pickup": "⬇️", "hover_pickup": "⏱️", "ascend_pickup": "⬆️",
        "plan_deliver": "📍", "fly_deliver": "📦", "descend_deliver": "⬇️",
        "drop_package": "📭", "ascend_deliver": "⬆️", "return_home": "🏠",
        "descend_home": "⬇️", "emergency_return": "⚠️", "emergency_descend": "⚠️⬇️"
    }
    
    emoji = state_emoji.get(delivery.state, "")
    reroute = " 🔄" if delivery.replan_cooldown > 0 else ""
    weather_warn = "" if weather_safe else " ⚠️ UNSAFE"
    protection = " 🛡️" if (protection_bubble and protection_bubble.visible) else ""
    package_mode = " 📦PKG" if delivery.is_package_delivery else ""
    queue_info = f" | Queue: {len(delivery.order_queue)}" if len(delivery.order_queue) > 0 else ""
    
    status_label.text = f"{emoji} {delivery.state.upper()}{reroute}{weather_warn}{protection}{package_mode}{queue_info} | {weather} ({wind_speed:.1f}) | Battery: {drone['battery']:.1f}%"

    blink += 1
    if blink % 20 == 0:
        nav_light.visible = not nav_light.visible
    nav_light.pos = drone["body"].pos + vector(0,1.2,0)